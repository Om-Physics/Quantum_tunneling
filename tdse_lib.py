"""
tdse_lib.py -- core routines for 1D wave-packet dynamics (units: eV, nm, fs; electron mass).

    hbar          = 0.6582119569 eV fs
    hbar^2/(2 m)  = 0.0380998212 eV nm^2
"""
import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla
import scipy.linalg as sla

HBAR = 0.6582119569        # eV fs
CKIN = 0.0380998212        # hbar^2/(2 m_e)  [eV nm^2]
H_PLANCK = 4.135667696     # eV fs

# half-stencils c_0, c_1, ..., c_r of the central second-derivative formula
HALF_STENCIL = {
    2: [-2.0, 1.0],
    4: [-5/2, 4/3, -1/12],
    6: [-49/18, 3/2, -3/20, 1/90],
    8: [-205/72, 8/5, -1/5, 8/315, -1/560],
}

# ----------------------------------------------------------------------------------
# operators
# ----------------------------------------------------------------------------------
def d2_matrix(N, dx, order=8):
    """Sparse second-derivative matrix, Dirichlet (psi=0 outside the grid)."""
    c = HALF_STENCIL[order]
    diags, offs = [np.full(N, c[0])], [0]
    for p in range(1, len(c)):
        diags += [np.full(N - p, c[p])] * 2
        offs += [p, -p]
    return sp.diags(diags, offs, format="csr") / dx**2

def hamiltonian(x, V, order=8):
    dx = x[1] - x[0]
    T = -CKIN * d2_matrix(len(x), dx, order)
    return (T + sp.diags(np.asarray(V, dtype=complex))).tocsc()

def build_propagator(H, dt, M):
    """A^{-1} B with A = sum_{n<=M} (i s H)^n/n!, B = sum_{n<=M} (-i s H)^n/n!, s = dt/(2 hbar)."""
    N = H.shape[0]
    s = dt / (2.0 * HBAR)
    X = (1j * s) * H
    P = sp.identity(N, format="csc", dtype=complex)
    A, B = P.copy(), P.copy()
    for n in range(1, M + 1):
        P = (P @ X) / n
        A = A + P
        B = B + ((-1) ** n) * P
    lu = spla.splu(A.tocsc(), permc_spec="NATURAL")     # NATURAL keeps the band structure
    return lu, B.tocsr()

def gaussian(x, x0, sigma, k0):
    return (2 * np.pi * sigma**2) ** (-0.25) * np.exp(-(x - x0) ** 2 / (4 * sigma**2) + 1j * k0 * x)

def cap_potential(x, eta, L, n=2):
    """Complex absorbing potential -i*eta*((|x|-x_c)/L)^n on both edges (x_c = x_max - L)."""
    xc = np.abs(x).max() - L
    d = np.clip((np.abs(x) - xc) / L, 0.0, None)
    return -1j * eta * d**n

# ----------------------------------------------------------------------------------
# potentials
# ----------------------------------------------------------------------------------
def v_double_barrier(x, V0=1.0, w=0.5, d=0.5):
    ax = np.abs(x)
    V = np.zeros_like(x)
    V[(ax > d / 2) & (ax < d / 2 + w)] = V0
    return V

def v_double_well(x, a=1.0, b=0.4, Vb=1.0, Vw=5.0):
    ax = np.abs(x)
    V = np.full_like(x, Vw)
    V[ax < b / 2 + a] = 0.0
    V[ax < b / 2] = Vb
    return V

# ----------------------------------------------------------------------------------
# propagation
# ----------------------------------------------------------------------------------
def propagate(x, V, psi0, dt, nsteps, M=8, order=8, rec=10, obs=5, masks=None):
    dx = x[1] - x[0]
    H = hamiltonian(x, V, order)
    lu, B = build_propagator(H, dt, M)
    psi = psi0.astype(complex).copy()
    masks = masks or {}
    out = {"t_rec": [], "rho": [], "t_obs": [], "norm": [], "psi_final": None}
    for k in masks:
        out[k] = []
    for n in range(nsteps + 1):
        if n % rec == 0:
            out["t_rec"].append(n * dt)
            out["rho"].append(np.abs(psi) ** 2)
        if n % obs == 0:
            r = np.abs(psi) ** 2
            out["t_obs"].append(n * dt)
            out["norm"].append(r.sum() * dx)
            for k, m in masks.items():
                out[k].append(r[m].sum() * dx)
        if n < nsteps:
            psi = lu.solve(B @ psi)
    for k in ("t_rec", "rho", "t_obs", "norm") + tuple(masks):
        out[k] = np.array(out[k])
    out["psi_final"] = psi
    return out

# ----------------------------------------------------------------------------------
# exact references
# ----------------------------------------------------------------------------------
def transmission_exact(E, layers):
    """Exact T(E) for piecewise-constant layers [(width, V), ...] embedded in V=0 (same mass)."""
    E = np.atleast_1d(np.asarray(E, dtype=float))
    out = np.empty_like(E)
    for i, e in enumerate(E):
        k = np.sqrt(e / CKIN + 0j)
        M = np.eye(2, dtype=complex)
        for wdt, V in layers:
            q = np.sqrt((e - V) / CKIN + 0j)
            if abs(q) < 1e-12:
                q = 1e-12
            Ml = np.array([[np.cos(q * wdt), np.sin(q * wdt) / q],
                           [-q * np.sin(q * wdt), np.cos(q * wdt)]])
            M = Ml @ M
        # left of x=0: e^{ikx} + r e^{-ikx};  right of x=W: t e^{ikx}; take x_L = 0, x_R = W -> phases cancel in |t|^2
        W = sum(l[0] for l in layers)
        # unknowns (r, t'):  M [1+r, ik(1-r)]^T = [t' , ik t']^T,  t' = t e^{ikW}
        # row1: M00 (1+r) + M01 ik (1-r) - t' = 0
        # row2: M10 (1+r) + M11 ik (1-r) - ik t' = 0
        a = np.array([[M[0, 0] - 1j * k * M[0, 1], -1.0],
                      [M[1, 0] - 1j * k * M[1, 1], -1j * k]])
        rhs = -np.array([M[0, 0] + 1j * k * M[0, 1], M[1, 0] + 1j * k * M[1, 1]])
        r, tp = np.linalg.solve(a, rhs)
        out[i] = abs(tp) ** 2
    return out

def lowest_eigs(x, V, nev=4, order=8):
    """Lowest eigenpairs of a real, Hermitian H using the banded eigen-solver."""
    dx = x[1] - x[0]
    c = HALF_STENCIL[order]
    r = len(c) - 1
    ab = np.zeros((r + 1, len(x)))
    ab[0] = -CKIN * c[0] / dx**2 + np.real(V)
    for p in range(1, r + 1):
        ab[p, : len(x) - p] = -CKIN * c[p] / dx**2
    w, v = sla.eig_banded(ab, lower=True, select="i", select_range=(0, nev - 1))
    v = v / np.sqrt(dx)          # sum |v|^2 dx = 1
    return w, v
