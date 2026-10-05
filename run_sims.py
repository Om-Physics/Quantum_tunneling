#!/usr/bin/env python3
import numpy as np, pickle, json, time, sys, traceback
from scipy.optimize import minimize_scalar, brentq
from tdse_lib import *

t0 = time.time()
def log(*a): print(f"[{time.time()-t0:7.1f}s]", *a, flush=True)

DX = 0.02
def grid(L, dx=DX):                      # grid points at integer multiples of dx
    J = int(round(L / dx)); return dx * np.arange(-J, J + 1)
def grid_half(L, dx=DX):                 # staggered grid (points at odd multiples of dx/2)
    J = int(round(L / dx)); return dx * (np.arange(-J, J) + 0.5)

R, J = {}, {}
CAP_ETA, CAP_L = 0.5, 10.0
V0, W, D = 1.0, 0.5, 0.5
LAYERS = [(W, V0), (D, 0.0), (W, V0)]

# ------------------------------------------------------------------ helpers
def probe_run(x, V, psi0, dt, nsteps, idx, M=8, order=8):
    H = hamiltonian(x, V, order); lu, B = build_propagator(H, dt, M)
    psi = psi0.astype(complex); sig = np.zeros((len(idx), nsteps + 1), complex)
    for n in range(nsteps + 1):
        sig[:, n] = psi[idx]
        if n < nsteps: psi = lu.solve(B @ psi)
    return sig

def ft(sig, dt, E):
    t = np.arange(sig.shape[-1]) * dt
    out = np.empty((sig.shape[0], len(E)), complex)
    for i in range(0, len(E), 100):
        ph = np.exp(1j * np.outer(E[i:i + 100], t) / HBAR) * dt
        out[:, i:i + 100] = (ph @ sig.T).T
    return out

def taper(n, dt, frac=0.5):
    """Unity until frac*T_total, then a raised-cosine decay to zero: suppresses spectral leakage of long-lived ringing."""
    t = np.arange(n) * dt; t1 = frac * t[-1]; w = np.ones(n); m = t > t1
    w[m] = 0.5 * (1 + np.cos(np.pi * (t[m] - t1) / (t[-1] - t1))); return w

def T_from_probes(sw, sf, dt, E):
    """T(E)=|F[psi_with(x_d)]|^2/|F[psi_free(x_d)]|^2 ; R(E) from the scattered field at x_l."""
    wn = taper(sw.shape[1], dt); sw, sf = sw * wn, sf * wn
    Fw, Ff, Fs = ft(sw, dt, E), ft(sf, dt, E), ft(sw - sf, dt, E)
    return abs(Fw[1]) ** 2 / abs(Ff[1]) ** 2, abs(Fs[0]) ** 2 / abs(Ff[0]) ** 2

# ------------------------------------------------------------------ 0. resonance (exact)
def sec0():
    f = lambda E: -transmission_exact(E, LAYERS)[0]
    Er = minimize_scalar(f, bounds=(0.40, 0.48), method="bounded", options={"xatol": 1e-11}).x
    g = lambda E: transmission_exact(E, LAYERS)[0] - 0.5
    a = brentq(g, Er - 0.05, Er); b = brentq(g, Er, Er + 0.05)
    R["Er"], R["Gamma"], R["tau_exact"] = Er, b - a, HBAR / (b - a)
    # above-barrier resonance
    f2 = lambda E: -transmission_exact(E, LAYERS)[0]
    R["Er2"] = minimize_scalar(f2, bounds=(1.35, 1.5), method="bounded", options={"xatol": 1e-11}).x
    R["Eoff"] = 0.70
    R["T_off_exact"] = float(transmission_exact(0.70, LAYERS)[0])
    log("resonance", R["Er"], "Gamma", R["Gamma"], "tau", R["tau_exact"], "Er2", R["Er2"])

# ------------------------------------------------------------------ 1. numerical validation
def sec1():
    # (a) spatial convergence: harmonic oscillator, E0 = 0.5 eV exactly
    Vho = 0.25 / CKIN
    dxs = np.array([0.2, 0.15, 0.1, 0.075, 0.05, 0.04, 0.03, 0.02])
    err = {o: [] for o in (2, 4, 6, 8)}
    for o in err:
        for dx in dxs:
            n = int(round(4 / dx)); x = dx * np.arange(-n, n + 1)
            w, _ = lowest_eigs(x, Vho * x**2, 1, order=o)
            err[o].append(abs(w[0] - 0.5))
    R["sp_dx"], R["sp_err"] = dxs, {o: np.array(v) for o, v in err.items()}
    # (b) temporal convergence against exact spectral propagation of the same discrete H
    #     (coarse grid dx=0.1 nm keeps cond(A) small, so the roundoff floor lies far below the truncation error)
    dx = 0.1; x = dx * np.arange(-100, 101)
    st = lambda u: 0.5 * (1 + np.tanh(u / 0.25))          # smooth (tanh) edges: removes edge-generated high-k content
    V = V0 * (st(abs(x) - D / 2) - st(abs(x) - D / 2 - W)); psi0 = gaussian(x, -4.0, 0.5, 3.4)
    H = hamiltonian(x, V).toarray().real
    E, U = np.linalg.eigh(H)
    Tf = 16.0
    ref = U @ (np.exp(-1j * E * Tf / HBAR) * (U.T @ psi0))
    dts = [0.8, 0.4, 0.2, 0.1, 0.05, 0.025]
    terr = {}
    for M in (1, 2, 4, 6, 8):
        terr[M] = []
        for dt in dts:
            out = propagate(x, V, psi0, dt, int(round(Tf / dt)), M=M, rec=10**9, obs=10**9)
            terr[M].append(np.sqrt(np.sum(abs(out["psi_final"] - ref) ** 2) * dx))
        terr[M] = np.array(terr[M])
    R["dts"], R["terr"] = np.array(dts), terr
    # (c) norm drift (no CAP, reflecting walls) and decay with CAP
    nd = {}
    for M in (1, 4, 8):
        out = propagate(x, V, psi0, 0.1, 20000, M=M, rec=10**9, obs=100)
        nd[M] = (out["t_obs"], out["norm"])
    outc = propagate(x, V + cap_potential(x, CAP_ETA, 3.0), psi0, 0.1, 20000, M=8, rec=10**9, obs=100)
    R["norm_drift"], R["norm_cap"] = nd, (outc["t_obs"], outc["norm"])
    # production-grid norm drift (dx=0.02, dt=0.05, no CAP)
    xp = grid(30); o = propagate(xp, v_double_barrier(xp), gaussian(xp, -12, 0.3, 4.0), 0.05, 8000, M=8, rec=10**9, obs=200)
    J["norm_drift_production"] = float(abs(o["norm"] - 1).max())
    log("sec1 done; temporal errors M=8:", terr[8])

# ------------------------------------------------------------------ 2. CAP characterisation
def sec2():
    x = grid(30); cap = cap_potential(x, CAP_ETA, CAP_L)
    res = {}
    for k0 in (1.5, 3.0, 5.0):
        v = 0.1158 * k0; Tt = (25 + 40) / v; n = int(Tt / 0.05)
        out = propagate(x, cap, gaussian(x, 0, 1.0, k0), 0.05, n, rec=10**9, obs=n)
        res[k0] = float(out["norm"][-1])
    R["cap_refl"] = res; log("CAP residual norm", res)

# ------------------------------------------------------------------ 3. T(E) curve and sweeps (time-domain probes)
def sec3():
    x = grid(30); cap = cap_potential(x, CAP_ETA, CAP_L)
    dt = 0.05; nst = int(450 / dt)
    psi0 = gaussian(x, -12, 0.3, 4.0)
    idx = [np.argmin(abs(x + 6)), np.argmin(abs(x - 6))]
    sf = probe_run(x, cap, psi0, dt, nst, idx)
    sw = probe_run(x, cap + v_double_barrier(x, V0, W, D), psi0, dt, nst, idx)
    E = np.linspace(0.15, 1.9, 176)
    Tn, Rn = T_from_probes(sw, sf, dt, E)
    Tx = transmission_exact(E, LAYERS)
    R["TE"] = dict(E=E, Tn=Tn, Rn=Rn, Tx=Tx)
    # dense exact curves
    Ed = np.linspace(0.02, 2.2, 1500)
    R["TE_dense"] = dict(E=Ed, T=transmission_exact(Ed, LAYERS),
                         T1=transmission_exact(Ed, [(W, V0)]))
    J["TE_max_abs_err"] = float(abs(Tn - Tx).max()); J["TE_rms_err"] = float(np.sqrt(np.mean((Tn - Tx) ** 2)))
    J["TE_unitarity_max"] = float(abs(Tn + Rn - 1).max())
    log("T(E) max err", J["TE_max_abs_err"], "T+R-1", J["TE_unitarity_max"])
    # sweeps at E = 0.5 eV
    Ec = np.array([0.5])
    def Tat(V):
        sw_ = probe_run(x, cap + V, psi0, dt, int(300 / dt), idx); n = sw_.shape[1]; wn = taper(n, dt)
        return float(abs(ft(sw_ * wn, dt, Ec)[1, 0]) ** 2 / abs(ft(sf[:, :n] * wn, dt, Ec)[1, 0]) ** 2)
    ws = np.round(np.arange(0.1, 1.01, 0.1), 2)
    Tw_n = np.array([Tat(v_double_barrier(x, V0, w_, D)) for w_ in ws])
    Tw_x = np.array([transmission_exact(0.5, [(w_, V0), (D, 0), (w_, V0)])[0] for w_ in ws])
    T1_x = np.array([transmission_exact(0.5, [(w_, V0)])[0] for w_ in ws])
    R["sweep_w"] = dict(w=ws, Tn=Tw_n, Tx=Tw_x, T1=T1_x); log("width sweep", Tw_n, Tw_x)
    V0s = np.round(np.arange(0.6, 3.01, 0.2), 2)
    # finer grid (dx = 0.5/49 nm, edges again on half-points) because T(V0) at fixed E crosses a sharp resonance
    dxf = 0.5 / 49; x = grid(30, dxf); cap = cap_potential(x, CAP_ETA, CAP_L)
    psi0 = gaussian(x, -12, 0.3, 4.0); idx = [np.argmin(abs(x + 6)), np.argmin(abs(x - 6))]
    sf = probe_run(x, cap, psi0, dt, nst, idx)
    Tv_n = np.array([Tat(v_double_barrier(x, v_, W, D)) for v_ in V0s])
    Tv_x = np.array([transmission_exact(0.5, [(W, v_), (D, 0), (W, v_)])[0] for v_ in V0s])
    T1v = np.array([transmission_exact(0.5, [(W, v_)])[0] for v_ in V0s])
    R["sweep_V"] = dict(V=V0s, Tn=Tv_n, Tx=Tv_x, T1=T1v); log("height sweep", Tv_n, Tv_x)

# ------------------------------------------------------------------ 4. dx / stencil convergence of the scattering problem
def sec4():
    out = {}
    dt = 0.05; nst = int(300 / dt); Ec = np.array([0.5])
    for order in (2, 8):
        out[order] = []
        for n in (2, 4, 8, 12, 24):
            dx = 0.5 / (2 * n + 1)
            x = grid(30, dx); cap = cap_potential(x, CAP_ETA, CAP_L)
            psi0 = gaussian(x, -12, 0.3, 4.0); idx = [np.argmin(abs(x - 6))]
            sf = probe_run(x, cap, psi0, dt, nst, idx, order=order)
            sw = probe_run(x, cap + v_double_barrier(x, V0, W, D), psi0, dt, nst, idx, order=order)
            wn = taper(sw.shape[1], dt)
            Tn = abs(ft(sw * wn, dt, Ec)[0, 0]) ** 2 / abs(ft(sf * wn, dt, Ec)[0, 0]) ** 2
            out[order].append((dx, float(Tn)))
            log("dx conv", order, dx, Tn)
    R["dx_conv"] = out; R["T05_exact"] = float(transmission_exact(0.5, LAYERS)[0])

# ------------------------------------------------------------------ 5. double barrier: wave-packet dynamics
def run_packet(sigma, k0, x0, Lx, Lcap, Tfin, rec=20, obs=5, eta=CAP_ETA):
    x = grid(Lx); V = v_double_barrier(x, V0, W, D)
    Vt = V + cap_potential(x, eta, Lcap)
    masks = {"left": x < -0.75, "struct": abs(x) <= 0.75, "well": abs(x) <= 0.25, "right": x > 0.75}
    out = propagate(x, Vt, gaussian(x, x0, sigma, k0), 0.05, int(Tfin / 0.05), rec=rec, obs=obs, masks=masks)
    out["x"], out["V"] = x, V
    return out

def sec5():
    kr = np.sqrt(R["Er"] / CKIN); ko = np.sqrt(R["Eoff"] / CKIN)
    R["dyn_res"] = run_packet(5.0, kr, -25.0, 115, 12, 200.0)
    log("res run", R["dyn_res"]["right"][-1], R["dyn_res"]["left"][-1])
    R["dyn_off"] = run_packet(5.0, ko, -25.0, 115, 12, 200.0)
    log("off run", R["dyn_off"]["right"][-1], R["dyn_off"]["left"][-1])
    for nm in ("dyn_res", "dyn_off"):
        o = R[nm]
        J[nm] = dict(T_final=float(o["right"][-1]), R_final=float(o["left"][-1]),
                     struct_final=float(o["struct"][-1]), absorbed=float(1 - o["norm"][-1]),
                     max_well=float(o["well"].max()), max_struct=float(o["struct"].max()))
    # mean-energy transmission expected for the packet (energy-weighted exact T)
    for nm, k in (("dyn_res", kr), ("dyn_off", ko)):
        s = 5.0; Ek = np.linspace(0.02, 2.0, 4000); kk = np.sqrt(Ek / CKIN)
        wgt = np.exp(-2 * s**2 * (kk - k) ** 2) / np.sqrt(Ek)    # |phi(k)|^2 dk/dE
        Tex = transmission_exact(Ek, LAYERS)
        J[nm]["T_expected"] = float(np.sum(wgt * Tex) / np.sum(wgt))
    log("expected", J["dyn_res"]["T_expected"], J["dyn_off"]["T_expected"])
    # lifetime run with short packet
    o = run_packet(1.5, kr, -12.0, 40, 12, 300.0, rec=20, obs=2)
    R["life"] = o
    t, P = o["t_obs"], o["struct"]; ip = np.argmax(P)
    sel = (t > t[ip] + 25) & (P > P[ip] * 1e-4)
    p = np.polyfit(t[sel], np.log(P[sel]), 1)
    R["life_fit"] = dict(p=p, t1=float(t[sel][0]), t2=float(t[sel][-1]), tpk=float(t[ip]))
    J["tau_tdse"] = float(-1 / p[0]); J["tau_exact"] = float(R["tau_exact"])
    log("lifetime TDSE", J["tau_tdse"], "exact", R["tau_exact"])

# ------------------------------------------------------------------ 6. double well
def dw_run(Vb, nper=4.0, keep=False, a=1.0, b=0.4):
    x = grid_half(4.0); V = v_double_well(x, a, b, Vb)
    w, v = lowest_eigs(x, V, 40)
    dE = w[1] - w[0]; Tper = H_PLANCK / dE
    psi0 = gaussian(x, -(b / 2 + a / 2), 0.18, 0.0)
    dt = 0.05; nst = int(nper * Tper / dt)
    masks = {"R": x > 0}
    o = propagate(x, V, psi0, dt, nst, rec=10 if keep else 10**9, obs=2, masks=masks)
    t = o["t_obs"]; PR = o["R"]
    # spectral reference
    c = (v.T @ psi0) * (x[1] - x[0])
    PRs = np.array([np.sum(abs(v[x > 0] @ (c * np.exp(-1j * w * tt / HBAR))) ** 2) * (x[1] - x[0]) for tt in t])
    # frequency from TDSE P_R(t) by windowed, zero-padded FFT + parabolic interpolation
    s = (PR - PR.mean()) * np.hanning(len(PR)); dts = t[1] - t[0]; nfft = 2**17
    F = abs(np.fft.rfft(s, nfft)); f = np.fft.rfftfreq(nfft, dts)
    f0 = dE / H_PLANCK; band = (f > 0.5 * f0) & (f < 1.5 * f0)
    i = np.where(band)[0][np.argmax(F[band])]
    y0, y1, y2 = np.log(F[i - 1:i + 2] + 1e-300); dl = 0.5 * (y0 - y2) / (y0 - 2 * y1 + y2)
    fpk = f[i] + dl * (f[1] - f[0])
    res = dict(Vb=Vb, E=w, dE=dE, dE_tdse=H_PLANCK * fpk, Tper=Tper, t=t, PR=PR, PRs=PRs,
               c=c, x=x, V=V, phi=v)
    if keep: res["rho"], res["t_rec"] = o["rho"], o["t_rec"]
    # WKB (Landau-Lifshitz) estimate for the square double well
    Em = 0.5 * (w[0] + w[1]); kap = np.sqrt((Vb - Em) / CKIN)
    hw = 0.5 * ((w[2] + w[3]) - (w[0] + w[1]))
    res["dE_wkb"] = hw / np.pi * np.exp(-kap * b)
    return res

def sec6():
    R["dw_main"] = dw_run(1.0, nper=4.0, keep=True)
    m = R["dw_main"]
    J["dw_main"] = dict(dE_eig=m["dE"], dE_tdse=m["dE_tdse"], dE_wkb=m["dE_wkb"], period=m["Tper"],
                        maxdev_spectral=float(abs(m["PR"] - m["PRs"]).max()), E0=m["E"][0],
                        c0sq=float(abs(m["c"][0])**2), c1sq=float(abs(m["c"][1])**2),
                        c01=float(abs(m["c"][0])**2 + abs(m["c"][1])**2))
    log("DW main", J["dw_main"])
    Vbs = np.array([0.3, 0.4, 0.5, 0.6, 0.8, 1.0, 1.2, 1.5, 2.0, 2.5])
    sweep = []
    for Vb in Vbs:
        if Vb <= 2.0:
            r = dw_run(Vb, nper=4.0); sweep.append((Vb, r["dE"], r["dE_tdse"], r["dE_wkb"]))
        else:
            x = grid_half(4.0); w, _ = lowest_eigs(x, v_double_well(x, Vb=Vb), 4)
            Em = 0.5 * (w[0] + w[1]); kap = np.sqrt((Vb - Em) / CKIN); hw = 0.5 * ((w[2] + w[3]) - (w[0] + w[1]))
            sweep.append((Vb, w[1] - w[0], np.nan, hw / np.pi * np.exp(-kap * 0.4)))
        log("DW sweep", sweep[-1])
    R["dw_sweep"] = np.array(sweep)

# ------------------------------------------------------------------ run
import os
todo = sys.argv[1:] or ["sec0", "sec1", "sec2", "sec3", "sec4", "sec5", "sec6"]
if os.path.exists("results.pkl"): R.update(pickle.load(open("results.pkl", "rb")))
if os.path.exists("results.json"): J.update({k: v for k, v in json.load(open("results.json")).items()})
for name in todo:
    try:
        globals()[name]()
        pickle.dump(R, open("results.pkl", "wb"))
        keep = {kk: R[kk] for kk in ("Er", "Gamma", "tau_exact", "Er2", "T_off_exact", "cap_refl", "T05_exact") if kk in R}
        json.dump({**J, **keep}, open("results.json", "w"), indent=1, default=float)
    except Exception:
        traceback.print_exc(); log("FAILED", name)
log("ALL DONE")
