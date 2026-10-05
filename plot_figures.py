#!/usr/bin/env python3
import pickle, json, numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
from tdse_lib import *

plt.rcParams.update({
    "font.family": "serif", "mathtext.fontset": "cm", "font.size": 8.5, "axes.labelsize": 9,
    "axes.titlesize": 9, "legend.fontsize": 7.5, "xtick.labelsize": 8, "ytick.labelsize": 8,
    "axes.linewidth": 0.7, "lines.linewidth": 1.3, "figure.dpi": 150, "savefig.dpi": 300,
    "savefig.bbox": "tight", "axes.grid": False, "xtick.direction": "in", "ytick.direction": "in",
    "xtick.top": True, "ytick.right": True,
})
BL, OR, GR, RD, GY, PU = "#1f4e9c", "#d9741a", "#2a8f4e", "#c0392b", "#6b6b6b", "#7b4fa3"
R = pickle.load(open("results.pkl", "rb"))
OUT = "figures/"

def tag(ax, s, x=0.02, y=0.96, c="k"):
    ax.text(x, y, s, transform=ax.transAxes, ha="left", va="top", fontsize=9, fontweight="bold", color=c)

def save(fig, name):
    fig.savefig(OUT + name + ".pdf"); fig.savefig(OUT + name + ".png", dpi=170); plt.close(fig)

# ------------------------------------------------------------------ Fig 1: potentials
def fig1():
    fig, ax = plt.subplots(1, 2, figsize=(7.0, 2.7), constrained_layout=True)
    x = np.linspace(-2.2, 2.2, 4401)
    Vd = v_double_barrier(x)
    a = ax[0]
    a.fill_between(x, 0, Vd, color=BL, alpha=0.25, lw=0); a.plot(x, Vd, color=BL)
    for E, lab in ((R["Er"], r"$E_r=%.3f$ eV" % R["Er"]), (R["Eoff"], r"$E=%.2f$ eV" % R["Eoff"])):
        a.axhline(E, color=OR if E < 0.5 else GR, ls="--", lw=0.9); 
        a.text(2.15, E + 0.02, lab, ha="right", va="bottom", color=OR if E < 0.5 else GR, fontsize=7.5)
    a.annotate("", (-0.75, 1.1), (-0.25, 1.1), arrowprops=dict(arrowstyle="<->", lw=0.7)); a.text(-0.5, 1.13, r"$w$", ha="center")
    a.annotate("", (-0.25, 1.1), (0.25, 1.1), arrowprops=dict(arrowstyle="<->", lw=0.7)); a.text(0, 1.13, r"$d$", ha="center")
    a.annotate("", (0.25, 1.1), (0.75, 1.1), arrowprops=dict(arrowstyle="<->", lw=0.7)); a.text(0.5, 1.13, r"$w$", ha="center")
    a.set_xlim(-2.2, 2.2); a.set_ylim(-0.05, 1.3); a.set_xlabel("$x$ (nm)"); a.set_ylabel("$V(x)$ (eV)"); tag(a, "(a)")
    a.set_title("Double barrier: $V_0=1$ eV, $w=d=0.5$ nm", fontsize=8.5)
    b = ax[1]
    m = R["dw_main"]; xx, V = m["x"], m["V"]
    b.fill_between(xx, 0, np.minimum(V, 1.6), color=PU, alpha=0.22, lw=0); b.plot(xx, np.minimum(V, 1.6), color=PU)
    for n, c in ((0, BL), (1, OR)):
        E = m["E"][n]; ph = m["phi"][:, n]; ph = ph * np.sign(ph[np.argmin(abs(xx + 0.7))])
        b.axhline(E, color=c, lw=0.7, ls=":"); b.plot(xx[abs(xx) <= 1.35], (E + 0.2 * ph)[abs(xx) <= 1.35], color=c, label=r"$E_%d=%.3f$ eV" % (n, E))
    b.set_xlim(-2.0, 2.0); b.set_ylim(-0.05, 1.3); b.set_xlabel("$x$ (nm)"); b.set_yticklabels([]); tag(b, "(b)")
    b.legend(loc="upper center", frameon=False, ncol=2, bbox_to_anchor=(0.5, 0.78))
    b.set_title("Double well: $V_b=1$ eV, $b=0.4$ nm, $a=1.0$ nm", fontsize=8.5)
    save(fig, "fig1_potentials")

# ------------------------------------------------------------------ Fig 2: numerical validation
def fig2():
    fig, ax = plt.subplots(1, 3, figsize=(7.2, 2.5), constrained_layout=True)
    a = ax[0]; dx = R["sp_dx"]
    for o, c, mk in ((2, GY, "s"), (4, RD, "^"), (6, GR, "D"), (8, BL, "o")):
        e = np.maximum(R["sp_err"][o], 1e-14); sl = np.polyfit(np.log(dx[:6]), np.log(e[:6]), 1)[0]
        a.loglog(dx, e, color=c, marker=mk, ms=3.5, label=r"order %d (obs. %.1f)" % (o, sl))
    a.set_xlabel(r"$\Delta x$ (nm)"); a.set_ylabel(r"$|E_0-E_0^{\rm exact}|$ (eV)"); a.set_ylim(1e-12, 3e1); a.legend(frameon=False, loc="upper left", fontsize=6.5); tag(a, "(a)", x=0.02, y=0.2)
    b = ax[1]; dts = R["dts"]
    cols = {1: GY, 2: OR, 4: RD, 6: GR, 8: BL}
    for M, e in R["terr"].items():
        kk = np.where(e[1:] > 1e-12)[0]; sl = max(np.log2(e[k] / e[k + 1]) for k in kk)      # observed order from the finest pair above the roundoff floor
        b.loglog(dts, np.maximum(e, 1e-15), color=cols[M], marker="o", ms=3.2, label=r"$M=%d$ (order %.1f)" % (M, sl))
    b.set_xlabel(r"$\Delta t$ (fs)"); b.set_ylabel(r"$\|\Psi-\Psi_{\rm ref}\|$ at $t=16$ fs"); b.set_ylim(1e-14, 3e6); b.legend(frameon=False, ncol=1, loc="upper left", fontsize=6.5); tag(b, "(b)", x=0.02, y=0.2)
    c = ax[2]
    for M, (t, n) in R["norm_drift"].items():
        c.semilogy(t, np.maximum(abs(n - 1), 1e-17), color=cols[M], label=r"$M=%d$, no CAP" % M, lw=1.0)
    t, n = R["norm_cap"]; c.semilogy(t, np.maximum(abs(n - 1), 1e-17), color="k", ls="--", label="with CAP")
    c.set_xlabel("$t$ (fs)"); c.set_ylabel(r"$|\int|\Psi|^2dx-1|$"); c.set_ylim(1e-16, 3); c.legend(frameon=False, loc="center right", fontsize=7); tag(c, "(c)", y=0.35)
    save(fig, "fig2_validation")

# ------------------------------------------------------------------ Fig 3: double-barrier wave-packet dynamics
def fig3():
    fig = plt.figure(figsize=(7.2, 5.6), constrained_layout=True)
    gs = fig.add_gridspec(2, 2)
    runs = [("dyn_res", r"resonant, $E_0=E_r$"), ("dyn_off", r"off-resonant, $E_0=0.70$ eV")]
    vmax = max(R[k]["rho"].max() for k, _ in runs)
    for i, (k, lab) in enumerate(runs):
        o = R[k]; ax = fig.add_subplot(gs[0, i]); x = o["x"]; t = o["t_rec"]
        im = ax.imshow(o["rho"], origin="lower", aspect="auto", extent=[x[0], x[-1], t[0], t[-1]], cmap="magma", vmin=0, vmax=0.14)
        ax.axvspan(-0.75, 0.75, color="w", alpha=0.0); ax.axvline(0, color="c", lw=0.6, ls=":")
        ax.set_xlim(-60, 60); ax.set_xlabel("$x$ (nm)"); ax.set_ylabel("$t$ (fs)" if i == 0 else ""); ax.set_title(lab, fontsize=8.5)
        tag(ax, "(%s)" % "ab"[i], c="w")
    cb = fig.colorbar(im, ax=fig.axes[:2], shrink=0.9, pad=0.01, location="right", extend="max"); cb.set_label(r"$|\Psi(x,t)|^2$ (nm$^{-1}$)")
    ax = fig.add_subplot(gs[1, 0]); o = R["dyn_res"]; x = o["x"]; sel = abs(x) <= 3.0; t = o["t_rec"]; sel_t = t >= 50
    im2 = ax.imshow(o["rho"][sel_t][:, sel], origin="lower", aspect="auto", extent=[x[sel][0], x[sel][-1], t[sel_t][0], t[-1]], cmap="magma", vmin=0)
    for xe in (-0.75, -0.25, 0.25, 0.75): ax.axvline(xe, color="c", lw=0.7, ls="--")
    ax.set_xlabel("$x$ (nm)"); ax.set_ylabel("$t$ (fs)"); ax.set_title("resonant case, zoom on the structure", fontsize=8.5); tag(ax, "(c)", c="w")
    cb2 = fig.colorbar(im2, ax=ax, pad=0.01); cb2.set_label(r"$|\Psi|^2$ (nm$^{-1}$)")
    ax = fig.add_subplot(gs[1, 1])
    for k, c in (("dyn_res", BL), ("dyn_off", OR)):
        o = R[k]; ax.plot(o["t_obs"], o["right"], color=c, label=("$T$, " + ("res." if k == "dyn_res" else "off-res.")))
        ax.plot(o["t_obs"], o["left"], color=c, ls="--", label=("$R$, " + ("res." if k == "dyn_res" else "off-res.")))
        ax.plot(o["t_obs"], o["struct"], color=c, ls=":", lw=1.0, label=("in structure, " + ("res." if k == "dyn_res" else "off-res.")))
    ax.plot(R["dyn_res"]["t_obs"], R["dyn_res"]["norm"], color="k", lw=0.8, ls="-.", label="total norm")
    ax.set_xlabel("$t$ (fs)"); ax.set_ylabel("probability"); ax.legend(frameon=False, ncol=1, loc="center left", bbox_to_anchor=(0.0, 0.62), fontsize=6.5); tag(ax, "(d)", y=0.35)
    ax.set_ylim(-0.03, 1.05)
    save(fig, "fig3_db_dynamics")

# ------------------------------------------------------------------ Fig 4: T(E), resonance and lifetime
def fig4():
    fig, ax = plt.subplots(1, 3, figsize=(7.4, 2.6), constrained_layout=True)
    a = ax[0]; d = R["TE_dense"]; te = R["TE"]
    a.semilogy(d["E"], d["T1"], color=GY, ls="--", lw=0.9, label="single barrier")
    a.semilogy(d["E"], d["T"], color=BL, label="exact (transfer matrix)")
    a.semilogy(te["E"][::4], te["Tn"][::4], "o", color=OR, ms=2.6, mfc="none", label="TDSE")
    a.set_xlim(0.1, 2.0); a.set_ylim(1e-5, 2); a.set_xlabel("$E$ (eV)"); a.set_ylabel("$T(E)$"); a.legend(frameon=False, loc="lower right", fontsize=7); tag(a, "(a)")
    a.axvline(R["Er"], color=GY, lw=0.5, ls=":"); a.axvline(R["Er2"], color=GY, lw=0.5, ls=":")
    b = ax[1]
    b.plot(te["E"], te["Tn"] - te["Tx"], color=OR, lw=0.9, label=r"$T_{\rm TDSE}-T_{\rm exact}$")
    b.plot(te["E"], te["Tn"] + te["Rn"] - 1, color=BL, lw=0.9, label=r"$T+R-1$")
    b.axhline(0, color="k", lw=0.4); b.set_xlabel("$E$ (eV)"); b.set_ylabel("deviation"); b.legend(frameon=False, loc="lower left", fontsize=7); tag(b, "(b)")
    c = ax[2]; o = R["life"]; t, P = o["t_obs"], o["struct"]; f = R["life_fit"]
    c.semilogy(t, P, color=BL, label="TDSE")
    tt = np.linspace(f["tpk"], 300, 200)
    c.semilogy(tt, np.exp(np.polyval(f["p"], tt)), color=OR, ls="--", label=r"fit, $\tau=%.1f$ fs" % (-1 / f["p"][0]))
    c.semilogy(tt, np.exp(np.polyval(f["p"], f["tpk"] + 25)) * np.exp(-(tt - f["tpk"] - 25) / R["tau_exact"]), color=GR, ls=":", label=r"$\hbar/\Gamma=%.1f$ fs" % R["tau_exact"])
    c.set_ylim(1e-8, 1); c.set_xlim(0, 300); c.set_xlabel("$t$ (fs)"); c.set_ylabel(r"$P_{|x|<0.75\,{\rm nm}}(t)$"); c.legend(frameon=False, loc="upper right", fontsize=7); tag(c, "(c)", y=0.96)
    save(fig, "fig4_resonance")

# ------------------------------------------------------------------ Fig 5: sweeps
def fig5():
    fig, ax = plt.subplots(1, 2, figsize=(7.0, 2.7), constrained_layout=True)
    a = ax[0]; s = R["sweep_w"]
    a.semilogy(s["w"], s["T1"], color=GY, ls="--", label="single barrier (exact)")
    a.semilogy(s["w"], s["Tx"], color=BL, label="double barrier (exact)")
    a.semilogy(s["w"], s["Tn"], "o", color=OR, ms=3.5, mfc="none", label="TDSE")
    kap = np.sqrt((1.0 - 0.5) / CKIN); ww = np.linspace(0.4, 1.0, 20)
    j = np.argmin(abs(s["w"] - 0.5)); a.semilogy(ww, s["Tx"][j] * np.exp(-4 * kap * (ww - 0.5)), color=RD, ls=":", lw=0.9, label=r"$\propto e^{-4\kappa w}$")
    a.set_xlabel("barrier width $w$ (nm)"); a.set_ylabel(r"$T(E=0.5\,{\rm eV})$"); a.legend(frameon=False, fontsize=7, loc="lower left"); tag(a, "(a)", y=0.55)
    b = ax[1]; s = R["sweep_V"]
    b.semilogy(s["V"], s["T1"], color=GY, ls="--"); b.semilogy(s["V"], s["Tx"], color=BL); b.semilogy(s["V"], s["Tn"], "o", color=OR, ms=3.5, mfc="none")
    b.set_xlabel(r"barrier height $V_0$ (eV)"); b.set_ylabel(r"$T(E=0.5\,{\rm eV})$"); tag(b, "(b)", y=0.55)
    save(fig, "fig5_sweeps")

# ------------------------------------------------------------------ Fig 6: double-well dynamics
def fig6():
    m = R["dw_main"]; fig, ax = plt.subplots(1, 2, figsize=(7.0, 2.8), constrained_layout=True)
    a = ax[0]; x = m["x"]; t = m["t_rec"]; sel = abs(x) <= 2.0
    im = a.imshow(m["rho"][:, sel], origin="lower", aspect="auto", extent=[x[sel][0], x[sel][-1], t[0], t[-1]], cmap="magma", vmin=0)
    for xe in (-1.2, -0.2, 0.2, 1.2): a.axvline(xe, color="c", lw=0.6, ls="--")
    a.set_xlabel("$x$ (nm)"); a.set_ylabel("$t$ (fs)"); cb = fig.colorbar(im, ax=a, pad=0.01); cb.set_label(r"$|\Psi|^2$ (nm$^{-1}$)"); tag(a, "(a)", c="w")
    b = ax[1]; tt = m["t"]
    b.plot(tt, m["PR"], color=BL, label="TDSE"); b.plot(tt, m["PRs"], color=OR, ls="--", lw=0.9, label="spectral reference")
    b.plot(tt, 0.5 * (1 - np.cos(m["dE"] * tt / HBAR)), color=GR, ls=":", lw=1.0, label=r"two-level, $\Delta E=%.1f$ meV" % (1e3 * m["dE"]))
    b.set_xlabel("$t$ (fs)"); b.set_ylabel(r"$P_R(t)=\int_{x>0}|\Psi|^2dx$"); b.set_ylim(-0.02, 1.25); b.legend(frameon=False, fontsize=7, loc="upper right", ncol=1); tag(b, "(b)", y=0.5)
    save(fig, "fig6_doublewell")

# ------------------------------------------------------------------ Fig 7: splitting vs barrier height
def fig7():
    s = R["dw_sweep"]; fig, a = plt.subplots(figsize=(3.6, 2.8), constrained_layout=True)
    a.semilogy(s[:, 0], 1e3 * s[:, 1], color=BL, label="eigenvalues")
    a.semilogy(s[:, 0], 1e3 * s[:, 3], color=GR, ls="--", label="WKB estimate")
    ok = ~np.isnan(s[:, 2]); a.semilogy(s[ok, 0], 1e3 * s[ok, 2], "o", color=OR, ms=3.8, mfc="none", label="TDSE ($P_R$ spectrum)")
    a.set_xlabel(r"central barrier height $V_b$ (eV)"); a.set_ylabel(r"$\Delta E=E_1-E_0$ (meV)"); a.legend(frameon=False, fontsize=7, loc="upper right")
    a2 = a.twinx(); lo, hi = a.get_ylim(); a2.set_yscale("log"); a2.set_ylim(H_PLANCK / (lo * 1e-3), H_PLANCK / (hi * 1e-3)); a2.set_ylabel(r"tunnelling period $h/\Delta E$ (fs)")
    save(fig, "fig7_splitting")

# ------------------------------------------------------------------ Fig 8: 3D surfaces
def fig8():
    fig = plt.figure(figsize=(7.4, 3.4))
    o = R["life"]; x, t = o["x"], o["t_rec"]; sx = (x >= -14) & (x <= 12); st = t <= 120
    X, T = np.meshgrid(x[sx][::4], t[st]); Z = o["rho"][st][:, sx][:, ::4]
    a = fig.add_axes([0.0, 0.02, 0.47, 0.92], projection="3d")
    a.plot_surface(X, T, Z, cmap="viridis", rstride=1, cstride=1, linewidth=0, antialiased=True)
    a.set_xlabel("$x$ (nm)"); a.set_ylabel("$t$ (fs)"); a.set_zlabel(r"$|\Psi|^2$ (nm$^{-1}$)", labelpad=4); a.view_init(32, -58); a.set_title("(a) double barrier", fontsize=8.5)
    m = R["dw_main"]; x, t = m["x"], m["t_rec"]; sx = abs(x) <= 1.6; sel = t <= 260
    X, T = np.meshgrid(x[sx][::2], t[sel]); Z = m["rho"][sel][:, sx][:, ::2]
    b = fig.add_axes([0.5, 0.02, 0.47, 0.92], projection="3d")
    b.plot_surface(X, T, Z, cmap="viridis", rstride=1, cstride=1, linewidth=0, antialiased=True)
    b.set_xlabel("$x$ (nm)"); b.set_ylabel("$t$ (fs)"); b.set_zlabel(r"$|\Psi|^2$ (nm$^{-1}$)", labelpad=4); b.view_init(32, -58); b.set_title("(b) double well", fontsize=8.5)
    save(fig, "fig8_surfaces")

if __name__ == "__main__":
    import sys
    for f in (sys.argv[1:] or ["fig1", "fig2", "fig3", "fig4", "fig5", "fig6", "fig7", "fig8"]):
        try: globals()[f](); print("ok", f)
        except Exception as e:
            import traceback; traceback.print_exc()
