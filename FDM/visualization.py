"""
visualization.py
================
11 publication-quality figures for Merton HJB-FDM analysis.
All figures saved to ./figures/.

Fig 01  — π*(W) at t=0:  FDM vs Analytical (1-D)
Fig 02  — π*(t) at W=1:  time path of optimal policy
Fig 03  — 3-D surface:   π*(t, W)  [FDM + analytical plane]
Fig 04  — 3-D surface:   V(t, W)   [FDM vs Analytical side-by-side]
Fig 05  — V(W) cross-sections at multiple time slices
Fig 06  — Absolute error |V_fdm - V_ana| heatmap (log scale)
Fig 07  — Policy error   |π_fdm - π_ana| vs W at multiple t
Fig 08  — Sensitivity of π* to risk aversion γ
Fig 09  — Terminal wealth distribution: optimal vs sub-optimal π
Fig 10  — Mean–Std efficient frontier (varying π)
Fig 11  — FDM convergence: error vs grid size
"""

import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from mpl_toolkits.mplot3d import Axes3D          # noqa
from matplotlib.colors import LogNorm

FIGURES_DIR = "./figures"
os.makedirs(FIGURES_DIR, exist_ok=True)

# ── Global style ──────────────────────────────────────────────────────────────
plt.rcParams.update({
    "font.family":    "DejaVu Sans",
    "font.size":      11,
    "axes.titlesize": 13,
    "axes.labelsize": 11,
    "legend.fontsize":10,
    "figure.dpi":     120,
    "axes.grid":      True,
    "grid.alpha":     0.30,
    "grid.linestyle": "--",
    "lines.linewidth":2.0,
})

COLOR_FDM  = "#2563EB"    # blue
COLOR_ANA  = "#DC2626"    # red
COLOR_ERR  = "#7C3AED"    # purple
CMAP_SURF  = "viridis"

def _save(name: str):
    path = os.path.join(FIGURES_DIR, name)
    plt.savefig(path, bbox_inches="tight", dpi=150)
    plt.close()
    print(f"  ✓  Saved: {path}")

def _W_mask(solver, lo=0.1, hi=15.0):
    return (solver.W_grid >= lo) & (solver.W_grid <= hi)


# ══════════════════════════════════════════════════════════════════════════════
# Fig 01  — π*(W) at t=0: FDM vs Analytical
# ══════════════════════════════════════════════════════════════════════════════
def plot_pi_vs_W(params, solver):
    mask = _W_mask(solver, 0.1, 15)
    W    = solver.W_grid[mask]
    pi_f = solver.pi_slice_t(0.0)[mask]
    pi_a = np.full_like(W, params.pi_star)

    fig, ax = plt.subplots(figsize=(9, 5))
    ax.plot(W, pi_f, color=COLOR_FDM, lw=2.4, label="FDM  π*(W)")
    ax.plot(W, pi_a, "--", color=COLOR_ANA, lw=2.0,
            label=f"Analytical π* = {params.pi_star:.5f}")
    ax.fill_between(W, pi_f, pi_a, alpha=0.12, color=COLOR_ERR, label="|error|")
    ax.set_xlabel("Wealth  W")
    ax.set_ylabel("Optimal risky proportion  π*")
    ax.set_title("Fig 1 — Optimal Policy π*(W) at t = 0  [FDM vs Analytical]")
    ax.legend()
    ax.set_xlim([W[0], W[-1]])
    ax.set_ylim([-0.05, 1.10])
    # Annotate π*
    ax.axhline(params.pi_star, color=COLOR_ANA, lw=0.8, ls=":")
    ax.text(W[-1]*0.05, params.pi_star + 0.02,
            f"π* = {params.pi_star:.4f}", color=COLOR_ANA, fontsize=10)
    fig.tight_layout()
    _save("fig01_pi_vs_W.png")


# ══════════════════════════════════════════════════════════════════════════════
# Fig 02  — π*(t) at W=1: time path
# ══════════════════════════════════════════════════════════════════════════════
def plot_pi_vs_t(params, solver):
    t    = solver.t_grid
    pi_f = solver.pi_slice_W(1.0)
    pi_a = np.full_like(t, params.pi_star)

    fig, ax = plt.subplots(figsize=(9, 5))
    ax.plot(t, pi_f, color=COLOR_FDM, lw=2.4, label="FDM  π*(t, W=1)")
    ax.plot(t, pi_a, "--", color=COLOR_ANA, lw=2.0,
            label=f"Analytical π* = {params.pi_star:.5f}")
    ax.fill_between(t, pi_f, pi_a, alpha=0.12, color=COLOR_ERR)
    ax.set_xlabel("Time  t  (years)");  ax.set_ylabel("π*")
    ax.set_title("Fig 2 — Optimal Policy π*(t) at W = 1  [time path]")
    ax.set_ylim([-0.05, 1.10])
    ax.legend()
    fig.tight_layout()
    _save("fig02_pi_vs_t.png")


# ══════════════════════════════════════════════════════════════════════════════
# Fig 03  — 3-D surface: π*(t, W)
# ══════════════════════════════════════════════════════════════════════════════
def plot_3d_pi(params, solver):
    t_idx = np.linspace(0, solver.Nt, 55, dtype=int)
    w_idx = np.where(_W_mask(solver, 0.2, 10))[0][::max(1, len(solver.W_grid)//150)]

    T_sub  = solver.t_grid[t_idx]
    W_sub  = solver.W_grid[w_idx]
    Pi_sub = solver.pi_fdm[np.ix_(t_idx, w_idx)]
    Pi_ana = np.full_like(Pi_sub, params.pi_star)

    TT, WW = np.meshgrid(T_sub, W_sub, indexing="ij")

    fig = plt.figure(figsize=(12, 7))
    ax  = fig.add_subplot(111, projection="3d")
    surf = ax.plot_surface(TT, WW, Pi_sub, cmap=CMAP_SURF, alpha=0.90, linewidth=0)
    ax.plot_surface(TT, WW, Pi_ana, alpha=0.20, color="red")
    fig.colorbar(surf, ax=ax, shrink=0.45, label="π*")
    ax.set_xlabel("t");  ax.set_ylabel("W");  ax.set_zlabel("π*")
    ax.set_title("Fig 3 — 3-D Surface: Optimal π*(t, W)  [FDM; red = analytical]")
    ax.view_init(elev=28, azim=-55)
    fig.tight_layout()
    _save("fig03_3d_pi_surface.png")


# ══════════════════════════════════════════════════════════════════════════════
# Fig 04  — 3-D surface: V(t, W)  FDM vs Analytical
# ══════════════════════════════════════════════════════════════════════════════
def plot_3d_V(params, solver):
    t_idx = np.linspace(0, solver.Nt, 55, dtype=int)
    w_idx = np.where(_W_mask(solver, 0.2, 8))[0][::max(1, len(solver.W_grid)//150)]

    T_sub  = solver.t_grid[t_idx]
    W_sub  = solver.W_grid[w_idx]
    V_fdm  = solver.V[np.ix_(t_idx, w_idx)]
    V_ana  = params.analytical_V(T_sub[:, None], W_sub[None, :])

    TT, WW = np.meshgrid(T_sub, W_sub, indexing="ij")

    fig = plt.figure(figsize=(14, 6))
    for i, (data, title, cmap) in enumerate([
        (V_fdm, "FDM   V(t,W)",        "plasma"),
        (V_ana, "Analytical   V(t,W)", "cividis"),
    ]):
        ax = fig.add_subplot(1, 2, i+1, projection="3d")
        ax.plot_surface(TT, WW, data, cmap=cmap, alpha=0.90, linewidth=0)
        ax.set_xlabel("t");  ax.set_ylabel("W");  ax.set_zlabel("V")
        ax.set_title(title);  ax.view_init(elev=28, azim=-55)

    fig.suptitle("Fig 4 — 3-D Value Function  V(t, W)", fontsize=14)
    fig.tight_layout()
    _save("fig04_3d_V_surface.png")


# ══════════════════════════════════════════════════════════════════════════════
# Fig 05  — V(W) cross-sections at multiple t
# ══════════════════════════════════════════════════════════════════════════════
def plot_V_slices(params, solver):
    t_vals = [0.0, 0.25, 0.50, 0.75, params.T]
    mask   = _W_mask(solver, 0.1, 10)
    W      = solver.W_grid[mask]
    colors = cm.plasma(np.linspace(0.10, 0.85, len(t_vals)))

    fig, axes = plt.subplots(1, 2, figsize=(13, 5), sharey=False)
    for t_val, col in zip(t_vals, colors):
        lbl   = f"t={t_val:.2f}"
        v_fdm = solver.V_slice_t(t_val)[mask]
        v_ana = params.analytical_V(
            np.full_like(W, t_val), W)
        axes[0].plot(W, v_fdm, color=col, label=lbl)
        axes[1].plot(W, v_ana, "--", color=col, label=lbl)

    for ax, title in zip(axes, ["FDM", "Analytical"]):
        ax.set_xlabel("Wealth  W");  ax.set_ylabel("V(t, W)")
        ax.set_title(f"{title}  —  V(t, W)");  ax.legend(title="t")

    fig.suptitle("Fig 5 — Value Function Cross-Sections at Multiple Times", fontsize=13)
    fig.tight_layout()
    _save("fig05_V_slices.png")


# ══════════════════════════════════════════════════════════════════════════════
# Fig 06  — |V_fdm - V_ana| heatmap
# ══════════════════════════════════════════════════════════════════════════════
def plot_V_error_heatmap(params, solver):
    t_idx = np.linspace(0, solver.Nt, 80, dtype=int)
    w_idx = np.where(_W_mask(solver, 0.2, 10))[0][::max(1, sum(_W_mask(solver,0.2,10))//200)]

    T_sub  = solver.t_grid[t_idx]
    W_sub  = solver.W_grid[w_idx]
    V_fdm  = solver.V[np.ix_(t_idx, w_idx)]
    V_ana  = params.analytical_V(T_sub[:, None], W_sub[None, :])
    err    = np.abs(V_fdm - V_ana)

    vmin = max(err.min(), 1e-12)
    vmax = err.max()

    fig, ax = plt.subplots(figsize=(10, 5))
    im = ax.pcolormesh(T_sub, W_sub, err.T, cmap="hot_r",
                       norm=LogNorm(vmin=vmin, vmax=vmax))
    fig.colorbar(im, ax=ax, label="|V_fdm − V_ana|  (log scale)")
    ax.set_xlabel("Time  t");  ax.set_ylabel("Wealth  W")
    ax.set_title("Fig 6 — Value-Function Error Heatmap  |V_fdm − V_ana|")
    fig.tight_layout()
    _save("fig06_V_error_heatmap.png")


# ══════════════════════════════════════════════════════════════════════════════
# Fig 07  — |π_fdm - π_ana| vs W at multiple t slices
# ══════════════════════════════════════════════════════════════════════════════
def plot_pi_error(params, solver):
    t_vals  = [0.0, 0.25, 0.50, 0.75]
    mask    = _W_mask(solver, 0.2, 10)
    W       = solver.W_grid[mask]
    colors  = cm.plasma(np.linspace(0.15, 0.85, len(t_vals)))

    fig, ax = plt.subplots(figsize=(9, 5))
    for t_val, col in zip(t_vals, colors):
        pi_fdm = solver.pi_slice_t(t_val)[mask]
        err = np.abs(pi_fdm - params.pi_star)
        err = np.where(err > 1e-12, err, 1e-12)
        ax.semilogy(W, err, color=col, label=f"t={t_val:.2f}")

    ax.set_xlabel("Wealth  W");  ax.set_ylabel("|π_fdm − π*|  (log scale)")
    ax.set_title("Fig 7 — Policy Error |π_fdm − π*| vs Wealth  (multiple t)")
    ax.legend(title="Time  t")
    fig.tight_layout()
    _save("fig07_pi_error.png")


# ══════════════════════════════════════════════════════════════════════════════
# Fig 08  — Sensitivity of π* to γ
# ══════════════════════════════════════════════════════════════════════════════
def plot_pi_sensitivity_gamma(params, solver):
    from fdm_main import MertonParams, MertonFDMSolver

    gamma_vals = np.linspace(1.1, 10.0, 80)
    pi_ana = (params.mu - params.r) / (gamma_vals * params.sigma**2)

    gamma_fdm = np.linspace(1.5, 9.0, 10)
    pi_fdm_pts = []
    for g in gamma_fdm:
        p2 = MertonParams(gamma=g, mu=params.mu, sigma=params.sigma,
                          r=params.r, T=params.T)
        s2 = MertonFDMSolver(p2, Nx=150, Nt=60, x_min=-2, x_max=4,
                             pi_min=0.0, pi_max=1.0)
        s2.solve(verbose=False)
        pi_fdm_pts.append(s2.pi_at(0.0, 1.0))

    fig, ax = plt.subplots(figsize=(9, 5))
    ax.plot(gamma_vals, pi_ana, color=COLOR_ANA, label="Analytical π*(γ)")
    ax.scatter(gamma_fdm, pi_fdm_pts, color=COLOR_FDM, zorder=5, s=70,
               label="FDM π*(γ) at W=1, t=0")
    ax.axvline(params.gamma, ls="--", color="gray", lw=1.3,
               label=f"Base  γ={params.gamma}")
    ax.axhline(params.pi_star, ls=":", color="gray", lw=1.0)
    ax.set_xlabel("Risk Aversion  γ  (RRA)");  ax.set_ylabel("π*")
    ax.set_title("Fig 8 — Sensitivity of π* to Risk Aversion γ")
    ax.set_ylim([0, 1])
    ax.legend()
    fig.tight_layout()
    _save("fig08_pi_sensitivity_gamma.png")


# ══════════════════════════════════════════════════════════════════════════════
# Fig 09  — Terminal wealth distribution: optimal vs sub-optimal π
# ══════════════════════════════════════════════════════════════════════════════
def plot_wealth_distribution(params, solver, n_paths=40_000):
    rng   = np.random.default_rng(42)
    dt_mc = params.T / 252
    steps = 252

    def simulate(pi_val):
        W = np.ones(n_paths)
        for _ in range(steps):
            dZ = rng.normal(0.0, np.sqrt(dt_mc), n_paths)
            W *= np.exp(
                (params.r + pi_val*(params.mu - params.r)
                 - 0.5*(pi_val*params.sigma)**2) * dt_mc
                + pi_val*params.sigma*dZ
            )
        return W

    W_opt   = simulate(params.pi_star)
    W_low   = simulate(max(params.pi_star * 0.3, 0.05))
    W_high  = simulate(min(params.pi_star * 1.7, 0.99))

    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    kw = dict(bins=120, density=True, alpha=0.60, edgecolor="none")
    for ax in axes:
        ax.hist(np.log(W_opt),  color="#16A34A",
                label=f"π* = {params.pi_star:.3f} (optimal)", **kw)
        ax.hist(np.log(W_low),  color=COLOR_FDM,
                label=f"π  = {params.pi_star*0.3:.3f} (conservative)", **kw)
        ax.hist(np.log(W_high), color=COLOR_ANA,
                label=f"π  = {params.pi_star*1.7:.3f} (aggressive)", **kw)
        ax.set_xlabel("log W(T)");  ax.set_ylabel("Density")
        ax.legend(fontsize=9)

    axes[0].set_title("Linear Y scale")
    axes[1].set_yscale("log");  axes[1].set_title("Log Y scale")

    fig.suptitle(f"Fig 9 — Terminal Wealth Distribution  (MC, {n_paths:,} paths)", fontsize=13)
    fig.tight_layout()
    _save("fig09_wealth_distribution.png")


# ══════════════════════════════════════════════════════════════════════════════
# Fig 10  — Efficient frontier: E[W(T)] vs Std[W(T)]
# ══════════════════════════════════════════════════════════════════════════════
def plot_efficient_frontier(params, n_paths=20_000):
    rng     = np.random.default_rng(0)
    dt_mc   = params.T / 252
    steps   = 252
    pi_range= np.linspace(0.0, 1.0, 50)
    means, stds = [], []

    for pi in pi_range:
        W = np.ones(n_paths)
        for _ in range(steps):
            dZ = rng.normal(0.0, np.sqrt(dt_mc), n_paths)
            W *= np.exp(
                (params.r + pi*(params.mu - params.r)
                 - 0.5*(pi*params.sigma)**2) * dt_mc
                + pi*params.sigma*dZ
            )
        means.append(W.mean());  stds.append(W.std())

    means, stds = np.array(means), np.array(stds)

    fig, ax = plt.subplots(figsize=(9, 6))
    sc = ax.scatter(stds, means, c=pi_range, cmap="plasma", s=50, zorder=3)
    fig.colorbar(sc, ax=ax, label="π (risky fraction)")
    ax.plot(stds, means, color="gray", lw=0.7, alpha=0.5)

    idx_star = int(np.argmin(np.abs(pi_range - params.pi_star)))
    ax.scatter([stds[idx_star]], [means[idx_star]], s=220, marker="*",
               color="red", zorder=6, label=f"π* = {params.pi_star:.3f}")
    ax.scatter([0], [np.exp(params.r*params.T)], s=130, marker="D",
               color="green", zorder=6, label="Risk-free  (π=0)")

    ax.set_xlabel("Std[ W(T) ]");  ax.set_ylabel("E[ W(T) ]")
    ax.set_title("Fig 10 — Mean–Std Efficient Frontier  (varying π)")
    ax.legend()
    fig.tight_layout()
    _save("fig10_efficient_frontier.png")


# ══════════════════════════════════════════════════════════════════════════════
# Fig 11  — Convergence: FDM error vs grid size
# ══════════════════════════════════════════════════════════════════════════════
def plot_convergence(params):
    from fdm_main import MertonFDMSolver
    import time

    grid_Nx = [50, 100, 150, 200, 300, 500]
    err_V, err_pi, runtimes = [], [], []

    V_exact  = float(params.analytical_V(
        np.array([0.0]), np.array([1.0]))[0])
    pi_exact = params.pi_star

    for N in grid_Nx:
        t0 = time.perf_counter()
        s  = MertonFDMSolver(params, Nx=N, Nt=max(50, N//2),
                             x_min=-2, x_max=4,
                             pi_min=0.0, pi_max=1.0)
        s.solve(verbose=False)
        runtimes.append(time.perf_counter() - t0)
        err_V.append(abs(s.V_at(0.0, 1.0) - V_exact))
        err_pi.append(abs(s.pi_at(0.0, 1.0) - pi_exact))

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    # Error in V
    axes[0].loglog(grid_Nx, err_V, "o-", color=COLOR_FDM)
    ref1 = err_V[0] * (grid_Nx[0] / np.array(grid_Nx))**1
    ref2 = err_V[0] * (grid_Nx[0] / np.array(grid_Nx))**2
    axes[0].loglog(grid_Nx, ref1, "--k", alpha=0.4, label="O(h¹)")
    axes[0].loglog(grid_Nx, ref2, ":k",  alpha=0.4, label="O(h²)")
    axes[0].set_xlabel("Nx");  axes[0].set_ylabel("|V_fdm − V_ana|")
    axes[0].set_title("Value Function Convergence");  axes[0].legend()

    # Error in π
    err_pi_plot = [max(e, 1e-16) for e in err_pi]
    axes[1].loglog(grid_Nx, err_pi_plot, "s-", color=COLOR_ANA)
    axes[1].loglog(grid_Nx, ref1, "--k", alpha=0.4, label="O(h¹)")
    axes[1].loglog(grid_Nx, ref2, ":k",  alpha=0.4, label="O(h²)")
    axes[1].set_xlabel("Nx");  axes[1].set_ylabel("|π_fdm − π*|")
    axes[1].set_title("Policy Convergence");  axes[1].legend()

    # Runtime
    axes[2].plot(grid_Nx, runtimes, "^-", color=COLOR_ERR)
    axes[2].set_xlabel("Nx");  axes[2].set_ylabel("Wall time  (s)")
    axes[2].set_title("Solver Runtime vs Grid Size")

    fig.suptitle("Fig 11 — FDM Convergence Study", fontsize=13)
    fig.tight_layout()
    _save("fig11_convergence.png")


# ══════════════════════════════════════════════════════════════════════════════
# Master runner
# ══════════════════════════════════════════════════════════════════════════════
def generate_all_plots(params, solver):
    entries = [
        ("Fig  1: π*(W) at t=0",            lambda: plot_pi_vs_W(params, solver)),
        ("Fig  2: π*(t) at W=1",             lambda: plot_pi_vs_t(params, solver)),
        ("Fig  3: 3-D π*(t,W) surface",      lambda: plot_3d_pi(params, solver)),
        ("Fig  4: 3-D V(t,W) surface",       lambda: plot_3d_V(params, solver)),
        ("Fig  5: V(W) slices over time",     lambda: plot_V_slices(params, solver)),
        ("Fig  6: V error heatmap",           lambda: plot_V_error_heatmap(params, solver)),
        ("Fig  7: π error vs W",              lambda: plot_pi_error(params, solver)),
        ("Fig  8: π sensitivity to γ",        lambda: plot_pi_sensitivity_gamma(params, solver)),
        ("Fig  9: Wealth distribution (MC)",  lambda: plot_wealth_distribution(params, solver)),
        ("Fig 10: Efficient frontier (MC)",   lambda: plot_efficient_frontier(params)),
        ("Fig 11: Convergence study",         lambda: plot_convergence(params)),
    ]
    print("\n" + "═"*60)
    print("  Generating all 11 figures")
    print("═"*60)
    for label, fn in entries:
        print(f"\n→  {label}")
        fn()
    print("\n" + "═"*60)
    print(f"  All figures saved to  {FIGURES_DIR}/")
    print("═"*60)


if __name__ == "__main__":
    from fdm_main import build_default_solver
    params, solver = build_default_solver(Nx=500, Nt=252)
    generate_all_plots(params, solver)
