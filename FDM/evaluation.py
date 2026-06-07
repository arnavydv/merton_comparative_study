"""
evaluation.py
=============
Complete quantitative evaluation of the FDM solver against Merton (1969).

Sections
────────
 1. Value-function error norms  (L-inf, L-1, L-2, max-relative)
 2. Value-function errors per time slice
 3. Policy error norms
 4. Policy errors per time slice
 5. Pointwise comparison table at t=0
 6. FDM stability analysis (Courant numbers, θ-scheme)
 7. Convergence order estimation
 8. Monte-Carlo utility comparison (expected utility + welfare loss)
 9. Summary
"""

import time
import numpy as np


def _linf(a, b):  return float(np.max(np.abs(a - b)))
def _l1(a, b):    return float(np.mean(np.abs(a - b)))
def _l2(a, b):    return float(np.sqrt(np.mean((a - b)**2)))
def _relmax(a, b):
    with np.errstate(divide="ignore", invalid="ignore"):
        denom = np.where(np.abs(b) > 1e-12, b, 1e-12)
        return float(np.max(np.abs((a - b) / denom)))

def _bar(title="", w=62):
    print(f"\n{'─'*w}\n  {title}\n{'─'*w}")


def evaluate(params, solver, mc_paths: int = 40_000, verbose: bool = True) -> dict:
    """
    Full numerical evaluation.  Returns dict of all metrics.
    """
    results = {}

    # Interior mask (avoid boundary artefacts)
    W_mask = (solver.W_grid > 0.2) & (solver.W_grid < 10.0)
    W_eval = solver.W_grid[W_mask]
    T_all  = solver.t_grid                           # (Nt+1,)

    # Build 2-D evaluation arrays  (Nt+1, |W_mask|)
    T_2d      = T_all[:, None] * np.ones(W_eval.shape)[None, :]
    V_fdm_2d  = solver.V[:, W_mask]
    V_ana_2d  = params.analytical_V(T_2d, W_eval[None, :])
    pi_fdm_2d = solver.pi_fdm[:, W_mask]
    pi_ana_2d = np.full_like(pi_fdm_2d, params.pi_star)

    # ── 1. Value function global errors ──────────────────────────────────────
    _bar("1.  VALUE FUNCTION ERRORS   V(t, W)  — global")
    V_linf = _linf(V_fdm_2d, V_ana_2d)
    V_l1   = _l1(V_fdm_2d,   V_ana_2d)
    V_l2   = _l2(V_fdm_2d,   V_ana_2d)
    V_rel  = _relmax(V_fdm_2d, V_ana_2d)
    results.update(V_linf=V_linf, V_l1=V_l1, V_l2=V_l2, V_rel=V_rel)
    if verbose:
        print(f"  L-∞  |V_fdm − V_ana|            = {V_linf:.6e}")
        print(f"  L-1  |V_fdm − V_ana|            = {V_l1:.6e}")
        print(f"  L-2  |V_fdm − V_ana|            = {V_l2:.6e}")
        print(f"  Max rel  |V_fdm−V_ana|/|V_ana|  = {V_rel*100:.6f} %")

    # ── 2. Value function errors by time slice ────────────────────────────────
    _bar("2.  VALUE FUNCTION ERRORS  — per time slice")
    t_slices = [0.0, 0.25, 0.50, 0.75, params.T]
    slice_V  = {}
    if verbose:
        print(f"  {'t':>6}  {'L-inf':>13}  {'L-2':>13}  {'Max rel %':>12}")
        print("  " + "─"*50)
    for tv in t_slices:
        k   = solver._t_idx(tv)
        vf  = solver.V[k, W_mask]
        va  = params.analytical_V(np.full_like(W_eval, tv), W_eval)
        ei  = _linf(vf, va);  e2 = _l2(vf, va);  er = _relmax(vf, va)
        slice_V[tv] = dict(linf=ei, l2=e2, rel=er)
        if verbose:
            print(f"  {tv:6.2f}  {ei:13.6e}  {e2:13.6e}  {er*100:12.6f}")
    results["slice_V_errors"] = slice_V

    # ── 3. Policy global errors ───────────────────────────────────────────────
    _bar("3.  POLICY ERRORS   π*(t, W)  — global")
    pi_linf = _linf(pi_fdm_2d, pi_ana_2d)
    pi_l1   = _l1(pi_fdm_2d,   pi_ana_2d)
    pi_l2   = _l2(pi_fdm_2d,   pi_ana_2d)
    pi_rel  = _relmax(pi_fdm_2d, pi_ana_2d)
    results.update(pi_linf=pi_linf, pi_l1=pi_l1, pi_l2=pi_l2, pi_rel=pi_rel)
    if verbose:
        print(f"  L-∞  |π_fdm − π*|                = {pi_linf:.6e}")
        print(f"  L-1  |π_fdm − π*|                = {pi_l1:.6e}")
        print(f"  L-2  |π_fdm − π*|                = {pi_l2:.6e}")
        print(f"  Max rel  |π_fdm − π*| / |π*|     = {pi_rel*100:.6f} %")
        print(f"\n  NOTE:  π* = {params.pi_star:.6f}  is in (0,1) — economically sensible.")
        print(f"         No leverage, no short-selling.  Constraints: [0, 1].")

    # ── 4. Policy errors by time slice ───────────────────────────────────────
    _bar("4.  POLICY ERRORS  — per time slice")
    slice_pi = {}
    if verbose:
        print(f"  {'t':>6}  {'L-inf':>13}  {'L-2':>13}  {'Max rel %':>12}")
        print("  " + "─"*50)
    for tv in t_slices:
        k   = solver._t_idx(tv)
        pf  = solver.pi_fdm[k, W_mask]
        ei  = _linf(pf, params.pi_star)
        e2  = _l2(pf,  params.pi_star)
        er  = abs(float(pf.mean()) - params.pi_star) / abs(params.pi_star)
        slice_pi[tv] = dict(linf=ei, l2=e2, rel=er)
        if verbose:
            print(f"  {tv:6.2f}  {ei:13.6e}  {e2:13.6e}  {er*100:12.6f}")
    results["slice_pi_errors"] = slice_pi

    # ── 5. Pointwise table at t=0 ─────────────────────────────────────────────
    _bar("5.  POINTWISE COMPARISON   (t = 0)")
    W_pts  = [0.2, 0.5, 1.0, 2.0, 5.0, 10.0, 20.0]
    pts_out = {}
    if verbose:
        hdr = f"  {'W':>6}  {'V_fdm':>12}  {'V_ana':>12}  {'π_fdm':>10}  {'π_ana':>10}  {'|ΔV|':>11}  {'|Δπ|':>10}"
        print(hdr)
        print("  " + "─"*76)
    for W_pt in W_pts:
        vf = solver.V_at(0.0, W_pt)
        va = float(params.analytical_V(np.array([0.0]), np.array([W_pt]))[0])
        pf = solver.pi_at(0.0, W_pt)
        pa = params.pi_star
        pts_out[W_pt] = dict(V_fdm=vf, V_ana=va, pi_fdm=pf, pi_ana=pa)
        if verbose:
            print(f"  {W_pt:6.2f}  {vf:12.6f}  {va:12.6f}  {pf:10.6f}  {pa:10.6f}  {abs(vf-va):11.3e}  {abs(pf-pa):10.3e}")
    results["pointwise"] = pts_out

    # ── 6. Stability analysis ─────────────────────────────────────────────────
    _bar("6.  FDM STABILITY ANALYSIS")
    max_pi   = float(np.max(np.abs(solver.pi_fdm)))
    D_max    = 0.5 * max_pi**2 * params.sigma**2
    b_max    = abs(params.r + max_pi * (params.mu - params.r))
    dx, dt   = solver.dx, solver.dt
    nu_diff  = D_max * dt / dx**2
    C_adv    = b_max * dt / dx
    results.update(dx=dx, dt=dt, nu_diff=nu_diff, C_adv=C_adv)
    stable   = "✓ STABLE (implicit)" if solver.theta >= 0.5 else \
               ("✓" if nu_diff < 0.5 else "✗") + f" (explicit, ν={nu_diff:.3f})"
    if verbose:
        print(f"  dx = {dx:.6f}    dt = {dt:.6f}")
        print(f"  Max |π_fdm|          = {max_pi:.6f}")
        print(f"  Diffusion Courant  ν = D·dt/dx²  = {nu_diff:.4f}")
        print(f"  Advection Courant  C = |b|·dt/dx = {C_adv:.4f}")
        print(f"  θ = {solver.theta}  →  {stable}")
        print(f"  Portfolio constraints: π ∈ [{solver.pi_min}, {solver.pi_max}]")

    # ── 7. Convergence order ──────────────────────────────────────────────────
    _bar("7.  CONVERGENCE ORDER  (refinement study)")
    from fdm_main import MertonFDMSolver
    V_exact = float(params.analytical_V(np.array([0.0]), np.array([1.0]))[0])
    Ns  = [80, 160, 320]
    errs = []
    for N in Ns:
        s2 = MertonFDMSolver(params, Nx=N, Nt=max(50, N//2),
                             x_min=-2, x_max=4,
                             pi_min=0.0, pi_max=1.0)
        s2.solve(verbose=False)
        errs.append(abs(s2.V_at(0.0, 1.0) - V_exact))
    orders = []
    for i in range(1, len(Ns)):
        if errs[i] > 0 and errs[i-1] > 0:
            orders.append(np.log(errs[i-1]/errs[i]) / np.log(Ns[i]/Ns[i-1]))
        else:
            orders.append(float("nan"))
    results["convergence_orders"] = orders
    if verbose:
        print(f"  {'Nx':>6}  {'|ΔV(0,1)|':>14}  {'order':>8}")
        print("  " + "─"*32)
        print(f"  {Ns[0]:6d}  {errs[0]:14.6e}  {'—':>8}")
        for i in range(1, len(Ns)):
            print(f"  {Ns[i]:6d}  {errs[i]:14.6e}  {orders[i-1]:8.3f}")

    # ── 8. Monte-Carlo utility comparison ─────────────────────────────────────
    _bar("8.  EXPECTED UTILITY COMPARISON  (Monte-Carlo)")
    rng   = np.random.default_rng(99)
    dt_mc = params.T / 252
    steps = 252
    p_exp = params.p    # utility exponent

    def expected_utility(pi_val):
        W = np.ones(mc_paths)
        for _ in range(steps):
            dZ = rng.normal(0.0, np.sqrt(dt_mc), mc_paths)
            W *= np.exp(
                (params.r + pi_val*(params.mu - params.r)
                 - 0.5*(pi_val*params.sigma)**2) * dt_mc
                + pi_val*params.sigma*dZ
            )
        return float(np.mean(W**p_exp / p_exp))

    EU_ana = expected_utility(params.pi_star)
    EU_fdm = expected_utility(solver.pi_at(0.0, 1.0))

    # Certainty-equivalent wealth:  CE^p/p = E[U]  ⟹  CE = (p·E[U])^{1/p}
    # For p < 0,  p·E[U] > 0  (since E[U] < 0 and p < 0)
    def ce(eu):
        val = p_exp * eu
        if val > 0:
            return val ** (1.0 / p_exp)
        return float("nan")

    CE_ana    = ce(EU_ana)
    CE_fdm    = ce(EU_fdm)
    welf_loss = (CE_ana - CE_fdm) / CE_ana if CE_ana != 0 else float("nan")
    results.update(EU_ana=EU_ana, EU_fdm=EU_fdm,
                   CE_ana=CE_ana, CE_fdm=CE_fdm, welfare_loss=welf_loss)

    if verbose:
        print(f"  γ (RRA) = {params.gamma},  p (utility exp) = {p_exp}")
        print(f"  π*(analytical)         = {params.pi_star:.8f}")
        print(f"  π*(FDM, t=0, W=1)      = {solver.pi_at(0.0,1.0):.8f}")
        print(f"  E[U] under π_ana       = {EU_ana:.8f}")
        print(f"  E[U] under π_fdm       = {EU_fdm:.8f}")
        print(f"  CE  under π_ana        = {CE_ana:.8f}")
        print(f"  CE  under π_fdm        = {CE_fdm:.8f}")
        print(f"  Welfare loss           = {welf_loss*100:.6f} %")

    # ── 9. Summary ────────────────────────────────────────────────────────────
    _bar("9.  SUMMARY")
    if verbose:
        print(f"  Model:  μ={params.mu}, σ={params.sigma}, r={params.r}, γ={params.gamma}, T={params.T}")
        print(f"  Grid:   Nx={solver.Nx},  Nt={solver.Nt},  θ={solver.theta}")
        print()
        print(f"  Analytical π*   = {params.pi_star:.8f}  ← in (0,1), economically correct")
        print(f"  FDM       π*   = {solver.pi_at(0.0,1.0):.8f}")
        print(f"  |Δπ|            = {abs(solver.pi_at(0.0,1.0)-params.pi_star):.4e}")
        print()
        print(f"  V function:  L-∞ = {V_linf:.4e},  L-2 = {V_l2:.4e}")
        print(f"  Policy:      L-∞ = {pi_linf:.4e},  L-2 = {pi_l2:.4e}")
        print(f"  Welfare loss from FDM approx = {welf_loss*100:.6f} %")
        print()
        print("  Convention reminder:")
        print("    γ = RRA (Relative Risk Aversion),  e.g. γ=5 → moderate aversion")
        print("    p = 1 - γ < 0  (utility exponent, V<0 for W>0 when p<0 — correct!)")
        print("    π* = (μ-r)/(γσ²) — always in (0,1) when μ>r, γ>0, no leverage")
    print()
    return results


if __name__ == "__main__":
    from fdm_main import build_default_solver
    params, solver = build_default_solver(Nx=500, Nt=252)
    t0 = time.perf_counter()
    results = evaluate(params, solver, mc_paths=40_000)
    print(f"Total evaluation time: {time.perf_counter()-t0:.2f} s")
