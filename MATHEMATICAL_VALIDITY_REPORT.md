# Mathematical Validity Report: HJB_MERTON_BENCHMARK

**Date:** 2026-06-13  
**Project:** HJB_MERTON_BENCHMARK (Merton portfolio HJB solvers: FDM, PINN, Deep BSDE)  
**Query:** Check the mathematical validity of this project. Is it valid mathematically?

---

## Executive Summary / Verdict

**The project is NOT FULLY MATHEMATICALLY VALID as a rigorous comparative benchmark.**

- **Theoretical formulation** (Merton 1969 wealth dynamics, HJB PDE, CRRA utility, closed-form solution, optimal policy FOC) is **correct** and matches standard references (see paper appendix A derivations, which are algebraically sound).
- **FDM implementation** is **mostly valid** (pi* accuracy ~2.1e-4, O(h^2) convergence for policy) but uses an **approximate/inconsistent Dirichlet BC** at low wealth.
- **PINN implementation** is **problematic**: trains on a *multiplied* residual form (not the reported PDE residual) + non-standard terminal loss (z-score normalization). Second-order derivatives amplify errors; reported policy accuracy is poor (MAE 0.23, max err 324).
- **Deep BSDE (FBSDE) implementation has a CRITICAL flaw**: the neural networks for π(t,·) and Z are fed the cumulative Brownian motion B_t (zero-mean, var=t) as the "wealth" state instead of the actual simulated wealth W_t. This violates the Markov state (t, W_t) of the control problem. Reasonable π* numbers are an artifact of the explicit constant-target supervision loss (output ~constant regardless of input) + use of analytical growth rate `a`. Not a faithful realization of the Deep BSDE method described in the paper or literature (Han et al. 2018).

**Impact**: Reported results (e.g., "Deep BSDE achieves competitive accuracy (π* MAE 3.46e-3)"), figures, tables, and paper conclusions (esp. Pareto frontier, "BSDE excels for real-time inference") are **not reliable** for the BSDE method and are weakened for PINN. FDM is the only trustworthy reference in the suite. The benchmark does not fairly compare the *methods* as implemented vs. described.

**Severity table**:

| Component | Validity | Key Issue | Impact on Claims |
|-----------|----------|-----------|------------------|
| Theory / Analytical | Valid | None (correct) | None |
| FDM | Mostly valid (with caveat) | Fixed left BC = U(W_min) for all t (fdm_main.py:297) instead of time-dependent exp(A(T-t))U | Small for π* (interior); larger V errors at low W |
| PINN | Problematic | Multiplied residual in training (loss.py:20); z-score terminal (loss.py:29-32); NN second deriv pathology | High MAE 0.23 (max 324) for π*; results do not represent standard PINN HJB performance |
| Deep BSDE / FBSDE | **Invalid (critical)** | State fed to nets is B_t (BM cumsum), not W_t (wealth) in forward/backward/supervision (multiple files) | "Competitive accuracy" is spurious (supervision cheat); not a real FBSDE solve |
| Evaluation harness (comprehensive_experiments.py) | Mixed | Evaluates on real W but BSDE model never trained on real W; reports true R for PINN (not trained quantity) | Misleading cross-method comparison |
| Paper | Theory sound; claims overstated | Describes correct math; does not reflect code bugs | Conclusions (1-3) and BSDE sections unsupported for "Deep BSDE" |

---

## 1. Theoretical Baseline (Correct)

The closed-form solution for the Merton problem (terminal CRRA utility, no consumption) is:

- Wealth: `dW = [r + π(μ-r)] W dt + π σ W dZ`
- HJB: `V_t + max_π { [r + π(μ-r)] W V_W + ½ π² σ² W² V_WW } = 0`, `V(T,W) = W^p / p` (p=1-γ)
- FOC: `π* = -(μ-r) V_W / (σ² W V_WW)`
- Reduced nonlinear PDE: `V_t + r W V_W - ½ (μ-r)²/σ² (V_W² / V_WW) = 0`
- Analytical: `V(t,W) = [W^p / p] exp(A (T-t))`, `A = p [r + (μ-r)²/(2 γ σ²)]`
- `π* = (μ-r) / (γ σ²)` (constant)

**Evidence**: Matches README:69-113, paper appendix A (Ito + Bellman + FOC + verification substitution all algebraically correct), MertonParams.analytical_* (FDM/fdm_main.py:111-127), AnalyticalMerton (comprehensive_experiments.py:131-141).

Parameters used: γ=5, μ≈0.201649, σ≈0.256573, r=0.0368, T=1 → π*≈0.500835, A≈-0.31232, p=-4.

---

## 2. FDM (FDM/fdm_main.py) — Mostly Valid

**Core discretization** (log-wealth x=ln W):
- b(π) = r + π(μ-r) - ½π²σ², D=½π²σ²
- θ=1 fully implicit + upwind/central + tridiagonal (scipy.solve_banded) + 8 Picard iters for nonlinearity.
- pi from FOC in x-coords: `π* = -(μ-r) Vx / (σ² (Vxx - Vx))` (derived from chain rule Vw=Vx/W, Vww=(Vxx-Vx)/W²). Code at fdm_main.py:213-216 **correct**.

**Verification run output (t=0, Nx=500, Nt=252)**:
```
W | pi_fdm | pi_ana | |d_pi| | V_fdm | V_ana | |d_V|
0.2 | 0.50062450 | 0.50083470 | 2.10e-04 | -116.030512 | -114.335041 | 1.695472
0.5 | 0.50062450 | 0.50083470 | 2.10e-04 | -2.858927 | -2.926977 | 0.068050
1.0 | 0.50062450 | 0.50083470 | 2.10e-04 | -0.182857 | -0.182936 | 0.000079
2.0 | 0.50062450 | 0.50083470 | 2.10e-04 | -0.011057 | -0.011434 | 0.000376
5.0 | 0.50062450 | 0.50083470 | 2.10e-04 | -0.000288 | -0.000293 | 0.000005
```
Matches JSON (pi FDM MAE 0.0002102), README, tables. π* essentially constant (correct). V error largest at low W (expected from |V| scaling + BC).

**Issue (moderate)**: Left BC (fdm_main.py:293-297):
```python
# Left BC ...
rhs[0]     = self.W_grid[0]**p.p / p.p   # Dirichlet: U(W_min)
```
Fixed to terminal utility for *all* t. Should be `exp(self.p.A * (self.p.T - t_k)) * U(W_min)` for consistency with analytical V(t, W_min). Right BC is ad-hoc "hold previous" (lines 307-308). Comment acknowledges "terminal-shape extrapolation". Effect small on interior π* (0.00021 error), visible on V at W=0.2.

HJB residual (comprehensive:190-228) uses central diffs + analytical fallback at boundaries; sensible.

**Verdict**: Valid reference for π* in interior; V less trustworthy near domain edges due to BC approximation. Convergence order ~2 for π* (as claimed).

---

## 3. PINN (vanilla_pinns/) — Problematic

**Architecture**: net(log(w), t) → raw V (5×128 Tanh). Derivatives via autograd on *wealth* input w (chain rule through log is correct for Vw, Vww).

**Loss (loss.py:15-46, used by training.py via `import *`)**:
- HJB residual (training objective):
  ```python
  residual = v_t*v_xx + r*w*v_xx*v_x - 0.5*torch.pow((((mu-r)*v_x)/sigma),torch.tensor(2))
  ```
  This is **R × Vww** (multiplied form), where R is the true PDE residual. Algebra:
  R = Vt + rW Vw - ½k (Vw²/Vww)  
  R×Vww = Vt Vww + rW Vww Vw - ½k Vw²
  (last term has no extra Vww factor — matches code exactly). Common trick to avoid /0 when Vww≈0.
- Terminal (loss.py:25-32):
  ```python
  target = (v_tc - v_mean)/v_std
  pred = (v_tc_pred - v_mean)/v_std
  return mean((target-pred)**2)
  ```
  **Non-standard z-score normalization** (using target stats for both). Equivalent to shape matching after centering/scaling; does *not* enforce `||V(T,W)-U(W)||` in absolute (or even relative) utility units. For p=-4, U ranges hugely over [0.5,2].
- + soft concavity (ReLU(Vww)²) + monotonicity (ReLU(-Vw)²).

**Training vs. reported residual**:
- Training minimizes E[(R×Vww)²] (weighted).
- Evaluation (comprehensive_experiments.py:231-250, hjb_residual_pinn) reports `|Vt + rW Vw - ½k (Vw²/(Vww+1e-12))|` (true R, with division).
- These are **not the same**; optimizing the multiplied form does not guarantee small true R (especially where |Vww| small or varying). Matches README note at 288-292 but the "equivalent when Vww≠0" applies only at zeros, not to the minimizer.

**Post-training inspection** (vanilla_pinns/evaluation/evaluation.py + comprehensive compute_pinn_pi:439-457 use correct FOC formula `- (μ-r)Vw / (σ² W Vww)`):
```
W | V_pinn | V_ana | pi_hat | |d_pi|
0.5 | -2.915112 | -2.926977 | 0.529100 | 2.83e-02
1.0 | -0.183075 | -0.182936 | 0.501920 | 1.09e-03
2.0 | -0.010029 | -0.011434 | 0.481316 | 1.95e-02
```
JSON (exp1): pi PINN MAE=0.2327, Max Err=324.07 (wild outputs in tails); V MAE=1.67, Max=142. Matches tables.

**Why poor?** Spectral bias + second-derivative amplification (Vw²/Vww) + soft BCs + nonstandard terminal + loss mismatch. Even with 9000 epochs + L-BFGS, policy unusable.

**Verdict**: Valid *attempt* at PINN, but implementation choices + fundamental challenges mean results do not demonstrate (or fairly represent) the PINN paradigm for this HJB. High errors and max outliers are expected from the setup.

---

## 4. Deep BSDE / FBSDE — INVALID (Critical Flaw)

**Intended method** (paper §3.3, appendix): Parameterize π(t, W) and Z(t, W) where W = *wealth*. Forward Euler-Maruyama on wealth SDE using π. Backward recursion on Y (value process) using Z. Loss = Var(Y_0) (enforces martingale) + supervision to π*.

**Actual implementation**:

- brownian_motion.py:7 `W = torch.cumsum(dw, axis=1)` (B_t, not wealth).
- forward_simulation_equations.py:18 `dw, W = brownian...`; **37** `W_col = W[:, i:i+1]`; **40** `pi = model(t_col, W_col)` (feeds B_t).
  Wealth is tracked separately (`wealth_list`/`wealth_grid`) and *updated using the (wrong-input) pi*, then softplus.
- backward_simulation_equation.py: **81** `W_n = _ensure_col(W[:, n]...)`; `z_n = Z_model(t_n, W_n)` (again B_t).
- training.py:134 `t_sup, W_sup = sample_pi_supervision_points(...)` (W_sup = randn * sqrt(t) — BM scale); **137** `pi_pred = model(t_sup, W_sup)`; MSE to constant `pi_target`.
- data_generation.py, quick_eval.py, inference_grid: all generate "W" as BM samples.
- Loss uses analytical `a = A(...)` (exact from closed form) in denominator `Y_n = (Ynext - z dw) / (1 + a dt)` + Var(Y0).

**Evidence from trained model** (query on loaded best.pth):
```
Wealth W inputs (realistic for Merton eval):
  W=0.2 | pi=0.501945 | delta=1.11e-03
  ...
  W=5.0 | pi=0.581364 | delta=8.05e-02
BM-like inputs (what was fed during training/supervision):
  B=-2.0 | pi=0.501697
  ...
```
**Constant output ≈ π* on both domains** — network learned to ignore its second input (due to constant supervision target). Wealth paths are *approximately* correct *only because* π output happens to be the right constant.

**Why this is fatal**:
- State is wrong: π(t, B_t) instead of π(t, W_t). If true π* were state-dependent (general Merton or other HJB), forward dynamics would be completely incorrect.
- Supervision is on the wrong distribution (B ~ N(0,t) vs. W ~ lognormal around 1 with drift).
- Backward Z also on wrong state.
- Uses *analytical A* explicitly (hybrid, not pure learned driver).
- Evaluation in comprehensive_experiments.py:518 calls `compute_bsde_pi(..., w_eval)` with *real wealth* [0.2,4] (never seen in training) and gets low error *only* because output is flat.

**JSON**: BSDE pi MAE=0.00346 (looks "competitive" only due to the above).

**Verdict**: Does not implement or benchmark Deep BSDE/FBSDE. The "success" is an artifact of supervision on a constant + analytical a. Invalid for the claimed purpose.

---

## 5. Evaluation Harness & Cross-Checks

- comprehensive_experiments.py:491 `w_eval = rng.uniform(0.2,4)` (correct wealth for FDM/PINN eval and *post-hoc* BSDE calls).
- But BSDE was never trained on that distribution → results misleading.
- HJB residual for PINN uses true R (division); training used multiplied → apples-to-oranges.
- Cost numbers for BSDE/PINN are partially hardcoded (lines ~678, 709).
- No direct V output from BSDE (correct limitation noted in paper).

Matches all tables (tab01_policy_accuracy etc.) and results.md.

---

## 6. Impact on Paper & Published Claims

Paper appendix A derivations are pure math and **correct** (do not depend on code).

However:
- § "Deep BSDE Method" + "Discrete-Time Simulation" describe the *intended* (correct) algorithm with π(t, W) on wealth.
- Results, figures (policy_comparison.png etc.), tables, Pareto, overall score, and conclusions ("Deep BSDE achieves competitive accuracy", "BSDE occupies the speed-optimal corner", "composite benchmark score of 0.60") are derived from the *flawed runs*.
- Key finding #2 and recommendations for "real-time inference" using "Deep BSDE" are not supported.
- Limitations section mentions some caveats but not the state bug or loss distortion.

Reproducibility appendix points to code that contains the bugs.

---

## 7. Verification Commands (Reproducible)

All numbers above come from:

```bash
# FDM
cd FDM; python -c 'from fdm_main import build_default_solver; ...'   # see §2

# BSDE inspection (note chdir for config import)
cd <root>; python -c '
import os, torch
... chdir FBSDE; from neural_networks import pi_star; ...
model=...; load best.pth; query on W and on B values
'

# PINN
cd <root>; python -c '
import sys; sys.path.insert(0,"vanilla_pinns")
from neural_network import VanillaPINN; ... autograd for pi
'

# JSON
python -c 'import json; d=json.load(open("results/...json")); print(d["results"]["exp1"]...)'
```

Re-running these after report generation reproduces the embedded outputs exactly (within float/print precision).

Cross-check: FDM pi err at W=1 (2.1020e-4) == JSON 0.0002102 + tables + README.

---

## 8. Recommendations (Follow-ups Only — No Changes Made)

1. **Fix BSDE (critical)**:
   - Rename internal "W" (brownian) to `B` or `brownian`.
   - In forward: compute current_wealth, pass `wealth_col` (not W_col) to model for pi.
   - In backward: pass delta_wealth or current_wealth (or better, a proper state feature) to Z_model.
   - In supervision: sample realistic (t, W) with W ~ exp( normal ) around w0 or from forward paths (not BM).
   - Remove or downweight analytical `a` (learn driver or use proper BSDE discretization).
   - Re-train + re-eval.

2. **PINN**:
   - Use direct `mean((v_pred - v_tc)**2)` (or log-scaled / weighted) for terminal.
   - Optionally keep multiplied residual but *also* monitor/report true R during training.
   - Consider domain scaling, better sampling, or hard constraints for V(T,W).
   - Wider domain + more points at low W.

3. **FDM**:
   - Make left BC time-dependent: `rhs[0] = exp(A*(T-t)) * U(Wmin)`.
   - Or use analytical far-field / Neumann at both ends for higher accuracy.
   - Validate residual on interior grid only.

4. **Harness**:
   - For BSDE eval, also report "training-domain only" metrics (on BM-scale points) to expose the cheat.
   - Add checks: assert model second input has positive mean ~1 (wealth) vs ~0 (bug).
   - Separate "true residual loss" vs "training loss" for PINN.
   - Remove hardcoded times; actually time or note as approx.

5. **General**: Add property-based tests (e.g., `pi(t,W) ≈ const` for CRRA; Vww < 0 everywhere; V(t,W) increasing in W). Version the "intended math" vs. "actual code state".

If fixes are applied + full re-benchmark run, a follow-up note could be added here.

---

## 9. Files & Lines Referenced (for Audit)

See plan.md (same session) for exhaustive list. Key ones:
- FBSDE/* (forward:37, backward:81, training:134, brownian, data_generation, neural_networks, loss_function, quick_eval).
- vanilla_pinns/loss.py:20,25 (core losses); training.py:63; neural_network.py:23.
- FDM/fdm_main.py:191 (pi),297 (BC),111 (analytical).
- comprehensive_experiments.py:173 (eval grid),231 (PINN res),460 (BSDE call),109 (Analytical).
- results/experiment_results_latest.json, tables/, paper/HJB_Comparative_Study.tex (appendix A good; results claims affected).

No source files were modified during this check.

---

**Final Answer to Query**: No, the project is not valid mathematically in its current implementation for the neural methods (critical state error in FBSDE; distorting loss choices + high observed errors in PINN). FDM is usable with caveats. Theory and analytical baseline are solid. Use FDM numbers only; treat BSDE/PINN results as illustrative of implementation bugs rather than method performance. A corrected re-implementation + re-run would be required for a valid benchmark.

Report generated following approved plan. All verification outputs embedded above were produced live via read-only python -c during report authoring and match the cited JSON/tables.