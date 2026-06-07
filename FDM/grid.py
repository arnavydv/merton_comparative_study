import numpy as np
import matplotlib.pyplot as plt
from scipy.linalg import solve_banded

# Import the 10 visualization functions
from fdm_visualization import (
    plot_value_function_surface, plot_value_function_error_heatmap,
    plot_policy_surface, plot_policy_error_heatmap,
    plot_value_function_slices, plot_policy_slices,
    plot_marginal_value_surface, plot_implied_risk_aversion_heatmap,
    plot_policy_iteration_diagnostics, plot_forward_wealth_paths
)

# ==========================================
# 1. MODEL & GRID PARAMETERS
# ==========================================
gamma = 5.0
mu = (0.201649)
sigma = ( 0.256573)
r = (0.0368)
T = 1.0

merton_pi_analytical = (mu - r) / (gamma * sigma**2)

N_w = 10000
N_tau = 252
w_max = 2

dw = w_max / N_w
dtau = T / N_tau

w_grid = np.linspace(0.5, w_max, N_w + 1)
tau_grid = np.linspace(0, T, N_tau + 1)
t_grid = T - tau_grid # Forward time grid (0 to T) for plotting

# Initialize arrays
V = np.zeros((N_tau + 1, N_w + 1))
pi_history = np.zeros((N_tau + 1, N_w + 1)) # Save pi at every time step!
iterations_history = []

epsilon = 1e-10
V[0, :] = (np.maximum(w_grid, epsilon)**(1 - gamma)) / (1 - gamma)
V[:, 0] = 0.0 

pi = np.full(N_w + 1, merton_pi_analytical)
pi[0] = 0.0

# ==========================================
# 2. FDM SOLVER LOOP
# ==========================================
print("Starting FDM Time Marching...")
for n in range(N_tau):
    V_old = V[n, :].copy()
    V_new = V_old.copy()
    
    max_iter = 20
    tol = 1e-6
    iters = 0
    
    for iteration in range(max_iter):
        iters += 1
        pi_old = pi.copy()
        
        w_interior = w_grid[1:N_w]
        pi_interior = pi[1:N_w]
        
        a = r * w_interior + pi_interior * w_interior * (mu - r)
        b = 0.5 * (pi_interior * w_interior * sigma)**2
        
        L = dtau * (a / (2 * dw) - b / (dw**2))
        D = 1.0 + dtau * (2 * b / (dw**2))
        U = -dtau * (a / (2 * dw) + b / (dw**2))
        
        # Ghost point elimination for upper boundary
        L[-1] = L[-1] - U[-1]
        D[-1] = D[-1] + 2 * U[-1]
        U[-1] = 0.0
        
        ab = np.zeros((3, N_w - 1))
        ab[0, 1:] = U[:-1]
        ab[1, :] = D
        ab[2, :-1] = L[1:]
        
        V_new_interior = solve_banded((1, 1), ab, V_old[1:N_w])
        
        V_new[1:N_w] = V_new_interior
        V_new[0] = 0.0
        V_new[N_w] = 2 * V_new[N_w - 1] - V_new[N_w - 2]
        
        # Calculate derivatives for policy update
        V_w = (V_new[2:] - V_new[:-2]) / (2 * dw)
        V_ww = (V_new[2:] - 2 * V_new[1:-1] + V_new[:-2]) / (dw**2)
        
        pi_new_interior = - (mu - r) * V_w / (w_interior * sigma**2 * (V_ww - 1e-10))
        pi[1:N_w] = np.clip(pi_new_interior, -2.0, 2.0)
        pi[0] = 0.0
        pi[N_w] = pi[N_w - 1]
        
        if np.max(np.abs(pi - pi_old)) < tol:
            break
            
    V[n + 1, :] = V_new
    pi_history[n + 1, :] = pi
    iterations_history.append(iters)

print("FDM Solution Complete!")

# ==========================================
# 3. POST-PROCESSING & ANALYTICAL BENCHMARKS
# ==========================================
# Reverse arrays to be in forward time t (0 to T)
V_t = V[::-1, :]
pi_t = pi_history[::-1, :]

# Create meshes for 3D/Heatmap plotting
t_mesh, w_mesh = np.meshgrid(t_grid, w_grid, indexing='ij')

# Analytical Value Function
k = (1 - gamma) * r + ((1 - gamma) * (mu - r)**2) / (2 * gamma * sigma**2)
V_exact = (np.maximum(w_mesh, epsilon)**(1 - gamma)) / (1 - gamma) * np.exp(k * (T - t_mesh))

# Calculate FDM derivatives on the full grid (for visualization)
V_w_fdm = np.zeros_like(V_t)
V_ww_fdm = np.zeros_like(V_t)
V_w_fdm[:, 1:-1] = (V_t[:, 2:] - V_t[:, :-2]) / (2 * dw)
V_ww_fdm[:, 1:-1] = (V_t[:, 2:] - 2 * V_t[:, 1:-1] + V_t[:, :-2]) / (dw**2)
# Simple boundary extrapolation for derivatives
V_w_fdm[:, 0], V_w_fdm[:, -1] = V_w_fdm[:, 1], V_w_fdm[:, -2]
V_ww_fdm[:, 0], V_ww_fdm[:, -1] = V_ww_fdm[:, 1], V_ww_fdm[:, -2]

# Implied Risk Aversion: gamma = -w * V_ww / V_w
gamma_implied = -w_mesh * V_ww_fdm / (V_w_fdm + 1e-10)

# ==========================================
# 4. FORWARD MONTE CARLO SIMULATION
# ==========================================
num_paths = 1000
num_steps = 252
dt = T / num_steps
time_sim = np.linspace(0, T, num_steps + 1)

def simulate_paths(pi_val, num_paths, num_steps, dt):
    wealth = np.full((num_paths, num_steps + 1), w_grid[0])
    for i in range(num_steps):
        dZ = np.random.randn(num_paths) * np.sqrt(dt)
        drift = r + pi_val * (mu - r)
        diffusion = pi_val * sigma
        wealth[:, i+1] = wealth[:, i] * (1 + drift * dt + diffusion * dZ)
    return wealth

# Extract the FDM policy at t=0 for the simulation
pi_fdm_t0 = pi_t[-1, int(w_grid[0] / dw)] 
wealth_exact = simulate_paths(merton_pi_analytical, num_paths, num_steps, dt)
wealth_fdm = simulate_paths(pi_fdm_t0, num_paths, num_steps, dt)
wealth_rf = simulate_paths(0.0, num_paths, num_steps, dt)

# ==========================================
# 5. GENERATE ALL 10 GRAPHS
# ==========================================
print("Generating 10 Diagnostic Graphs...")

plot_value_function_surface(t_mesh, w_mesh, V_t)
plot_value_function_error_heatmap(t_mesh, w_mesh, V_t, V_exact)
plot_policy_surface(t_mesh, w_mesh, pi_t, merton_pi_analytical)
plot_policy_error_heatmap(t_mesh, w_mesh, pi_t, merton_pi_analytical)
plot_value_function_slices(t_grid, w_grid, V_t, V_exact)
plot_policy_slices(t_grid, w_grid, pi_t, merton_pi_analytical)
plot_marginal_value_surface(t_mesh, w_mesh, V_w_fdm)
plot_implied_risk_aversion_heatmap(t_mesh, w_mesh, gamma_implied, gamma)
plot_policy_iteration_diagnostics(iterations_history)
plot_forward_wealth_paths(time_sim, wealth_fdm, wealth_exact, wealth_rf)

print("All graphs generated successfully!")