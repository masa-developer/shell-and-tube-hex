# -*- coding: utf-8 -*-
"""
Created on Mon Nov 17 14:15:42 2025

@author: VK
"""

import math
import matplotlib.pyplot as plt

# =========================
# USER INPUTS (edit these)
# =========================
m_dot_h_total = 120.0    # kg/s  -> HOT fluid mass flow (in shell)
m_dot_c = 120.0          # kg/s  -> COLD fluid mass flow (in tubes)
N_tubes = 80            # number of parallel tubes

# geometry / configuration (fixed per your earlier requests)
N_shell_passes = 1      # 1 shell pass
N_tube_passes = 6       # 6 tube passes per tube
tube_length_per_pass = 5.0  # m
tube_total_length = N_tube_passes * tube_length_per_pass

# tube dims & pitch
D_inner = 0.042   # m (42 mm ID)
D_outer = 0.054   # m (54 mm OD)
pitch_transverse = 0.075   # m (75 mm)
pitch_longitudinal = 0.075 # m (75 mm)

# shell geometry & baffles
D_shell = 0.5             # m (assumed)
baffle_spacing = 0.25     # m (assumed)

# fluid & material properties (approximate, constant)
T_hot_in = 120.0   # °C (shell inlet)
T_cold_in = 35.0   # °C (tube inlet)
Cp_h = 4180.0      # J/kg.K
Cp_c = 4180.0
rho = 1000.0       # kg/m3
mu = 0.001         # Pa.s
k = 0.6            # W/m.K
Pr = 7.0

# tube wall & fouling
k_wall = 45.0      # W/m.K
t_wall = (D_outer - D_inner) / 2.0
R_f = 0.0002       # m2.K/W

# correction factor for 1 shell pass, 6 tube passes (approx)
F = 0.86

# solver settings
max_iter = 200
tol = 1e-6
relax = 0.5

# =========================
# Helper functions
# =========================
def safe_lmtd(dT1, dT2):
    eps = 1e-9
    dT1 = max(dT1, eps)
    dT2 = max(dT2, eps)
    if abs(dT1 - dT2) < 1e-12:
        return dT1
    return (dT1 - dT2) / math.log(dT1 / dT2)

# =========================
# Basic geometry & flows
# =========================

# tube flow area and per-tube mass flow (cold fluid now inside tubes)
A_flow_tube = math.pi * (D_inner**2) / 4.0
m_dot_per_tube = m_dot_c / N_tubes      # cold flow per tube (kg/s)

# tube velocity (cold in tubes)
velocity_tube = m_dot_per_tube / (rho * A_flow_tube)

# total outer area of tubes (for heat transfer)
A_total = math.pi * D_outer * tube_total_length * N_tubes

# cell area & free fraction (bundle packing)
cell_area = pitch_transverse * pitch_longitudinal
tube_proj_area = math.pi * (D_outer**2) / 4.0
cell_free_area = max(1e-9, cell_area - tube_proj_area)
free_fraction = max(0.01, cell_free_area / cell_area)

# shell cross-flow area approximation
A_shell_flow = math.pi * D_shell * baffle_spacing * free_fraction
Dh_shell = 4.0 * cell_free_area / (math.pi * D_outer)  # approximate hydraulic diameter

# =========================
# Tube-side (cold in tubes) heat transfer: Dittus-Boelter
# =========================
Re_tube = rho * velocity_tube * D_inner / mu if velocity_tube > 0 else 0.0
if Re_tube >= 3000:
    Nu_tube = 0.023 * (Re_tube**0.8) * (Pr**0.4)
elif Re_tube > 0:
    Nu_tube = 4.36  # laminar approx (fully developed)
else:
    Nu_tube = 0.0

h_i = Nu_tube * k / D_inner if Nu_tube > 0 else 0.0

# =========================
# Shell-side (hot in shell) convective coefficient estimate (Kern-like)
# =========================
if A_shell_flow <= 0:
    raise ValueError("Shell flow area computed <= 0. Check tube pitch and shell/baffle geometry.")

G_shell = m_dot_h_total / A_shell_flow
Re_shell = G_shell * Dh_shell / mu if Dh_shell > 0 else 0.0

if Re_shell > 100:
    Nu_shell = 0.36 * (Re_shell**0.55) * (Pr**0.33)
else:
    Nu_shell = max(0.2 * (Re_shell**0.5) * (Pr**0.33), 0.0)

h_o = Nu_shell * k / Dh_shell if Dh_shell > 0 else 0.0

# =========================
# Overall U from resistances
# =========================
R_total = 0.0
R_total += (1.0 / h_i) if h_i > 0 else 1e9
R_total += t_wall / k_wall
R_total += (1.0 / h_o) if h_o > 0 else 1e9
R_total += R_f
U = 1.0 / R_total

# =========================
# Input validation & warnings
# =========================
if N_tubes <= 0:
    raise ValueError("N_tubes must be > 0")
if m_dot_h_total <= 0 or m_dot_c <= 0:
    raise ValueError("Mass flows must be positive (kg/s).")

if m_dot_per_tube > 50:
    print("WARNING: mass flow per tube is very large (>50 kg/s). Check units or N_tubes.")
if velocity_tube > 10:
    print("NOTE: velocity in tubes is high (>10 m/s). Check for erosion/pumping concerns.")

# =========================
# Iterative solve for T_cold_out (tube side) and T_hot_out (shell side)
# =========================
# Start with small changes
T_cold_out = T_cold_in + 0.1
T_hot_out = T_hot_in - 0.1

for it in range(max_iter):
    # Heat removed from cold tubes (energy balance)
    Q_tubes = m_dot_c * Cp_c * (T_cold_out - T_cold_in)  # positive if cold warms up
    # Heat removed from shell (should equal in magnitude but opposite sign)
    # Here Q_area will be positive when heat flows from shell->tubes
    # LMTD pairing for 1 shell pass, multi-tube-pass approx:
    dT1 = T_hot_in - T_cold_out   # hot inlet - tube outlet
    dT2 = T_hot_out - T_cold_in   # hot outlet - tube inlet

    # safe LMTD
    deltaT_lm = safe_lmtd(dT1, dT2)
    deltaT_corr = F * deltaT_lm

    Q_area = U * A_total * deltaT_corr  # predicted heat transfer from area

    # Update temperatures based on energy balances:
    # use Q_area to get new cold outlet (from tube energy balance)
    if m_dot_c * Cp_c > 0:
        T_cold_out_new = T_cold_in + Q_area / (m_dot_c * Cp_c)
    else:
        T_cold_out_new = T_cold_out

    # use Q_area to get new hot outlet
    if m_dot_h_total * Cp_h > 0:
        T_hot_out_new = T_hot_in - Q_area / (m_dot_h_total * Cp_h)
    else:
        T_hot_out_new = T_hot_out

    # enforce physical limits:
    T_cold_out_new = min(T_cold_out_new, T_hot_in - 1e-9)   # tube outlet cannot exceed hot inlet
    T_cold_out_new = max(T_cold_out_new, T_cold_in)       # cannot be below inlet
    T_hot_out_new = max(T_hot_out_new, T_cold_in + 1e-9)  # hot outlet cannot be below cold inlet
    T_hot_out_new = min(T_hot_out_new, T_hot_in)          # cannot be above hot inlet

    # relaxation
    T_cold_out = (1 - relax) * T_cold_out + relax * T_cold_out_new
    T_hot_out = (1 - relax) * T_hot_out + relax * T_hot_out_new

    # convergence check
    err = max(abs(T_cold_out_new - T_cold_out), abs(T_hot_out_new - T_hot_out))
    if err < tol:
        break

# final heat duty (positive = heat from hot shell to cold tubes)
Q_final = U * A_total * F * safe_lmtd(max(T_hot_in - T_cold_out, 1e-9), max(T_hot_out - T_cold_in, 1e-9))

# =========================
# Print results
# =========================
print("\n=== INPUTS ===")
print(f"Hot (shell) mass flow m_dot_h_total = {m_dot_h_total:.6g} kg/s")
print(f"Cold (tube) mass flow m_dot_c      = {m_dot_c:.6g} kg/s")
print(f"N_parallel_tubes = {N_tubes}")
print(f"Passes per tube = {N_tube_passes}, tube length per pass = {tube_length_per_pass} m")
print()

print("=== GEOMETRY & HYDRAULICS ===")
print(f"Tube inner diameter D_i = {D_inner*1000:.1f} mm, outer D_o = {D_outer*1000:.1f} mm")
print(f"Mass flow per tube (cold) = {m_dot_per_tube:.6g} kg/s")
print(f"Tube flow area A_flow_tube = {A_flow_tube:.6g} m^2")
print(f"Tube velocity (cold) = {velocity_tube:.6g} m/s")
print(f"Re_tube = {Re_tube:.3g}")
print()

print("=== SHELL-SIDE ESTIMATE ===")
print(f"Cell area = {cell_area:.6g} m^2, tube projected area = {tube_proj_area:.6g} m^2")
print(f"Free fraction in bundle = {free_fraction:.4f}")
print(f"Shell cross-flow area A_shell_flow = {A_shell_flow:.6g} m^2")
print(f"Shell Dh approx = {Dh_shell:.6g} m")
print(f"G_shell = {G_shell:.6g} kg/(m2 s), Re_shell = {Re_shell:.3g}")
print()

print("=== HEAT TRANSFER COEFFICIENTS & U ===")
print(f"Nu_tube = {Nu_tube:.6g}, h_i (tube, cold) = {h_i:.3g} W/m2.K")
print(f"Nu_shell = {Nu_shell:.6g}, h_o (shell, hot) = {h_o:.3g} W/m2.K")
print(f"Tube wall thickness t_wall = {t_wall:.6g} m, fouling R_f = {R_f:.6g}")
print(f"Calculated overall U = {U:.6g} W/m2.K")
print(f"Total outer area A_total = {A_total:.6g} m^2")
print()

print("=== RESULTS ===")
print(f"Hot inlet T_hot_in  = {T_hot_in:.2f} °C")
print(f"Hot outlet T_hot_out = {T_hot_out:.4f} °C")
print(f"Cold inlet T_cold_in = {T_cold_in:.2f} °C")
print(f"Cold outlet T_cold_out = {T_cold_out:.4f} °C")
print(f"Heat duty Q_final = {Q_final/1000.0:.6f} kW")
print(f"Iterations = {it+1}")
print()

# =========================
# Simple temperature plot
# =========================
x = [0, 1]
plt.figure(figsize=(7,4))
plt.plot(x, [T_hot_in, T_hot_out], marker='o', label='Hot (shell)')
plt.plot(x, [T_cold_in, T_cold_out], marker='o', label='Cold (tubes)')
plt.xlabel('Normalized exchanger length')
plt.ylabel('Temperature (°C)')
plt.grid(True)
plt.legend()
plt.title('Temperature distribution (hot in shell, cold in tubes)')
plt.show()
