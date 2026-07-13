"""
Separation_Project_Model_v2_0.py

Computational Lab Project - Separation Processes (Technion, 0540309)
Models the exam question written for Assignment #1 (Sections A - D):
a continuous centrifuge followed by an ultrafiltration (UF) membrane,
used to purify a protein produced inside cells.

Author: Nadav, ChemEng undergraduate (3rd year)
Date: July 2026

Changelog:
v1.0 - initial full build
v2.0 - simplified: no global plot styling, section numbers only
       (no original question text), all data collected in one place
       at the top of the file, figures shown with plt.show() instead
       of being saved to disk
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp


# ======================================================================
# SECTION 1 : DATA
# ======================================================================

# ---- General data (shared by the whole process) ----
rho = 1000.0                # liquid density [kg/m^3]
rho_p = 1150.0               # debris particle density [kg/m^3]
mu = 1e-3                     # liquid viscosity [Pa*s]
Qf = 1.2 / 3600               # feed volumetric flow rate [m^3/s] (1.2 m^3/h)
Cf_cells = 2.0                # debris concentration in the feed [kg/m^3]
Cf_protein = 2.0 * 1000       # protein concentration in the feed [g/m^3] (2 g/L)

# ---- Section A data (centrifugation) ----
Dp = 100e-6                   # minimal particle diameter to separate [m]
omega = 2000.0                # angular velocity of the centrifuge [rad/s]
r1 = 0.02                     # liquid surface radius [m]
r2 = 0.06                     # bowl wall radius [m]
K_stokes_max = 3.3            # upper K boundary of the Stokes regime
K_newton_min = 43.6           # lower K boundary of the Newton regime

# ---- Section B data (ultrafiltration) ----
R_protein = 0.93              # true membrane rejection towards the protein [-]
permeate_fraction = 0.30      # fraction of feed leaving as permeate, Qp/Qf [-]

# ---- Section D data (fouling and feasibility) ----
alpha = 5e13                  # specific cake resistance [m/kg]
Rm = 2e12                     # intrinsic membrane resistance [1/m]
A_mem = 2.22                  # membrane area [m^2]
J_target = 1.5e-4             # target permeate flux to maintain [m/s]
t_2h = 2 * 3600                # 2 hours, in seconds [s]
membrane_pressure_limit = 10.0  # safe operating pressure limit of a UF membrane [bar]


# ======================================================================
# SECTION 2 : PART A - CENTRIFUGATION
# ======================================================================

# --- A1: identify the flow regime at the two extreme radii ---
drho = rho_p - rho

ae1 = omega ** 2 * r1
ae2 = omega ** 2 * r2

K1 = Dp * (rho * drho * ae1 / mu ** 2) ** (1 / 3)
K2 = Dp * (rho * drho * ae2 / mu ** 2) ** (1 / 3)

print("Section A1")
print(f"K1 = {K1:.2f}, K2 = {K2:.2f}")
if K_stokes_max < K1 and K2 < K_newton_min:
    print("Both values fall in the Intermediate regime.")
print()

# --- A2: minimal residence time in the Intermediate regime ---
# Settling correlation: ut = 0.153 * ae^0.71 * Dp^1.14 * drho^0.71 * rho^-0.29 * mu^-0.43
# Since ae = omega^2 * r, this reduces to: dr/dt = C * r^0.71
C = 0.153 * (omega ** 2) ** 0.71 * Dp ** 1.14 * drho ** 0.71 * rho ** -0.29 * mu ** -0.43
print("Section A2")
print(f"C = {C:.3f} m^0.29/s (settling equation: dr/dt = C * r^0.71)")


def settling_ode(t, r):
    """Right-hand side of the settling equation dr/dt = C * r^0.71."""
    return C * r ** 0.71


solution_A = solve_ivp(settling_ode, t_span=(0, 0.05), y0=[r1], max_step=1e-5, dense_output=True)

t_values = np.linspace(0, 0.05, 200000)
r_values = solution_A.sol(t_values)[0]
t_min = t_values[np.argmax(r_values >= r2)]
print(f"t_min = {t_min:.4f} s (time for a particle to travel from r1 to r2)")
print()

# Figure A1: K vs radius, with the regime bands marked
r_range = np.linspace(r1, r2, 300)
ae_range = omega ** 2 * r_range
K_range = Dp * (rho * drho * ae_range / mu ** 2) ** (1 / 3)

fig_a1, ax_a1 = plt.subplots(figsize=(7, 5))
ax_a1.axhspan(0, K_stokes_max, color="tab:blue", alpha=0.15, label="Stokes regime")
ax_a1.axhspan(K_stokes_max, K_newton_min, color="tab:green", alpha=0.15, label="Intermediate regime")
ax_a1.axhspan(K_newton_min, K_newton_min + 20, color="tab:red", alpha=0.15, label="Newton regime")
ax_a1.plot(r_range * 1000, K_range, color="black", label="K(r)")
ax_a1.scatter([r1 * 1000, r2 * 1000], [K1, K2], color="black")
ax_a1.set_xlabel("Radial position, r [mm]")
ax_a1.set_ylabel("K [-]")
ax_a1.set_title("Section A1: flow regime across the centrifuge")
ax_a1.set_ylim(0, K_newton_min + 20)
ax_a1.legend()
plt.show()

# Figure A2: particle trajectory r(t), with t_min marked
t_plot = np.linspace(0, t_min * 1.3, 400)
r_plot = solution_A.sol(t_plot)[0]

fig_a2, ax_a2 = plt.subplots(figsize=(7, 5))
ax_a2.plot(t_plot * 1000, r_plot * 1000, color="tab:blue", label="r(t)")
ax_a2.axhline(r2 * 1000, color="gray", linestyle="--")
ax_a2.axvline(t_min * 1000, color="tab:red", linestyle="--")
ax_a2.scatter([t_min * 1000], [r2 * 1000], color="tab:red")
ax_a2.annotate(f"t_min = {t_min * 1000:.1f} ms", (t_min * 1000, r2 * 1000),
               textcoords="offset points", xytext=(-90, -30))
ax_a2.set_xlabel("Time, t [ms]")
ax_a2.set_ylabel("Radial position, r [mm]")
ax_a2.set_title("Section A2: minimal residence time")
ax_a2.legend()
plt.show()


# ======================================================================
# SECTION 3 : PART B - ULTRAFILTRATION
# ======================================================================

# --- B1: protein mass balance across the membrane ---
Qp = permeate_fraction * Qf
Qr = Qf - Qp
Cp_protein = Cf_protein * (1 - R_protein)
Cr_protein = (Qf * Cf_protein - Qp * Cp_protein) / Qr

print("Section B1")
print(f"Qp = {Qp * 3600:.3f} m^3/h, Qr = {Qr * 3600:.3f} m^3/h")
print(f"Cp_protein = {Cp_protein:.1f} g/m^3")
print(f"Cr_protein = {Cr_protein:.0f} g/m^3")
print()

# --- B2: flux vs. pressure, showing the osmotic-pressure ceiling ---
P_osmotic_onset = 20.0   # bar, where the flux plateaus
flux_slope = 1.0          # L/(m^2 h) per bar, in the pressure-controlled region
flux_plateau = 18.0       # L/(m^2 h), plateau value

P_range = np.linspace(0, 80, 400)
J_range = np.minimum(flux_slope * P_range, flux_plateau)
J_onset = min(flux_slope * P_osmotic_onset, flux_plateau)

print("Section B2")
print(f"Flux plateaus once pressure exceeds about {P_osmotic_onset:.0f} bar.")
print()

fig_b, ax_b = plt.subplots(figsize=(7, 5))
ax_b.plot(P_range, J_range, color="tab:blue", label="Water flux")
ax_b.axvline(P_osmotic_onset, color="tab:red", linestyle="--")
ax_b.scatter([P_osmotic_onset], [J_onset], color="tab:red")
ax_b.annotate(f"onset ~ {P_osmotic_onset:.0f} bar", (P_osmotic_onset, J_onset),
              textcoords="offset points", xytext=(10, -25))
ax_b.set_xlabel("Applied pressure, P [bar]")
ax_b.set_ylabel("Flux, J [L/(m^2 h)]")
ax_b.set_title("Section B2: flux vs. pressure")
ax_b.legend()
plt.show()


# ======================================================================
# SECTION 4 : PART C - PERFORMANCE DECLINE (discussion only)
# ======================================================================

print("Section C1")
print("Centrifuge: accumulated solids on the bowl wall reduce the free")
print("volume available for settling, shortening the actual residence")
print("time and allowing small particles to escape with the liquid.")
print("Membrane: fouling (cake formation and pore blocking) increases")
print("the total hydraulic resistance, which lowers the flux at fixed")
print("pressure over time.")
print()


# ======================================================================
# SECTION 5 : PART D - FOULING GROWTH AND FEASIBILITY
# ======================================================================

# --- D1: pressure required to maintain flux after 2 hours ---
Mc_2h = Qf * Cf_cells * t_2h
Rc_2h = alpha * Mc_2h / A_mem
Rtot_2h = Rm + Rc_2h
dP_2h_bar = J_target * mu * Rtot_2h / 1e5

print("Section D1")
print(f"Mc(2h) = {Mc_2h:.2f} kg")
print(f"Rc(2h) = {Rc_2h:.3e} 1/m")
print(f"Rtot(2h) = {Rtot_2h:.3e} 1/m")
print(f"dP(2h) = {dP_2h_bar:.0f} bar")
print()

# --- D2: is this pressure feasible? ---
print("Section D2")
print(f"Membrane pressure limit is about {membrane_pressure_limit:.0f} bar, "
      f"which is much lower than the {dP_2h_bar:.0f} bar required after 2 hours.")
print("Removing the centrifuge to save cost is therefore not reasonable,")
print("since it would push the membrane far beyond its safe operating range.")
print()

t_range = np.linspace(1, t_2h * 1.4, 400)
Mc_range = Qf * Cf_cells * t_range
Rc_range = alpha * Mc_range / A_mem
Rtot_range = Rm + Rc_range
dP_range_bar = J_target * mu * Rtot_range / 1e5

fig_d, ax_d = plt.subplots(figsize=(7, 5))
ax_d.plot(t_range / 3600, dP_range_bar, color="tab:blue", label="Required pressure")
ax_d.axhline(membrane_pressure_limit, color="tab:green", linestyle="--", label="Membrane limit")
ax_d.axvline(2, color="gray", linestyle=":")
ax_d.scatter([2], [dP_2h_bar], color="tab:red")
ax_d.annotate(f"t = 2 h, dP = {dP_2h_bar:.0f} bar", (2, dP_2h_bar),
              textcoords="offset points", xytext=(-140, -10))
ax_d.set_xlabel("Operating time, t [h]")
ax_d.set_ylabel("Required pressure, dP [bar]")
ax_d.set_title("Section D1-D2: pressure required vs. operating time")
ax_d.set_yscale("log")
ax_d.legend()
plt.show()


# ======================================================================
# SECTION 6 : SUMMARY
# ======================================================================

print("Summary")
print(f"A1: K1 = {K1:.2f}, K2 = {K2:.2f} -> Intermediate regime")
print(f"A2: t_min = {t_min * 1000:.1f} ms")
print(f"B1: Cr_protein = {Cr_protein:.0f} g/m^3")
print(f"B2: osmotic-pressure onset ~ {P_osmotic_onset:.0f} bar")
print(f"D1: dP(2h) = {dP_2h_bar:.0f} bar")
print("D2: not feasible, dP(2h) far exceeds the membrane pressure limit")