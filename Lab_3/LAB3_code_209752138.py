import numpy
from scipy import special, optimize
from scipy.optimize import fsolve
import matplotlib.pyplot as plt
import numpy as np
import math as math
from scipy.integrate import solve_ivp
import pandas as pd

# TEST 3 part A
L_COL = 100  # m
TF = 50  # h
D_L = 2.2  # m2/h
Q_IN = 0.21  # m3/h
T2=12 #m
Z2=41 #m



# TEST 3 part B+C

VL = 100
W = 88
C0_PT2 = 0.44

n1 = 1.15
n2 = 0.85

Q_PT3 = 12
VS = 2
KCA = 0.066
TF2 = 50

# ---------- Further Variables – Part A ----------
EPS     = 0.37       # Bed void fraction                             [-]
A_COL   = 0.3        # Column cross-sectional area                   [m2]
C0_COL  = 0.72       # Column inlet concentration                    [mol/m3]
K_EQ    = 5.2e-3     # Equilibrium constant (column)                 [-]
N_STEPS = 1000       # Discretisation points for all plots
Z_OR_T_START = 0.001 # Minimum z or t                                [m]
T_REQ   = 9.42       # Requested evaluation time, Part A Task 1      [h]
Z_REQ   = 37.37      # Requested evaluation position, Part A Task 1  [m]

# ---------- Further Variables – Part A2 Bonus ----------
C2_BONUS = 1.0       # Second-phase inlet concentration              [mol/m3]

# Derived interstitial (pore) velocity
u_col = Q_IN / (A_COL * EPS)   # [m/h]


# ==============================================================
#  PART A – Column Adsorption Functions
# ==============================================================

def t_R(z):
    """
    Calculates the retardation time (tR) at axial position z. 
    Can calculate for a specific or an array of values.

    Parameters:
        z (float | ndarray): axial position/s [m]
    Returns:
        float | ndarray: retardation time/s [h]
    """
    return (z / u_col) * (1.0 + (1.0 - EPS) / EPS * K_EQ)


def C_col(t, z, C0=C0_COL):
    """
    Analytical concentration profile using erf function.
    Can calculate for a specific or an array of values.
    
    Parameters:
        t  (float | ndarray): time [h]
        z  (float | ndarray): position [m]
        C0 (float): inlet concentration [mol/m3]
    Returns:
        float | ndarray: concentration [mol/m3]
    """
    tR    = t_R(z)
    sigma = 2.0 * tR * np.sqrt(D_L / (z * u_col))   

    return 0.5 * C0 * (1.0 + special.erf((t - tR) / sigma))


def t_breakthrough(z, ratio):
    """
    Calculates the breakout time at the given ratio from the original column concentration (ratio = C/C0).
    Uses the invert of the erf (erf^-1 = erfinv).

    Parameters:
        z     (float): axial position [m]
        ratio (float): C/C0 threshold (e.g. 0.01 for 1%, 0.9999 for saturation)
    Returns:
        float: breakthrough time [h]
    """
    tR    = t_R(z)
    sigma = 2.0 * tR * np.sqrt(D_L / (z * u_col))
    return tR + sigma * special.erfinv(2.0 * ratio - 1.0)


# ==============================================================
#  BONUS – Part A2: Used-Column Functions
# ==============================================================

def t_saturation():
    """
    Time when the column exit reaches 99.99 % of C0 (column considered 'full').
    Uses t_breakthrough with ratio = 0.9999.
    """
    return t_breakthrough(L_COL, ratio=0.9999)


def C_used_col(t_arr, z, C1=C0_COL, C2=C2_BONUS, t_sat=None):
    """
    Concentration at position z versus absolute time, for a two-phase process:
      Phase 1 (t ≤ t_sat) : C1 flows through a fresh column.
      Phase 2 (t  > t_sat): C2 injected into the C1-saturated column.

    For a linear (Henry) isotherm the concentration increment (C2-C1) propagates
    with the same retardation tR as C1 on a fresh column, so the phase-2 front
    is simply an erf shifted by t_sat:

        C_used = C1 + (C2-C1) * 0.5 * (1 + erf((tau - tR) / sigma))
    where tau = t - t_sat.

    Parameters:
        t_arr (ndarray): absolute time vector [h]
        z     (float): observation position [m]
        C1    (float): first-phase concentration [mol/m3]
        C2    (float): second-phase concentration [mol/m3]
        t_sat (float): column saturation time [h] (computed if None)
    Returns:
        ndarray: concentration [mol/m3]
    """
    if t_sat is None:
        t_sat = t_saturation()
    t_arr = np.asarray(t_arr, dtype=float)
    tR    = t_R(z)
    sigma = 2.0 * tR * np.sqrt(D_L / (z * u_col))

    # Phase 1: C1 flowing – standard erf breakthrough at z
    C_phase1 = 0.5 * C1 * (1.0 + special.erf((t_arr - tR) / sigma))
    # Phase 2: tau = time elapsed since C2 injection started
    tau      = t_arr - t_sat
    C_phase2 = C1 + (C2 - C1) * 0.5 * (1.0 + special.erf((tau - tR) / sigma))

    return np.where(t_arr <= t_sat, C_phase1, C_phase2)


# ==============================================================
#  PART B – Batch Adsorption: Isotherm Functions
# ==============================================================

def isotherm_henry(C, K):
    """
    Henry (linear) isotherm.

    q = K * C

    Parameters:
        C (float | ndarray): liquid concentration [g/L]
        K (float): Henry constant [L/g]
    Returns:
        float | ndarray: solid loading q [g/g]
    """
    return K * C


def isotherm_langmuir(C, qm, K):
    """
    Langmuir isotherm – monolayer adsorption with finite capacity.

    q = qm * K*C / (1 + K*C)

    Parameters:
        C  (float | ndarray): liquid concentration [g/L]
        qm (float): maximum solid loading [g/g]
        K  (float): equilibrium (affinity) constant [L/g]
    Returns:
        float | ndarray: solid loading q [g/g]
    """
    return qm * K * C / (1.0 + K * C)


def isotherm_freundlich(C, K, n):
    """
    Freundlich isotherm – empirical power-law model.

    q = K * C^n

    n > 1 : unfavourable (convex isotherm, broadening front)
    n < 1 : favourable  (concave isotherm, sharpening front)

    Parameters:
        C (float | ndarray): liquid concentration [g/L]
        K (float): Freundlich constant [L/g]
        n (float): Freundlich exponent [-]
    Returns:
        float | ndarray: solid loading q [g/g]
    """
    return K * C ** n


def operation_line(C, C0=C0_PT2, q0=0.0):
    """
    Mass-balance (operation) line for a batch stirred-tank adsorber.

    q = q0 + (VL/W) * (C0 - C)

    Derived from  C0*VL + q0*W = C*VL + q*W  (total-mass balance).

    Parameters:
        C  (float | ndarray): liquid concentration [g/L]
        C0 (float): initial liquid concentration [g/L]
        q0 (float): initial solid loading [g/g]
    Returns:
        float | ndarray: solid loading q [g/g]
    """
    return q0 + (VL / W) * (C0 - C)


# ==============================================================
#  PART C – Continuous Stirred-Tank ODE System
# ==============================================================

def make_ode_rhs(iso_inv_func):
    """
    Build the ODE right-hand side for the continuous stirred-tank adsorber
    given an inverse isotherm  C* = iso_inv_func(q).

    Governing equations (from the lecture):
      Liquid: VL * dC/dt = Q*(C0-C) - VS*(dq/dt)
      Solid:  VS * dq/dt = KCA*(VL+VS)*(C - C*)
      Equilibrium: q = f(C*)  →  C* = f^{-1}(q)

    Combining: dq/dt = KCA*(VL+VS)*(C - C*) / VS
               dC/dt = [ Q*(C0-C) - VS*dq/dt ] / VL

    Parameters:
        iso_inv_func (callable): inverse isotherm  q → C*  [g/L]
    Returns:
        callable: ode_rhs(t, [C, q]) for solve_ivp
    """
    def ode_rhs(t, y):
        C, q   = y
        C_star = iso_inv_func(q)                            # equilibrium liq. conc.
        dq_dt  = KCA * (VL + VS) * (C - C_star) / VS       # solid-phase rate
        dC_dt  = (Q_PT3 * (C0_PT2 - C) - VS * dq_dt) / VL  # liquid-phase rate
        return [dC_dt, dq_dt]
    return ode_rhs


def linear_interp_crossing(t_arr, q_arr, q_target):
    """
    Find the first time q_arr crosses q_target using linear interpolation
    between the two bracketing points.

    Parameters:
        t_arr    (ndarray): time vector
        q_arr    (ndarray): solid-loading vector
        q_target (float): target solid loading [g/g]
    Returns:
        float: interpolated crossing time, or np.nan if not reached
    """
    for i in range(len(q_arr) - 1):
        if q_arr[i] <= q_target <= q_arr[i + 1]:
            # Linear interpolation: t_cross = t_i + (q_target - q_i)/Δq * Δt
            dt = t_arr[i + 1] - t_arr[i]
            dq = q_arr[i + 1] - q_arr[i]
            return t_arr[i] + (q_target - q_arr[i]) / dq * dt
    return np.nan


# ==============================================================
#  MAIN PIPELINE
# ==============================================================

if __name__ == "__main__":

    # ==========================================================
    #  PART A – Column Adsorption
    # ==========================================================

    # ----------------------------------------------------------
    # Task 1: Concentration at a specific point
    # ----------------------------------------------------------
    C_req = C_col(T_REQ, Z_REQ)
    print(f"[Part A | Task 1]  C(t={T_REQ} h, z={Z_REQ} m) = {C_req:.3f} mol/m3")
    
    # ----------------------------------------------------------
    # Task 2: C vs Time – 10 % column-length steps
    # ----------------------------------------------------------
    z_vals = np.linspace(L_COL / 10, L_COL, 10)   # z = 10%, 20%, ... 100% of L
    t_vals  = np.linspace(Z_OR_T_START, TF, N_STEPS)

    fig, ax = plt.subplots(figsize=(9, 5))
    for z_val in z_vals:
        ax.plot(t_vals, C_col(t_vals, z_val), label=f"z={z_val:.1f} m")
    ax.set_xlabel("Time [hours]")
    ax.set_ylabel("Concentration [mol/m3]")
    ax.set_title("Concentration vs Time")
    ax.legend()
    ax.grid(True)
    plt.tight_layout()
    plt.show()

    # ----------------------------------------------------------
    # Task 3: C vs Position – 10 % TF steps
    # ----------------------------------------------------------
    t_vals = np.linspace(TF / 10, TF, 10)          # t = 10%, 20%, ... 100% of Total Run Time
    z_vals  = np.linspace(Z_OR_T_START, L_COL, N_STEPS)

    fig, ax = plt.subplots(figsize=(9, 5))
    for t_val in t_vals:
        ax.plot(z_vals, C_col(t_val, z_vals), label=f"t={t_val:.1f} hours")
    ax.set_xlabel("Position [m]")
    ax.set_ylabel("Concentration [mol/m3]")
    ax.set_title("Concentration vs Position")
    ax.legend()
    ax.grid(True)
    plt.tight_layout()
    plt.show()

    # ----------------------------------------------------------
    # Task 4: Breakthrough time (C/C0 = 1 %) at column exit
    # ----------------------------------------------------------
    t_BT = t_breakthrough(L_COL, ratio=0.01)
    print(f"[Part A | Task 4]  Breakthrough time for C/C0=1% at z=L: {t_BT:.3f} h")
    
    # ==========================================================
    #  BONUS – Part A2: Used Column
    # ==========================================================

    # Task 1: Saturation time (column exit = 99.99 % of C0)
    t_sat      = t_saturation()
    t_sat_ceil = math.ceil(t_sat)
    print(f"\n[Bonus  | Task 1]  Saturation time (99.99 % of C0): {t_sat:.5f} h")
    print(f"[Bonus  | Task 1]  Rounded-up saturation time      : {t_sat_ceil} h")

    # ----------------------------------------------------------
    # Task 2: C vs Time at z = Z2 (3 curves, axis = 2*ceil(t_sat))
    # ----------------------------------------------------------
    t_max   = 2 * t_sat_ceil
    t_b_arr = np.linspace(0.001, t_max, N_STEPS)
    tR_Z2   = t_R(Z2)
    sig_Z2  = 2.0 * tR_Z2 * np.sqrt(D_L / (Z2 * u_col))

    # Curves 1 & 2: fresh-column breakthroughs (C1 and C2)
    C1_vs_t = 0.5 * C0_COL   * (1.0 + special.erf((t_b_arr - tR_Z2) / sig_Z2))
    C2_vs_t = 0.5 * C2_BONUS * (1.0 + special.erf((t_b_arr - tR_Z2) / sig_Z2))
    # Curve 3: C2 on the C1-saturated column
    Cu_vs_t = C_used_col(t_b_arr, Z2, C0_COL, C2_BONUS, t_sat)

    fig, ax = plt.subplots(figsize=(9, 5))
    ax.plot(t_b_arr, C1_vs_t, label=f"z={Z2}.0 [m] C1 on fresh column")
    ax.plot(t_b_arr, C2_vs_t, label=f"z={Z2}.0 [m] C2 on fresh column")
    ax.plot(t_b_arr, Cu_vs_t, label=f"z={Z2}.0 [m] C2 on used column")
    ax.set_xlabel("Time [hours]")
    ax.set_ylabel("Concentration [mol/m3]")
    ax.set_title("Test 3 – Concentration vs Time\n@ change in initial concentration")
    ax.legend()
    ax.grid(True)
    plt.tight_layout()
    plt.show()

    # ----------------------------------------------------------
    # Task 3: C vs Position at t = T2 (3 curves)
    # T2 is the elapsed time in the C2-injection phase for the used column,
    # and the absolute time from injection start for the fresh-column references.
    # ----------------------------------------------------------
    z_b_arr = np.linspace(0.001, L_COL, N_STEPS)
    tR_z    = t_R(z_b_arr)                                      # array over positions
    sig_z   = 2.0 * tR_z * np.sqrt(D_L / (z_b_arr * u_col))

    C1_vs_z  = 0.5 * C0_COL   * (1.0 + special.erf((T2 - tR_z) / sig_z))
    C2_vs_z  = 0.5 * C2_BONUS * (1.0 + special.erf((T2 - tR_z) / sig_z))
    # Used column: starts at C1, the (C2-C1) increment has propagated for T2 h
    Cu_vs_z  = C0_COL + (C2_BONUS - C0_COL) * 0.5 * (1.0 + special.erf((T2 - tR_z) / sig_z))

    fig, ax = plt.subplots(figsize=(9, 5))
    ax.plot(z_b_arr, Cu_vs_z, label=f"t={T2}.0 [h] C2 on used column")
    ax.plot(z_b_arr, C2_vs_z, label=f"t={T2}.0 [h] C2 on fresh column")
    ax.plot(z_b_arr, C1_vs_z, label=f"t={T2}.0 [h] C1 on fresh column")
    ax.set_xlabel("Position [m]")
    ax.set_ylabel("Concentration [mol/m3]")
    ax.set_title("Test 3 – Concentration vs Position\n@ change in initial concentration")
    ax.legend()
    ax.grid(True)
    plt.tight_layout()
    plt.show()

    # ==========================================================
    #  PART B – Batch Adsorption in a Stirred Tank
    # ==========================================================

    # ----------------------------------------------------------
    # Task 1: Read CSV – Langmuir data in Lineweaver-Burk form
    #         1/q = (1/(qm*K)) * (1/C) + 1/qm
    # ----------------------------------------------------------
    df     = pd.read_csv("synthetic_data_models.csv")
    # Row 0 in the data contains the sub-headers '1/C' / '1/q' → skip it
    inv_C  = df["Test 3"].iloc[1:].astype(float).values
    inv_q  = df["Unnamed: 5"].iloc[1:].astype(float).values

    # Linear regression: slope = 1/(qm*K),  intercept = 1/qm
    slope_lw, intercept_lw = np.polyfit(inv_C, inv_q, 1)
    qm_fit = 1.0 / intercept_lw            # max adsorption capacity [g/g]
    K_fit  = intercept_lw / slope_lw       # equilibrium constant     [L/g]
    print(f"\n[Part B | Task 1]  Langmuir fit (unrounded): qm = {qm_fit:.4f} g/g, "
          f"K = {K_fit:.4f} L/g")

    # Round to 2 decimal places for use in isotherm models
    qm = round(qm_fit, 2)
    K  = round(K_fit,  2)
    print(f"[Part B | Task 1]  Langmuir constants (rounded to 2 d.p.): "
          f"qm = {qm} g/g,  K = {K} L/g")

    # ----------------------------------------------------------
    # Tasks 2–4: Build isotherms, plot, find intersections
    # All four models share the same K (and qm for Langmuir),
    # rounded from the Lineweaver-Burk fit above.
    # ----------------------------------------------------------
    C_plot = np.linspace(0.0, C0_PT2, N_STEPS)   # concentration axis for plot

    q_op   = operation_line(C_plot)
    q_hen  = isotherm_henry(C_plot, K)
    q_lang = isotherm_langmuir(C_plot, qm, K)
    q_fr1  = isotherm_freundlich(C_plot, K, n1)
    q_fr2  = isotherm_freundlich(C_plot, K, n2)

    fig, ax = plt.subplots(figsize=(8, 6))
    ax.plot(C_plot, q_op,   label="Operation line",                           color="blue")
    ax.plot(C_plot, q_hen,  label="Equilibrium line (Henry's law)",           color="orange")
    ax.plot(C_plot, q_lang, label="Equilibrium line (Langmuir model)",        color="green")
    ax.plot(C_plot, q_fr1,  label=f"Equilibrium line (Freundlich model) n={n1}", color="red")
    ax.plot(C_plot, q_fr2,  label=f"Equilibrium line (Freundlich model) n={n2}", color="purple")

    # Four models with their functions (for intersection search)
    isotherms_b = {
        "Henry":               lambda C: isotherm_henry(C, K),
        "Langmuir":            lambda C: isotherm_langmuir(C, qm, K),
        f"Freundlich n={n1}":  lambda C: isotherm_freundlich(C, K, n1),
        f"Freundlich n={n2}":  lambda C: isotherm_freundlich(C, K, n2),
    }

    print("\n[Part B | Task 4]  Equilibrium intersections (max recovery):")
    for name, iso_func in isotherms_b.items():
        # Solve iso(C) = operation_line(C) in the interval (0, C0)
        def residual(C, f=iso_func):
            return f(C) - operation_line(C)

        C_eq = optimize.brentq(residual, 1e-9, C0_PT2 - 1e-9)
        q_eq = operation_line(C_eq)
        print(f"  {name:28s}  C_eq = {C_eq:.5f} g/L,  q_eq = {q_eq:.5f} g/g")
        ax.scatter(C_eq, q_eq, color="black", zorder=5, s=50)   # mark on plot

    ax.set_xlabel("C (g/L)")
    ax.set_ylabel("q (g/g)")
    ax.set_title("Test 3 – Batch adsorption in a stirred tank")
    ax.legend(fontsize=8)
    ax.grid(True)
    plt.tight_layout()
    plt.show()

    # ==========================================================
    #  PART C – Continuous Stirred-Tank Adsorber (ODE)
    # ==========================================================

    t_c_arr = np.linspace(0.0, TF2, N_STEPS)   # time vector [s]
    y0      = [0.0, 0.0]                        # initial state: C(0)=0, q(0)=0

    # Inverse isotherms C* = f^{-1}(q) – needed by the ODE solid balance
    inv_iso_funcs = {
        "Henry":               lambda q: q / K,
        "Langmuir":            lambda q: q / (K * (qm - q)),     # C*=q/(K*(qm-q))
        f"Freundlich n={n1}":  lambda q: (q / K) ** (1.0 / n1),
        f"Freundlich n={n2}":  lambda q: (q / K) ** (1.0 / n2),
    }

    # 25 % of Langmuir qm is the target solid loading
    q_target = 0.25 * qm
    print(f"\n[Part C | Task 3]  Target: 25% of qm = {q_target} g/g")

    fig_c, axes_c = plt.subplots(1, 2, figsize=(13, 5))

    # No-adsorption reference: simple CSTR dC/dt = Q*(C0-C)/VL
    C_no_ads = C0_PT2 * (1.0 - np.exp(-Q_PT3 / VL * t_c_arr))
    axes_c[0].plot(t_c_arr, C_no_ads, color="black", label="no_adsorption")

    best_name, best_time = None, np.inf

    for name, inv_func in inv_iso_funcs.items():
        sol = solve_ivp(
            make_ode_rhs(inv_func),
            t_span=(0.0, TF2),
            y0=y0,
            t_eval=t_c_arr,
            method="RK45",
            rtol=1e-8,
            atol=1e-10,
        )
        C_sol = sol.y[0]
        q_sol = sol.y[1]

        # Plot liquid concentration
        axes_c[0].plot(t_c_arr, C_sol, label=name)
        # Plot solid loading
        axes_c[1].plot(t_c_arr, q_sol, label=name)

        # Linear interpolation to find when q first reaches q_target
        t_cross = linear_interp_crossing(t_c_arr, q_sol, q_target)
        print(f"  {name:28s}  t(q = 25% qm) = {t_cross:.5f} s")

        if t_cross < best_time:
            best_time = t_cross
            best_name = name

    print(f"[Part C | Task 3]  Fastest model to reach q_target: {best_name} "
          f"(t = {best_time:.5f} s)")

    axes_c[0].set_xlabel("Time [s]")
    axes_c[0].set_ylabel("C (g/L)")
    axes_c[0].set_title("Test 3 – Concentration in liquid over time")
    axes_c[0].legend(fontsize=8)
    axes_c[0].grid(True)

    axes_c[1].set_xlabel("Time [s]")
    axes_c[1].set_ylabel("q (g/g)")
    axes_c[1].set_title("Test 3 – Concentration on solid over time")
    axes_c[1].legend(fontsize=8)
    axes_c[1].grid(True)

    plt.tight_layout()
    plt.show()
    