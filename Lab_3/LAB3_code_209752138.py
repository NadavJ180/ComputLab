import numpy
from scipy import special
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

# ---------- Further Variables - Part A ----------
EPS     = 0.37       # Bed void fraction                             [-]
A_COL   = 0.3        # Column cross-sectional area                   [m2]
C0_COL  = 0.72       # Column inlet concentration                    [mol/m3]
K_EQ    = 5.2e-3     # Equilibrium constant (column)                 [-]
N_STEPS = 1000       # Discretisation points for all plots
Z_OR_T_START = 0.001 # Minimum z or t                                [m]
T_REQ   = 9.42       # Requested evaluation time, Part A Task 1      [h]
Z_REQ   = 37.37      # Requested evaluation position, Part A Task 1  [m]

# ---------- Further Variables - Part A2 Bonus ----------
C2_BONUS = 1.0       # Second-phase inlet concentration              [mol/m3]

# Derived interstitial (pore) velocity
u_col = Q_IN / (A_COL * EPS)   # [m/h]


# ==================== Part A - Column Adsorption Functions ============================

# -------------------- Q1: Find concentration at a specific point --------------------

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
    tR    = t_R(z)                                      # Retardation time at z
    sigma = 2.0 * tR * np.sqrt(D_L / (z * u_col))   

    return 0.5 * C0 * (1.0 + special.erf((t - tR) / sigma))

# -------------------- Q4: Breakthrough time (C/C0 = 1%) --------------------

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
    tR    = t_R(z)                                      # Retardation time at z
    sigma = 2.0 * tR * np.sqrt(D_L / (z * u_col))
    return tR + sigma * special.erfinv(2.0 * ratio - 1.0)


# ==================== BONUS - Part A2: Used-Column Functions ====================

# -------------------- Q1: Saturation time --------------------

def t_saturation():
    """
    Time when the column exit reaches 99.99 % of C0.
    Uses t_breakthrough from part A with ratio = 0.9999.
    """
    return t_breakthrough(L_COL, ratio=0.9999)

# -------------------- Q2 + Q3: C vs Time, Position at z = Z2, t = T2 --------------------

def C_used_col(t_arr, z, C1=C0_COL, C2=C2_BONUS):
    """
    Concentration at point z versus absolute time, for a two-phase process:
      Phase 1 (t <= t_sat) : C1 flows through a fresh column.
      Phase 2 (t  > t_sat): C2 is injected into the C1-saturated column.

    Since we are using a similar approximation to part A using the erf, we are 
    essentially shifting the erf by C2-C1, with C0 = C1 and the time in the erf is
    the shifted time from time of new injection (for phase 2).

    Parameters:
        t_arr (ndarray): absolute time vector [h]
        z     (float): position [m]
        C1    (float): first-phase concentration [mol/m3]
        C2    (float): second-phase concentration [mol/m3]
        t_sat (float): column saturation time [h] 
    Returns:
        ndarray: concentration [mol/m3]
    """
    
    t_sat = t_saturation()                                                      # Saturation time for 99.99% C0
    t_arr = np.asarray(t_arr, dtype=float)                                      # Makes sure this is np array
    tR    = t_R(z)                                                              # Retardation time at z
    sigma = 2.0 * tR * np.sqrt(D_L / (z * u_col))

    C_phase1 = 0.5 * C1 * (1.0 + special.erf((t_arr - tR) / sigma))             # C1 flowing - standard erf breakthrough at z
    tau      = t_arr - t_sat                                                    # Time elapsed since C2 injection
    C_phase2 = C1 + (C2 - C1) * 0.5 * (1.0 + special.erf((tau - tR) / sigma))   # C2 flowing - shifted erf  

    return np.where(t_arr <= t_sat, C_phase1, C_phase2)                         # Similar to if-else, returns new array of the combined concetnrations


# ==================== PART B - Batch Adsorption: Isotherm Functions ====================

# -------------------- Q1: Read CSV, find q_m & K --------------------

def import_CSV(Test_Num=None, q_inv_col=None):
    """
    Imports the CSV for the specified test of the linearized Langmuir isotherm (Linweaver-Burk)
    
    1/q = 1/(C*q_m*K) + 1/q_m 

    We import without row 0 (contains titles). If the test number is not supplied, asks for manual input of test number.

    Parameters:
        Test_Num (int): the test number we are currently running
    Returns:
        inv_C (ndarray): inverted concentration of the relevant test [L/g]
        inv_q (ndarray): inverted final content in adsorbant [g/g] 
    """
    if Test_Num is None:
        Test_Num = int(input("Input relevant test number (1 / 2 / 3): "))
    

    df = pd.read_csv("synthetic_data_models.csv") 
    col_num_dict = {1: 1, 2: 3}
    inv_C  = df[f"Test {Test_Num}"].iloc[1:].astype(float).values           # Skip row 0, contains titles
    inv_q  = df[f"Unnamed: {2*Test_Num-1}"].iloc[1:].astype(float).values   # Imports the row that corresponds to the test number

    return inv_C, inv_q

# -------------------- Q2: Functions for Henry, Langmuir and Freundlich (n>1 + n<1) adsorption models --------------------

def henry_isotherm(C, K):
    """
    Function for the Henry isotherm.

    Parameters:
        C (float | ndarray): liquid concentration [g/L]
        K (float): Henry constant [L/g]
    Returns:
        q (float | ndarray): final content in adsorbant [g/g]
    """
    q = K * C

    return q

def langmuir_isotherm(C, qm, K):
    """
    Function for the Langmuir isotherm.

    Parameters:
        C  (float | ndarray): liquid concentration [g/L]
        qm (float): maximum content in adsorbant [g/g]
        K  (float): equilibrium (affinity) constant [L/g]
    Returns:
        q (float | ndarray): final content in adsorbant [g/g]
    """
    q = qm * K * C / (1.0 + K * C)

    return q 

def freundlich_isotherm(C, K, n):
    """
    Function for the Freundlich isotherm.

    Parameters:
        C (float | ndarray): liquid concentration [g/L]
        K (float): Freundlich constant [L/g]
        n (float): Freundlich exponent [-]
    Returns:
        q (float | ndarray): final content in adsorbant [g/g]
    """
    C_safe = np.maximum(C, 0.0)         # Avoids unphyical concetrations (negetive values) in the numerical solutions
    q = K * C_safe ** n

    return q

# -------------------- Q3-4: q vs C in all models + their maximum retentions --------------------

def operation_line(C, C0=C0_PT2, q0=0.0):
    """
    Operation line used to find the maximal recovery when equilibrim is reached.
    
    Parameters:
        C  (float | ndarray): liquid concentration [g/L]
        C0 (float): initial liquid concentration [g/L]
        q0 (float): initial content in adsorbant [g/g]
    Returns:
        q (float | ndarray): final content in the adsorbant (at a specific liquid concetration) [g/g]
    """
    q = q0 + (VL / W) * (C0 - C)

    return q

def isotherm_residual(C, func, model_args):
    """
    Calculates the residual between the equilibrium isotherm and the operation line. 
    Used as input for fsolve to find the intersection between them.

    Parameters:
        C  (float | ndarray): liquid concentration [g/L]
        func (function): one of the isotherms (henry/langmuir/freundlich) [g/L]
        model_args (tuple): further arguments for the model equation (q_m / K / n)

    Returns:
        Res (float): Residual between isotherm and operation line
    """
    Res = func(C, *model_args) - operation_line(C)
    
    return Res 

# ==================== PART C - Continuous Stirred-Tank ODE functions ====================

def henry_inv(q, K):
    """
    Function for the inverted Henry isotherm.

    Parameters:
        q (float | ndarray): final content in adsorbant [g/g]    
        K (float): Henry constant [L/g]
    Returns:
        C (float | ndarray): liquid concentration [g/L]
    """
    C = q / K

    return C

def langmuir_inv(q, K, q_m):
    """
    Function for the iverted Langmuir isotherm.
    np.maximum is used to avoid numerical error of division by zero.

    Parameters:
        q (float | ndarray): final content in adsorbant [g/g]
        qm (float): maximum content in adsorbant [g/g]
        K  (float): equilibrium (affinity) constant [L/g]
    Returns:
        C  (float | ndarray): liquid concentration [g/L]
    """
    C = q / (K * np.maximum(q_m - q, 1e-10))
    
    return C 

def freundlich_inv(q, K, n):
    """
    Function for the inverted Freundlich isotherm.
    q_safe is used to avoid unphysical values (q<0).

    Parameters:
        q (float | ndarray): final content in adsorbant [g/g]
        K (float): Freundlich constant [L/g]
        n (float): Freundlich exponent [-]
    Returns:
        C (float | ndarray): liquid concentration [g/L]
    """
    q_safe = np.maximum(q, 0.0)
    C = (q_safe / K) ** (1.0 / n)
    
    return C

def adsorber_ode(t, y, inv_func, model_args):
    """
    The initial value ODE of the mass balance in the liquid and solid that we will solve using solve.ivp.
    We calculate C* using the inverted isotherms.
    """
    C, q   = y
    
    C_star = inv_func(q, *model_args)                  # Equilibrium liquid concentration at specific q 
    
    dq_dt  = KCA * (VL + VS) * (C - C_star) / VS       # Solid-phase rate
    dC_dt  = (Q_PT3 * (C0_PT2 - C) - VS * dq_dt) / VL  # Liquid-phase rate
    
    return [dC_dt, dq_dt]

def linear_interp_crossing(t_arr, q_arr, q_target):
    """
    Find the first time q_arr crosses q_target using linear interpolation
    between the two bracketing points.

    Parameters:
        t_arr    (ndarray): time vector
        q_arr    (ndarray): solid-loading vector
        q_target (float): target content in the adsorbant [g/g]
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
# ===================== MAIN PIPELINE ==========================
# ==============================================================

if __name__ == "__main__":

    # ==================== PART A - Column Adsorption ====================

    # -------------------- Q1: Find concentration at a specific point --------------------
    C_req = C_col(T_REQ, Z_REQ)
    print(f"[Part A | Q1]  C(t={T_REQ} [h], z={Z_REQ} [m]) = {C_req:.3f} [mol/m3]")
    
    # -------------------- Q2: C vs Time - 10 % column-length steps --------------------

    z_vals = np.linspace(L_COL / 10, L_COL, 10)   # z jumps of 10%
    t_vals_c  = np.linspace(Z_OR_T_START, TF, N_STEPS)

    fig, ax = plt.subplots(figsize=(9, 5))
    for z_val in z_vals:                                                    # Plot a curve for each position increment
        ax.plot(t_vals_c, C_col(t_vals_c, z_val), label=f"z={z_val:.1f} [m]")
    ax.set_xlabel("Time [hours]")
    
    ax.set_ylabel("Concentration [mol/m3]")
    ax.set_title("Concentration vs Time")
    ax.legend()
    ax.grid(True)
    plt.tight_layout()
    plt.show()

    
    # -------------------- Q3: C vs Position - 10 % TF steps --------------------

    t_vals_c = np.linspace(TF / 10, TF, 10)          # t jumps of 10%
    z_vals  = np.linspace(Z_OR_T_START, L_COL, N_STEPS)

    fig, ax = plt.subplots(figsize=(9, 5))
    for t_val in t_vals_c:                                                        # Plot a curve for each time increment
        ax.plot(z_vals, C_col(t_val, z_vals), label=f"t={t_val:.1f} hours")
    
    ax.set_xlabel("Position [m]")
    ax.set_ylabel("Concentration [mol/m3]")
    ax.set_title("Concentration vs Position")
    ax.legend()
    ax.grid(True)
    plt.tight_layout()
    plt.show()

    
    # -------------------- Q4: Breakthrough time (C/C0 = 1%) --------------------
    
    t_BT = t_breakthrough(L_COL, ratio=0.01)
    print(f"[Part A | Q4]  Breakthrough time for C/C0=1% at z=L: {t_BT:.3f} [h]")
    

    # ==================== BONUS - Part A2: Used Column ====================

    # -------------------- Q1: Saturation time --------------------
    t_sat      = t_saturation()                                                 # Time for exit concentration to br 99.99% C0
    t_sat_ceil = math.ceil(t_sat)
    print(f"\n[Bonus  | Q1]  Saturation time (99.99 % of C0): {t_sat:.5f} [h]")
    print(f"[Bonus  | Q1]  Rounded-up saturation time      : {t_sat_ceil} [h]")


    # -------------------- Q2: C vs Time at z = Z2 --------------------

    t_max   = 2 * t_sat_ceil                            
    t_vals_c = np.linspace(Z_OR_T_START, t_max, N_STEPS)       

    C1_vs_t = C_col(t_vals_c, Z2, C0_COL)                              # Entrance concentration of C0
    C2_vs_t = C_col(t_vals_c, Z2, C2_BONUS)                            # Entrrance concentration of C2
    C_used_vs_t = C_used_col(t_vals_c, Z2, C0_COL, C2_BONUS)           # Entrance concentration C2 on used column

    fig, ax = plt.subplots(figsize=(9, 5))
    ax.plot(t_vals_c, C1_vs_t, label=f"z={Z2:.1f} [m] C1 on fresh column")
    ax.plot(t_vals_c, C2_vs_t, label=f"z={Z2:.1f} [m] C2 on fresh column")
    ax.plot(t_vals_c, C_used_vs_t, label=f"z={Z2:.1f} [m] C2 on used column")
    
    ax.axvspan(0, t_sat_ceil, color='gold', alpha=0.15)                     # color background up to t_sat
    ax.axvspan(t_sat_ceil, t_vals_c[-1], color='gray', alpha=0.15)            # color background from t_sat
    ax.set_xlabel("Time [hours]")
    ax.set_ylabel("Concentration [mol/m3]")
    ax.set_title("Concentration vs Time\n@ change in initial concentration")
    ax.legend()
    ax.grid(True)
    plt.tight_layout()
    plt.show()


    # -------------------- Q3: C vs Position at t = T2 --------------------
    
    z_vals = np.linspace(Z_OR_T_START, L_COL, N_STEPS)

    C1_vs_z  = C_col(T2, z_vals, C0_COL)                                # Entrance concentration of C0
    C2_vs_z  = C_col(T2, z_vals, C2_BONUS)                              # Entrrance concentration of C2
    C_used_vs_z  = C_used_col(t_sat + T2, z_vals, C0_COL, C2_BONUS)     # Entrance concentration C2 on used column

    fig, ax = plt.subplots(figsize=(9, 5))
    ax.plot(z_vals, C_used_vs_z, label=f"t={T2:.1f} [h] C2 on used column")
    ax.plot(z_vals, C2_vs_z, label=f"t={T2:.1f} [h] C2 on fresh column")
    ax.plot(z_vals, C1_vs_z, label=f"t={T2:.1f} [h] C1 on fresh column")
    
    ax.set_xlabel("Position [m]")
    ax.set_ylabel("Concentration [mol/m3]")
    ax.set_title("Concentration vs Position\n@ change in initial concentration")
    ax.legend()
    ax.grid(True)
    plt.tight_layout()
    plt.show()

    # ==================== PART B - Batch Adsorption in a Stirred Tank ====================

    # -------------------- Q1: Read CSV, find q_m & K --------------------
    
    VL_dict = {50: 1, 122: 2, 100: 3}     # Dictionary of which solution volume corresponds to which test number
    Test_Num = VL_dict.get(VL, None)      # Finds the test number using the solution volume, if not one of the OG tests, returns None
    inv_C, inv_q = import_CSV(Test_Num)

    slope, intercept = np.polyfit(inv_C, inv_q, 1)                  # Linear regression for Linweaver Berk
    
    q_m = 1.0 / intercept                                           # max adsorption capacity [g/g]
    K = intercept / slope                                           # equilibrium constant [L/g]        K = (1/qm) / (1/qm*K) 

    print(f"\n[Part B | Q1] Linear Langmuir constants (unrounded): qm = {q_m} [g/g], "
          f"K = {K} [L/g]")
    
    q_m_round = round(q_m, 2)                                 
    K_round  = round(K, 2)                                
    

    # -------------------- Q3-4: q vs C in all models + their maximum recovery --------------------

    C_vals = np.linspace(0.0, C0_PT2, N_STEPS)   # concentration values for plot

    q_op   = operation_line(C_vals)
    q_hen  = henry_isotherm(C_vals, K_round)
    q_lang = langmuir_isotherm(C_vals, q_m_round, K_round)
    q_fr1  = freundlich_isotherm(C_vals, K_round, n1)
    q_fr2  = freundlich_isotherm(C_vals, K_round, n2)

    fig, ax = plt.subplots(figsize=(8, 6))
    ax.plot(C_vals, q_op,   label="Operation line", color="blue")
    ax.plot(C_vals, q_hen,  label="Equilibrium line (Henry's law)", color="orange")
    ax.plot(C_vals, q_lang, label="Equilibrium line (Langmuir model)", color="green")
    ax.plot(C_vals, q_fr1,  label=f"Equilibrium line (Freundlich model) n={n1}", color="red")
    ax.plot(C_vals, q_fr2,  label=f"Equilibrium line (Freundlich model) n={n2}", color="purple")

    # Dictionary for the model functions and their arguments
    isotherms = {
        "Henry": (henry_isotherm, (K_round,)),
        "Langmuir": (langmuir_isotherm, (q_m_round, K_round)),
        f"Freundlich n={n1}": (freundlich_isotherm, (K_round, n1)),
        f"Freundlich n={n2}": (freundlich_isotherm, (K_round, n2)),
    }

    print("\n[Part B | Q4]  Equilibrium intersections (max recovery):")
    
    for name, (func, model_args) in isotherms.items():
        initial_guess = C0_PT2 / 2                                          # Initial guess in middle of operating range
        
        C_eq_arr = fsolve(isotherm_residual, 
                          initial_guess, args=(func, model_args))           # Finds the intersection point
        C_eq = C_eq_arr[0]                                                  # Turns the array output of fsolve into a float
        
        q_eq = operation_line(C_eq)

        print(f"{name}: C_eq = {C_eq:.5f} [g/L],  q_eq = {q_eq:.5f} [g/g]")
        ax.scatter(C_eq, q_eq, color="black", zorder=5)                     # Plot the intersetion points on the plot

    ax.set_xlabel("C (g/L)")
    ax.set_ylabel("q (g/g)")
    ax.set_title("Batch adsorption in a stirred tank")
    ax.legend()
    ax.grid(True)
    plt.tight_layout()
    plt.show()

    # ==================== PART C - Continuous Stirred-Tank Adsorber (ODE) ====================

    t_vals_c = np.linspace(0.0, TF2, N_STEPS)    # Time values [s]
    y0 = [0.0, 0.0]                              # Initial conditions: C(0)=0, q(0)=0

    # Dictionary for the inverse model functions and their arguments
    inv_iso_funcs = {
        "Henry": (henry_inv, (K_round,)),
        "Langmuir": (langmuir_inv, (K_round, q_m_round)),
        f"Freundlich n={n1}": (freundlich_inv, (K_round, n1)),
        f"Freundlich n={n2}": (freundlich_inv, (K_round, n2)),
    }

    q_target = 0.25 * q_m_round
    print(f"\n[Part C | Task 3]  Target: 25% of qm = {q_target} g/g")

    fig_c, axes_c = plt.subplots(1, 2, figsize=(13, 5))

    # No-adsorption reference
C_no_ads = C0_PT2 * (1.0 - np.exp(-Q_PT3 / VL * t_vals_c))
axes_c[0].plot(t_vals_c, C_no_ads, color="black", label="no_adsorption")

best_name, best_time = None, np.inf

for name, (inv_func, model_args) in inv_iso_funcs.items():
    # Pass the function and its extra arguments cleanly into solve_ivp
    sol = solve_ivp(
        adsorber_ode,
        t_span=(0.0, TF2),
        y0=y0,
        t_eval=t_vals_c,
        method="RK45",
        rtol=1e-8,
        atol=1e-10,
        args=(inv_func, model_args)  # <-- This safely bridges variables to the ODE
    )
    
    C_sol = sol.y[0]
    q_sol = sol.y[1]

    # Plotting
    axes_c[0].plot(t_vals_c, C_sol, label=name)
    axes_c[1].plot(t_vals_c, q_sol, label=name)

    # Intersection calculation
    t_cross = linear_interp_crossing(t_vals_c, q_sol, q_target)
    print(f"  {name:28s}  t(q = 25% qm) = {t_cross:.5f} s")

    if t_cross < best_time:
        best_time = t_cross
        best_name = name

print(f"[Part C | Task 3]  Fastest model to reach q_target: {best_name} (t = {best_time:.5f} s)")

# Graph decorations
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
    