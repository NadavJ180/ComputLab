import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp


# ======================================================================
# DATA and PARAMTERS
# ======================================================================

# ---- General data ----
rho = 1000.0                # liquid density [kg/m^3]
rho_p = 1150.0               # debris particle density [kg/m^3]
mu = 1e-3                     # liquid viscosity [Pa*s]
Qf = 1.2 / 3600               # feed volumetric flow rate [m^3/s]
Cf_cells = 2.0                # debris concentration in the feed [kg/m^3]
Cf_protein = 2.0e3            # protein concentration in the feed [g/m^3]

# ---- Section A data ----
Dp = 100e-6                   # minimal particle diameter to separate [m]
omega = 2000.0                # angular velocity of the centrifuge [rad/s]
r1 = 0.02                     # liquid surface radius [m]
r2 = 0.06                     # centrifuge outer radius [m]
K_stokes_max = 3.3            # upper K boundary of the Stokes regime
K_newton_min = 43.6           # lower K boundary of the Newton regime

# ---- Section B data ----
R_protein = 0.93              # membrane rejection towards the protein [-]
permeate_fraction = 0.30      # fraction of feed leaving as permeate, Qp/Qf [-]

# ---- Section D data ----
alpha = 5e13                  # specific cake resistance [m/kg]
Rm = 2e12                     # intrinsic membrane resistance [1/m]
A_mem = 2.22                  # membrane area [m^2]
membrane_pressure_limit = 10.0  # safe operating pressure limit of an UF membrane [bar]


if __name__ == "__main__":
    # ======================================================================
    # PART A - CENTRIFUGATION
    # ======================================================================

    # --- A1: identify the flow regime at the two extreme radii ---

    delta_rho = rho_p - rho         # density difference between particles and fluid [kg/m^3]

    ae1 = omega ** 2 * r1           # centrifugal acceleration at r1 [m/s^2]
    ae2 = omega ** 2 * r2           # centrifugal acceleration at r2 [m/s^2]

    K1 = Dp * (rho * delta_rho * ae1 / mu ** 2) ** (1 / 3)      # K parameter at r1
    K2 = Dp * (rho * delta_rho * ae2 / mu ** 2) ** (1 / 3)      # K parameter at r2

    print("="*20 + " Section A1 " + "="*20)
    print(f"The K parameter of the centrifuge falls in the following range:")
    print(f"K1 = {K1:.2f} <= K <= K2 = {K2:.2f}\n")

    if K_stokes_max < K1 and K2 < K_newton_min:
        print("Both values fall in the Intermediate regime (3.3 <= K <= 43.6).")
    print()

    # Figure A1: K vs radius, with the regime bands marked

    r_range = np.linspace(r1, r2, 300)
    ae_range = (omega ** 2) * r_range
    K_range = Dp * ( (rho * delta_rho * ae_range) / (mu ** 2) ) ** (1 / 3)

    fig_a1, ax_a1 = plt.subplots(figsize=(7, 5))
    ax_a1.axhspan(0, K_stokes_max, color="tab:blue", alpha=0.15, label="Stokes regime")
    ax_a1.axhspan(K_stokes_max, K_newton_min, color="tab:green", alpha=0.15, label="Intermediate regime")
    ax_a1.axhspan(K_newton_min, K_newton_min + 20, color="tab:red", alpha=0.15, label="Newton regime")
    ax_a1.plot(r_range * 1000, K_range, color="black", label="K(r)")
    ax_a1.scatter([r1 * 1000, r2 * 1000], [K1, K2], color="black", zorder=5)
    
    ax_a1.set_xlabel("Radial position, r [mm]")
    ax_a1.set_ylabel("K [-]")
    ax_a1.set_title("Section A1: flow regime across the centrifuge")
    ax_a1.set_ylim(0, K_newton_min + 20)
    ax_a1.legend()
    ax_a1.grid()
    plt.show()


    # --- A2: minimal residence time in the Intermediate regime ---

    C = 0.153 * ((omega ** 2) ** 0.71) * (Dp ** 1.14) * (delta_rho ** 0.71) \
            * (rho ** -0.29) * (mu ** -0.43)                                      
    # Units - [m^0.29/2]

    def settling_ode(t, r):
        """Right-hand side of the settling equation dr/dt = C * r^0.71."""
        return C * r ** 0.71


    solution_A = solve_ivp(settling_ode, t_span=(0, 0.05), y0=[r1], max_step=1e-5, dense_output=True)

    t_values = np.linspace(0, 0.05, 200000)
    r_values = solution_A.sol(t_values)[0]
    t_min = t_values[np.argmax(r_values >= r2)]

    print("="*20 + " Section A2 " + "="*20)
    print(f"Minimal time for the waste particles to travel from r1={r1} [m] to r2={r2} [m] is:")
    print(f"t_min = {t_min:.4f} [s]")
    print()


    # Figure A2: particle trajectory r(t), with t_min marked

    t_plot = np.linspace(0, t_min * 1.3, 400)
    r_plot = solution_A.sol(t_plot)[0]

    fig_a2, ax_a2 = plt.subplots(figsize=(7, 5))
    ax_a2.plot(t_plot, r_plot, color="tab:blue", label="r(t)")
    ax_a2.axhline(r2, color="gray", linestyle="--")
    ax_a2.scatter([t_min], [r2], color="tab:red", zorder=5)
    
    ax_a2.annotate(f"t_min = {t_min:.3f} [s]", (t_min, r2),
                textcoords="offset points", xytext=(-90, 5))
    ax_a2.annotate(f"r2 = {r2:.3f} [m]", (0, r2),
                textcoords="offset points", xytext=(-10, 5))
    ax_a2.set_xlabel("Retention Time, t [s]")
    ax_a2.set_ylabel("Radial position, r [m]")
    ax_a2.set_title("Section A2: minimal residence time")
    ax_a2.grid()
    plt.show()


    # ======================================================================
    # PART B - ULTRAFILTRATION
    # ======================================================================

    # --- B1: protein mass balance across the membrane ---

    Qp = permeate_fraction * Qf                             # permeate volumetric flow rate [m^3/s]
    Qr = Qf - Qp                                            # retentate volumetric flow rate [m^3/s]
    Cp_protein = Cf_protein * (1 - R_protein)               # protein concentration in permeate [g/m^3]
    Cr_protein = (Qf * Cf_protein - Qp * Cp_protein) / Qr   # protein concentration in retentate [g/m^3]

    print("="*20 + " Section B1 " + "="*20)
    print(f"The concentration of the protein in the retentate is:")
    print(f"Cr_protein = {Cr_protein:.0f} [g/m^3]")
    print()

    

    # --- B2: retentate concentration vs. processing time ---
    
    V_feed = 1.2                                        # volume of feed after 1 hour at OG pressure [m^3] 

    Qp_normal = permeate_fraction * Qf  * 3600          # normal operating pressure [m^3/h]
    Qp_fast = Qp_normal * 1.5                           # 50% increase in Qp due to pressure increase [m^3/s]
    
    t_max_hours = (0.30 * V_feed) / Qp_normal           # max time to reach the given 30% recovery under normal conditions
    time_hours = np.linspace(0, t_max_hours, 400)
    
    recovery_normal = (Qp_normal * time_hours) / V_feed
    recovery_fast = (Qp_fast * time_hours) / V_feed
    
    # Calculate retentate concentration over time
    Cr_normal = (Cf_protein - recovery_normal * Cp_protein) / (1 - recovery_normal)
    Cr_fast = (Cf_protein - recovery_fast * Cp_protein) / (1 - recovery_fast)

    print("="*20 + " Section B2 " + "="*20)
    print(f"Plotting concentration over time for a {V_feed} [m^3] batch.")
    print(f"We can explicitly see in the plot, that increasing the recovery (higher permeate flow)\n"
          f"will directly correlate to less time needed to reach the same retentate concentration.")
    print()

    # --- Plotting ---
    fig_b, ax_b = plt.subplots(figsize=(7, 5))
    
    # Plot both scenarios to show how increasing pressure speeds up the process
    ax_b.plot(time_hours, Cr_normal, color="tab:blue", label="Normal Pressure (Standard Qp)")
    ax_b.plot(time_hours, Cr_fast, color="tab:orange", label="Increased Pressure (Higher Qp)")
    ax_b.axhline(Cr_protein, color="black", linestyle="--")
    ax_b.annotate(f"Cr_protein = {Cr_protein:.0f} [g/m^3]", 
                  (0, Cr_protein), xytext=(0, 5), textcoords="offset points")
    
    ax_b.set_xlabel("Processing Time, t [hours]")
    ax_b.set_ylabel("Retentate protein concentration, Cr [g/m^3]")
    ax_b.set_title("Section B2: Retentate Concentration vs. Time")
    ax_b.legend()
    ax_b.grid()
    plt.show()


    # ======================================================================
    # PART C - PERFORMANCE DECLINE OVER TIME
    # ======================================================================

    print("="*20 + " Section C1 " + "="*20)
    print(f"Assuming that 0.1% of the cell debris reaches the membrane, we can see a non-negligibale decline\n"
           "in permeate flux after only 3 hours, becuase of cake resistance (fouling).")

    P_fixed_bar = membrane_pressure_limit                   # fixed applied pressure [bar]
    t_range_C = np.linspace(1, 3 * 3600, 400)               # operating time [s]
    Mc_range_C = Qf * (Cf_cells * 0.001) * t_range_C        # accumulated mass in the cake [kg] (assuming 0.1% of OG cell concent.)
    Rc_range_C = alpha * Mc_range_C / A_mem                 # resistance of the cake [1/m]
    Rtot_range_C = Rm + Rc_range_C                          # total resistance [1/m]
    J_range_C = (P_fixed_bar * 1e5) / (mu * Rtot_range_C)   # flux [m/s]
    J_range_C_Lm2h = J_range_C * 3600                       # flux [m^3/(m^2 h)]

    fig_c, ax_c = plt.subplots(figsize=(7, 5))
    ax_c.plot(t_range_C / 3600, J_range_C_Lm2h, color="tab:blue")
    ax_c.set_xlabel("Operating time, t [h]")
    ax_c.set_ylabel("Permeate flux, J [m^3/(m^2 h)]")
    ax_c.set_title(f"Section C1: flux decline over time at fixed pressure ({P_fixed_bar:.0f} bar)")
    ax_c.grid()
    plt.show()


    # ======================================================================
    # PART D - FOULING GROWTH AND FEASIBILITY
    # ======================================================================

    # --- D1: pressure required to maintain flux after 2 hours ---

    _2h_to_s = 2 * 3600                         # 2 hours, in seconds [s]

    Mc_2h = Qf * Cf_cells * _2h_to_s            # mass of depris accumulated on the membrane after 2 hours [kg]
    Rc_2h = alpha * Mc_2h / A_mem               # cake resistance after 2 hours [1/m]
    Rtot_2h = Rm + Rc_2h                        # total resistance after 2 hours [1/m]

    J_p = (Qf * permeate_fraction) / A_mem      # permeate flux [m/s]
    dP_2h_bar = (J_p * mu * Rtot_2h) / 1e5      # pressure required to maintain the target flux after 2 hours [bar]

    print("="*20 + " Section D1 " + "="*20)
    print(f"The difference in pressure required to maintain the target flux (J={J_p:.6f} [m/s]) after 2 hours, is:")
    print(f"dP(t=2h) = {dP_2h_bar:.0f} bar")
    print()

    # --- D2: is this pressure feasible? ---

    print("="*20 + " Section D2 " + "="*20)
    print(f"Membrane pressure limit is about {membrane_pressure_limit:.0f} bar,\n"
        f"we can see that without the centrifuge we reach this limit in less than"
        f" half an hour!")
    print("Removing the centrifuge to save cost is therefore not reasonable,")
    print("since it would push the membrane far beyond its safe operating range.")
    print()

    t_range = np.linspace(1, _2h_to_s * 1.4, 400)
    Mc_range = Qf * Cf_cells * t_range              # mass of depris accumulated on the membrane [kg]
    Rc_range = alpha * Mc_range / A_mem             # cake resistance [1/m]
    Rtot_range = Rm + Rc_range                      # total resistance [1/m]
    dP_range_bar = J_p * mu * Rtot_range / 1e5      # pressure required to maintain the target flux [bar]

    fig_d, ax_d = plt.subplots(figsize=(7, 5))
    ax_d.plot(t_range / 3600, dP_range_bar, color="tab:blue", label="Required pressure")
    ax_d.axvline(2, color="gray", linestyle=":")
    ax_d.axhspan(membrane_pressure_limit, dP_range_bar[-1] + 50, color="tab:red", alpha=0.15, label="Membrane Failure Zone!")
    ax_d.scatter([2], [dP_2h_bar], color="tab:red", zorder=5)
    ax_d.annotate(f"t = 2 h, dP = {dP_2h_bar:.0f} bar", (2, dP_2h_bar),
                textcoords="offset points", xytext=(-100, 7))
    ax_d.set_xlabel("Operating time, t [h]")
    ax_d.set_ylabel("Required pressure, dP [bar]")

    ax_d.set_title("Section D1-D2: pressure required vs. operating time")
    ax_d.set_yscale("log")
    ax_d.set_ylim(1, dP_range_bar[-1] + 50)
    ax_d.minorticks_on()
    ax_d.grid(which='minor', axis='y', color='gray', linestyle='-.', linewidth=0.5)
    ax_d.grid(which='major', axis='both', color='gray', linestyle='-', linewidth=0.6)
    ax_d.legend()
    plt.show()