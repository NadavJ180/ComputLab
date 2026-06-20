import numpy
from scipy.optimize import fsolve
import matplotlib.pyplot as plt
import numpy as np
import math as math
from scipy.integrate import solve_ivp

hfluid = 6         # [m]
hbed = 2           # [m]
MIU = 9 * 10 ** -4 # [Kg / (m * s)]
rho = 777          # [Kg / m^3]
dpeff = 0.001      # [m]
g = 9.81           # [m / sec^2]
tf_graph = 2000

# More paramters
EPS = 0.37                          # Void Fraction
D_TANK = 10                         # Tank diameter [m]
A_cross = (math.pi * D_TANK**2) / 4 # Tank cross sectional area [m^2]
P_IN = 101325                       # inlet pressure [Pa]
P_OUT = 101325                      # outlet pressure [Pa]

epsilon1, epsilon2 = 0.45, 0.29
h1, h2 = 1, 1           # [m]

# ======================= Part 1 =======================

# ----------------------- Q1: Ergun equation -----------------------    

def ergun_laminar(v0: float, h: float, epsilon=EPS) -> float:
    """
    Calculate the pressure drop across the packed bed using the laminar term of the Ergun equation.

    Parameters:
    v0 (float): Superficial velocity of the fluid [m/s]
    h (float): Height of the packed bed [m]
    epsilon (float, optional): the void fraction of the matrial, default is EPS parameter
    
    Returns:
    float: Pressure drop across the packed bed [Pa]
    """
    return (150 * MIU * (1 - epsilon) ** 2 * v0 * h) / (dpeff ** 2 * epsilon ** 3)

def ergun_turbulent(v0: float, h:float, epsilon=EPS) -> float:
    """
    Calculate the pressure drop across the packed bed using the turbulent term of the Ergun equation.

    Parameters:
    v0 (float): Superficial velocity of the fluid [m/s]
    h (float): Height of the packed bed [m]
    epsilon (float, optional): the void fraction of the matrial, default is EPS parameter
    
    Returns:
    float: Pressure drop across the packed bed [Pa]
    """
    return (1.75 * rho * (1 - epsilon) * v0 ** 2 * h) / (dpeff * epsilon ** 3)

def ergun_total(v0: float, h: float, epsilon=EPS) -> float:
    """
    Calculate the total pressure drop across the packed bed using the Ergun equation.

    Parameters:
    v0 (float): Superficial velocity of the fluid [m/s]
    h (float): Height of the packed bed [m]
    epsilon (float, optional): the void fraction of the matrial, default is EPS parameter

    Returns:
    float: Total pressure drop across the packed bed [Pa]
    """
    
    return ergun_laminar(v0, h, epsilon) + ergun_turbulent(v0, h, epsilon)

# ----------------------- Q2: find the operational flow rate ----------------------- 

def pressure_balance(v0: float, h_bed: float, h_fluid: float) -> float:
    """
    Residual of the pressure balance to be zeroed by fsolve.
 
    Residual = hydrostatic driving pressure - Ergun friction pressure drop
             = rho * g * hfluid  -  ergun_total(v0, bed_height)
 
    When this equals zero the system is at its operating (working) point.
 
    Parameters
    ----------
    v0 : float – superficial velocity [m/s] 
    h_bed : float - height of the packed bed [m]
    h_fluid : float - height of the fluid above the bed [m]

    Returns
    -------
    float – residual [Pa]  (fsolve drives this to 0)
    """

    delta_p_hydro = (P_IN - P_OUT) + rho * g * h_fluid    # [Pa]
    delta_p_friction = ergun_total(v0, h_bed)             # [Pa]

    return delta_p_hydro - delta_p_friction

def find_superficial_velocity(initial_guess: float, h_bed: float, h_fluid: float) -> float:
    """
    Find the superficial velocity (v0) that satisfies the pressure balance using fsolve.

    Parameters
    ----------
    initial_guess : float – initial guess for the superficial velocity [m/s]  
    h_bed : float - height of the packed bed [m]
    h_fluid : float - height of the fluid above the bed [m] 

    Returns:
    float: Superficial velocity (v0) at the operating point [m/s]
    """

    v0_solution = fsolve(pressure_balance, initial_guess, args=(h_bed, h_fluid))

    return v0_solution.item() # return the velocity as a float instead of array

# ----------------------- Q3: graph pressure vs bed height ----------------------- 

def pressure_diff(v0: float, h: float) -> tuple[list[float], np.ndarray]:
    """
    Calculates the pressure difference throughout the bed.

    Parameters
    ----------  
    v0 : float - superficial velocity [m/s]
    h : float - height of the packed bed [m] 
    
    Returns:
    list of pressure differences and array of height values
    """
    h_values = np.arange(0, h + 0.1, 0.1)                                        # Bed height with 0.1 [m] increments
    p_values_Pa = np.array([(P_OUT + ergun_total(v0, h_curr))                    # Pressure drop of each bed height [Pa]
                for h_curr in h_values
                ])                                                               
    p_values_atm = p_values_Pa / 101325                                           # [Pa] -> [atm]

    return p_values_atm, h_values

def graph_p_vs_hbed(v0: float, h: float) -> None:
    """
    Graph the pressure along the bed as a funtion of the bed height
    
    Parameters
    ----------  
    h : float - height of the packed bed [m] 
    v0 : float - superficial velocity [m/s]
    
    Returns:
    graph of pressure vs bed height
    """

    p_values_atm, h_values = pressure_diff(v0, h)
    
    fig, ax = plt.subplots(figsize=(8, 5))

    ax.scatter(h_values, p_values_atm, color='mediumorchid', zorder=5)

    ax.set_xlabel('bed height [m]', fontsize=12)
    ax.set_ylabel('Pressure [atm]', fontsize=12)
    ax.set_title('Pressure along the filtration bed', fontsize=13)

    ax.invert_xaxis()
    ax.grid(True, linestyle='--', alpha=0.6)

    plt.tight_layout()
    plt.show()


# ======================= Part 2 =======================

# ----------------------- Q1 + Q2: Superficial velocity and Reynolds vs Fluid height graphs -----------------------   

def calculate_reynolds(v0: float) -> float:
    """
    Calculate the particle Reynolds number for the packed bed.
    """
    return (rho * v0 * dpeff) / (MIU * (1 - EPS))

def graph_v0_and_Re_vs_hfluid(initial_v0_guess: float, h_bed: float, h_fluid: float) -> None:
    """
    Creates two subplot in one figure -> the superficial velocity and the Reynolds number as functions of the fluid height
    Assumptions: Momentary Steady-State -> Height of fluid ≈ constant
    
    Parameters
    ----------  
    initial_guess : float – initial guess for the superficial velocity [m/s]
    h_bed : float - height of the packed bed [m] 
    h_fluid : float - height of the fluid above the bed [m] 
    
    Returns:
    2 graphs of superficial velocity / Reyolds vs fluid height
    """
    h_fluid_values = np.arange(h_fluid, -0.1, -0.1)                                     # fluid height intervals from top of bed
                                                                                        
    v0_values = np.array([find_superficial_velocity(initial_v0_guess, h_bed, h_fluid_i) # superficial velocities values for height intervals
                          for h_fluid_i in h_fluid_values])

    Re_values = np.array([calculate_reynolds(v0_i) for v0_i in v0_values])      # Reynolds values for height intervals

    fig, (ax1, ax2) = plt.subplots(nrows=2, ncols=1, figsize=(8,5))

    ax1.scatter(h_fluid_values, v0_values, color='blue', zorder=5)                        # first plot - superficial velocity
    ax1.set_title('Superficial velocity at different fluid heights')
    ax1.set_xlabel('Fluid height [m]')
    ax1.set_ylabel('Superficial fluid velocity [m/s]')

    ax1.grid()
    ax1.invert_xaxis()

    ax2.scatter(h_fluid_values, Re_values, color='green', zorder=5)                       # second plot - Reynolds number
    ax2.set_title('Reynolds number at different fluid heights')
    ax2.set_xlabel('Fluid height [m]')
    ax2.set_ylabel('Reynolds Number')
    
    ax2.invert_xaxis()
    ax2.grid()

    plt.tight_layout()
    plt.show()

# ----------------------- Q3: Pressure drop (laminer/turbulent/total) vs Fluid height graph -----------------------

def graph_Pdrop_vs_hfluid(initial_v0_guess: float, h_bed: float, h_fluid: float) -> None:
    """
    Creates a graph of the pressure drop from the laminer/turbulent/total vs fluid height
    Assumptions: Momentary Steady-State -> Height of fluid ≈ constant

    Parameters
    ----------  
    initial_guess : float – initial guess for the superficial velocity [m/s]
    h_bed : float - height of the packed bed [m] 
    h_fluid : float - height of the fluid above the bed [m] 
    
    Returns:
    Graph with 3 plots of turbulent / laminar / total pressure drop vs fluid height
    """

    h_fluid_values = np.arange(h_fluid, -0.1, -0.1)                                         # fluid height intervals from top of bed

    v0_values = np.array([find_superficial_velocity(initial_v0_guess, h_bed, h_fluid_i)     # superficial velocities values for height intervals
                          for h_fluid_i in h_fluid_values])
    
    Laminar_dP_values = np.array([ergun_laminar(v0_i, h_bed)                                # [Pa] -> [atm]
                                  for v0_i in v0_values]) / 101325       
    Turbulent_dP_values = np.array([ergun_turbulent(v0_i, h_bed)                            # [Pa] -> [atm]
                                    for v0_i in v0_values]) / 101325   
    Total_dP_values = (Laminar_dP_values + Turbulent_dP_values)                             # Already converted to [atm]

    fig, ax = plt.subplots(figsize=(8,6))

    ax.scatter(h_fluid_values, Total_dP_values, color='blue', zorder=5, label='▲Pf(Total)')
    ax.scatter(h_fluid_values, Laminar_dP_values, color='black', zorder=5, label='▲Pf(Laminer)')
    ax.scatter(h_fluid_values, Turbulent_dP_values, color='red', zorder=5, label='▲Pf(Turbulent)')

    ax.set_title('Friction pressure drop at different fluid heights')
    ax.set_xlabel('Fluid height [m]')
    ax.set_ylabel('Pressure drop [atm]')

    ax.legend()
    ax.grid()
    ax.invert_xaxis()
    plt.tight_layout()
    plt.show()


# ======================= Part 3A =======================

# ----------------------- Q1 + Q3: Solve dhdt IVP + Graphing full and laminar part of solution -----------------------
    
def dhdt(t: float, h: float) -> float:
    """
    Computes value of derivative of h_total at a specific time. 

     Parameters
    ----------  
    t : float – time for computing dhdt [s]
    h : float - total height (liquid + packed bed) [m] 
    
    Returns:
    Value of dhdt at a speficic time
    """
    
    alpha = (hbed / (g * rho)) * (150 * MIU / dpeff**2) * ((1 - EPS)**2 / EPS**3)
    beta =  (hbed / g) * (1.75 / dpeff) * ((1 - EPS) / EPS**3)
    gamma = hbed

    with np.errstate(invalid='ignore'): # ignores the runtime error becuase of a negetive discriminant from numerical noise.
        dh_dt = ( alpha - np.sqrt( alpha**2 - 4 * beta * (gamma - h) ) ) / (2 * beta)
    
    return  dh_dt

def dhdt_laminar(t: float, h: float) -> float:
    """
    Computes only the laminar value of derivative of h_total at a specific time. 

     Parameters
    ----------  
    t : float – time for computing dhdt [s]
    h : float - total height (liquid + packed bed) [m] 
    
    Returns:
    Value of laminar part of dhdt at a speficic time
    """

    alpha = (hbed / (g * rho)) * (150 * MIU / dpeff**2) * ((1 - EPS)**2 / EPS**3)
    gamma = hbed

    dh_dt_laminar = (h - gamma) / (- alpha)

    return dh_dt_laminar 

def solve_dhdt(stop_option=None) -> np.ndarray:
    """
    Solves the IVP, finds h(t).
    
     Parameters
    ----------  
    stop_option : a stopping paramter for the solver. Defaults to none it none supplied


    Returns:
    An array of the solution of the ODE where
    sol.t    -> array of times
    sol.h[0] -> array of the corresponding total height
    """

    tspan = np.arange(0, tf_graph, 1) 
    h_init = hfluid + hbed

    sol = solve_ivp(dhdt, (tspan[0], tspan[-1]), [h_init], t_eval=tspan, events=stop_option)

    return sol

def solve_dhdt_laminar() -> np.ndarray:
    """
    Solves the only the laminar part of the IVP, finds h(t).
    
    Returns:
    An array of the solution of the ODE where
    sol_laminar.t    -> array of times
    sol_laminar.h[0] -> array of the corresponding total height
    """

    tspan = np.arange(0, tf_graph, 1) 
    h_init = hfluid + hbed
    
    sol_laminar = solve_ivp(dhdt_laminar, (tspan[0], tspan[-1]), [h_init], t_eval=tspan)

    return sol_laminar

def graph_dhdt_solution() -> None:
    """
    Graphs the solution of of the full and laminar part of the ODE of the height of fluid.
    If the 

    Returns:
    Graph of h(t) and h_laminar(t) as a function of t
    """

    sol = solve_dhdt()
    sol_laminar = solve_dhdt_laminar()

    plt.figure(figsize=(8, 5))
    plt.plot(sol.t, sol.y[0], label="ODE Full solution", color="red")
    plt.plot(sol_laminar.t, sol_laminar.y[0], label="ODE - Laminar only", color="blue")
    plt.xlabel("Time (s)")
    plt.ylabel("Height (m)")
    plt.title("Total height over Time")
    plt.legend()
    plt.grid(True)
    plt.show()

# ----------------------- Q2: halfway height of the fluid -----------------------

def half_drained_bed(t: float, h: np.ndarray) -> float:
    """
    Checks if h_fluid = htotal - hbed equal to half the original height of hfluid.
    Used as a stopping paramtere for th IVP solver.
    
    Parameters
    ----------  
    t : float – time for computing dhdt [s]
    h : float - total height (liquid + packed bed) [m] 
    
    Returns:
    number (True) if h_fluid != 1/2 h_initial and 0 (False) when h_fluid = 1/2 h_initial
    """

    return (h[0] - hbed) - (hfluid / 2)

# ======================= Part 3B =======================

# ----------------------- 2 layers with 2 different ε ----------------------- 

def pressure_balance_2_layers(v0: float, h1: float, h2: float, h_fluid: float) -> float:
    """
    Residual of the pressure balance to be zeroed by fsolve of 2 layers with different porosities.
 
    Residual = hydrostatic driving pressure - Ergun friction pressure drop
             = rho * g * hfluid  -  ergun_total(v0, bed_height)
 
    When this equals zero the system is at its operating (working) point.
 
    Parameters
    ----------
    v0 : float – superficial velocity [m/s] 
    h1 : float - height of the packed bed of first material [m]
    h2 : float - height of the packed bed of second material [m]
    h_fluid : float - height of the fluid above the bed [m]

    Returns
    -------
    float – residual [Pa]  (fsolve drives this to 0)
    """

    delta_p_hydro = (P_IN - P_OUT) + rho * g * h_fluid              # [Pa]
    delta_p_friction = ergun_total(v0, h1, epsilon1) + ergun_total(v0, h2, epsilon2)    # [Pa]

    return delta_p_hydro - delta_p_friction

def find_superficial_velocity_2_layers(initial_guess: float, h1: float, h2: float, h_fluid: float) -> float:
    """
    Find the superficial velocity (v0) that satisfies the pressure balance using fsolve.
    For 2 layers of different material with different porosities

    Parameters
    ----------
    initial_guess : float – initial guess for the superficial velocity [m/s]  
    h1 : float - height of the packed bed of first material [m]
    h2 : float - height of the packed bed of second material [m]
    h_fluid : float - height of the fluid above the bed [m] 

    Returns:
    float: Superficial velocity (v0) at the operating point [m/s]
    """

    v0_solution = fsolve(pressure_balance_2_layers, initial_guess, args=(h1, h2, h_fluid))

    return v0_solution.item() # return the velocity as a float instead of array

def pressure_diff_2_layers(v0: float, h1: float, h2: float) -> tuple[list[float], np.ndarray]:
    """
    Calculates the pressure difference throughout the bed for 2 different materials with different porosities.

    Parameters
    ----------  
    v0 : float - superficial velocity [m/s]
    h1 : float - height of the packed bed of first material [m]
    h2 : float - height of the packed bed of second material [m] 
    
    Returns:
    list of pressure differences and array of height values
    """
    dh = 0.1
    h_values = np.arange(0, h1 + h2 + dh, dh)                    # Bed height with dh [m] increments
    p_values_atm = []
    for h_curr in h_values:
        if h_curr <= h2:                                        # Bottom half of the bed
            p_tot = P_OUT + ergun_total(v0, h_curr, epsilon2)
            p_values_atm.append(p_tot / 101325)                 # Pa -> atm
        else:                                                   # Top half of the bed
            p_bot = ergun_total(v0, h2, epsilon2)               # Donation from all of the bottom half 
            p_top = ergun_total(v0, h_curr - h2, epsilon1)      # Donation of partial top half
            p_tot = P_OUT + p_top + p_bot
            p_values_atm.append(p_tot / 101325)                 # Pa -> atm

    return p_values_atm, h_values

# ----------------------- continuous ε ----------------------- 

def epsilon_func(h: float)-> float:
    return 0.55 * np.exp( - ( hbed - h ) / hbed )

def ergun_total_continuous(v0: float, h_bed: float) -> float:
    """
    Calculate the total pressure drop across the packed bed using the Ergun equation for continuous porosities.

    Parameters:
    v0 (float): Superficial velocity of the fluid [m/s]
    h_bed (float): Height of the packed bed [m]

    Returns:
    float: Total pressure drop across the packed bed [Pa]
    """
    dh = 0.01
    h_values = np.arange(0, h_bed, dh)
    p_tot = 0

    for h_curr in h_values:
        eps_curr = epsilon_func(h_curr) # Friction gets greater the more bed there is beneath
        p_tot += ergun_total(v0, dh, eps_curr)

    return p_tot # [Pa]

def pressure_balance_continuous(v0: float, h_bed:float, h_fluid: float) -> float:
    """
    Residual of the pressure balance to be zeroed by fsolve of continuous porosities.
 
    Residual = hydrostatic driving pressure - Ergun friction pressure drop
             = rho * g * hfluid  -  ergun_total(v0, bed_height)
 
    When this equals zero the system is at its operating (working) point.
 
    Parameters
    ----------
    v0 : float – superficial velocity [m/s] 
    h_bed : float - height of the packed bed [m]
    h_fluid : float - height of the fluid above the bed [m]

    Returns
    -------
    float – residual [Pa]  (fsolve drives this to 0)
    """

    delta_p_hydro = (P_IN - P_OUT) + rho * g * h_fluid     # [Pa]
    delta_p_friction = ergun_total_continuous(v0, h_bed)   # [Pa]

    return delta_p_hydro - delta_p_friction

def find_superficial_velocity_continuous(initial_guess: float, h_bed: float, h_fluid: float) -> float:
    """
    Find the superficial velocity (v0) that satisfies the pressure balance using fsolve.
    For continuous porosities.

    Parameters
    ----------
    initial_guess : float – initial guess for the superficial velocity [m/s]  
    h_bed : float - height of the packed bed [m]
    h_fluid : float - height of the fluid above the bed [m] 

    Returns:
    float: Superficial velocity (v0) at the operating point [m/s]
    """

    v0_solution = fsolve(pressure_balance_continuous, initial_guess, args=(h_bed, h_fluid))

    return v0_solution.item() # return the velocity as a float instead of array

def pressure_diff_continuous(v0: float, h_bed: float) -> tuple[list[float], np.ndarray]:
    """
    Calculates the pressure difference throughout the bed for continuous porosities.

    Parameters
    ----------  
    v0 : float - superficial velocity [m/s]
    h_bed : float - height of the packed bed [m]
    
    Returns:
    list of pressure differences and array of height values
    """

    dh = 0.1
    h_values = np.arange(0, h_bed + dh, dh)           # Bed height with dh [m] increments
    p_values_atm = [P_OUT / 101325]                   # Initial pressure diff at h = 0
    
    for h_curr in h_values[1:]:
        p_tot = P_OUT + ergun_total_continuous(v0, h_curr)
        p_values_atm.append(p_tot / 101325)           # Pa -> atm


    return p_values_atm, h_values

# ----------------------- Plot -----------------------

def graph_p_vs_hbed_diff_eps(v0_1: float, v0_2: float, v0_cont: float, h1: float, h2: float) -> None:
    """
    Graph the height of the bed as a funtion of the pressure difference for different void fraction: constant, 2 values, continuous

    
    Parameters
    ----------  
    h1 : float - height of the packed bed of first material [m]
    h2 : float - height of the packed bed of second material [m] 
    v0_1 : float - superficial velocity of 1 layer [m/s]
    v0_2 : float - superficial velocity of 2 layers [m/s]
    v0_cont : float - superficial velocity of continuous porosity [m/s] 
    
    Returns:
    graph of bed height vs pressure for different ε values
    """
    h_bed = h1 + h2
    
    p_values_1_layer, h_values_1_layer = pressure_diff(v0_1, h_bed)
    p_values_2_layers, h_values_2_layers = pressure_diff_2_layers(v0_2, h1, h2)
    p_values_cont, h_values_cont = pressure_diff_continuous(v0_cont, h_bed)
    
    fig, ax = plt.subplots(figsize=(8, 5))

    ax.scatter(p_values_1_layer, h_values_1_layer, color='turquoise', zorder=5, label='constant = ε')
    ax.scatter(p_values_2_layers, h_values_2_layers, color='blue', zorder=5, label='different ε in 2 sections of filtration bed')
    ax.scatter(p_values_cont, h_values_cont, color='red', zorder=5, label='changed ε through filtration bed')
    
    ax.set_ylabel('Bed Height [m]', fontsize=12)
    ax.set_xlabel('Pressure [atm]', fontsize=12)
    ax.set_title('Pressure along filtration bed', fontsize=13)

    ax.invert_xaxis()
    ax.grid(True, linestyle='--', alpha=0.6)

    plt.legend()
    plt.tight_layout()
    plt.show()

#===================================================================================================
#===================================     Main Pipeline     =========================================
#===================================================================================================

if __name__ == "__main__":
    
    #==================================== Part 1 ====================================================
    
    #-------------- Q2: find the working flow of the system ------------------
    
    initial_guess = 0.01   # [m/s]
    
    v0_op = find_superficial_velocity(initial_guess, hbed, hfluid)

    Q_op = v0_op  * A_cross # [m3/s]

    print(f"Operating volumetric flow rate (Q): {Q_op:.3f} [m^3/s]")

    #-------------- Q3: graph pressure vs bed height ------------------

    graph_p_vs_hbed(v0_op, hbed)
    
    #==================================== Part 2 =======================================================

    #-------------- Q1 + Q2: Superficial velocity and Reynolds vs Fluid height graphs ------------------

    initial_guess = 0.01   # [m/s]
    
    graph_v0_and_Re_vs_hfluid(initial_guess, hbed, hfluid)

    #-------------- Q3: Pressure drop (laminer/turbulent/total) vs Fluid height graph ------------------

    initial_guess = 0.01   # [m/s]

    graph_Pdrop_vs_hfluid(initial_guess, hbed, hfluid)
  
    #==================================== Part 3A =======================================================

    # ----------------------- Q1 + Q3: Solve dhdt IVP + Graphing full and laminar part of solution -----------------------

    graph_dhdt_solution()

    # ----------------------- Q2: halfway height of the fluid -----------------------

    half_drained_bed.terminal = True                            #initializa the stopping conditions
    sol_half_way = solve_dhdt(stop_option=half_drained_bed)

    print(
        f"Time needed to reach half of the original height: {sol_half_way.t_events[0][0]:.3f} [s]"
    )
    
    #==================================== Part 3B =======================================================

    initial_guess = 0.01   # [m/s]
    
    v0_op_1_layer = find_superficial_velocity(initial_guess, hbed, hfluid)
    v0_op_2_layers = find_superficial_velocity_2_layers(initial_guess, h1, h2, hfluid)
    v0_op_cont = find_superficial_velocity_continuous(initial_guess, hbed, hfluid)

    graph_p_vs_hbed_diff_eps(v0_op_1_layer, v0_op_2_layers, v0_op_cont, h1, h2)
