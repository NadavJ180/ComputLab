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

# ======================= Part 1 =======================

# ----------------------- Q1: Ergun equation -----------------------    

def ergun_laminar(v0: float, h: float) -> float:
    """
    Calculate the pressure drop across the packed bed using the laminar term of the Ergun equation.

    Parameters:
    v0 (float): Superficial velocity of the fluid [m/s]
    h (float): Height of the packed bed [m]

    Returns:
    float: Pressure drop across the packed bed [Pa]
    """
    return (150 * MIU * (1 - EPS) ** 2 * v0 * h) / (dpeff ** 2 * EPS ** 3)

def ergun_turbulant(v0: float, h:float) -> float:
    """
    Calculate the pressure drop across the packed bed using the turbulent term of the Ergun equation.

    Parameters:
    v0 (float): Superficial velocity of the fluid [m/s]
    h (float): Height of the packed bed [m]

    Returns:
    float: Pressure drop across the packed bed [Pa]
    """
    return (1.75 * rho * (1 - EPS) * v0 ** 2 * h) / (dpeff * EPS ** 3)

def ergun_total(v0: float, h: float) -> float:
    """
    Calculate the total pressure drop across the packed bed using the Ergun equation.

    Parameters:
    v0 (float): Superficial velocity of the fluid [m/s]
    h (float): Height of the packed bed [m]

    Returns:
    float: Total pressure drop across the packed bed [Pa]
    """
    
    return ergun_laminar(v0, h) + ergun_turbulant(v0, h)

# ----------------------- Q2: finf the operational flow rate ----------------------- 

def pressure_balance(v0: float, h_bed: float, h_fluid: float) -> float:
    """
    Residual of the pressure balance to be zeroed by fsolve.
 
    Residual = hydrostatic driving pressure - Ergun friction pressure drop
             = ρ·g·hfluid  -  ergun_total(v0, bed_height)
 
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

    h_values = np.arange(0, h + 0.1, 0.1)                                        # Bed height with 0.1 [m] increments
    p_values_Pa = np.array([(P_OUT + ergun_total(v0, h_curr))                    # Pressure drop of each bed height [Pa]
                for h_curr in h_values
                ])                                                               
    p_values_atm = p_values_Pa / 101325                                           # [Pa] -> [atm]
    
    fig, ax = plt.subplots(figsize=(8, 5))

    ax.scatter(h_values, p_values_atm, color='mediumorchid', zorder=5)

    ax.set_xlabel('bed height [m]', fontsize=12)
    ax.set_ylabel('pressure [atm]', fontsize=12)
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

# ----------------------- Q3: Pressure drop (laminer/turbulant/total) vs Fluid height graph -----------------------

def graph_Pdrop_vs_hfluid(initial_v0_guess: float, h_bed: float, h_fluid: float) -> None:
    """
    Creates a graph of the pressure drop from the laminer/turbulant/total vs fluid height
    Assumptions: Momentary Steady-State -> Height of fluid ≈ constant

    Parameters
    ----------  
    initial_guess : float – initial guess for the superficial velocity [m/s]
    h_bed : float - height of the packed bed [m] 
    h_fluid : float - height of the fluid above the bed [m] 
    
    Returns:
    Graph with 3 plots of turbulant / laminar / total pressure drop vs fluid height
    """

    h_fluid_values = np.arange(h_fluid, -0.1, -0.1)                                         # fluid height intervals from top of bed

    v0_values = np.array([find_superficial_velocity(initial_v0_guess, h_bed, h_fluid_i)     # superficial velocities values for height intervals
                          for h_fluid_i in h_fluid_values])
    
    Laminar_dP_values = np.array([ergun_laminar(v0_i, h_bed)                                # [Pa] -> [atm]
                                  for v0_i in v0_values]) / 101325       
    Turbulant_dP_values = np.array([ergun_turbulant(v0_i, h_bed)                            # [Pa] -> [atm]
                                    for v0_i in v0_values]) / 101325   
    Total_dP_values = (Laminar_dP_values + Turbulant_dP_values)                             # Already converted to [atm]

    fig, ax = plt.subplots(figsize=(8,6))

    ax.scatter(h_fluid_values, Total_dP_values, color='blue', zorder=5, label='▲Pf(Total)')
    ax.scatter(h_fluid_values, Laminar_dP_values, color='black', zorder=5, label='▲Pf(Laminer)')
    ax.scatter(h_fluid_values, Turbulant_dP_values, color='red', zorder=5, label='▲Pf(Turbulant)')

    ax.set_title('Friction pressure drop at different fluid heights')
    ax.set_xlabel('Fluid height [m]')
    ax.set_ylabel('Pressure drop [atm]')

    ax.legend()
    ax.grid()
    ax.invert_xaxis()
    plt.tight_layout()
    plt.show()

    
# ====================================================================================================
# ====================================== Main Pipline ================================================
# ====================================================================================================

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

    #-------------- Q3: Pressure drop (laminer/turbulant/total) vs Fluid height graph ------------------

    initial_guess = 0.01   # [m/s]

    graph_Pdrop_vs_hfluid(initial_guess, hbed, hfluid)
