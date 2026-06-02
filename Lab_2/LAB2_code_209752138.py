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
P_ATM = 101325                      # Atmospheric pressure [Pa]
# ======================= Part 1 =======================

# ----------------------- Validation -----------------------  

def calculate_reynolds(v0: float) -> float:
    """
    Calculate the particle Reynolds number for the packed bed.
    """
    return (rho * v0 * dpeff) / (MIU * (1 - EPS))

def check_reynolds(Re: float) -> None:
    """
    Check that the Reynolds number is within the range for the Ergun Equation
     1.0 <= Re_p <= 4000.0

    Parameters:
    v_0 (float): Superficial velocity of the fluid [m/s]

    Returns:
    None: Raises ValueError if Reynolds number is out of range
    """
    
    if not (1.0 <= Re <= 4000.0):
        raise ValueError(f"Reynolds number (Re_p={Re:.2f}) out of Ergun range (1.0 <= Re_p <= 4000.0)")

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
    Checks if the Reynolds number for the given parameters is in the range of the Ergun equation.

    Parameters:
    v0 (float): Superficial velocity of the fluid [m/s]
    h (float): Height of the packed bed [m]

    Returns:
    float: Total pressure drop across the packed bed [Pa]
    """
    check_reynolds(calculate_reynolds(v0))
    
    return ergun_laminar(v0, h) + ergun_turbulant(v0, h)

# ----------------------- Q2: finf the operational flow rate ----------------------- 

def pressure_balance(v0: float, h: float) -> float:
    """
    Residual of the pressure balance to be zeroed by fsolve.
 
    Residual = hydrostatic driving pressure - Ergun friction pressure drop
             = ρ·g·hfluid  -  ergun_total(v0, bed_height)
 
    When this equals zero the system is at its operating (working) point.
 
    Parameters
    ----------
    v0 : float – superficial velocity [m/s] 
    h : float - height of the packed bed [m]

    Returns
    -------
    float – residual [Pa]  (fsolve drives this to 0)
    """
    delta_p_hydro = rho * g * hfluid        # [Pa]
    delta_p_friction = ergun_total(v0, h)  # [Pa]

    return delta_p_hydro - delta_p_friction

def find_superficial_velocity(initial_guess: float, h: float) -> float:
    """
    Find the superficial velocity (v0) that satisfies the pressure balance using fsolve.

    Parameters
    ----------
    initial_guess : float – initial guess for the superficial velocity [m/s]  
    h : float - height of the packed bed [m] 

    Returns:
    float: Superficial velocity (v0) at the operating point [m/s]
    """

    v0_solution = fsolve(pressure_balance, initial_guess, args=(h,))

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
    p_values_Pa = np.array([(P_ATM + ergun_total(v0, h_curr))                    # Pressure drop of each bed height [Pa]
                for h_curr in h_values
                ])                                                               
    p_values_atm = p_values_Pa / P_ATM                                           # Pressure drop of each bed height [atm]
    
    fig, ax = plt.subplots(figsize=(8, 5))

    ax.scatter(h_values, p_values_atm, color='mediumorchid', zorder=5)

    ax.set_xlabel('bed height [m]', fontsize=12)
    ax.set_ylabel('pressure drop [atm]', fontsize=12)
    ax.set_title('Pressure drop along the filtration bed', fontsize=13)

    ax.invert_xaxis()
    ax.grid(True, linestyle='--', alpha=0.6)

    plt.tight_layout()
    plt.show()

# ============================================
# =============== Main Pipline ================
# ============================================

if __name__ == "__main__":
    

    #==================================== Part A =======================================
    
    #-------------- Q2: find the working flow of the system ------------------

    initial_guess = 0.01 # [m/s]
    bed_height = hbed    #[m]
    v0_op = find_superficial_velocity(initial_guess, bed_height)
    
    Re = calculate_reynolds(v0_op)
    check_reynolds(Re)                #Validation if Reynolds is in range for Ergun equation

    Q_op = v0_op  * A_cross # [m3/s]

    print(f"Operating Re number: {Re:.2f}")
    print(f"Operating superficial velocity (v0): {v0_op:.3f} [m/s]")
    print(f"Operating volumetric flow rate (Q): {Q_op:.3f} [m^3/s]")

    #-------------- Q3: graph pressure vs bed height ------------------

    graph_p_vs_hbed(v0_op, bed_height)




