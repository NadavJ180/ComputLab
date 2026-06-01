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
EPS = 0.4          # Void Fraction

# ======================= Part 1 =======================

# ----------------------- Q1: Ergun equation -----------------------

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
        raise ValueError(f"Reynolds number ($Re_p$={Re:.2f}) out of Ergun range (1.0 <= Re_p <= 4000.0)")

def ergun_laminar(v0: float, L: float) -> float:
    """
    Calculate the pressure drop across the packed bed using the laminar term of the Ergun equation.

    Parameters:
    v0 (float): Superficial velocity of the fluid [m/s]
    L (float): Length of the packed bed [m]

    Returns:
    float: Pressure drop across the packed bed [Pa]
    """
    return (150 * MIU * (1 - EPS) ** 2 * v0 * L) / (dpeff ** 2 * EPS ** 3)

def ergun_turbulant(v0: float, L:float) -> float:
    """
    Calculate the pressure drop across the packed bed using the turbulent term of the Ergun equation.

    Parameters:
    v0 (float): Superficial velocity of the fluid [m/s]
    L (float): Length of the packed bed [m]

    Returns:
    float: Pressure drop across the packed bed [Pa]
    """
    return (1.75 * rho * (1 - EPS) * v0 ** 2 * L) / (dpeff * EPS ** 3)

def ergun_total(v0: float, L: float) -> float:
    """
    Calculate the total pressure drop across the packed bed using the Ergun equation.
    Checks if the Reynolds number for the given parameters is in the range of the Ergun equation.

    Parameters:
    v0 (float): Superficial velocity of the fluid [m/s]
    L (float): Length of the packed bed [m]

    Returns:
    float: Total pressure drop across the packed bed [Pa]
    """
    check_reynolds(calculate_reynolds(v0))
    
    return ergun_laminar(v0, L) + ergun_turbulant(v0, L)



# ============================================
# =============== Main Pipline ================
# ============================================

if __name__ == "__main__":
    





