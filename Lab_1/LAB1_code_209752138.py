from scipy.optimize import fsolve
import matplotlib.pyplot as plt
import numpy as np
import math as math
from scipy.integrate import solve_ivp


# Don't change this parametrs names, only valuses if needed!!
F = 0.0073 #[m3/s]
L_PIPE = 66 #[m]
D_PIPE = 0.072 #[m]
EPSILON = 1 * 10 ** -5 #[m]
NU = 3 * 10 ** -7 #[m2/s]
Z_PUMP= -0.5 #[m]

# Needed parameters that were not included in original file
NUMBER_OF_TURNS = 7 #[no units]
TURN_FRICTION_FACTOR = 0.75 #[no units]
P_IN = 222915 #[Pa]
P_OUT = 182385 #[Pa]
RHO = 777 #[kg/m3]
KINETIC_ALPHA = 1 #[no units]
DELTA_HEIGHT = 32 #[m]
P_VAPOR = 5000 #[Pa]
g = 9.81 # gravity [m/s^2]

eD = [0.00001, 0.0001, 0.001] # Relative roughness values to be used in the graph
Re_Laminar = np.geomspace(1e3, 4000, 50) # Reynolds numbers from 10^2 to 4000
Re_Turbulent = np.geomspace(4001, 1e8, 50) # Reynolds numbers from 4000 (not including) to 10^8



# ===============================================
# ================== Part A =====================
# ================================================


# ============ Q1: Find Fanning friction factor ============================

def Check_Roughness(eD:float) -> bool: 
    
    """
    Checks if the relative roughnes in within the bounds for the approximation.
    """
    
    if eD < 0 or eD > 0.01:
        return False
    else:
        return True

def Check_Reynolds_Type(Re:float) -> str | bool:
     
     """
     Checks which flow regime the system is in. 
     
     If Reynolds number is negative or too high for the approximation to be valid, returns False.
     Otherwise, returns 'Laminar' for laminar flow and 'Turbulent' for turbulent flow.

     Reynolds number range: 0 < Re <= 1e8
     """

     if Re <= 0 or Re > 1e8: 
        return False
    
     elif 0 < Re <= 4000: 
        return 'Laminar'
    
     elif 4000 < Re <= 1e8: 
          return 'Turbulent'

def f_fanning(Re: float, eD: float) -> float:

    """
    Calculates the Fanning friction factor for a given Reynolds number and relative roughness.
    
    If the relative roughness or Reynolds number are not valid, returns ff=0 and an error message
    If flow laminar, calculate ff unsing the formula: 16/Re
    If flow turbulent, calculate ff using the formula: 1.375 * 1e-3 * (1 + (2 * 1e4 * eD + 1e6 / Re) ** (1/3))
    """

    Roughness = Check_Roughness(eD)
    Flow = Check_Reynolds_Type(Re)

    if not Roughness or not Flow: 
        ff = 0
        print('error in one of the inserted values:')
        return ff

    if Flow == 'Laminar':
        ff = 16 / Re
        return ff
    
    elif Flow == 'Turbulent':
        ff = 1.375 * 1e-3 * (1 + (2 * 1e4 * eD + 1e6 / Re) ** (1/3))
        return ff

# ============ Q2: Graph F_f vs Re =========================================

def graph_F_f_vs_Re(eD: np.ndarray, Re_Laminar: np.ndarray, Re_Turbulent: np.ndarray):

    """
    Graphs the Fanning friction factor as a function of the relative roughness and the Reynolds number 
    in the laminar and turbulent regimes (The Moody Diagram).

    Output is a single figure with one plot for the laminar regime and a plot for each relative roughness in the turbulent regime.
    
    The graph is in loglog scale 
    """

    fig, ax = plt.subplots(figsize=(10, 6))

    f_Laminar = [f_fanning(Re, 0) for Re in Re_Laminar] # friction factors for the laminar regime (roughness doesn't affect laminar flow)

    ax.loglog(Re_Laminar, f_Laminar, color='brown')

    color = ['royalblue', 'green', 'mediumpurple'] # similar colors for the turbulent plots as provided in the test case

    for i, eD_value in enumerate(eD): # Claculate the fanning number for each eD values for turbulant Re
        f_Turbulent = [f_fanning(Re, eD_value) for Re in Re_Turbulent]
        ax.loglog(Re_Turbulent, f_Turbulent, color=color[i] if i < len(color) else None) #if more eD values are added, generate random colors
        ax.annotate(f'{eD_value}', xy=(4e7, f_Turbulent[-1]),
                    xytext=(0, 2),  
                    textcoords='offset points', 
                    va='bottom',
                    ha='center',fontsize=9)
    
    ax.set_xlabel('Reynolds Number')
    ax.set_ylabel('Fanning Friction Factor')
    ax.set_title('Moody Diagram')
    ax.set_xlim([1e3, 1e8])
    ax.set_ylim([2e-3, 1.25e-2])
    ax.text(1.03, 0.5, 'e/D', transform=ax.transAxes, 
            rotation=90, 
            va='center', 
            fontsize=10)
    ax.grid(True, which="both", ls="-", alpha=0.3)
    plt.tight_layout()
    plt.show()

# ============ Q3: Frictional losses as a function of flow rate ============

def find_flow_velocity(F: float) -> float: 

    """
    Calculates the average flow velocity for a given flow rate using the flow rate and pipe cross section: 
    A_cross_section_pipe = π * r^2 = π * (D/2)^2 

    Flow rate units (input) -> [m^3/s]
    Pipe diameter -> [m]
    Pipe Area -> [m^2]
    Average fluid velocity units (output) -> [m/s] 
    """

    A = math.pi * (D_PIPE / 2) ** 2 
    V = F / A 
    return V 

def find_reynolds_number(F: float) -> float: 

    """
    Calculates the Reynolds number in a pipe by a given flow rate using kinematic viscocity

    Flow rate units (input) -> [m^3/s]
    Average fluid velocity -> [m/s]
    Pipe diameter -> [m]
    Kinematic viscocity / NU -> [m^2/s]
    Reynolds (output) -> [no units]
    """

    V = find_flow_velocity(F)
    Re = V * D_PIPE / NU
    return Re

def find_h_LT(F: float) -> float: 

    """
    Calcualtes the total frictional head loss as a function of the flow rate and pipe parameters.

    Flow rate units (input) -> [m^3/s]
    Average fluid velocity -> [m/s]
    Reynolds -> [no units]
    Fanning friction factor -> [no units]
    Absolute pipe roughness / Epsilon -> [m]
    Pipe diameter -> [m]
    Pipe length -> [m]
    No. of turns & turn friction factor -> [no units]
    Total frictional head loss -> [m^2/s^2]
    """

    V = find_flow_velocity(F) 
    Re = find_reynolds_number(F) 
    eD = EPSILON / D_PIPE 
    ff = f_fanning(Re, eD) 
     
    h_LT = (ff * (L_PIPE / D_PIPE) + 
            NUMBER_OF_TURNS * TURN_FRICTION_FACTOR) * 2 * V ** 2 
    
    return h_LT 


# ===============================================
# ================== Part B =====================
# ================================================

# ============= Q1: Import impeller data and approximate to 2nd degree polynomial =============

"""
Load the impellers data from the CSV file, 
and create a dictionary for the different impeller sizes and their corresponding column in the CSV file
"""
from pathlib import Path
DATA_DIR = Path(__file__).parent
impeller_data = np.loadtxt(DATA_DIR / 'lab1_impellers.csv', delimiter=',', skiprows=2)

impeller_size_dict = { 
    '5&11/16': 2,
    '6': 0,
    '6&9/16': 4,
    '6&7/8': 6
} 

def find_pump_head(impeller_data: np.ndarray, impeller_size:str) -> tuple[np.ndarray, int]: 

    """
    Calculates the pump head by taking the raw impeller data and str of impeller size (in inches) 
    and returns a 2nd order polynomial approximation.
    Valid impeller sizes are: 5&11/16", 6", 6&9/16", 6&7/8" [inch] (str!!!)

    ------ Units ------
    Flow rate in csv (input) -> [m^3/hr]
    Head in csv (input) -> [m]
    Impeller size (input) -> str [inches]
    Impeller size col (output) -> [no units]
    Flow rate in polynomial (output) -> [m^3/s]
    """
    
    if str(impeller_size) not in impeller_size_dict: # Check if the impeller size is valid
        raise ValueError(f"Invalid impeller size. Valid sizes are: {', '.join(impeller_size_dict.keys())}")

    col = impeller_size_dict[str(impeller_size)]
    x = impeller_data[:, col] 
    x_m3s = x / 3600 # Convert flow units (m3/h -> m3/s)
    y = impeller_data[:, col+1] 
    
    coeffs = np.polyfit(x_m3s, y, 2) # Fit a 2nd degree polynomial 
    polynomial = np.poly1d(coeffs) # Create a polynomial from coefficients
    
    return polynomial, col

# ============= Q2: Find the system head and graph the pump and system curve ===================

def find_system_head(F: float) -> float:
    
    """
    Calculates the system head using the known bernoulli equation 
    with the frictional head losses function.
    
    ---- Units -----
    Flow rate (input) -> [m^3/s]
    H_system (output) -> [m]
    """

    v_ave_in = 0 # fluid at top of tank is static [m/s]
    v_ave_out = find_flow_velocity(F) # Average flow velocity [m/s]
    h_LT = find_h_LT(F) # Frictional head loss [m^2/s^2]

    H_system = (P_OUT - P_IN) / (RHO * g) + DELTA_HEIGHT + (KINETIC_ALPHA * (v_ave_out**2 - v_ave_in**2) / (2 * g) + (h_LT / g))  # Total system head

    return H_system

def graph_pump_and_system_curves(impeller_data: np.ndarray, impeller_size:str): 
    
    """
    Graphs the pump curve and system curve on the same plot for a single impeller size. 
    Also plots intersection between the two curves (working point).

    ----- Units -----
    Flow rate in csv (input) -> [m^3/hr]
    Head in csv (input) -> [m]
    Impeller size (input) -> str [inches]]
    Flow rate -> [m^3/s]
    """

    delta_H_pump, impeller_size_col = find_pump_head(impeller_data, impeller_size) # Get the polynomial function for the specified impeller size
    
    F_values = impeller_data[:, impeller_size_col]  # Flow rates from csv data for the specified impeller size 
    H_pump = [delta_H_pump(F/3600) for F in F_values] # Flow rate [m^3/hr] -> [m^3/s]
    H_system = [find_system_head(F/3600) for F in F_values] # Flow rate [m^3/hr] -> [m^3/s]
    
    working_point = find_working_point(impeller_data, impeller_size) # Find the working point of the pump

    plt.figure(figsize=(10, 6))
    
    plt.plot(F_values, H_pump, label='▲Hpump', color='red') # Plot pump curve
    plt.plot(F_values, H_system, label='▲Hsys', color='green') # Plot system curve
    plt.scatter(working_point[0], working_point[1], color='black', zorder=5) # Plot the working point
    
    plt.xlim(-3, 67)
    plt.ylim(0, 80)
    
    plt.xlabel('Q (m3/hr)')
    plt.ylabel('h (m)')
    plt.title(f'Pump and System Curves for Impeller Size {impeller_size} inches')
    plt.annotate(f'{impeller_size}"', xy=(0, H_pump[0]), xytext=(0,H_pump[0]+2), fontsize=10, color='black')
    plt.legend(loc='lower right')
    plt.grid()
    plt.show()

# ============= Q3: Find the working point of the pump =========================================

def find_working_point(impeller_data: np.ndarray, impeller_size: str) -> tuple[float, float]: 
    
    """
    Calculates the working point of the pump (delta_H_pump = delta_H_system) for the global system setting
    and a specified impeller size.
    Working point calculated using fsolve() of the difference between system and pump heads. 
    Tolerence for fsolve() = 0.001, initial  guess = 0.005 [m^3/s]

    ----- Units -----
    Flow rate in csv (input) -> [m^3/hr]
    Head in csv (input) -> [m]
    Impeller size (input) -> str [inches]
    Flow rate -> [m^3/s]
    Flow rate at working point (output) -> [m^3/hr]
    Head at working point (output) -> [m]
    """
    
    delta_H_pump, _ = find_pump_head(impeller_data, impeller_size) # gets polynomial approx for pump head

    def head_difference(F: float) -> tuple[np.ndarray, np.ndarray]: # head_difference = delta_H_pump - delta_H_system

        H_pump = delta_H_pump(F) # Flow in [m3/s]
        H_system = find_system_head(F) # Flow in [m3/s]
        
        return H_pump - H_system 
    
    F_at_working_point = fsolve(head_difference, x0=0.005, xtol=0.001) # Head_difference and x0 in [m3/s]
    Head_at_working_point = delta_H_pump(F_at_working_point) # F_at_working_point in [m3/s]
    
    return F_at_working_point*3600, Head_at_working_point # Convert flow rate units [m3/s] -> [m3/h] 


# ===============================================
# ================== Part C =====================
# ================================================


# ============= Q1: graph the pump and system curves for all impeller sizes on the same plot =============

def delta_H_pump_for_all_impeller_sizes(impeller_data: np.ndarray) -> dict:
    
    """
    Calculates the polynomial approximations for the pump heads of each impeller size in the csv.
    The function returns a dictionary with the impeller sizes as keys and the corresponding 
    polynomial functions for the pump head as values.
    
    ----- Units -----
    Flow rate in csv (input) -> [m^3/hr]
    Head in csv (input) -> [m]
    Impeller sizes.keys() -> str [inches]
    Impeller sizes.item() -> int
    Flow rate in polynomials (output)-> [m^3/s]
    """

    impeller_head_functions = {} # Dictionary to store the polynomial functions for each impeller size
    
    for impeller_size in impeller_size_dict.keys(): # Loop through all impeller sizes
        delta_H_pump, _ = find_pump_head(impeller_data, impeller_size) # returns polynomial with flow in [m^3/s]
        impeller_head_functions[impeller_size] = delta_H_pump # adds poly to dict (impeller_sizee -> keys, delta_H_pump -> items)

    return impeller_head_functions

def graph_all_impeller_sizes(impeller_data: np.ndarray):
    
    """
    Graphs the pump curve and system curve on the same plot for all impeller sizes on a single plot. 
    Also plots working points for each impeller size, as well as the efficiency plots from the 
    relevent csv.
    Boolean flag used for the pump curve used to not create multiple identical legend entries.
    NPSH Required axis was also added (log scale) to show the working points NPSH Required.
    Flow rate list form the csv is cut at 65 [m^3/hr] for aesthetic cutoff of all curves at same point.
   
    ----- Units -----
    Flow rate in csv (input) -> [m^3/hr]
    Head in csv (input) -> [m]
    Impeller size (input) -> str [inches]
    Flow rate -> [m^3/s]
    Flow rate in graph -> [m^3/hr]
    Head in graph -> [m]
    """
    
    delta_H_pump_functions = delta_H_pump_for_all_impeller_sizes(impeller_data) # Dictionary of polynomials for all impeller sizes 
    
    fig, ax1 = plt.subplots(figsize=(10, 6))

    eff_label_added = False
    
    efficiency_list = import_efficiency_data() # Import the efficiency data for all impeller sizes from the CSV file
    for eff, (x_eff, y_eff) in efficiency_list.items(): # Plot the efficiency curves for all impeller sizes
        
        eff_label = 'efficiencies' if not eff_label_added else '_nolegend_' # Single label for the pump curve in the legend
        eff_label_added = True
        
        ax1.plot(x_eff, y_eff, label=eff_label, color='blue') # Plot efficiency curve
        ax1.annotate(f'{eff}', xy=(x_eff[0], y_eff[0]), xytext=(x_eff[0],y_eff[0]), fontsize=9, color='black')


    H_pump_label_added = False #  Check if label for the pump curve is added to the legend
    max_F = 65 # [m^3/hr] Maximum flow rate for plotting, based on the test cases

    for impeller_size, delta_H_pump in delta_H_pump_functions.items():
    
        F_values = impeller_data[:, impeller_size_dict[impeller_size]]  # Flow rates from csv data for the specified impeller size [m^3/hr]
        F_values = F_values.copy() # Create a copy of the flow rate values to modify for plotting
        F_values = F_values[F_values <= max_F]  # Filter flow rates to be within the maximum flow rate for plotting
        F_values[-1] = max_F 

        H_pump = [delta_H_pump(F/3600) for F in F_values] # Calculate pump head for each flow rate (Flow rate converted to [m^3/s])
        H_system = [find_system_head(F/3600) for F in F_values] # Calculate system head for each flow rate (Flow rate converted  to [m^3/s])
        working_point = find_working_point(impeller_data, impeller_size) # Find the working point of the pump

        pump_label = '▲Hpump' if not H_pump_label_added else '_nolegend_' # Single label for the pump curve in the legend
        H_pump_label_added = True
        
        ax1.plot(F_values, H_pump, label=pump_label, color='red') # Plot pump curve
        ax1.scatter(working_point[0], working_point[1], color='black', zorder=5) # Plot the working point
        ax1.annotate(f'{impeller_size}"', xy=(0, H_pump[0]), xytext=(0,H_pump[0]+1), fontsize=10, color='black')
    
    ax1.plot(F_values, H_system, label='▲Hsys', color='green') # Plot system curve
    ax1.set_xlim(-3, 67)
    ax1.set_ylim(0, 80)

    ax2 = ax1.twiny() # Create a second x-axis for the system curve
    ax2.set_xscale('log') # Set the second x-axis to logarithmic scale
    ax2.set_xlim(0.5,17) # Set the limits of the second x-axis to match the tests

    ax1.set_xlabel('Q (m3/hr)')
    ax1.set_ylabel('h (m)')
    ax2.set_xlabel('NPSH Required (m)')

    ax2.set_xticks([2, 4, 6, 8, 10, 12, 14, 16]) # NPSH_R values for the axis as seen in the tests
    ax2.set_xticklabels(['2', '4', '6', '8', '10', '12', '14', '16'])
    
    plt.title(f'Pump and System Curves for all Impeller Sizes')
    ax1.legend(loc='lower right')
    ax1.grid()
    plt.tight_layout()
    plt.show()

# ============= Q2: Find the working points for all impeller sizes =======================================

def find_working_points_all_impeller_sizes(impeller_data: np.ndarray) -> dict:
    
    """
    Calculates the working points for each impeller size and inserts in a dictionary,
    where the impeller sizes (in str) are the keys and the working point coordinates are the items. 
    The working points are given as a tuple, with x coord = Flow rate @ working point [m^3/hr] and
    y coord = Head @ working point [m]
    
    ----- Units -----
    Flow rate in csv (input) -> [m^3/hr]
    Head in csv (input) -> [m]
    Impeller size -> str [inches]
    Flow rate @ working points -> [m^3/hr]
    Head @ working points -> [m]
    """

    working_points = {} 
    
    for impeller_size in impeller_size_dict.keys():
        working_point = find_working_point(impeller_data, impeller_size) # Tuple of working point coords (Flow rate [m^3/hr], Head [m])
        working_points[impeller_size] = working_point
    
    return working_points

# ============= Q3: NPSH_Required and Available for all impeller sizes ===================================

def NPSH_Required(F: float) -> float:
    """
    Calculates the required net positive suction head for a given flow rate
    using the given correlation. 
    
    ----- Units -----
    Flow rate (input) -> [m^3/s]
    Flow rate for correlation -> [m^3/hr]
    NPSH Required (output) -> [m]
    """
    F_m3_hr = F * 3600 # [m^3/s] -> [m^3/hr]
    NPSH_Required_result = 0.5761 * np.exp(0.0511 * F_m3_hr )
    
    return NPSH_Required_result

def NPSH_Available(F: float) -> float:
    
    """
    Calculates the available net positive suction head for a given flow rate. 
    The input Flow Rate must be in [m^3/hr]
    Approximates the head loss before the pump (h_LT_pump) as 25% of the total head loss (h_LT) for the system 
    (pump is located a quarter of the way of the full pipe length).

    ----- Units -----
    Flow rate (input) -> [m^3/s]
    NPSH Available (output) -> [m]
    """

    h_LT_pump = 0.25 * find_h_LT(F) #  approximation of h_LT for the system before the pump [m^2/s^2] (Flow rate converted  to [m^3/s])
    NPSH_Available = (P_IN - P_VAPOR) / (RHO * g) - Z_PUMP - h_LT_pump / g  # NPSH_Available in [m]
    
    return NPSH_Available

def Check_NPSH_for_all_impeller_sizes(impeller_data: np.ndarray) -> None:
    
    """
    Calculates the NPSH Required for the working points for each size of impeller and prints their values.
    Also claculates and prints NPSH Available and checks if it's larger than NPSH Required. 
    If it is, prints all clear message. If not, prints error message (cavitation risk).
    Uses the working point dictionary from Part C Q2 to find the working points.

    ----- Units -----
    Flow rate in csv (input) -> [m^3/hr]
    Head in csv (input) -> [m]
    NPSH Required (printed output) -> [m]
    NPSH Available (printed output) -> [m]
    """

    working_points = find_working_points_all_impeller_sizes(impeller_data)
    
    for impeller_size, working_point in working_points.items():
        F_at_working_point = working_point[0].item() # Flow rate at the working point in [m^3/hr]
        NPSH_req = NPSH_Required(F_at_working_point/3600)    #[m^3/hr] -> [m^3/s]
        NPSH_avail = NPSH_Available(F_at_working_point/3600) #[m^3/hr] -> [m^3/s]
        
        if NPSH_avail > NPSH_req:
            print(f'For impeller size {impeller_size} inches: the NPSH Available at the working point is {NPSH_avail:.3f} [m] -> No cavitation risk\n')
        else:
            print(f'For impeller size {impeller_size} inches: the NPSH Available at the working point is {NPSH_avail:.3f} [m] -> Cavitation risk\n')

# ----- Q4: Add Efficiencies to the pump curves for all impeller sizes --------

def import_efficiency_data() -> dict: 
    
    """
    Imports the given CSV data of the efficiency, returns dictionary with efficiency precentage as keys
    and (x: Flow rate[m3/hr] ,y: Head[m]) coordinates as values.
    """

    from pathlib import Path
    DATA_DIR = Path(__file__).parent
    efficiency_data = np.loadtxt(DATA_DIR / 'lab1_eff1.csv', delimiter=',', skiprows=2) # Load the efficiency data from the CSV file
    
    efficiency_list = np.loadtxt(DATA_DIR / 'lab1_eff1.csv', delimiter=',', dtype=str, max_rows=1)[1::2]
    efficiency_list = np.char.add(efficiency_list, '%')

    efficiency = {} # Dictionary to store the data for each efficiency curve
    for i, eff in enumerate(efficiency_list):
        x = efficiency_data[:, 2*i] # Flow rates in [m^3/s]
        y = efficiency_data[:, 2*i + 1] # Head losses in [m]
        efficiency[eff] = (x, y) # Store the data in the dictionary with efficiency as key

    return efficiency


# ================================================
# ============== Question to hand in =============
# ================================================

"""
Assuming we cannot change the pump paramters, we will have to modify the system. First, we replace the pipe with a pipe with a larger diameter.
This decreases the system resistance by decreasing the average flow velocity of the fluid. This 'strectches' and 'flattens' the system curve's
arch, which results with an interscection (working point) with the pump curve at a higher flow rate, while sacrificing head.

Secondly, we decrease the pressure gradient between out and in of the system by lowering the fluid tank pressure. 
This results in the lowering of the starting point of the system head curve, which also results in in a higher flow rate for the working point.
Using these two methods, we can potentially recover the original working point flow rate rate with the 6&7/8" impeller.
""" 

def find_flow_velocity_modified(F: float) -> float: 

    """
    Modified flow rate using a larger diameter pipe

    Flow rate units (input) -> [m^3/s]
    New pipe diameter (input) -> [m]
    Pipe diameter -> [m]
    Pipe Area -> [m^2]
    Average fluid velocity units (output) -> [m/s] 
    """

    A = math.pi * (D_PIPE_NEW / 2) ** 2 
    V = F / A 
    return V 

def find_h_LT_modified(F: float) -> float: 

    """
    Finds the modified h_LT using the new larger pipe diameter

    Flow rate units (input) -> [m^3/s]
    New pipe diameter (nput) -> [m]
    Average fluid velocity -> [m/s]
    Reynolds -> [no units]
    Fanning friction factor -> [no units]
    Absolute pipe roughness / Epsilon -> [m]
    Pipe diameter -> [m]
    Pipe length -> [m]
    No. of turns & turn friction factor -> [no units]
    Total frictional head loss -> [m^2/s^2]
    """

    V = find_flow_velocity(F) 
    Re = find_reynolds_number(F) 
    eD = EPSILON / D_PIPE_NEW 
    ff = f_fanning(Re, eD) 
     
    h_LT = (ff * (L_PIPE / D_PIPE_NEW) + 
            NUMBER_OF_TURNS * TURN_FRICTION_FACTOR) * 2 * V ** 2 
    
    return h_LT

def find_system_head_modified(F: float) -> float:
    
    """
    Calculates the system head using the known bernoulli equation 
    with the frictional head losses function.
    Here we use the modified parameters to try and change to working point of the 5&11/16" impeller
    to the working point of 6&7/8" impeller (in terms of flow rate)
    
    ---- Units -----
    Flow rate (input) -> [m^3/s]
    New pipe diameter (input) -> [m]
    New inlet pressure (input) -> [Pa]
    H_system (output) -> [m]
    """

    v_ave_in = 0 # fluid at top of tank is static [m/s]
    v_ave_out = find_flow_velocity_modified(F) # Average flow velocity [m/s]
    h_LT = find_h_LT_modified(F) # Frictional head loss [m^2/s^2]

    H_system = (P_OUT - P_IN_NEW) / (RHO * g) + DELTA_HEIGHT + (KINETIC_ALPHA * (v_ave_out**2 - v_ave_in**2) / (2 * g) + (h_LT / g))  # Total system head

    return H_system

def find_working_point_modified(impeller_data: np.ndarray, impeller_size: str) -> tuple[float, float]: 
    
    """
    Calculates the working point for the modified system (different pressure at inlet and pipe diameter)

    ----- Units -----
    Flow rate in csv (input) -> [m^3/hr]
    Head in csv (input) -> [m]
    New pipe diameter (input) -> [m]
    New inlet pressure (input) -> [Pa]
    Impeller size (input) -> str [inches]
    Flow rate -> [m^3/s]
    Flow rate (output) -> [m^3/hr]
    """
    
    delta_H_pump, _ = find_pump_head(impeller_data, impeller_size) # gets polynomial approx for pump head

    def head_difference(F: float) -> tuple[np.ndarray, np.ndarray]: # head_difference = delta_H_pump - delta_H_system

        H_pump = delta_H_pump(F) # Flow in [m3/s]
        H_system = find_system_head_modified(F) # Flow in [m3/s]
        
        return H_pump - H_system 
    
    F_at_working_point = fsolve(head_difference, x0=0.005, xtol=0.001) # Head_difference and x0 in [m3/s]
    Head_at_working_point = delta_H_pump(F_at_working_point) # F_at_working_point in [m3/s]
    
    return F_at_working_point*3600, Head_at_working_point # Convert flow rate units [m3/s] -> [m3/h] 

def find_new_parameters(impeller_data: np.ndarray) -> tuple[float, float]:
    """
    Calculates the new pressure difference and pipe diameter needed to keep the flow rate the same for the
    defective 6&7/8" impeller that works like a 5&11/16" impeller
    """
    global D_PIPE_NEW, P_IN_NEW # define the new paramters as global parameters
    P_IN_NEW = P_IN
    D_PIPE_NEW = D_PIPE

    F_at_working_point_6_and_7_8, _ = find_working_point(impeller_data, '6&7/8')
    F_at_working_point_5_and_11_16, _ = find_working_point(impeller_data, '5&11/16')

    F_diff = (F_at_working_point_6_and_7_8 - F_at_working_point_5_and_11_16)
    F_tol = 0.01 
    iter, curr_iter = 1e5, 0
    
    incr_P = 100 # [Pa]
    incr_D = 0.0005 # [m]

    while (abs(F_diff) > F_tol) and (curr_iter <= iter):
        if F_diff > 0:
            P_IN_NEW += incr_P
            D_PIPE_NEW += incr_D
            
            if P_IN_NEW <= P_VAPOR:
                incr_P = 0
            if D_PIPE_NEW <= 5e-4:
                incr_D = 0
            
            F_current, _ = find_working_point_modified(impeller_data, '5&11/16')
            F_diff = (F_at_working_point_6_and_7_8 - F_current)
        
        else:
            P_IN_NEW -= incr_P
            D_PIPE_NEW -= incr_D
            
            incr_P /= 2
            incr_D /= 2
        curr_iter += 1

    return D_PIPE_NEW, P_IN_NEW

def graph_all_impeller_sizes_modified(impeller_data: np.ndarray) -> None:
    
    """
    Creates the graph for the original system, and then adds the system curve for the
    modified system (new pressure and pipe diameter)
   
    ----- Units -----
    Flow rate in csv (input) -> [m^3/hr]
    Head in csv (input) -> [m]
    Impeller size (input) -> str [inches]
    Flow rate -> [m^3/s]
    Flow rate in graph -> [m^3/hr]
    Head in graph -> [m]
    """
    
    delta_H_pump_functions = delta_H_pump_for_all_impeller_sizes(impeller_data) # Dictionary of polynomials for all impeller sizes 
    
    fig, ax1 = plt.subplots(figsize=(10, 6))

    eff_label_added = False
    
    efficiency_list = import_efficiency_data() # Import the efficiency data for all impeller sizes from the CSV file
    for eff, (x_eff, y_eff) in efficiency_list.items(): # Plot the efficiency curves for all impeller sizes
        
        eff_label = 'efficiencies' if not eff_label_added else '_nolegend_' # Single label for the pump curve in the legend
        eff_label_added = True
        
        ax1.plot(x_eff, y_eff, label=eff_label, color='blue') # Plot efficiency curve
        ax1.annotate(f'{eff}', xy=(x_eff[0], y_eff[0]), xytext=(x_eff[0],y_eff[0]), fontsize=9, color='black')


    H_pump_label_added = False #  Check if label for the pump curve is added to the legend
    max_F = 65 # [m^3/hr] Maximum flow rate for plotting, based on the test cases

    for impeller_size, delta_H_pump in delta_H_pump_functions.items():
    
        F_values = impeller_data[:, impeller_size_dict[impeller_size]]  # Flow rates from csv data for the specified impeller size [m^3/hr]
        F_values = F_values.copy() # Create a copy of the flow rate values to modify for plotting
        F_values = F_values[F_values <= max_F]  # Filter flow rates to be within the maximum flow rate for plotting
        F_values[-1] = max_F 

        H_pump = [delta_H_pump(F/3600) for F in F_values] # Calculate pump head for each flow rate (Flow rate converted to [m^3/s])
        H_system = [find_system_head(F/3600) for F in F_values] # Calculate system head for each flow rate (Flow rate converted  to [m^3/s])
        working_point = find_working_point(impeller_data, impeller_size) # Find the working point of the pump

        pump_label = '▲Hpump' if not H_pump_label_added else '_nolegend_' # Single label for the pump curve in the legend
        H_pump_label_added = True
        
        ax1.plot(F_values, H_pump, label=pump_label, color='red') # Plot pump curve
        ax1.scatter(working_point[0], working_point[1], color='black', zorder=5) # Plot the working point
        ax1.annotate(f'{impeller_size}"', xy=(0, H_pump[0]), xytext=(0,H_pump[0]+1), fontsize=10, color='black')
    
    ax1.plot(F_values, H_system, label='▲Hsys', color='green') # Plot system curve

    H_system_modified = [find_system_head_modified(F/3600) for F in F_values]
    modified_working_point = find_working_point_modified(impeller_data, '5&11/16')

    ax1.plot(F_values, H_system_modified, label='Modified ▲Hsys', color='green', linestyle='--') # Plot the modifed system curve
    
    ax1.annotate(f'$\\mathbf{{{'System \\ modification'}}}$\n New Pipe Diameter: {D_PIPE_NEW:.3f} [m]\n New Inlet Pressure: {P_IN_NEW:.3f} [Pa]',
                 xy=(F_values[0],H_system_modified[0]), xytext=(2,3),
                 fontsize=10, color='black', bbox=dict(
                boxstyle='round,pad=0.3',    
                facecolor='lightblue',           
                edgecolor='gray',            
                alpha=0.9,                   
                linewidth=0.5))

    ax1.scatter(modified_working_point[0], modified_working_point[1], color='black', zorder=5) # Plot the working point of the modified system with 5&11/16"
    ax1.axvline(modified_working_point[0], color='black', linestyle=':')
    ax1.annotate('Q of modified working point', xy=(modified_working_point[0],75), 
                 xytext=(modified_working_point[0]-10,75), fontsize=10, color='black', 
                 ha='center', va='center', arrowprops=dict(facecolor='black', shrink=0.01))

    ax1.set_xlim(-3, 67)
    ax1.set_ylim(0, 80)

    ax2 = ax1.twiny() # Create a second x-axis for the system curve
    ax2.set_xscale('log') # Set the second x-axis to logarithmic scale
    ax2.set_xlim(0.5,17) # Set the limits of the second x-axis to match the tests

    ax1.set_xlabel('Q (m3/hr)')
    ax1.set_ylabel('h (m)')
    ax2.set_xlabel('NPSH Required (m)')

    ax2.set_xticks([2, 4, 6, 8, 10, 12, 14, 16]) # NPSH_R values for the axis as seen in the tests
    ax2.set_xticklabels(['2', '4', '6', '8', '10', '12', '14', '16'])
    
    plt.title(f'Pump and System Curves for all Impeller Sizes\n with a modification for pipe diameter and inlet pressure')
    ax1.legend(loc='lower right')
    ax1.grid()
    plt.tight_layout()
    plt.show()


# ------ Checks for PartA ------

def Check_PartA_Q2():
    graph_F_f_vs_Re(eD, Re_Laminar, Re_Turbulent)

def Check_PartA_Q3():
    h_LT_value = find_h_LT(F)
    print(f'Frictional head loss (h_LT) for flow rate F={F:.3f} [m^3/s]: {h_LT_value:.3f} [m^2/s^2]\n')

# ----- Checks for PartB -----

def Check_PartB_Q2():
    print(f'The system Head for flow rate = {F} [m^3/s] is: {find_system_head(F):.3f} [m]\n')

def Check_PartB_Q3():
    graph_pump_and_system_curves(impeller_data, "6") # Graph for 6 inch impeller
    working_point = find_working_point(impeller_data, "6") # Check the working point for the 6 inch impeller
    F_at_working_point, Head_at_working_point = working_point[0].item(), working_point[1].item()
    print(f'The flow rate at the working point is {F_at_working_point:.3f} [m^3/hr] \nThe head at the working point is {Head_at_working_point:.3f} [m]\n') 

# ----- Checks for PartC ------

def Check_PartC_Q1():
    graph_all_impeller_sizes(impeller_data)

def Check_PartC_Q2():
    working_points_all_impeller_sizes = find_working_points_all_impeller_sizes(impeller_data)
    
    for impeller_size, working_point in working_points_all_impeller_sizes.items(): # Extract working points from dictionary and print them
        F_at_working_point, Head_at_working_point = working_point[0].item(), working_point[1].item()
        print(f'For impeller size {impeller_size} inches: \nThe flow rate at the working point is {F_at_working_point:.3f} [m^3/hr] \nThe head at the working point is {Head_at_working_point:.3f} [m]\n')

def Check_PartC_Q3():
    Check_NPSH_for_all_impeller_sizes(impeller_data)

# ----- Checks for Open Question -----
def Check_Open_Q():
    find_new_parameters(impeller_data)
    graph_all_impeller_sizes_modified(impeller_data)


"""
Main Pipline to run the code when opened
"""

if __name__ == '__main__':
    
    Check_PartA_Q2(), Check_PartA_Q3()
    Check_PartB_Q2(), Check_PartB_Q3()
    Check_PartC_Q1(), Check_PartC_Q2(), Check_PartC_Q3()
    Check_Open_Q()



