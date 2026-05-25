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
Re_Laminar = np.geomspace(1e3, 4000, 20) # Reynolds numbers from 10^2 to 4e3
Re_Turbulent = np.geomspace(4001, 1e8, 50) # Reynolds numbers from 10^4 to 10^8
# 4000 was ommited from the turbulent range to create the discontinuity in the 
# graph between laminar and turbulent flow

# -------- Part A --------

# ------- Validation ---------
def Check_Roughness(eD): #Checks if the relative roughnes in within the bounds for the approximation
    
    if eD < 0 or eD > 0.01:
        return False
    else:
        return True

def Check_Reynolds_Type(Re): #Checks in which flow regime the system is in
     
    if Re <= 0 or Re > 1e8: # Reynolds is not physical
        return False
    
    elif 0 < Re < 4001: # Laminar flow
        return 'Laminar'
    
    elif 4001 <= Re <= 1e8: # Turbulent flow
          return 'Turbulent'
    

# ------ Q1: Find Fanning friction factor --------

def f_fanning(Re, eD):
    
    Roughness = Check_Roughness(eD)
    Flow = Check_Reynolds_Type(Re)

    if not Roughness or (Flow != 'Laminar' and Flow != 'Turbulent'): # If the input values are not valid, return 0
        return 0

    if Flow == 'Laminar':
        ff = 16 / Re
        return ff
    
    elif Flow == 'Turbulent':
        ff = 1.375 * 1e-3 * (1 + (2 * 1e4 * eD + 1e6 / Re) ** (1/3))
        return ff


# ------ Q2: Graph F_f vs Re  ---------

def graph_F_f_vs_Re(eD, Re_Laminar, Re_Turbulent):
    fig, ax = plt.subplots(figsize=(10, 6))
    f_Laminar = [f_fanning(Re, 0) for Re in Re_Laminar] # Roughness doesn't affect laminar flow
    ax.loglog(Re_Laminar, f_Laminar, color='brown')

    color = ['royalblue', 'green', 'mediumpurple'] # similar colours as provided in the test case

    for i, eD_value in enumerate(eD): #Claculate the fanning number for each eD values
        f_Turbulent = [f_fanning(Re, eD_value) for Re in Re_Turbulent]
        ax.loglog(Re_Turbulent, f_Turbulent, color=color[i])
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

# ------ Q3: Frictional losses as a function of flow rate --------

def find_flow_velocity(F): # Finds the average flow velocity for a given flow rate
    A = math.pi * (D_PIPE / 2) ** 2 #pipe cross-section area
    V = F / A 
    return V

def find_reynolds_number(F): # Finds the Reynolds number for a given flow rate
    V = find_flow_velocity(F)
    Re = V * D_PIPE / NU
    return Re

def find_h_LT(F): #calculates the friction loss as a function of flow [m^3/s]
    
    V = find_flow_velocity(F) #average flow velocity
    Re = find_reynolds_number(F) # Reynolds number
    eD = EPSILON / D_PIPE # Relative roughness
    ff = f_fanning(Re, eD) # Fanning friction factor
     
    h_LT = (ff * (L_PIPE / D_PIPE) + 
            NUMBER_OF_TURNS * TURN_FRICTION_FACTOR) * 2* V ** 2 # Frictional head loss
    
    return h_LT #[m^2/s^2]


# ------ Checks for Q1 ------
def Check_PartA_Q1():
    Check1_Re, Check1_eD = 1000, 0.03
    Check2_Re, Check2_eD = 0, 0.001
    Check3_Re, Check3_eD = 1000, 0.001
    Check4_Re, Check4_eD = 4000, 0.0007
    Check5_Re, Check5_eD = 10e6, 0.00003
    Check6_Re, Check6_eD = 153475, 0.000576

    print(f'ff={round(f_fanning(Check1_Re, Check1_eD),3)}')
    print(f'ff={round(f_fanning(Check2_Re, Check2_eD),3)}')
    print(f'ff={round(f_fanning(Check3_Re, Check3_eD),3)}')
    print(f'ff={round(f_fanning(Check4_Re, Check4_eD),3)}')
    print(f'ff={round(f_fanning(Check5_Re, Check5_eD),3)}')
    print(f'ff={round(f_fanning(Check6_Re, Check6_eD),3)}')

# ------ Checks for Q2 ------
def Check_PartA_Q2():
    graph_F_f_vs_Re(eD, Re_Laminar, Re_Turbulent)

#------ Checks for Q3 ------
def Check_PartA_Q3():
    h_LT_value = find_h_LT(F)
    print(f'Frictional head loss (h_LT) for flow rate F={F:.3f} [m^3/s]: {h_LT_value:.3f} [m^2/s^2]')

# Checks
Check_PartA_Q1(), Check_PartA_Q2(), Check_PartA_Q3()

# ------ Part B --------

# ----- Q1: Import impeller data and approximate to 2nd degree polynomial --------

# Load the impeller data from the CSV file
impeller_data = np.loadtxt('lab1_impellers.csv', delimiter=',', skiprows=2) 

# Dictionary to map impeller sizes to their corresponding column indices in the CSV data
impeller_size_dict = { 
    '5&11/16': 2,
    '6': 0,
    '6&9/16': 4,
    '6&7/8': 6
} 

def find_pump_head(impeller_data, impeller_size:str): 
    
    """
    Function that takes the raw impeller data and str of impeller size (in inches) 
    and returns a 2nd order polynomial approximation.
    Flow rate in csv is in m^3/hr, head is in [m]
    Flow rate in polynomial is in m^3/s
    Valid impeller sizes are: 6, 5&11/16, 6&9/16, 6&7/8 [inch] (str!!!)
    """
    
    if str(impeller_size) not in impeller_size_dict: # Check if the impeller size is valid
        raise ValueError(f"Invalid impeller size. Valid sizes are: {', '.join(impeller_size_dict.keys())}")

    col = impeller_size_dict[str(impeller_size)]
    x = impeller_data[:, col] # Flow rate (Q) [m3/h]
    x_m3s = x / 3600 # Convert flow units (m3/h -> m3/s)
    y = impeller_data[:, col+1] # Head (H) [m]
    
    coeffs = np.polyfit(x_m3s, y, 2) # Fit a 2nd degree polynomial 
    polynomial = np.poly1d(coeffs) # Create a polynomial from coefficients
    
    return polynomial, col # Polynomial in [m^3/s]

# ----- Q2: Find the system head and graph the pump and system curve --------

def find_system_head(F): # Function of flow rate (F) [m^3/s] 
    
    v_ave_in = 0 # fluid at top of tank is static [m/s]
    v_ave_out = find_flow_velocity(F) # Average flow velocity [m/s]
    h_LT = find_h_LT(F) # Frictional head loss [m^2/s^2]

    H_system = (P_OUT - P_IN) / (RHO * g) + DELTA_HEIGHT + (KINETIC_ALPHA * (v_ave_out**2 - v_ave_in**2) / (2 * g) + (h_LT / g))  # Total system head

    return H_system

def graph_pump_and_system_curves(impeller_data, impeller_size): # Graph the pump curve and system curve on the same plot
    delta_H_pump, impeller_size_col = find_pump_head(impeller_data, impeller_size) # Get the polynomial function for the specified impeller size
    
    F_values = impeller_data[:, impeller_size_col]  # Flow rates from csv data for the specified impeller size [m3/h]
    H_pump = [delta_H_pump(F/3600) for F in F_values] # Calculate pump head for each flow rate(Flow rate converted  to [m^3/s])
    H_system = [find_system_head(F/3600) for F in F_values] # Calculate system head for each flow rate (Flow rate converted  to [m^3/s])
    
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

# ----- Q3: Find the working point of the pump --------

def find_working_point(impeller_data, impellar_size): # Working point -> delta_H_pump = delta_H_system
    delta_H_pump, _ = find_pump_head(impeller_data, impellar_size) 

    def head_difference(F): # head_difference = delta_H_pump - delta_H_system

        H_pump = delta_H_pump(F) # Flow in [m3/s]
        H_system = find_system_head(F) # Flow in [m3/s]
        
        return H_pump - H_system 
    
    F_at_working_point = fsolve(head_difference, x0=0.005, xtol=0.001) # Head_difference and x0 in [m3/s]
    Head_at_working_point = delta_H_pump(F_at_working_point) # F_at_working_point in [m3/s]
    
    return F_at_working_point*3600, Head_at_working_point # Convert flow rate units [m3/s] -> [m3/h] 

# ----- Checks -----

def Check_PartB():
    graph_pump_and_system_curves(impeller_data, "6") # Graph for 6 inch impeller
    working_point = find_working_point(impeller_data, "6") # Check the working point for the 6 inch impeller
    F_at_working_point, Head_at_working_point = round(working_point[0].item(),3), round(working_point[1].item(),3)
    print(f'The flow rate at the working point is {F_at_working_point} [m^3/h] \nThe head at the working point is {Head_at_working_point} [m]') 

#Check_PartB()

# ----- Part C --------

# ----- Q1: graph the pump and system curves for all impeller sizes on the same plot --------

def delta_H_pump_for_all_impeller_sizes(impeller_data): # Get the polynomial function for all impeller sizes
    '''
    This function returns a dictionary with the impeller sizes as keys and the corresponding polynomial functions for the pump head as values.
    '''
    impeller_head_functions = {} # Dictionary to store the polynomial functions for each impeller size
    
    for impeller_size in impeller_size_dict.keys(): # Loop through all impeller sizes
        delta_H_pump, _ = find_pump_head(impeller_data, impeller_size)
        impeller_head_functions[impeller_size] = delta_H_pump

    return impeller_head_functions

def graph_all_impeller_sizes(impeller_data): # Graph the pump and system curves for all impeller sizes on the same plot
    '''
    Graphs the pump and system curves for all impeller sizes on the same plot.
    Uses the polynomial functions for the pump heads of all impeller sizes from the dictionary
    '''
    
    delta_H_pump_functions = delta_H_pump_for_all_impeller_sizes(impeller_data) # Dictionary of polynomials for all impeller sizes 
    
    fig, ax1 = plt.subplots(figsize=(10, 6))

    labels_added = False #  Check if label for the pump curve is added to the legend
    max_F = 65 # Maximum flow rate for plotting, based on the test cases

    for impeller_size, delta_H_pump in delta_H_pump_functions.items():
    
        F_values = impeller_data[:, impeller_size_dict[impeller_size]]  # Flow rates from csv data for the specified impeller size [m^3/hr]
        F_values = F_values.copy() # Create a copy of the flow rate values to modify for plotting
        F_values = F_values[F_values <= max_F]  # Filter flow rates to be within the maximum flow rate for plotting
        F_values[-1] = max_F 

        H_pump = [delta_H_pump(F/3600) for F in F_values] # Calculate pump head for each flow rate (Flow rate converted to [m^3/s])
        H_system = [find_system_head(F/3600) for F in F_values] # Calculate system head for each flow rate (Flow rate converted  to [m^3/s])
        working_point = find_working_point(impeller_data, impeller_size) # Find the working point of the pump

        pump_label = '▲Hpump' if labels_added == 0 else '__nolegend__' # Single label for the pump curve in the legend
        labels_added += 1
        
        ax1.plot(F_values, H_pump, label=pump_label, color='red') # Plot pump curve
        ax1.scatter(working_point[0], working_point[1], color='black', zorder=5) # Plot the working point
        ax1.annotate(f'{impeller_size}"', xy=(0, H_pump[0]), xytext=(0,H_pump[0]+2), fontsize=10, color='black')
    
    efficiency_list = import_efficiency_data() # Import the efficiency data for all impeller sizes from the CSV file
    for eff, (x_eff, y_eff) in efficiency_list.items(): # Plot the efficiency curves for all impeller sizes
        ax1.plot(x_eff, y_eff, color='blue') # Plot efficiency curve
        ax1.annotate(f'{eff}', xy=(x_eff[0], y_eff[0]), xytext=(x_eff[0],y_eff[0]), fontsize=9, color='black')
    
    ax1.plot(F_values, H_system, label='▲Hsys', color='green') # Plot system curve
    ax1.set_xlim(-3, 67)
    ax1.set_ylim(0, 80)

    ax2 = ax1.twiny() # Create a second x-axis for the system curve
    ax2.set_xscale('log') # Set the second x-axis to logarithmic scale
    ax2.set_xlim(0.5,17) # Set the limits of the second x-axis to match the tests

    ax1.set_xlabel('Q (m3/hr)')
    ax1.set_ylabel('h (m)')
    ax2.set_xlabel('NPSH Required (m)')

    ax2.set_xticks([2, 4, 6, 8, 10, 12, 14, 16])
    ax2.set_xticklabels(['2', '4', '6', '8', '10', '12', '14', '16'])
    
    plt.title(f'Pump and System Curves for all Impeller Sizes')
    ax1.legend(loc='lower right')
    ax1.grid()
    plt.tight_layout()
    plt.show()

# ----- Q2: Find the working points for all impeller sizes --------
def find_working_points_all_impeller_sizes(impeller_data): # Find the working points for all impeller sizes
    
    working_points = {}
    
    for impeller_size in impeller_size_dict.keys():
        working_point = find_working_point(impeller_data, impeller_size)
        working_points[impeller_size] = working_point
    
    return working_points

# ----- Q3: NPSH_Required and Available for all impeller sizes --------
def NPSH_Required(F): # Calculate NPSH_Required for a given impeller size
    '''
    Calculates the required net positive suction head for a given flow rate. The input Flow Rate must be in [m^3/hr]
    '''
    NPSH_Required = 0.5761 * np.exp(0.0511 * F)
    
    return NPSH_Required

def NPSH_Available(F): # Calculate NPSH_Available for a given flow rate
    '''
    Calculates the available net positive suction head for a given flow rate. 
    The input Flow Rate must be in [m^3/hr]
    Approximates the head loss before the pump (h_LT_pump) as 25% of the total head loss (h_LT) for the system 
    (pump is located a quarter way of the system).
    '''
    h_LT_pump = 0.25 * find_h_LT(F/3600) #  approximation of h_LT for the system befpre the pump [m^2/s^2] (Flow rate converted  to [m^3/s])
    NPSH_Available = (P_IN - P_VAPOR) / (RHO * g) - Z_PUMP - h_LT_pump / g  # NPSH_Available in [m]
    
    return NPSH_Available

def Check_NPSH_for_all_impeller_sizes(impeller_data): # Check if NPSH_Available is greater than NPSH_Required for all impeller sizes at their working points
    working_points = find_working_points_all_impeller_sizes(impeller_data)
    
    for impeller_size, working_point in working_points.items():
        F_at_working_point = working_point[0].item() # Flow rate at the working point in [m^3/hr]
        NPSH_req = NPSH_Required(F_at_working_point)
        NPSH_avail = NPSH_Available(F_at_working_point)
        
        print(f'For impeller size {impeller_size} inches: \nNPSH Required at the working point is {round(NPSH_req,3)} [m]')
        
        if NPSH_avail > NPSH_req:
            print(f'The NPSH Available at the working point is {round(NPSH_avail,3)} [m] -> No cavitation risk\n')
        else:
            print(f'The NPSH Available at the working point is {round(NPSH_avail,3)} [m] -> Cavitation risk\n')

# ----- Q4: Add Efficiencies to the pump curves for all impeller sizes --------

def import_efficiency_data(): # Import the efficiency data from the CSV file and return a dictionary with impeller sizes as keys and efficiency data as values
    '''
    Import the given CSV data of the efficiency, return dictionary with efficiency as keys
    and (x,y) coordinates as values. The x coordinates are the flow rates in [m^3/s] and the y coordinates are the head losses [m].
    '''
    efficiency_data = np.loadtxt('lab1_eff1.csv', delimiter=',', skiprows=2) # Load the efficiency data from the CSV file
    
    efficiency_list = ['68%', '70%', '71%'] # List of efficiency values corresponding to the columns in the CSV file

    efficiency = {} # Dictionary to store the data for each efficiency curve
    for i, eff in enumerate(efficiency_list):
        x = efficiency_data[:, 2*i] # Flow rates in [m^3/s]
        y = efficiency_data[:, 2*i + 1] # Head losses in [m]
        efficiency[eff] = (x, y) # Store the data in the dictionary with efficiency as key

    return efficiency



# ----- Checks -----
def Check_PartC_Q1():
    graph_all_impeller_sizes(impeller_data)

def Check_PartC_Q2():
    working_points_all_impeller_sizes = find_working_points_all_impeller_sizes(impeller_data)
    
    for impeller_size, working_point in working_points_all_impeller_sizes.items(): # Extract working points from dictionary and print them
        F_at_working_point, Head_at_working_point = round(working_point[0].item(),3), round(working_point[1].item(),3)
        print(f'For impeller size {impeller_size} inches: \nThe flow rate at the working point is {F_at_working_point} [m^3/h] \nThe head at the working point is {Head_at_working_point} [m]\n')

def Check_PartC_Q3():
    Check_NPSH_for_all_impeller_sizes(impeller_data)

Check_PartC_Q1(), Check_PartC_Q2(), Check_PartC_Q3()
