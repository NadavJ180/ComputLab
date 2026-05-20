import numpy
from scipy.optimize import fsolve
import matplotlib.pyplot as plt
import numpy as np
import math as math
from scipy.integrate import solve_ivp


# Don't change this parametrs names, only valuses if needed!!
F = 0.0073 #[m3/s]
L_PIPE = 25 #[m]
D_PIPE = 0.042 #[m]
EPSILON = 5 * 10 ** -6 #[m]
NU = 2 * 10 ** -6 #[m2/s]
Z_PUMP= -0.5 #[m]

NUMBER_OF_TURNS = 10 #[no units]
TURN_FRICTION_FACTOR = 0.75 #[no units]
P_IN = 86126.25 #[Pa]
P_OUT = 202650 #[Pa]
RHO = 921 #[kg/m3]
KINETIC_ALPHA = 1 #[no units]
DELTA_HEIGHT = 4 #[m]

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
    
    elif 0 < Re <= 4e3: # Laminar flow
        return 'Laminar'
    
    elif 4e3 < Re <= 1e8: # Turbulent flow
          return 'Turbulent'
    

# ------ Q1: Find Fanning friction factor --------

def f_fanning(Re, eD):
    
    Roughness = Check_Roughness(eD)
    Flow = Check_Reynolds_Type(Re)

    if not Roughness or Flow != 'Laminar' and Flow != 'Turbulent': # If the input values are not valid, return 0
        return 0

    if Flow == 'Laminar':
        ff = 16 / Re
        return ff
    
    elif Flow == 'Turbulent':
        ff = 1.375 * 1e-3 * (1 + (2 * 1e4 * eD + 1e6 / Re) ** (1/3))
        return ff


# ------ Q2: Graph F_f vs Re  ---------
'''
fig, ax = plt.subplots(figsize=(10, 6))
f_Laminar = [f_fanning(Re, 0) for Re in Re_Laminar] # Roughness doesn't affect laminar flow
ax.loglog(Re_Laminar, f_Laminar, color='brown')

color = ['royalblue', 'green', 'mediumpurple'] # similar colours as provided in the test case

for eD_value in eD: #Claculate the fanning number for each eD values
    f_Turbulent = [f_fanning(Re, eD_value) for Re in Re_Turbulent]
    ax.loglog(Re_Turbulent, f_Turbulent, color=color[eD.index(eD_value)])
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
'''
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
            NUMBER_OF_TURNS * TURN_FRICTION_FACTOR) * 2 * V ** 2  # Frictional head loss
    
    return h_LT #[m^2/s^2]


# ------ Checks for Q1 ------
'''
Check1_Re, Check1_eD = 1000, 0.03
Check2_Re, Check2_eD = 0, 0.001
Check3_Re, Check3_eD = 1000, 0.001
Check4_Re, Check4_eD = 4000, 0.0007
Check5_Re, Check5_eD = 10e6, 0.00003
Check6_Re, Check6_eD = 153475, 0.000576

ff_1 = print(f'ff={f_fanning(Check1_Re, Check1_eD)}')
ff_2 = print(f'ff={f_fanning(Check2_Re, Check2_eD)}')
ff_3 = print(f'ff={f_fanning(Check3_Re, Check3_eD)}')
ff_4 = print(f'ff={f_fanning(Check4_Re, Check4_eD)}')
ff_5 = print(f'ff={f_fanning(Check5_Re, Check5_eD)}')
ff_6 = print(f'ff={f_fanning(Check6_Re, Check6_eD)}')

#------ Checks for Q3 ------
h_LT_value = find_h_LT(F)
print(f'Frictional head loss (h_LT) for flow rate F={F} [m^3/s]: {h_LT_value} [m^2/s^2]')
'''


# Check if the function for fanning needs to print the ff number, as well as if it needs to print the error message!!!


# ------ Part B --------

# ----- Q1: Import impeller data and approximate to 2nd degree polynomial --------

# Load the impeller data from the CSV file
impeller_data = np.loadtxt('lab1_impellers.csv', delimiter=',', skiprows=2) 

# Dictionary to map impeller sizes to their corresponding column indices in the CSV data
impeller_columns = { 
    '6': 0,
    '5&11/16': 2,
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

    if str(impeller_size) not in impeller_columns: # Check if the impeller size is valid
        raise ValueError(f"Invalid impeller size. Valid sizes are: {', '.join(impeller_columns.keys())}")

    col = impeller_columns[str(impeller_size)]
    x = impeller_data[:, col] # Flow rate (Q) [m3/h]
    x_m3s = x / 3600 # Convert flow units (m3/h -> m3/s)
    y = impeller_data[:, col+1] # Head (H) [m]
    
    coeffs = np.polyfit(x_m3s, y, 2) # Fit a 2nd degree polynomial 
    polynomial = np.poly1d(coeffs) # Create a polynomial from coefficients
    
    return polynomial, col # Polynomial in [m^3/s]


# ----- Q2: Find the system head and graph the pump and system curve --------


def find_system_head(F): # Function of flow rate (F) [m^3/s] 
    
    g = 9.81 # gravity [m/s^2]
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
print(find_h_LT(F)) # Check the frictional head loss for the given flow rate F
print(find_system_head(F)) # Check the system head for the given flow rate F
print(find_pump_head(impeller_data, "6")) # Check the pump head for the given flow rate F
graph_pump_and_system_curves(impeller_data, "6") # Graph for 6 inch impeller
working_point = find_working_point(impeller_data, "6") # Check the working point for the 6 inch impeller
print(round(float(working_point[0]),3), round(float(working_point[1]),3)) # Check the working point for the 6 inch impeller