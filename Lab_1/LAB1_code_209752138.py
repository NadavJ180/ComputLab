import numpy
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

# ------ Q3: Frictional losses as a function of flow rate --------

def find_flow_velocity(F): # Finds the average flow velocity for a given flow rate
    A = math.pi * (D_PIPE / 2) ** 2 #pipe cross-section area
    V = F / A 
    return V

def find_reynolds_number(F): # Finds the Reynolds number for a given flow rate
    V = find_flow_velocity(F)
    Re = V * D_PIPE / NU
    return Re

def find_h_LT(F): #calculates the friction loss as a function of flow 
    
    V = find_flow_velocity(F) #average flow velocity
    Re = find_reynolds_number(F) # Reynolds number
    eD = EPSILON / D_PIPE # Relative roughness
    ff = f_fanning(Re, eD) # Fanning friction factor
     
    h_LT = (ff * (L_PIPE / D_PIPE) + 
            NUMBER_OF_TURNS * TURN_FRICTION_FACTOR) * 2 * V ** 2  # Frictional head loss
    
    return round(h_LT, 3) #[m^2/s^2]





# ------ Checks for Q1 ------
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
print(f'Frictional head loss (h_LT) for flow rate F={F} m3/s: {h_LT_value} m2/s2')



# Check if the function for fanning needs to print the ff number, as well as if it needs to print the error message!!!