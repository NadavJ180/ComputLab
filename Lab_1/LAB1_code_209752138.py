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

eD = [0.00001, 0.0001, 0.001] # Relative roughness values to be used in the graph
Re_Laminar = np.geomspace(1e3, 4000, 20) # Reynolds numbers from 10^2 to 4e3
Re_Turbulent = np.geomspace(4001, 1e8, 50) # Reynolds numbers from 10^4 to 10^8


# -------- Part A --------

# ------- Validation ---------
def Check_Roughness(eD):
    
    if eD < 0 or eD > 0.01:
        raise ValueError('ff= 0 error in one of the inserted values')

def Check_Reynolds_Type(Re):
     
    if Re <= 0 or Re > 1e8: # Reynolds is not physical
        raise ValueError('ff= 0 error in one of the inserted values')
    
    elif 0 < Re <= 4e3: # Laminar flow
        #print(f'Re = {Re:.2f}, laminar')
        return 'Laminar'
    
    elif 4e3 < Re <= 1e8: # Turbulent flow
          #print(f'Re = {Re:.2f}, turbulant')
          return 'Turbulent'
    

# ------ Q1: Find Fanning friction factor --------
def f_fanning(Re, eD):
    Check_Roughness(eD)
    Flow = Check_Reynolds_Type(Re)

    if Flow == 'Laminar':
        ff = 16 / Re
        print(f'ff= {ff}')
        return ff
    
    elif Flow == 'Turbulent':
        ff = 1.375 * 1e-3 * (1 + (2 * 1e4 * eD + 1e6 / Re) ** (1/3))
        print(f'ff= {ff}')
        return ff


# ------ Checks for part 1 ------
Check1_Re, Check1_eD = 1000, 0.03
Check2_Re, Check2_eD = 0, 0.001
Check3_Re, Check3_eD = 1000, 0.001
Check4_Re, Check4_eD = 4000, 0.0007
Check5_Re, Check5_eD = 10e6, 0.00003
Check6_Re, Check6_eD = 153475, 0.000576

#ff_1 = f_fanning(Check1_Re, Check1_eD) # Should raise ValueError for roughness
#ff_2 = f_fanning(Check2_Re, Check2_eD) # Should raise ValueError for Reynolds number
ff_3 = f_fanning(Check3_Re, Check3_eD) # Should return  0.016 for laminar flow
ff_4 = f_fanning(Check4_Re, Check4_eD) # Should return a value for turbulent flow
ff_5 = f_fanning(Check5_Re, Check5_eD) # Should raise ValueError for Reynolds number
ff_6 = f_fanning(Check6_Re, Check6_eD) # Should return a value for turbulent flow

# ------ Q2: Graph F_f vs Re  ---------
fig, ax = plt.subplots(figsize=(10, 6))
f_Laminar = [f_fanning(Re, 0) for Re in Re_Laminar] # Roughness doesn't affect laminar flow
ax.loglog(Re_Laminar, f_Laminar, color='brown')

color = ['royalblue', 'green', 'mediumpurple']
for eD_value in eD:
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



