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

