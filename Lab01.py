import numpy as np
import matplotlib.pyplot as plt
import math
from scipy.integrate import solve_ivp
from scipy.optimize import fsolve

#Exercise 1

data = np.loadtxt('plot-data.csv', delimiter=',', skiprows=1)

y = data[:, 1]
x = data[:, 0]

f1 = np.polyfit(x, y, 4)
p1 = np.poly1d(f1)
print("Value of Polynomial Fit at t=2:", p1(2)) #value of Polynomial at t=2

#Finding Roots of the polynomial

root1 = fsolve(p1, -3) #initial val at t=-3
root2 = fsolve(p1, 0) #initial val at t=0
root3 = fsolve(p1, 5) #initial val at t=5
root4 = fsolve(p1, 7) #initial val at t=7

print("Roots of the polynomial are: ", root1, root2, root3, root4)

#Finding Intersection Points t in (3,7)

def f2(t):
    return 42*np.sin(0.5*t) + 14

def intersection(t, ):
    return p1(t) - f2(t)

intersection_root = fsolve(intersection, 5) #initial val at t=5
print("Intersection point in the range of (3,7) is: ", intersection_root)

#Plot p1 and f2 in the range (-3,8)

#Root Labels

Roots_x = [root1, root2, root3, root4]
Roots_y = [0, 0, 0, 0]
Root_Labels = ['root1', 'root2', 'root3', 'root4']

plt.scatter(Roots_x, Roots_y, color='black')

for i, label in enumerate(Root_Labels):
    plt.annotate(label, (Roots_x[i], Roots_y[i]), xytext=(5, 5), textcoords='offset points')

#Intersetion label

Intersection_x = intersection_root
Intersection_y = f2(Intersection_x)

plt.scatter(Intersection_x, Intersection_y, color='none')
plt.annotate(f'[{float(Intersection_x):.4f}, {float(Intersection_y):.4f}]', (Intersection_x, Intersection_y), xytext=(5, 5), textcoords='offset points')

#Graph plot

t = np.linspace(-3, 8, 100)
plt.plot(t, p1(t), linestyle='--', label='f1= ?  ?  ?', color='red')
plt.plot(t, f2(t), linestyle='--',label='f2= 42sin(0.5t) + 14', color='blue')
plt.xlabel('t')
plt.ylabel('y')
plt.axhline(0, color='black')
plt.title('Lab 0 - exercise 1')
plt.legend()
plt.grid()
plt.show()

#Exercise 2

def f1(t): #f1 from exercise 1 is used here
    return p1(t)

#f2 defined on line 29

def system(t, y): #vecctor of y values
    y1, y2 = y

    dy1_dt = y1**0.2 + f1(t) + y2
    dy2_dt = (f2(t) * y2) / 50 

    return [dy1_dt, dy2_dt]

tspan = np.linspace(0, 10, 1001) #time steps
y1y2_0 = [1, 1] #initial conditions

sol = solve_ivp(system, [tspan[0], tspan[-1]], y1y2_0, t_eval=tspan)

plt.plot(sol.t, sol.y[0], label='y1(t)', color='cornflowerblue')
plt.plot(sol.t, sol.y[1], label='y2(t)', color='orange')
plt.xlabel('t')
plt.ylabel('y')
plt.title('Lab 0 - exercise 2')
plt.legend()
plt.grid()
plt.show()




