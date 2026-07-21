import numpy as np
import matplotlib.pyplot as plt

# --- System Parameters ---
alpha = 3.0
x_F = 0.20
x_D = 0.85
x_W = 0.05
q = 0.6

# --- Data Generation ---
# Generate x values from 0 to 1 for the curves
x = np.linspace(0, 1, 200)

# Equilibrium curve: y = (alpha * x) / (1 + (alpha - 1) * x)
y_eq = (alpha * x) / (1 + (alpha - 1) * x)

# --- Pinch Point Calculation ---
# q-line equation: y = -1.5x + 0.5
a_quad = 3.0
b_quad = 3.5
c_quad = -0.5

# Solve quadratic equation for x_pinch (taking the positive root)
x_pinch = (-b_quad + np.sqrt(b_quad**2 - 4 * a_quad * c_quad)) / (2 * a_quad)
y_pinch = -1.5 * x_pinch + 0.5

# --- Plotting ---
plt.figure(figsize=(9, 9), dpi=150)

# 1. Plot the equilibrium curve and the diagonal y=x line
plt.plot(x, y_eq, color='#1f77b4', linewidth=2)
plt.plot([0, 1], [0, 1], color='gray', linestyle='--', linewidth=1.5)

# 2. Plot the extended q-line (y = -1.5x + 0.5)
plt.plot([0, 1/3], [0.5, 0], color='#ff7f0e', linewidth=2, linestyle='-.')

# 3. Plot the Rectifying Operating Line for minimum reflux
plt.plot([x_D, x_pinch], [x_D, y_pinch], color='#2ca02c', linewidth=2, linestyle='-.')

# --- Point Markers and Explicit Numerical Annotations ---

# Distillate point (x_D, x_D)
plt.scatter([x_D], [x_D], color='black', zorder=5)
plt.annotate(f'$x_D = {x_D}$\n(Distillate)', (x_D, x_D), 
             textcoords="offset points", xytext=(35, -35), ha='center', fontsize=16)

# Feed point (x_F, x_F)
plt.scatter([x_F], [x_F], color='black', zorder=5)
plt.annotate(f'$x_F = {x_F:.2f}$\n(Feed)', (x_F, x_F), 
             textcoords="offset points", xytext=(50, -15), ha='center', fontsize=16)

# Bottoms point (x_W, x_W)
plt.scatter([x_W], [x_W], color='black', zorder=5)
plt.annotate(f'$x_W = {x_W}$\n(Bottoms)', (x_W, x_W), 
             textcoords="offset points", xytext=(50, -20), ha='center', fontsize=16)

# Pinch Point with an Arrow
plt.scatter([x_pinch], [y_pinch], color='red', zorder=6, s=60)
plt.annotate(f'Pinch Point\n($x \\approx {x_pinch:.2f}, y \\approx {y_pinch:.2f}$)', 
             xy=(x_pinch, y_pinch),                 
             xytext=(15, 150),                      
             textcoords="offset points", ha='center', color='darkred', fontsize=16, fontweight='bold',
             arrowprops=dict(arrowstyle="->", color='darkred', lw=1.5, connectionstyle="arc3,rad=0.1"))

# --- General Text & Line Annotations ---

# Top-Left Parameters Text Box
plt.text(0.01, 0.80, 
         f"Relative Volatility: $\\alpha = {alpha}$\nFeed quality: $q = {q}$", 
         fontsize=16, bbox=dict(facecolor='white', edgecolor='none', alpha=0.8))

# Equilibrium Curve Label & Arrow
x_eq_mid = 0.5
y_eq_mid = (alpha * x_eq_mid) / (1 + (alpha - 1) * x_eq_mid) # Calculates y on the curve at x=0.5
plt.annotate('Equilibrium Curve', 
             xy=(x_eq_mid, y_eq_mid), 
             xytext=(0.40, 0.90), 
             ha='center', fontsize=16,
             arrowprops=dict(arrowstyle="->", color='black', lw=1.0))

# Operating Line Label & Arrow
x_op_mid = 0.6
y_op_mid = 0.7528 * (x_op_mid - 0.85) + 0.85 
plt.annotate('Operating Line\nfor $R_{min}$', 
             xy=(x_op_mid, y_op_mid), 
             xytext=(0.75, 0.50), 
             ha='center', fontsize=16,
             arrowprops=dict(arrowstyle="->", color='black', lw=1.0))

# q-line Label & Arrow
plt.annotate('q-line', 
             xy=(0.25, -1.5*0.25 + 0.5), 
             xytext=(0.40, 0.15), 
             ha='center', fontsize=16,
             arrowprops=dict(arrowstyle="->", color='black', lw=1.0))

# --- Formatting ---
plt.xlabel('Mole Fraction of Ethanol in Liquid ($x$)', fontsize=16)
plt.ylabel('Mole Fraction of Ethanol in Vapor ($y$)', fontsize=16)

plt.xlim(0, 1)
plt.ylim(0, 1)
plt.xticks(np.arange(0, 1.1, 0.1))
plt.yticks(np.arange(0, 1.1, 0.1))

plt.grid(True, which='both', linestyle='-', linewidth=0.5, color='lightgray')

plt.tight_layout()
plt.savefig('mccabe_thiele_pinch_final.png', format='png', dpi=300, bbox_inches='tight')
plt.show()