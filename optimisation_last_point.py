from parameters_computing import dist_dernier_point,plot_cable,show_plot,solve_cosserat_ivp
from parameters_computing import T2, R2, forces, torques, T3,ax, ax2, ax3
import numpy as np
from scipy.optimize import minimize
import matplotlib.pyplot as plt


# Initial guess for the parameters
initial_guess = np.array([0.01, 10e8, 0.4, 1400])
initial_guess = np.array([9.27944556e-03, 1.00000000e+09 ,4.00000075e-01 ,1.40000000e+03,-0.06103516,
                           -0.18310547 ,-0.54931641,
                            -0.01336679 ,-0.00512704 ,-0.00457773])

initial_guess = np.array([-12,0,0,0,1.8,0])

bounds = [(0.001, 0.02), (10e5, 10e11), (0.1, 1), (900, 3000),(-1, 1), (-1, 1), (-1, 1), (-1, 1), (-1, 1), (-1, 1)]

bounds = [(-1,1),(-1,1),(-1,1),(-0.1,0.1),(-0.1,0.1),(-0.1,0.1)]
#d E poisson rho

# Options pour l'optimisation
options = {
    'maxiter': 1000,       # Nombre maximal d'itérations (par défaut : 100)
    'iprint': 2,          # Niveau de détail des affichages (0 = silence, 1 = affichage de base, 2+ = détaillé)
    'disp': True,         # Afficher les messages de convergence (True/False)
}

# Perform the optimization
result = minimize(dist_dernier_point, initial_guess)#,bounds=bounds)

# Print the results
print("Optimized parameters:", result.x)
print("Minimum value of dist_dernier_point:", result.fun)
"""
d,E,poisson,rho = result.x
sol = solve_cosserat_ivp(
    d=d, L=0.60, E=E, poisson=poisson, rho=rho,
    position=T2,
    rotation=R2,
    n0=forces[0],
    m0=torques[0]
)
"""

n0 = result.x[:3]
m0 = result.x[3:6]

(d, E, poisson, rho) = [0.01, 10e8, 0.4, 1400]


sol= solve_cosserat_ivp(
    d=0.01, L=0.60, E=10e8, poisson=0.48, rho=1400,
    position=T2,
    rotation=R2,
    n0=n0,
    m0=m0
    )

# Plot the optimized cable
plot_cable(sol, 'red', ax, ax2, ax3, T3)
show_plot()




"""


from parameters_computing import *

fig = plt.figure(figsize=(16, 10))

# 3D plot (large on the left)
ax = fig.add_subplot(121, projection='3d')
ax.set_xlim([-1, 1])
ax.set_ylim([-1, 1])
ax.set_zlim([-1, 1])
ax.set_box_aspect([1, 1, 1])  # For isotropic axes

# Frame 1: Identity
R1 = np.eye(3)
T1 = np.array([0, 0, 0])
plot_frame(ax, R1, T1, length=0.8, name="F0")

R2 = rotations2[0]
T2 = positions2[0]
plot_frame(ax, R2, T2, length=0.8, name="start")

R3 = rotations1[0]
T3 = positions1[0]
plot_frame(ax, R3, T3, length=0.8, name="end")



# Create additional 2D subplots for the cable visualization
ax2 = fig.add_subplot(222)  # z vs x
ax2.set_xlabel('x')
ax2.set_ylabel('z')
ax2.set_title('z vs x')
ax2.grid()

ax3 = fig.add_subplot(224)  # z vs y
ax3.set_xlabel('y')
ax3.set_ylabel('z')
ax3.set_title('z vs y')
ax3.grid()


plot_cable(sol, 'red', ax, ax2, ax3, T3)
show_plot()


"""