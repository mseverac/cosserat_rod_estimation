from utils import *
from txt_reader import *
from parameters_computing import solve_cosserat_ivp
from scipy.optimize import root
import numpy as np
import matplotlib.pyplot as plt
from both_ends_fixed import cosserat_get_cable_state    

from scipy.optimize import root

from scipy.optimize import least_squares,minimize

# Paramètres physiques
(d, E, poisson, rho) = (0.01, 10e6, 0.5, 1400)
L = 0.60

# Position et orientation initiales et finales
R2 = rotations2[0]
T2 = positions2[0]
R3 = rotations1[0]
T3 = positions1[0]

# Fonction résidu pour root (résidu vectoriel : position finale - cible)
from scipy.spatial.transform import Rotation as R
import time

pp_list = cosserat_get_cable_state(
    T2,T3,n_elem=49,E=1e7)

last_step = -1 
positions = np.array(pp_list["position"][last_step])  # Shape (3, n_elem+1)




def residu_dernier_point(x,print_=False,plot=False):
    n0 = x[:3]
    m0 = x[3:6]

    sol = solve_cosserat_ivp(
        d=d, L=L, E=E, poisson=poisson, rho=rho,
        position=T2,
        rotation=R2,
        n0=n0,
        m0=m0
    )

    # --- Résidu de position (3,)
    positions_calulées = np.array(sol.y[:3])

    

    residu = []

    for i in range(positions_calulées.shape[1]):
        residu.append(np.linalg.norm(positions_calulées[:, i] - positions[:, i])/(i+1))
        if print_:
            print(f"Position {i}: {positions_calulées[:, i]} vs Target: {positions[:, i]} -> Residual: {residu[-1]}")

        

    residu = np.array(residu)

    if plot:
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.plot(positions_calulées[0], positions_calulées[2], label='Calculated Position', marker='o')
        ax.plot(positions[0], positions[2], label='Target Position', marker='x')
        ax.set_xlabel('X Position')
        ax.set_ylabel('Z Position')
        ax.set_title('Cable Position Comparison')
        ax.legend()
        plt.grid()
        plt.show(block=True)

    return  residu



def residu_dernier_point_fin(x,print_=False,plot=False):
    n0 = x[:3]
    m0 = x[3:6]

    sol = solve_cosserat_ivp(
        d=d, L=L, E=E, poisson=poisson, rho=rho,
        position=T2,
        rotation=R2,
        n0=n0,
        m0=m0
    )

    # --- Résidu de position (3,)
    positions_calulées = np.array(sol.y[:3])

    

    residu = []

    for i in range(positions_calulées.shape[1]):
        residu.append(np.linalg.norm(positions_calulées[:, i] - positions[:, i])*(i+1))
        if print_:
            print(f"Position {i}: {positions_calulées[:, i]} vs Target: {positions[:, i]} -> Norm : {np.linalg.norm(positions_calulées[:, i] - positions[:, i])} Residual: {residu[-1]}")

        

    residu = np.array(residu)

    if plot:
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.plot(positions_calulées[0], positions_calulées[2], label='Calculated Position', marker='o')
        ax.plot(positions[0], positions[2], label='Target Position', marker='x')
        ax.set_xlabel('X Position')
        ax.set_ylabel('Z Position')
        ax.set_title('Cable Position Comparison')
        ax.legend()
        plt.grid()
        plt.show(block=True)

    return  residu





# Guess initial



def test_initial_guess(initial_guess, E=E,fin=False, xtol=1e-8,both=False):
    start_time = time.time()

    # Call the function and measure execution time
    

    
        
    """print("résidu : ",residu_dernier_point(initial_guess,print_=False,plot=False))
    print("passed")"""

    print("testing Initial guess:", initial_guess,"with E =", E)

    if both:
        fin = False


    if fin:
        result = least_squares(
            residu_dernier_point_fin,
            initial_guess,
            xtol=xtol,
        )

    else:
        result = least_squares(
            residu_dernier_point,
            initial_guess,
            xtol=xtol,
        )



    print("\nRésultat de l’optimisation (root):")
    print("Success:", result.success)
    print("Message:", result.message)
    if fin :
        print("Résidu final:", residu_dernier_point_fin(result.x,print_=False))
    else: 
        print("Résidu final:", residu_dernier_point(result.x,print_=False))
    print("Solution trouvée x =", numpy_array_to_string(np.array(result.x)))

    # Simulation avec les efforts trouvés
    n0_opt = result.x[:3]
    m0_opt = result.x[3:6]

    

    sol = solve_cosserat_ivp(
        d=d, L=L, E=E, poisson=poisson, rho=rho,
        position=T2,
        rotation=R2,
        n0=n0_opt,
        m0=m0_opt,
        print_=False
    )

    # Affichage
    fig = plt.figure(figsize=(16, 10))
    ax = fig.add_subplot(121, projection='3d')
    ax.set_xlim([-1, 1])
    ax.set_ylim([-1, 1])
    ax.set_zlim([-1, 1])
    ax.set_box_aspect([1, 1, 1])
    plot_frame(ax, np.eye(3), np.zeros(3), length=0.8, name="F0")
    plot_frame(ax, R2, T2, length=0.8, name="start")
    plot_frame(ax, R3, T3, length=0.8, name="end")

    # 2D views
    ax2 = fig.add_subplot(222)
    ax2.set_xlabel('x'); ax2.set_ylabel('z'); ax2.set_title('z vs x'); ax2.grid()
    ax3 = fig.add_subplot(224)
    ax3.set_xlabel('y'); ax3.set_ylabel('z'); ax3.set_title('z vs y'); ax3.grid()

    sol_initial = solve_cosserat_ivp(d=d, L=L, E=E, poisson=poisson, rho=rho,
        position=T2, rotation=R2, n0=initial_guess[:3], m0=initial_guess[3:6], print_=False)


    plot_cable(sol, 'green', ax, ax2, ax3, T3, n0=n0_opt, m0=m0_opt,E=E)
    plot_cable(positions, 'red', ax, ax2, ax3,T3)
    plot_cable(sol_initial, 'blue', ax, ax2, ax3, T3, n0=initial_guess[:3], m0=initial_guess[3:6],E=E)



    show_plot(block=True, title="res:"+numpy_array_to_string(result.x,print_=False)+"_init:"+numpy_array_to_string(initial_guess,print_=False),plot=False)
    
    end_time = time.time()
    execution_time = end_time - start_time
    print(f"Execution time: {execution_time:.4f} seconds")



    if both :
        print("testing with fin and already optimised parameters ")
        test_initial_guess(result.x, E=E, fin=True, xtol=xtol) 

"""initial_guesses = [
    np.array([-0.18283517, -0.0971982, -0.21271606, -0.00527533, 0.01916997, -0.01073625]),
    np.array([-0.17854694, -0.11992282, -0.19540045, -0.01104545, 0.01688951, -0.00278103]),
    np.array([-0.18069000, -0.10856051, -0.20405826, -0.00816039, 0.01802974, -0.00675864]),
    np.array([-0.18498000, -0.09500000, -0.21500000, -0.00450000, 0.02050000, -0.01150000]),
    np.array([-0.17600000, -0.12500000, -0.19000000, -0.01200000, 0.01600000, -0.00250000]),
    np.array([-0.18300000, -0.10000000, -0.21000000, -0.00600000, 0.01900000, -0.00900000]),
    np.array([-0.17900000, -0.11500000, -0.20000000, -0.01000000, 0.01700000, -0.00400000]),
    np.array([-0.18150000, -0.10500000, -0.20750000, -0.00750000, 0.01850000, -0.00750000]),
    np.array([-0.18500000, -0.09250000, -0.21750000, -0.00400000, 0.02100000, -0.01200000]),
    np.array([-0.17750000, -0.12250000, -0.19250000, -0.01150000, 0.01650000, -0.00300000]),
    np.array([-0.18200000, -0.09750000, -0.21250000, -0.00550000, 0.01950000, -0.01050000]),
    np.array([-0.17800000, -0.12000000, -0.19500000, -0.01100000, 0.01700000, -0.00300000]),
]"""

"""Es = np.logspace(6, 7.5, 3) 
"""

Es = [3.16e7,5e6]

Es = np.logspace(7, 8, 10)

initial_guesses = [
    np.array([-0.17834246, -0.09977787, -0.21975261, -0.00928889,  0.02773056,
 -0.01566086])]

for initial_guess in initial_guesses:
    for E in Es:
        
        test_initial_guess(initial_guess,E=E,both=True,xtol=1e-8)