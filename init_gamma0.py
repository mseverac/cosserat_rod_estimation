from utils import *
from parameters_computing import solve_cosserat_ivp
from scipy.optimize import root
import numpy as np
import matplotlib.pyplot as plt
from both_ends_fixed import cosserat_get_cable_state   
import time 

from scipy.optimize import root

from scipy.optimize import least_squares,minimize



def get_cosserat_gamma0(start,end,
                        R1 = np.matrix([[1,0,0],[0,1,0],[0,0,1]]),
                        R2 = np.matrix([[1,0,0],[0,1,0],[0,0,1]]),
                        init_shape=None ,
                        print_=False,plot=False, 
                        n_elem=49,E=3e7, poisson=0.5, rho=1400,
                        d=0.01, L=0.60,
                        xtol=1e-6, 
                        ):
    

    start_time = time.time()

    

    if init_shape is None:
        pp_list = cosserat_get_cable_state(
            start,end,n_elem=49,E=E)

        last_step = -1 
        init_shape = np.array(pp_list["position"][last_step])  



    # Initial guess for n0 and m0
    initial_guess = np.array([-0.17834246, -0.09977787, -0.21975261,
               -0.00928889,  0.02773056, -0.01566086])
    

    def residu_dernier_point(x,print_=False,plot=False):


        n0 = x[:3]
        m0 = x[3:6]



        sol = solve_cosserat_ivp(
            d=d, L=L, E=E, poisson=poisson, rho=rho,
            position=start,
            rotation=R1,
            n0=n0,
            m0=m0
        )

        # --- Résidu de position (3,)
        positions_calulées = np.array(sol.y[:3])

        

        residu = []

        for i in range(positions_calulées.shape[1]):

            residu.append(np.linalg.norm(positions_calulées[:, i] - init_shape[:, i])/(i+1))
            if print_:
                print(f"Position {i}: {positions_calulées[:, i]} vs Target: {init_shape[:, i]} -> Residual: {residu[-1]}")

            

        residu = np.array(residu)

        if plot:
            fig, ax = plt.subplots(figsize=(10, 6))
            ax.plot(positions_calulées[0], positions_calulées[2], label='Calculated Position', marker='o')
            ax.plot(init_shape[0], init_shape[2], label='Target Position', marker='x')
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
            position=start,
            rotation=R1,
            n0=n0,
            m0=m0
        )

        # --- Résidu de position (3,)
        positions_calulées = np.array(sol.y[:3])

        

        residu = []

        for i in range(positions_calulées.shape[1]):
            residu.append(np.linalg.norm(positions_calulées[:, i] - init_shape[:, i])*(i+1))
            if print_:
                print(f"Position {i}: {positions_calulées[:, i]} vs Target: {init_shape[:, i]} -> Norm : {np.linalg.norm(positions_calulées[:, i] - init_shape[:, i])} Residual: {residu[-1]}")

            

        residu = np.array(residu)

        if plot:
            fig, ax = plt.subplots(figsize=(10, 6))
            ax.plot(positions_calulées[0], positions_calulées[2], label='Calculated Position', marker='o')
            ax.plot(init_shape[0], init_shape[2], label='Target Position', marker='x')
            ax.set_xlabel('X Position')
            ax.set_ylabel('Z Position')
            ax.set_title('Cable Position Comparison')
            ax.legend()
            plt.grid()
            plt.show(block=True)

        return  residu


    # Use least_squares to find the optimal n0 and m0

    initermediate_result = least_squares(
            residu_dernier_point,
            initial_guess,
            xtol=xtol,
            )
    
    result = least_squares(
            residu_dernier_point_fin,
            initermediate_result.x,
            xtol=xtol,
            )
    
    if plot :
    
        n0_opt = result.x[:3]
        m0_opt = result.x[3:6]

        sol = solve_cosserat_ivp(
            d=d, L=L, E=E, poisson=poisson, rho=rho,
            position=start,
            rotation=R1,
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
        plot_frame(ax, R1, start, length=0.8, name="start")
        plot_frame(ax, R2, end, length=0.8, name="end")

        # 2D views
        ax2 = fig.add_subplot(222)
        ax2.set_xlabel('x'); ax2.set_ylabel('z'); ax2.set_title('z vs x'); ax2.grid()
        ax3 = fig.add_subplot(224)
        ax3.set_xlabel('y'); ax3.set_ylabel('z'); ax3.set_title('z vs y'); ax3.grid()

        sol_initial = solve_cosserat_ivp(d=d, L=L, E=E, poisson=poisson, rho=rho,
            position=start, rotation=R1, n0=initial_guess[:3], m0=initial_guess[3:6], print_=False)
        
        sol_inter = solve_cosserat_ivp(d=d, L=L, E=E, poisson=poisson, rho=rho,
            position=start, rotation=R1, n0=initermediate_result.x[:3], m0=initermediate_result.x[3:6], print_=False)


        print("Initial guess:", initial_guess)
        print("Intermediate result:", initermediate_result.x)
        print("Final result:", result.x)

        plot_cable(sol, 'green', ax, ax2, ax3, end, n0=n0_opt, m0=m0_opt,E=E)
        plot_cable(init_shape, 'red', ax, ax2, ax3,end)
        plot_cable(sol_inter, 'yellow', ax, ax2, ax3, end, n0=initermediate_result.x[:3], m0=initermediate_result.x[3:6],E=E)
        plot_cable(sol_initial, 'blue', ax, ax2, ax3, end, n0=initial_guess[:3], m0=initial_guess[3:6],E=E)



        show_plot(block=True,
                   title="res:"+numpy_array_to_string(result.x,print_=False)+"_init:"+numpy_array_to_string(initial_guess,print_=False),
                   plot=False,
                   folder="results_gamma0"
                   )
        


    if print_:

        end_time = time.time()
        execution_time = end_time - start_time
        print(f"Execution time: {execution_time:.4f} seconds")






    
