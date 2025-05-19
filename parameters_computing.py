import numpy as np
from scipy.spatial.transform import Rotation as R
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import geometry_msgs.msg

from test_solve_ivp import solve_cosserat_ivp
import random


#distance entre le capteur de force et le point d'attache de la corde
l=0.15
r=np.array([0,0,l])

sols = None

def parse_vector3(vector_str):
    """Parse a geometry_msgs.msg.Vector3 string into a NumPy array."""
    # print(f"Parsing Vector3: {vector_str}")

    components = vector_str.strip('geometry_msgs.msg.Vector3()').split(',')

    # for i, component in enumerate(components[:3]):
    #     print(f"Component {i}: {component}")
    #     print(f"Parsed value: {component.split('=')[1].strip(')')}")

    x, y, z = [float(component.split('=')[1].strip(")")) for component in components[:3]]
    
    result = np.array([x, y, z])
    # print(f"Parsed Vector3 result: {result}")
    return result

def parse_quaternion(quaternion_str):
    """Parse a geometry_msgs.msg.Quaternion string into a NumPy array."""
    # print(f"Parsing Quaternion: {quaternion_str}")

    components = quaternion_str.strip('geometry_msgs.msg.Quaternion()').split(',')

    # for i, component in enumerate(components[:4]):
    #     print(f"Component {i}: {component}")
    #     print(f"Parsed value: {component.split('=')[1].strip(')')}")

    x, y, z, w = [float(component.split('=')[1].strip(")")) for component in components[:4]]

    result = np.array([x, y, z, w])
    # print(f"Parsed Quaternion result: {result}")
    return result

def quaternion_to_rotation_matrix(quaternion):
    """Convert a quaternion to a rotation matrix."""
    # print(f"Converting Quaternion to Rotation Matrix: {quaternion}")
    rotation_matrix = R.from_quat(quaternion).as_matrix()
    # print(f"Rotation Matrix result: \n{rotation_matrix}")
    return rotation_matrix

def read_wrench_poses(file_path):
    """Read the wrench poses from the file and organize them into NumPy arrays."""
    positions1, rotations1 = [], []
    positions2, rotations2 = [], []
    forces, torques = [], []

    # print(f"Opening file: {file_path}")
    with open(file_path, 'r') as file:
        for line_number, line in enumerate(file, start=1):
            # print(f"Processing line {line_number}: {line.strip()}")
            # Split the line into components
            parts = line.split('geometry_msgs.msg.')
            # print("-------------------")
            # print("Split parts:")
            # for i, part in enumerate(parts):
            #     print(f"Part {i}:")
            #     print(part)
            #     print(part.strip())
            # print("-------------------")

            # Parse the first transform
            translation1 = parse_vector3(parts[2].strip())
            quaternion1 = parse_quaternion(parts[3].strip())
            # print(f"Parsed translation1: {translation1}")
            # print(f"Parsed quaternion1: {quaternion1}")
            positions1.append(translation1)
            rotations1.append(quaternion_to_rotation_matrix(quaternion1))

            # Parse the second transform
            translation2 = parse_vector3(parts[5].strip())
            quaternion2 = parse_quaternion(parts[6].strip())
            # print(f"Parsed translation2: {translation2}")
            # print(f"Parsed quaternion2: {quaternion2}")
            positions2.append(translation2)
            rotations2.append(quaternion_to_rotation_matrix(quaternion2))

            # Parse the wrench
            force = parse_vector3(parts[8].strip())
            torque = parse_vector3(parts[9].strip())
            # print(f"Parsed force: {force}")
            # print(f"Parsed torque: {torque}")
            forces.append(force)
            torques.append(torque)

    # Convert lists to NumPy arrays
    positions1 = np.array(positions1)
    rotations1 = np.array(rotations1)
    positions2 = np.array(positions2)
    rotations2 = np.array(rotations2)
    forces = np.array(forces)
    torques = np.array(torques)

    """print("Final shapes of arrays:")
    print(f"positions1 shape: {positions1.shape}")
    print(f"rotations1 shape: {rotations1.shape}")
    print(f"positions2 shape: {positions2.shape}")
    print(f"rotations2 shape: {rotations2.shape}")
    print(f"forces shape: {forces.shape}")
    print(f"torques shape: {torques.shape}")"""

    return positions1, rotations1, positions2, rotations2, forces, torques

# Example usage
file_path = 'wrench_poses.txt'
positions1, rotations1, positions2, rotations2, forces, torques = read_wrench_poses(file_path)



for i, (torque, force) in enumerate(zip(torques, forces)):
    print("--------------------")
    print("rotation:", rotations2[i])
    print("torque:", torque)
    print("force:", force)
    torques[i] = torque - np.dot(torque, force)
    print("torque after:", torques[i])
    print("---------------------")



# Print shapes of the arrays
"""print("positions1 shape:", positions1.shape)
print("rotations1 shape:", rotations1.shape)
print("positions2 shape:", positions2.shape)
print("rotations2 shape:", rotations2.shape)
print("forces shape:", forces.shape)
print("torques shape:", torques.shape)"""


def plot_frame(ax, R, T, length=1.0, name=None):
    """
    Trace un repère orthonormé 3D à partir d'une matrice de rotation R (3x3) et d'une translation T (3,)
    """
    # Origine
    origin = T.reshape(3)
    
    # Axes : colonnes de R
    x_axis = R[:, 0]
    y_axis = R[:, 1]
    z_axis = R[:, 2]
    
    # Tracer les 3 axes
    ax.quiver(*origin, *(x_axis * length), color='r', label='x' if name is None else f'{name}_x')
    ax.quiver(*origin, *(y_axis * length), color='g', label='y' if name is None else f'{name}_y')
    ax.quiver(*origin, *(z_axis * length), color='b', label='z' if name is None else f'{name}_z')
    
    # Optionnel : nom de l'origine
    if name:
        ax.text(*origin, f'{name}', fontsize=12, color='k')

# Exemple d'utilisation :


"""
sol = solve_cosserat_ivp(
    d=0.01, L=0.60, E=3e9, poisson=0.4, rho=1000,
    position=T2,
    rotation=R2,
    n0=forces[0],
    m0=torques[0]
)


sols = []
ds = []
Es = []
poissons = []
rhos = []
for _ in range(10):
    d = 0.01 + random.uniform(-0.001, 0.001)
    L = 0.60  # Keeping L constant
    E = pow(10,random.uniform(7,9))
    poisson = 0.4 + random.uniform(-0.2, 0.4)
    rho = 1000 + random.uniform(-100, 600)
    ds.append(d)
    Es.append(E)
    poissons.append(poisson)
    rhos.append(rho)
    
    sol = solve_cosserat_ivp(
        d=d, L=L, E=E, poisson=poisson, rho=rho,
        position=T2,
        rotation=R2,
        n0=forces[0],
        m0=torques[0]
    )

    sols.append(sol)
for i in range(2):
    d = 0.01 
    L = 0.60  
    E = 10e8
    poisson = 0.9
    rho = 1000 
    if i == 1:
        rho = 10000

    sols.append(solve_cosserat_ivp(
        d=d, L=L, E=E, poisson=poisson, rho=rho,
        position=T2,
        rotation=R2,
        n0=forces[0],
        m0=torques[0]
    ))

"""
fig = plt.figure(figsize=(16, 10))

# 3D plot (large on the left)
ax = fig.add_subplot(121, projection='3d')
ax.set_xlim([-1, 1])
ax.set_ylim([-1, 1])
ax.set_zlim([-1, 1])
ax.set_box_aspect([1, 1, 1])  # For isotropic axes

"""print("positions1:", positions1[0])
print("rotations1:", rotations1[0])
print("positions2:", positions2[0])
print("rotations2:", rotations2[0])"""

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


def plot_cable(sol, color, ax, ax2, ax3, T3,n0=None, m0=None):
    """
    Plot the cable represented by sol on the provided axes.

    Parameters:
    - sol: The solution object containing the cable data.
    - color: The color to use for the cable.
    - ax: The 3D axis for the cable plot.
    - ax2: The 2D axis for the z vs x plot.
    - ax3: The 2D axis for the z vs y plot.
    - ax4: The 2D axis for the x, y, z vs s plot.
    - T3: The reference point to scatter on the 2D plots.
    """
    # Plot the position of each point of the cable in the 3D figure
    ax.plot(sol.y[0], sol.y[1], sol.y[2], label='Cable', color=color, linewidth=2)

    # Optionally, scatter the points for better visualization
    #ax.scatter(sol.y[0], sol.y[1], sol.y[2], color=color, s=10, label='Cable Points')

    # Plot z vs x
    ax2.plot(sol.y[0], sol.y[2], color=color)
    ax2.scatter(T3[0], T3[2], color='green', marker='x', label='T3')

    # Plot z vs y
    ax3.plot(sol.y[1], sol.y[2], color=color,label=f"n0 : {n0} ,m0 : {m0}")
    ax3.scatter(T3[1], T3[2], color='green', marker='x')
    ax3.legend()
    

# Usage of the function

  # Generate 10 distinct colors using the Viridis colormap

def show_plot():
    plt.tight_layout()
    plt.show()

(d, E, poisson, rho) = [0.01, 10e8, 0.4, 1400]

n0s = np.linspace(np.array([-12,0,0]), np.array([-12,0,0]), 1)
print("nos shape:", n0s.shape)
m0s = np.linspace(np.array([0.0,1.8,0]), np.array([-0.0,1.8,0]),1)
print("mos shape:", m0s.shape)

sols = []
wrenches = []
for m0 in m0s : 
    for n0 in n0s:
        sol = solve_cosserat_ivp(
            d=d, L=0.60, E=E, poisson=poisson, rho=rho,
            position=T2,
            rotation=R2,
            n0=n0,
            m0=m0
        )
        sols.append(sol)
        wrenches.append((n0, m0))
        print("----------------------")
        print("n0:", n0)
        print("m0:", m0)


sols = None
if sols is not None:
    colors = plt.cm.viridis(np.linspace(0, 1, len(sols)))
    for sol, color,(n0,m0) in zip(sols, colors,wrenches):
        plot_cable(sol, color, ax, ax2, ax3, T3,n0=n0, m0=m0)


def dist_dernier_point(x,plot=False): 

    if len(x) == 4:

        (d, E, poisson, rho) = x
        n0=forces[0],
        m0=torques[0]

    elif len(x) == 10:
        (d,E,poisson,rho) = x[:4]
        n0 = x[4:7]
        m0 = x[7:10]

    elif len(x) == 6:
        (d, E, poisson, rho) = (0.01, 10e8, 0.48, 1400)

        n0 = x[:3]
        m0 = x[3:6]

    

    
    sol = solve_cosserat_ivp(
        d=d, L=0.60, E=E, poisson=poisson, rho=rho,
        position=T2,
        rotation=R2,
        n0=n0,
        m0=m0
    )

    #if plot :
     #   plot_cable(sol, 'red', ax, ax2, ax3, T3)

    #print("sol shape:", sol.y.shape)

    p = sol.y[:3,-1]
    #print("p:", p)
    #print("T3:", T3)
    return np.linalg.norm(p - T3)

#print(dist_dernier_point(0.01, 1e8, 0.4, 1000))


