from scipy.spatial.transform import Rotation as R
import numpy as np

def rotation_angle_between(R1, R2):
    # Rotation relative : R = R1^T * R2
    R_rel = R.from_matrix(R1.T @ R2)
    # Angle de rotation (entre -pi et pi)
    angle = R_rel.magnitude()
    return angle if angle <= np.pi else angle - 2 * np.pi


def numpy_array_to_string(array):
    array_str = np.array2string(array, separator=', ', precision=8, suppress_small=True)
    return f"np.array({array_str})"