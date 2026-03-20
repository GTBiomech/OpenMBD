# geometry_mesh_generator.py
# Citation: Tierney. OpenMBD: An Open-Source Multibody Dynamics Simulator for Biomechanics Research and Education. F1000Research, 2026.
# Version: 1.0 
# Research Contact: Dr Gregory Tierney (g.tierney@ulster.ac.uk)

import numpy as np
import math

def generate_ellipsoid_mesh(dims, transform, resolution=12):
    u = np.linspace(0, 2 * np.pi, resolution)
    v = np.linspace(0, np.pi, resolution)
    x = np.outer(np.cos(u), np.sin(v)) * dims[0]
    y = np.outer(np.sin(u), np.sin(v)) * dims[1]
    z = np.outer(np.ones(np.size(u)), np.cos(v)) * dims[2]
    points = np.stack([x.flatten(), y.flatten(), z.flatten(), np.ones(x.size)])
    transformed = transform @ points
    return (transformed[0, :].reshape(x.shape),
            transformed[1, :].reshape(y.shape),
            transformed[2, :].reshape(z.shape))

def rotation_matrix_from_euler(angles_deg):
    rad = np.radians(angles_deg)
    c0, c1, c2 = np.cos(rad[0]), np.cos(rad[1]), np.cos(rad[2])
    s0, s1, s2 = np.sin(rad[0]), np.sin(rad[1]), np.sin(rad[2])
    Rx = np.array([[1, 0, 0],
                   [0, c2, -s2],
                   [0, s2, c2]])
    Ry = np.array([[c1, 0, s1],
                   [0, 1, 0],
                   [-s1, 0, c1]])
    Rz = np.array([[c0, -s0, 0],
                   [s0, c0, 0],
                   [0, 0, 1]])
    return Rz @ Ry @ Rx

def rotation_matrix_from_euler_axis(axis, angle_deg):
    angle_rad = math.radians(angle_deg)
    c = np.cos(angle_rad)
    s = np.sin(angle_rad)
    R = np.identity(4)
    if axis == 'X':
        R[1:3, 1:3] = [[c, -s], [s, c]]
    elif axis == 'Y':
        R[0:3, 0:3] = [[c, 0, s], [0, 1, 0], [-s, 0, c]]
    elif axis == 'Z':
        R[0:3, 0:3] = [[c, -s, 0], [s, c, 0], [0, 0, 1]]
    return R