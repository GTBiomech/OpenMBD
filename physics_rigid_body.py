# physics_rigid_body.py
# Citation: Tierney. OpenMBD: An Open-Source Multibody Dynamics Simulator for Biomechanics Research and Education. F1000Research, 2026.
# Version: 1.0 
# Research Contact: Dr Gregory Tierney (g.tierney@ulster.ac.uk)

import numpy as np
from physics_utils import quat_to_matrix, matrix_to_quat

class RigidBody:
    def __init__(self, name, data, model_id):
        self.name = name
        self.model_id = model_id

        if 'mass' in data:
            self.mass = max(data['mass'], 0.001)
        else:
            self.mass = 1.0

        self.inv_mass = 1.0/self.mass if self.mass > 0 else 0.0

        if 'inertia' in data:
            ivec = data['inertia']
            if len(ivec) == 6:
                self.I_body = np.array([
                    [ivec[0], -ivec[3], -ivec[5]],
                    [-ivec[3], ivec[1], -ivec[4]],
                    [-ivec[5], -ivec[4], ivec[2]]
                ])
            elif len(ivec) == 3:
                self.I_body = np.diag(np.maximum(ivec, 1e-6))
            else:
                self.I_body = np.eye(3) * 0.01
        else:
            self.I_body = np.eye(3) * 0.01

        self.I_body = self.make_positive_definite(self.I_body)

        try:
            self.inv_I_body = np.linalg.inv(self.I_body)
        except np.linalg.LinAlgError:
            self.inv_I_body = np.linalg.pinv(self.I_body)
            self.I_body += 1e-6 * np.eye(3)
            self.inv_I_body = np.linalg.inv(self.I_body)

        if 'cg' in data:
            self.cg_local = np.array(data['cg'])
        elif 'center_of_mass' in data:
            self.cg_local = np.array(data['center_of_mass'])
        elif 'center_of_gravity' in data:
            self.cg_local = np.array(data['center_of_gravity'])
        elif 'com' in data:
            self.cg_local = np.array(data['com'])
        else:
            self.cg_local = np.array([0.0, 0.0, 0.0])

        from geometry_ellipsoid import EllipsoidGeometry
        self.ellipsoids = []
        if 'ellipsoids' in data:
            for ell_data in data['ellipsoids']:
                dims = ell_data['dims']
                local_T = ell_data.get('local_T', np.eye(4))
                name = ell_data.get('name', '')
                force_curve = ell_data.get('force_curve', None)
                unload_curve = ell_data.get('unload_curve', None)
                self.ellipsoids.append(EllipsoidGeometry(dims, local_T, name,
                                                         force_curve, unload_curve))

        self.pos = np.zeros(3)
        self.quat = np.array([1.0, 0.0, 0.0, 0.0])
        self.vel = np.zeros(3)
        self.ang_vel = np.zeros(3)
        self.R = np.eye(3)

        self.force_accum = np.zeros(3)
        self.torque_accum = np.zeros(3)

        self.prev_pos = np.zeros(3)
        self.prev_vel = np.zeros(3)
        self.prev_ang_vel = np.zeros(3)

    def make_positive_definite(self, I, min_eigenvalue=1e-6):
        I_sym = 0.5 * (I + I.T)
        eigenvalues, eigenvectors = np.linalg.eigh(I_sym)
        eigenvalues = np.maximum(eigenvalues, min_eigenvalue)
        return eigenvectors @ np.diag(eigenvalues) @ eigenvectors.T

    def get_body_transform(self):
        T = np.eye(4)
        T[:3, :3] = self.R
        T[:3, 3] = self.pos - self.R @ self.cg_local
        return T

    def set_velocity(self, vel, ang_vel=None):
        self.vel = vel.copy()
        if ang_vel is not None:
            self.ang_vel = ang_vel.copy()
        else:
            self.ang_vel = np.zeros(3)

    def set_state_from_transform(self, T_origin_world):
        self.R = T_origin_world[:3, :3]
        self.R = self.orthonormalize_rotation(self.R)
        self.quat = matrix_to_quat(self.R)
        body_origin_pos = T_origin_world[:3, 3]
        self.pos = body_origin_pos + self.R @ self.cg_local

    def orthonormalize_rotation(self, R):
        x = R[:, 0]
        x = x / (np.linalg.norm(x) + 1e-12)
        y = R[:, 1]
        y = y - np.dot(y, x) * x
        y = y / (np.linalg.norm(y) + 1e-12)
        z = np.cross(x, y)
        return np.column_stack([x, y, z])

    def update_derived_state(self):
        self.R = quat_to_matrix(self.quat)
        self.R = self.orthonormalize_rotation(self.R)
        self.quat = matrix_to_quat(self.R)

    def add_force(self, f, loc_world=None):
        self.force_accum += f
        if loc_world is not None:
            r = loc_world - self.pos
            self.torque_accum += np.cross(r, f)

    def get_world_inertia(self):
        return self.R @ self.I_body @ self.R.T

    def get_inv_world_inertia(self):
        return self.R @ self.inv_I_body @ self.R.T

    def get_velocity_at_point(self, point):
        return self.vel + np.cross(self.ang_vel, point - self.pos)

    def apply_impulse(self, impulse, point):
        if self.inv_mass > 0:
            self.vel += impulse * self.inv_mass
            r = point - self.pos
            self.ang_vel += self.get_inv_world_inertia() @ np.cross(r, impulse)

    def clear_forces(self):
        self.force_accum = np.zeros(3)
        self.torque_accum = np.zeros(3)

    def get_kinetic_energy(self):
        linear_ke = 0.5 * self.mass * np.dot(self.vel, self.vel)
        I_world = self.get_world_inertia()
        rotational_ke = 0.5 * self.ang_vel @ I_world @ self.ang_vel
        return linear_ke + rotational_ke

    def get_momentum(self):
        linear_momentum = self.mass * self.vel
        I_world = self.get_world_inertia()
        angular_momentum = I_world @ self.ang_vel
        return linear_momentum, angular_momentum