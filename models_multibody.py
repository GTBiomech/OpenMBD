# models_multibody.py
# Citation: Tierney. OpenMBD: An Open-Source Multibody Dynamics Simulator for Biomechanics Research and Education. F1000Research, 2026.
# Version: 1.0 
# Research Contact: Dr Gregory Tierney (g.tierney@ulster.ac.uk)

import numpy as np
from geometry_mesh_generator import rotation_matrix_from_euler_axis

class MultibodyBody:
    def __init__(self, name, body_id):
        self.name = name
        self.body_id = body_id
        self.global_transform = np.identity(4)
        self.ellipsoids = []
        self.children = []
        self.joint_name_to_parent = "None"
        self.joint_info = None

class MultibodyHumanModel:
    def __init__(self, model_file):
        self.load_from_json(model_file)

    def load_from_json(self, json_file):
        import json
        with open(json_file, 'r') as f:
            self.model_data = json.load(f)

        self.bodies = {}
        self.root_body = None
        self.joints_list = []
        self.joint_states = {}
        self.joint_infos = {}

        self._parse_bodies()
        self._parse_hierarchy()
        self._parse_geometry()

    def _parse_bodies(self):
        for body_name, body_data in self.model_data['bodies'].items():
            b = MultibodyBody(body_name, len(self.bodies))
            self.bodies[body_name] = b
            b.original_data = body_data

            if 'pelvis' in body_name.lower() or not self.root_body:
                self.root_body = b

        if not self.root_body and self.bodies:
            self.root_body = list(self.bodies.values())[0]

    def _parse_hierarchy(self):
        if 'joints' not in self.model_data:
            return

        children_map = {}
        joint_info_map = {}

        for joint_data in self.model_data['joints']:
            j_name = joint_data['name']
            parent_name = joint_data['parent']
            child_name = joint_data['child']
            joint_type = joint_data.get('type', 'fixed')

            if child_name not in self.bodies:
                print(f"⚠ Warning: Joint {j_name} references unknown child body {child_name}")
                continue

            child = self.bodies[child_name]

            T1 = np.array(joint_data['T1'])
            T2 = np.array(joint_data['T2'])
            T2_inv = np.array(joint_data['T2_inv'])

            joint_info = {
                'name': j_name,
                'type': joint_type,
                'T1': T1,
                'T2': T2,
                'T2_inv': T2_inv,
                'parent_name': parent_name,
                'child_name': child_name,
                'parent_anchor': T1[:3, 3],
                'child_anchor': T2[:3, 3],
                'child': child
            }

            if parent_name == "GROUND" or parent_name not in self.bodies:
                joint_info['parent'] = None
            else:
                joint_info['parent'] = self.bodies[parent_name]

            self.joint_infos[j_name] = joint_info
            joint_info_map[j_name] = joint_info

            if parent_name in self.bodies:
                if parent_name not in children_map:
                    children_map[parent_name] = []
                children_map[parent_name].append((child_name, j_name))

        for parent_name, children in children_map.items():
            if parent_name in self.bodies:
                parent = self.bodies[parent_name]
                for child_name, j_name in children:
                    if child_name in self.bodies and j_name in joint_info_map:
                        child = self.bodies[child_name]
                        child.joint_name_to_parent = j_name
                        self.joints_list.append(j_name)
                        parent.children.append((child, joint_info_map[j_name]))

        for body_name, body in self.bodies.items():
            if body_name not in children_map and not any(body_name == child for children in children_map.values() for child, _ in children):
                if 'root_joint' not in self.joint_infos:
                    root_joint_info = {
                        'name': 'root_joint',
                        'type': 'free',
                        'T1': np.identity(4),
                        'T2': np.identity(4),
                        'T2_inv': np.identity(4),
                        'parent_name': 'GROUND',
                        'child_name': body_name,
                        'parent_anchor': np.zeros(3),
                        'child_anchor': np.zeros(3),
                        'parent': None,
                        'child': body,
                        'is_root_joint': True
                    }
                    self.joint_infos['root_joint'] = root_joint_info
                    body.root_joint_info = root_joint_info

        ground_joints = []
        for j_name, joint_info in self.joint_infos.items():
            if joint_info['parent_name'] == 'GROUND':
                ground_joints.append(j_name)
                joint_info['is_root_joint'] = True
                if joint_info.get('type', 'fixed') == 'fixed':
                    joint_info['type'] = 'free'

        if ground_joints and 'root_joint' not in self.joint_infos:
            first_ground_joint = ground_joints[0]
            self.joint_infos['root_joint'] = self.joint_infos[first_ground_joint]
            if first_ground_joint not in self.joints_list:
                self.joints_list.append(first_ground_joint)
            self.joint_infos['root_joint']['is_root_joint'] = True

    def _parse_geometry(self):
        for body_name, body_data in self.model_data['bodies'].items():
            if body_name not in self.bodies:
                continue

            body = self.bodies[body_name]

            for ell_data in body_data.get('ellipsoids', []):
                dims = np.array(ell_data['dimensions'])

                local_T = np.array(ell_data['local_orientation'], dtype=float)
                if local_T.shape == (4, 4):
                    pass
                elif local_T.shape == (3, 3):
                    T = np.identity(4)
                    T[:3, :3] = local_T
                    local_T = T
                else:
                    local_T = np.identity(4)

                local_T[:3, 3] = np.array(ell_data['local_position'])

                from models_parser import extend_force_curve
                force_curve = extend_force_curve(
                    np.array(ell_data.get('force_curve', [[0,0],[0.01,10000]])))

                body.ellipsoids.append({
                    'dims': dims,
                    'local_T': local_T,
                    'force_curve': force_curve,
                    'name': ell_data.get('name', '')
                })

    def get_joint_rotation_matrix(self, angles_deg):
        R = np.identity(4)
        if abs(angles_deg[0]) > 0.01:
            R = R @ rotation_matrix_from_euler_axis('Z', angles_deg[0])
        if abs(angles_deg[1]) > 0.01:
            R = R @ rotation_matrix_from_euler_axis('Y', angles_deg[1])
        if abs(angles_deg[2]) > 0.01:
            R = R @ rotation_matrix_from_euler_axis('X', angles_deg[2])
        return R

    def update_kinematics(self, joint_states, root_pos, root_angles=[0, 0, 0]):
        if not self.root_body:
            return

        T_root = np.identity(4)
        T_root[:3, 3] = root_pos

        R_root = self.get_joint_rotation_matrix(root_angles)
        T_root[:3, :3] = R_root[:3, :3]

        self._update_body_kinematics(self.root_body, T_root, joint_states)

    def _update_body_kinematics(self, body, parent_T, joint_states):
        body.global_transform = parent_T

        for child, joint_info in body.children:
            j_name = joint_info['name']
            angles = joint_states.get(j_name, [0, 0, 0])

            R_joint = self.get_joint_rotation_matrix(angles)
            T_child = parent_T @ joint_info['T1'] @ R_joint @ joint_info['T2_inv']
            self._update_body_kinematics(child, T_child, joint_states)