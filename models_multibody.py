# models_multibody.py
# Citation: Tierney. OpenMBD: An Open-Source Multibody Dynamics Simulator for Biomechanics Research and Education. F1000Research, 2026.
# Version: 1.2
# Research Contact: Dr Gregory Tierney (g.tierney@ulster.ac.uk)

import numpy as np
from joint_conventions import bryant_matrix, joint_rotation_from_input
from models_parser import parse_ellipsoid

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
            # Optional joint restraint, about the parent body axes
            # (see physics_joint_restraints.py).
            #   stiffness N m/rad, neutral_angle deg, damping N m s/rad,
            #   friction N m  -- scalar or [Z, Y, X]
            #   elastic_curve [[deg, N m], ...] or {"Z": ..., "Y": ..., "X": ...}
            #   rom_limits    bool (built-in range-of-motion limit)
            for key in ('stiffness', 'neutral_angle', 'damping', 'friction'):
                if key in joint_data:
                    val = joint_data[key]
                    if isinstance(val, (list, tuple)):
                        joint_info[key] = [float(v) for v in val[:3]]
                    else:
                        joint_info[key] = float(val)
            if 'elastic_curve' in joint_data:
                joint_info['elastic_curve'] = joint_data['elastic_curve']
            if 'rom_limits' in joint_data:
                joint_info['rom_limits'] = bool(joint_data['rom_limits'])

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

        # Bodies that already appear as the CHILD of an explicit joint (including
        # one whose parent is GROUND) must not get a synthetic root joint.
        # children_map only holds joints whose parent is another BODY, so a
        # single-body system with an explicit GROUND joint -- a ball, a wall,
        # an impactor -- used to be given a SECOND root joint, doubling its
        # free DOFs and leaving 7 of them unconstrained and unintegrated.
        explicit_children = {jd['child'] for jd in self.model_data.get('joints', [])}

        for body_name, body in self.bodies.items():
            if (body_name not in children_map
                    and body_name not in explicit_children
                    and not any(body_name == child for children in children_map.values()
                                for child, _ in children)):
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
                # An EXPLICIT "fixed" joint to GROUND is a weld: the body is
                # rigidly attached to inertial space with zero DOFs.  This used
                # to be silently rewritten to 'free', so there was no way to
                # anchor anything -- an "anchor" body would just free-fall,
                # taking whatever was attached to it along.  Synthetic root
                # joints are created as 'free' already, so honouring 'fixed'
                # here does not affect them.
                if joint_info.get('type') is None:
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
                body.ellipsoids.append(parse_ellipsoid(ell_data, default_name=''))

    def get_joint_rotation_matrix(self, angles_deg):
        """
        4x4 rotation for user input angles [phi_Z, phi_Y, phi_X] (degrees) as
        Bryant angles (X, then Y', then Z''; right-handed).  See
        joint_conventions.py.  Used for the root orientation, whose
        parent axes are the global axes.
        """
        R = np.identity(4)
        R[:3, :3] = bryant_matrix(angles_deg)
        return R

    def update_kinematics(self, joint_states, root_pos, root_angles=[0, 0, 0],
                          joint_rotations=None, root_R=None):
        """
        Forward kinematics for the visual/preview model.

        joint_states    : {joint_name: [phi_Z, phi_Y, phi_X] deg} user inputs,
                          about the PARENT body axes.
        root_angles     : root Bryant angles about the global axes.
        joint_rotations : optional {joint_name: 3x3 internal joint rotation};
                          used by the physics engine to push its exact state
                          without an Euler-angle round trip.
        root_R          : optional 3x3 root orientation (overrides root_angles).
        """
        if not self.root_body:
            return

        T_root = np.identity(4)
        T_root[:3, 3] = root_pos
        if root_R is not None:
            T_root[:3, :3] = np.asarray(root_R, dtype=float)
        else:
            T_root[:3, :3] = bryant_matrix(root_angles)

        self._update_body_kinematics(self.root_body, T_root, joint_states,
                                     joint_rotations or {})

    def _update_body_kinematics(self, body, parent_T, joint_states,
                                joint_rotations=None):
        body.global_transform = parent_T
        joint_rotations = joint_rotations or {}

        for child, joint_info in body.children:
            j_name = joint_info['name']
            R_joint = np.identity(4)
            if j_name in joint_rotations:
                R_joint[:3, :3] = joint_rotations[j_name]
            elif joint_info.get('type', 'fixed') != 'fixed':
                angles = joint_states.get(j_name, [0, 0, 0])
                R_joint[:3, :3] = joint_rotation_from_input(joint_info, angles)
            T_child = parent_T @ joint_info['T1'] @ R_joint @ joint_info['T2_inv']
            self._update_body_kinematics(child, T_child, joint_states,
                                         joint_rotations)
