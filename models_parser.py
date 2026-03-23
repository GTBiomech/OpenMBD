# models_parser.py
# Citation: Tierney. OpenMBD: An Open-Source Multibody Dynamics Simulator for Biomechanics Research and Education. F1000Research, 2026.
# Version: 1.0 
# Research Contact: Dr Gregory Tierney (g.tierney@ulster.ac.uk)

import json
import numpy as np

def extend_force_curve(curve_array, pen_max_m=0.15, plateau_fraction=0.85):
    """
    Parameters
    ----------
    curve_array : np.ndarray (N, 2)  – original [pen, force] table
    pen_max_m   : float – maximum penetration to extend to (metres)
    plateau_fraction : float – where the plateau starts, as a fraction of pen_max_m

    Returns
    -------
    np.ndarray (M, 2) with M >= N.
    """
    curve = np.array(curve_array, dtype=float)
    last_pen   = float(curve[-1, 0])
    last_force = float(curve[-1, 1])

    if last_pen >= pen_max_m:
        return curve   # already covers the full range

    plat_pen = pen_max_m * plateau_fraction
    # Only add plateau start point if it's beyond the current end
    extra = []
    if last_pen < plat_pen:
        extra.append([plat_pen, last_force])
    extra.append([pen_max_m, last_force])

    return np.vstack([curve, extra])

def get_text_as_array(text):
    """Convert a space-separated string of numbers to a NumPy array."""
    if not text:
        return np.zeros(3)
    try:
        clean = text.replace('|', ' ').replace('\n', ' ').strip()
        return np.array([float(x) for x in clean.split()])
    except:
        return np.zeros(3)

class JSONModelParser:
    def __init__(self, json_path):
        with open(json_path, 'r') as f:
            self.model_data = json.load(f)

        self.bodies = {}
        self.joints = []

        self._parse_bodies()
        self._parse_joints()

    def _parse_bodies(self):
        for body_name, body_data in self.model_data['bodies'].items():
            inertia = np.array(body_data['inertia'])
            if len(inertia) < 6:
                inertia = np.array([0.01, 0.01, 0.005, 0, 0, 0])

            ellipsoids = []
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

                force_curve = extend_force_curve(
                    np.array(ell_data.get('force_curve', [[0, 0], [0.01, 10000]])))

                raw_unload = ell_data.get('unload_curve', None)
                unload_curve = (extend_force_curve(np.array(raw_unload))
                                if raw_unload is not None else None)

                ellipsoids.append({
                    'dims': dims,
                    'local_T': local_T,
                    'force_curve': force_curve,
                    'unload_curve': unload_curve,
                    'name': ell_data.get('name', 'unnamed')
                })

            self.bodies[body_name] = {
                'mass': body_data['mass'],
                'inertia': inertia,
                'cg': np.array(body_data['center_of_mass']),
                'ellipsoids': ellipsoids
            }

    def _parse_joints(self):
        for joint_data in self.model_data.get('joints', []):
            T1 = np.array(joint_data['T1'])
            T2 = np.array(joint_data['T2'])
            T2_inv = np.array(joint_data['T2_inv'])

            self.joints.append({
                'name': joint_data['name'],
                'type': joint_data.get('type', 'fixed'),
                'parent': joint_data['parent'],
                'child': joint_data['child'],
                'T1': T1,
                'T2': T2,
                'T2_inv': T2_inv
            })