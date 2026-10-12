# models_parser.py
# Citation: Tierney. OpenMBD: An Open-Source Multibody Dynamics Simulator for Biomechanics Research and Education. F1000Research, 2026.
# Version: 1.2
# Research Contact: Dr Gregory Tierney (g.tierney@ulster.ac.uk)

import json
import numpy as np


def extend_force_curve(curve_array, pen_max_m=None, plateau_fraction=None):
    """
    Validate a force-penetration table [[x, F], ...] and return it as an
    (N, 2) float array sorted by penetration, starting at (0, 0).

    The table is NOT padded.  Beyond its last point the contact model
    extrapolates linearly with the slope of the last segment, so the force
    keeps rising with the terminal stiffness of the characteristic instead
    of saturating.  (pen_max_m and plateau_fraction are accepted for
    backwards compatibility and ignored.)
    """
    curve = np.array(curve_array, dtype=float)
    if curve.ndim != 2 or curve.shape[1] != 2 or len(curve) < 2:
        raise ValueError(f"force curve must be [[x, F], ...] with at least "
                         f"two points, got {curve_array!r}")
    curve = curve[np.argsort(curve[:, 0], kind='stable')]
    if np.any(np.diff(curve[:, 0]) <= 0.0):
        raise ValueError("force curve penetration values must be strictly "
                         "increasing")
    if curve[0, 0] > 0.0:
        curve = np.vstack([[0.0, 0.0], curve])
    elif curve[0, 0] < 0.0:
        raise ValueError("force curve must not contain negative penetration")
    return curve


def get_text_as_array(text):
    """Convert a space-separated string of numbers to a NumPy array."""
    if not text:
        return np.zeros(3)
    try:
        clean = text.replace('|', ' ').replace('\n', ' ').strip()
        return np.array([float(x) for x in clean.split()])
    except (ValueError, AttributeError):
        return np.zeros(3)


def parse_ellipsoid(ell_data, default_name='unnamed'):
    """Common ellipsoid record used by the parser and the visual model."""
    dims = np.array(ell_data['dimensions'], dtype=float)

    local_T = np.array(ell_data.get('local_orientation', np.eye(4)), dtype=float)
    if local_T.shape == (4, 4):
        pass
    elif local_T.shape == (3, 3):
        T = np.identity(4)
        T[:3, :3] = local_T
        local_T = T
    else:
        local_T = np.identity(4)
    local_T[:3, 3] = np.array(ell_data.get('local_position', [0.0, 0.0, 0.0]),
                              dtype=float)

    force_curve = extend_force_curve(
        ell_data.get('force_curve', [[0, 0], [0.01, 10000]]))
    raw_unload = ell_data.get('unload_curve', None)
    unload_curve = (extend_force_curve(raw_unload)
                    if raw_unload is not None else None)

    rec = {
        'dims': dims,
        'local_T': local_T,
        'force_curve': force_curve,
        'unload_curve': unload_curve,
        'name': ell_data.get('name', default_name),
    }
    if 'hysteresis_slope' in ell_data:
        rec['hysteresis_slope'] = float(ell_data['hysteresis_slope'])
    if 'elastic_limit' in ell_data:
        rec['elastic_limit'] = float(ell_data['elastic_limit'])
    return rec


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

            ellipsoids = [parse_ellipsoid(e)
                          for e in body_data.get('ellipsoids', [])]

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
