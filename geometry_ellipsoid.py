# geometry_ellipsoid.py
# Citation: Tierney. OpenMBD: An Open-Source Multibody Dynamics Simulator for Biomechanics Research and Education. F1000Research, 2026.
# Version: 1.0 
# Research Contact: Dr Gregory Tierney (g.tierney@ulster.ac.uk)

import numpy as np

class EllipsoidGeometry:
    def __init__(self, dims, local_T, name="", force_curve=None):
        self.dims = np.array(dims)
        self.local_T = np.array(local_T)
        self.name = name
        self.force_curve = np.array(force_curve) if force_curve is not None else None

    def get_world_transform(self, body_transform):
        return body_transform @ self.local_T