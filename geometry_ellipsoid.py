# geometry_ellipsoid.py
# Citation: Tierney. OpenMBD: An Open-Source Multibody Dynamics Simulator for Biomechanics Research and Education. F1000Research, 2026.
# Version: 1.1
# Research Contact: Dr Gregory Tierney (g.tierney@ulster.ac.uk)

import numpy as np

class EllipsoidGeometry:
    """
    Contact ellipsoid attached to a rigid body.

    dims             : semi-axes (a, b, c) in metres
    local_T          : 4x4 pose of the ellipsoid in the body frame
    force_curve      : loading characteristic [[x, F], ...]
    unload_curve     : optional unloading characteristic [[x, F], ...]
    hysteresis_slope : optional unloading slope (N/m); default is the
                       steepest slope of the loading curve
    elastic_limit    : optional penetration (m) below which unloading
                       follows the loading curve; default 0
    """

    def __init__(self, dims, local_T, name="", force_curve=None, unload_curve=None,
                 hysteresis_slope=None, elastic_limit=None):
        self.dims = np.array(dims, dtype=float)
        self.local_T = np.array(local_T, dtype=float)
        self.name = name
        self.force_curve = np.array(force_curve, dtype=float) if force_curve is not None else None
        self.unload_curve = np.array(unload_curve, dtype=float) if unload_curve is not None else None
        self.hysteresis_slope = (float(hysteresis_slope)
                                 if hysteresis_slope is not None else None)
        self.elastic_limit = (float(elastic_limit)
                              if elastic_limit is not None else None)

    def get_world_transform(self, body_transform):
        return body_transform @ self.local_T
