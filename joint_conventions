# joint_conventions.py
# Citation: Tierney. OpenMBD: An Open-Source Multibody Dynamics Simulator for Biomechanics Research and Education. F1000Research, 2026.
# Version: 1.1
# Research Contact: Dr Gregory Tierney (g.tierney@ulster.ac.uk)
"""
User-input convention for joint angles, angular velocities and torques.

  * Rotations are RIGHT-HANDED: a positive angle, angular velocity or torque
    about an axis is anticlockwise when viewed looking back along that axis
    from its positive end.

  * Joint angles are Bryant (cardan) angles: phi_X about the X axis of the
    PARENT, then phi_Y about the resulting (floating) Y' axis, then phi_Z
    about the resulting Z'' axis of the CHILD:
        R_child_rel_parent = Rx(phi_X) @ Ry(phi_Y) @ Rz(phi_Z).
    For a single-axis input (e.g. a knee angle) this is simply a rotation
    about that axis of the parent.

  * The joint coordinate system is parallel to the parent body's local
    coordinate system and coincides with the child's at zero angle.  Model
    files may carry a rotated internal joint frame in T1/T2 (the bundled
    human models use diag(1,-1,-1), which flips Y and Z); that internal frame
    is NOT the axis set the user's numbers refer to.  Inputs are always
    resolved about the PARENT BODY axes and converted to the internal joint
    frame here.

  * Joint angular velocity and joint torque are components, about the
    parent-body axes, of the child's angular velocity relative to the parent
    and of the torque acting on the child.  For a free root joint the parent
    is the global reference frame.

  * Revolute joints rotate about the hinge axis stored in the model; the
    scalar input is the right-handed rotation about that axis expressed in
    the parent body.  The X/Y/Z fields are projected onto the axis, so
    entering +30 in the field of the axis the hinge lies along gives +30 deg
    about the parent's +axis regardless of how T1 orients the joint frame.

Storage order
-------------
Config files and the UI keep the three-slot order [Z, Y, X]
(``ModelConfig.joints``, ``joint_vels``, ``joint_torques``).
"""

import numpy as np


def zyx_slots_to_xyz(v):
    """[vZ, vY, vX] (UI/config slot order) -> np.array([vX, vY, vZ])."""
    v = [float(x) for x in list(v)[:3]] if v is not None else []
    v = v + [0.0] * (3 - len(v))
    return np.array([v[2], v[1], v[0]], dtype=float)


def _rx(a):
    c, s = np.cos(a), np.sin(a)
    return np.array([[1.0, 0.0, 0.0], [0.0, c, -s], [0.0, s, c]])


def _ry(a):
    c, s = np.cos(a), np.sin(a)
    return np.array([[c, 0.0, s], [0.0, 1.0, 0.0], [-s, 0.0, c]])


def _rz(a):
    c, s = np.cos(a), np.sin(a)
    return np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]])


def bryant_matrix(angles_zyx_deg):
    """
    Rotation of the child relative to the parent for Bryant angles supplied
    in slot order [phi_Z, phi_Y, phi_X] (degrees):
        R = Rx(phi_X) @ Ry(phi_Y) @ Rz(phi_Z)
    """
    ax, ay, az = np.radians(zyx_slots_to_xyz(angles_zyx_deg))
    return _rx(ax) @ _ry(ay) @ _rz(az)


def matrix_to_bryant_zyx_deg(R):
    """Inverse of bryant_matrix; returns [phi_Z, phi_Y, phi_X] in degrees."""
    R = np.asarray(R, dtype=float)
    sy = float(np.clip(R[0, 2], -1.0, 1.0))
    ay = np.arcsin(sy)
    if abs(np.cos(ay)) > 1e-8:
        ax = np.arctan2(-R[1, 2], R[2, 2])
        az = np.arctan2(-R[0, 1], R[0, 0])
    else:
        ax = np.arctan2(R[2, 1], R[1, 1])
        az = 0.0
    return list(np.degrees([az, ay, ax]))


def _rot(T):
    return np.asarray(T, dtype=float)[:3, :3]


def revolute_axis_parent(jinfo):
    """Unit hinge axis of a revolute joint in PARENT body (or global) axes."""
    ax = _rot(jinfo['T1'])[:, 2].copy()
    n = np.linalg.norm(ax)
    return ax / n if n > 1e-12 else np.array([0.0, 0.0, 1.0])


def revolute_scalar_from_input(jinfo, vec_zyx):
    """Project a [Z, Y, X] input (angle, rate or torque) onto the hinge axis."""
    return float(revolute_axis_parent(jinfo) @ zyx_slots_to_xyz(vec_zyx))


def joint_rotation_from_input(jinfo, angles_zyx_deg):
    """
    Internal joint rotation R_j (the matrix between T1 and T2_inv in
    T_child = T_parent @ T1 @ R_j @ T2_inv) for user input angles.

    Spherical/other:  R_parent->child = R_in @ R_T1 @ R_T2^T, with R_in the
    Bryant matrix about the parent axes, so  R_j = R_T1^T @ R_in @ R_T1.
    Revolute: R_j = Rz(theta) in the joint frame, theta = projected input.
    """
    if jinfo.get('type') == 'revolute':
        theta = np.radians(revolute_scalar_from_input(jinfo, angles_zyx_deg))
        return _rz(theta)
    R1 = _rot(jinfo['T1'])
    return R1.T @ bryant_matrix(angles_zyx_deg) @ R1


def relative_rotation(jinfo, R_j):
    """Orientation of the child body relative to the parent body."""
    return _rot(jinfo['T1']) @ np.asarray(R_j, dtype=float) @ _rot(jinfo['T2']).T


def user_frame_rotation(jinfo, R_j):
    """
    Joint rotation expressed about the parent-body axes with the zero-angle
    offset removed:  R_user = R_T1 @ R_j @ R_T1^T  (identity at zero input).
    """
    R1 = _rot(jinfo['T1'])
    return R1 @ np.asarray(R_j, dtype=float) @ R1.T


def parent_vector_to_child(jinfo, R_j, vec_zyx):
    """
    Convert a [Z, Y, X] vector given about the parent axes (angular velocity
    or torque) into child-body components, which is how the engine stores the
    spherical-joint generalised velocity:  v_child = R_pc^T @ v_parent.
    """
    return relative_rotation(jinfo, R_j).T @ zyx_slots_to_xyz(vec_zyx)
