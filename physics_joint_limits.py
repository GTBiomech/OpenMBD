# physics_joint_limits.py
# Citation: Tierney. OpenMBD: An Open-Source Multibody Dynamics Simulator for Biomechanics Research and Education. F1000Research, 2026.
# Version: 1.2 
# Research Contact: Dr Gregory Tierney (g.tierney@ulster.ac.uk)
"""
Physiologically realistic joint range-of-motion (ROM) enforcement.

Each joint has a soft limit implemented as a penalty spring-damper that
activates only when an anatomical joint angle leaves its physiological range.

Anatomical angles
-----------------
For a quaternion spherical joint the relative rotation R_j (the rotation
between the parent-side T1 frame and the child-side T2 frame) is decomposed
as an intrinsic Y-X-Z sequence

    R_j = Ry(a) @ Rx(b) @ Rz(c)

    a = rotation about the joint Y axis  -> sagittal plane  (flexion/extension)
    b = rotation about the joint X axis  -> frontal plane   (ab/adduction,
                                            lateral flexion, inversion/eversion)
    c = rotation about the joint Z axis  -> segment long axis (axial rotation)

This is the ordering of a joint coordinate system (flexion first, axial
rotation last).  Its singularity is at b = +-90 deg, i.e. 90 deg of frontal-
plane motion, which lies outside the range of every joint except the
shoulder.  At small angles a, b, c coincide with the UI's Y, X and Z joint
angle fields.

Sign conventions (bundled human models, verified against the model geometry
in the neutral standing pose; anterior = +x, left = +y, up = +z):
  a > 0 : distal end of a downward segment moves ANTERIOR
          (hip/shoulder/elbow flexion, knee HYPER-extension, ankle
          dorsiflexion); top of an upward segment moves POSTERIOR
          (spine/neck/head extension)
  b > 0 : distal end of a downward segment moves to the model's LEFT
          (left hip/shoulder abduction, right hip/shoulder adduction; left
          ankle eversion, right ankle inversion); top of an upward segment
          moves to the RIGHT (right lateral flexion)
  c > 0 : axial rotation about the segment long axis

Two (a, b, c) triplets describe every rotation.  The one with the lower
penalty energy is used, so large flexion (knee 140 deg, hip 130 deg, shoulder
180 deg) is not misread as a small flexion plus 180 deg of abduction and
axial rotation.

Generalised forces
------------------
The penalty torques tau_anat are conjugate to the angle RATES.  Their
generalised force on the joint's angular-velocity DOFs follows from virtual
power, tau_anat . d(angles)/dt = Q . omega_joint with omega_joint = E @ rates:

    Q_joint = E^{-T} @ tau_anat

which makes the elastic part of the limit exactly conservative.  E^{-T} is
evaluated exactly for |b| < ~78 deg and with a small Tikhonov damping
beyond that, so it stays bounded at b = +-90 deg.

History: the previous version used ZYX Euler angles with E^T instead of
E^{-T}, a ROM table whose Z/X anatomical assignments were swapped relative to
the model's joint frames, reversed knee and ankle sagittal signs (every knee
flexion was resisted from 0 deg: 126 N m at 60 deg), and arcsin-limited
middle angles that turned elbow/hip flexion beyond 90 deg into spurious
240-360 N m torques.

ROM sources:
  American Academy of Orthopaedic Surgeons (AAOS) normative values,
  supplemented with values from Kapandji (1987) and White & Panjabi (1990).

Joint matching
--------------
``_find_rom`` performs a case-insensitive substring search of the joint name
against the ordered keys of ROM_RAD.
"""

import numpy as np

# ---------------------------------------------------------------------------
#  ROM TABLE  (degrees, converted to radians below)
#  'axes': [(anatomical_name, dof_index, min_deg, max_deg), ...]
#     dof_index 0 = a (Y, sagittal), 1 = b (X, frontal), 2 = c (Z, axial)
#     for spherical joints; revolute joints use dof_index 0 only.
#  'stiffness': N m/rad, 'damping': N m s/rad
# ---------------------------------------------------------------------------

_ROM_DEG = {
    # ── Hip joints ────────────────────────────────────────────────────────
    'hipl': {
        'axes': [
            ('flexion(+)/extension(-)',     0,  -30, 130),
            ('abduction(+)/adduction(-)',   1,  -30,  45),
            ('axial rotation',              2,  -50,  50),
        ],
        'stiffness': 60.0, 'damping': 3.0,
    },
    'hipr': {
        'axes': [
            ('flexion(+)/extension(-)',     0,  -30, 130),
            ('adduction(+)/abduction(-)',   1,  -45,  30),
            ('axial rotation',              2,  -50,  50),
        ],
        'stiffness': 60.0, 'damping': 3.0,
    },

    # ── Knee joints (flexion is NEGATIVE: the shank swings posterior) ─────
    'kneel': {
        'axes': [
            ('extension(+)/flexion(-)',     0, -140,   0),
            ('varus/valgus',                1,  -10,  10),
            ('axial rotation',              2,  -10,  10),
        ],
        'stiffness': 120.0, 'damping': 5.0,
    },
    'kneer': {
        'axes': [
            ('extension(+)/flexion(-)',     0, -140,   0),
            ('varus/valgus',                1,  -10,  10),
            ('axial rotation',              2,  -10,  10),
        ],
        'stiffness': 120.0, 'damping': 5.0,
    },

    # ── Ankle joints ──────────────────────────────────────────────────────
    'anklel': {
        'axes': [
            ('dorsiflexion(+)/plantarflexion(-)', 0, -45, 20),
            ('eversion(+)/inversion(-)',          1, -35, 20),
            ('adduction/abduction',               2, -15, 15),
        ],
        'stiffness': 40.0, 'damping': 2.0,
    },
    'ankler': {
        'axes': [
            ('dorsiflexion(+)/plantarflexion(-)', 0, -45, 20),
            ('inversion(+)/eversion(-)',          1, -20, 35),
            ('adduction/abduction',               2, -15, 15),
        ],
        'stiffness': 40.0, 'damping': 2.0,
    },

    # ── Shoulder joints ───────────────────────────────────────────────────
    # NOTE: shoulder abduction passes through b = 90 deg, the singularity
    # of any three-angle decomposition; limits near that pose are approximate.
    'shoulderl': {
        'axes': [
            ('flexion(+)/extension(-)',     0,  -60, 180),
            ('abduction(+)/adduction(-)',   1,  -30, 180),
            ('int/ext rotation',            2,  -90,  90),
        ],
        'stiffness': 25.0, 'damping': 1.5,
    },
    'shoulderr': {
        'axes': [
            ('flexion(+)/extension(-)',     0,  -60, 180),
            ('adduction(+)/abduction(-)',   1, -180,  30),
            ('int/ext rotation',            2,  -90,  90),
        ],
        'stiffness': 25.0, 'damping': 1.5,
    },

    # ── Elbow joints ──────────────────────────────────────────────────────
    'elbow': {
        'axes': [
            ('flexion(+)/extension(-)',     0,    0, 145),
            ('varus/valgus',                1,  -10,  10),
            ('pronation/supination',        2,  -90,  90),
        ],
        'stiffness': 80.0, 'damping': 3.0,
    },

    # ── Wrist joints ──────────────────────────────────────────────────────
    # Symmetric ranges: the bundled hand geometry does not fix whether the
    # palm faces anteriorly or medially, so the radial/ulnar sign is not
    # assigned.
    'wrist': {
        'axes': [
            ('flexion/extension',           0,  -75,  75),
            ('radial/ulnar deviation',      1,  -30,  30),
            ('axial rotation',              2,  -20,  20),
        ],
        'stiffness': 20.0, 'damping': 1.0,
    },

    # ── Lumbar spine (lowerbackJnt) ───────────────────────────────────────
    'lowerback': {
        'axes': [
            ('extension(+)/flexion(-)',     0,  -45,  25),
            ('lateral flex R(+)/L(-)',      1,  -30,  30),
            ('axial rotation',              2,  -40,  40),
        ],
        'stiffness': 35.0, 'damping': 2.0,
    },
    # ── Thoracic spine (upperbackJnt) ─────────────────────────────────────
    'upperbackjnt': {
        'axes': [
            ('extension(+)/flexion(-)',     0,  -30,  20),
            ('lateral flex R(+)/L(-)',      1,  -25,  25),
            ('axial rotation',              2,  -35,  35),
        ],
        'stiffness': 35.0, 'damping': 2.0,
    },

    # ── Neck joints ───────────────────────────────────────────────────────
    'neck': {
        'axes': [
            ('extension(+)/flexion(-)',     0,  -70,  60),
            ('lateral flex R(+)/L(-)',      1,  -45,  45),
            ('axial rotation',              2,  -80,  80),
        ],
        'stiffness': 15.0, 'damping': 1.0,
    },
    'head': {
        'axes': [
            ('extension(+)/flexion(-)',     0,  -30,  30),
            ('lateral flexion',             1,  -20,  20),
            ('axial rotation',              2,  -40,  40),
        ],
        'stiffness': 10.0, 'damping': 0.5,
    },
}

# Convert degree limits to radians once at import time
ROM_RAD = {}
for key, entry in _ROM_DEG.items():
    ROM_RAD[key] = {
        'axes': [(name, dof, np.radians(lo), np.radians(hi))
                 for name, dof, lo, hi in entry['axes']],
        'stiffness': entry['stiffness'],
        'damping':   entry['damping'],
    }

# Tikhonov damping for E^{-T} / E^{-1} near the b = +-90 deg singularity.
# The damping is zero while |cos b| >= _INV_DAMPING_COS (|b| <= 72.5 deg), so
# the mapping is exact there, and ramps smoothly to _INV_DAMPING^2 at cos b = 0.
# With these values the gain of the damped E^{-T} never exceeds its exact
# value at |b| = 72.5 deg (4.7), so torques stay bounded through the shoulder's
# 90 deg abduction pose.  Only the shoulder range reaches |b| > 72.5 deg.
_INV_DAMPING = 0.15
_INV_DAMPING_COS = 0.3


def _find_rom(joint_name: str):
    """Return the ROM entry for a joint by substring match (case-insensitive)."""
    jl = joint_name.lower()
    for key, entry in ROM_RAD.items():
        if key in jl:
            return entry
    return None


def _wrap(x):
    return (x + np.pi) % (2.0 * np.pi) - np.pi


def _yxz_candidates(R):
    """
    Both intrinsic Y-X-Z solutions (a, b, c) of R = Ry(a) Rx(b) Rz(c).
    """
    sb = float(np.clip(-R[1, 2], -1.0, 1.0))
    b = np.arcsin(sb)
    cb = np.cos(b)
    if cb > 1e-9:
        a = np.arctan2(R[0, 2], R[2, 2])
        c = np.arctan2(R[1, 0], R[1, 1])
    else:
        # Gimbal lock: only a -/+ c is defined; put it all in a.
        c = 0.0
        a = np.arctan2(-R[2, 0], R[0, 0]) if sb > 0 else np.arctan2(R[2, 0], R[0, 0])
    first = np.array([a, b, c])
    second = np.array([_wrap(a + np.pi), _wrap(np.pi - b), _wrap(c + np.pi)])
    return first, second


def _E_joint_yxz(angles):
    """
    omega_joint = E @ [a_dot, b_dot, c_dot] for R = Ry(a) Rx(b) Rz(c), with
    omega_joint resolved in the rotated (child-side) joint frame.
    Columns: Y axis after Rx Rz, X axis after Rz, Z axis.
    """
    b, c = angles[1], angles[2]
    cb, sb = np.cos(b), np.sin(b)
    cc, sc = np.cos(c), np.sin(c)
    return np.array([
        [sc * cb,  cc, 0.0],
        [cc * cb, -sc, 0.0],
        [-sb,     0.0, 1.0],
    ])


def _violations(angles, rom):
    """Per-axis signed violation (rad): >0 above hi, <0 below lo, 0 inside."""
    v = np.zeros(3)
    for (_, dof_idx, lo, hi) in rom['axes']:
        if dof_idx >= 3:
            continue
        th = angles[dof_idx]
        if th > hi:
            v[dof_idx] = th - hi
        elif th < lo:
            v[dof_idx] = th - lo
    return v


def anatomical_angles(R_joint, rom):
    """
    Anatomical (a, b, c) angles of a joint rotation, choosing whichever of the
    two Y-X-Z solutions has the lower penalty energy for this ROM entry.
    """
    s1, s2 = _yxz_candidates(R_joint)
    v1, v2 = _violations(s1, rom), _violations(s2, rom)
    return s1 if float(v1 @ v1) <= float(v2 @ v2) else s2


def compute_joint_limit_torques(engine) -> np.ndarray:
    """
    Generalised force vector Q_rom (size nq) of soft penalty torques for every
    joint outside its physiological range.

    Quaternion spherical joints (dof = 4)
        angles   : anatomical (a, b, c) of the joint rotation R_j
        rates    : E^{-1} @ omega_joint,  omega_joint = R_T2^T @ qdot[s:s+3]
        torques  : tau = -k * violation - c * (rate in the violating direction)
        Q        : R_T2 @ E^{-T} @ tau   (virtual-power dual; see module doc)

    Returns Q_rom to be added to the generalised forces in assemble_A_and_B.
    """
    from physics_utils import quat_to_matrix

    q    = engine.state[:engine.nq]
    qdot = engine.state[engine.nq:]
    Q    = np.zeros(engine.nq)

    for (midx, jname, jinfo, dof) in engine.joint_list:
        if jinfo.get('is_root_joint', False):
            continue                    # no ROM on the free-floating root
        if dof == 0:
            continue

        rom = _find_rom(jname)
        if rom is None:
            continue

        s, _ = engine.joint_dof_map[(midx, jname)]
        k    = rom['stiffness']
        c    = rom['damping']

        if dof == 4:
            qjt = q[s:s+4]
            norm = float(np.linalg.norm(qjt))
            if norm < 1e-12:
                continue
            R = quat_to_matrix(qjt / norm)
            angles = anatomical_angles(R, rom)
            viol = _violations(angles, rom)
            if not np.any(viol):
                continue

            # qdot[s:s+3] is resolved in the CHILD body frame; the joint
            # rotation's own rate is omega_j = R_T2^T @ omega_child.
            R2T = np.asarray(jinfo['T2'], dtype=float)[:3, :3].T
            omega_j = R2T @ qdot[s:s+3]
            E = _E_joint_yxz(angles)
            cb2 = np.cos(angles[1]) ** 2
            lam2 = _INV_DAMPING ** 2 * max(0.0, 1.0 - cb2 / _INV_DAMPING_COS ** 2)
            EtE = E.T @ E + lam2 * np.eye(3)
            rates = np.linalg.solve(EtE, E.T @ omega_j)          # ~ E^{-1} omega

            tau = np.zeros(3)
            for i in range(3):
                if viol[i] > 0.0:
                    tau[i] = -k * viol[i] - c * max(0.0, rates[i])
                elif viol[i] < 0.0:
                    tau[i] = -k * viol[i] - c * min(0.0, rates[i])

            Q_joint = E @ np.linalg.solve(EtE, tau)                # ~ E^{-T} tau
            Q[s:s+3] += R2T.T @ Q_joint

        elif dof == 3:
            # --- Legacy Euler spherical joint (not used by bundled models) ---
            for (_, dof_idx, lo, hi) in rom['axes']:
                if dof_idx >= dof:
                    continue
                theta  = q[s + dof_idx]
                thetad = qdot[s + dof_idx]
                if theta > hi:
                    Q[s + dof_idx] -= k * (theta - hi) + c * max(0.0, thetad)
                elif theta < lo:
                    Q[s + dof_idx] += k * (lo - theta) - c * min(0.0, thetad)

        elif dof == 1:
            # --- Revolute joint ---
            for (_, dof_idx, lo, hi) in rom['axes']:
                if dof_idx >= dof:
                    continue
                theta  = q[s + dof_idx]
                thetad = qdot[s + dof_idx]
                if theta > hi:
                    Q[s + dof_idx] -= k * (theta - hi) + c * max(0.0, thetad)
                elif theta < lo:
                    Q[s + dof_idx] += k * (lo - theta) - c * min(0.0, thetad)

    return Q


def get_joint_limit_status(engine) -> list:
    """
    Return a list of (joint_name, axis_label, angle_deg, limit_deg,
    violation_deg) for every joint currently outside its physiological range.
    Useful for debugging and visualisation.
    """
    from physics_utils import quat_to_matrix

    q      = engine.state[:engine.nq]
    status = []

    for (midx, jname, jinfo, dof) in engine.joint_list:
        if jinfo.get('is_root_joint', False) or dof == 0:
            continue
        rom = _find_rom(jname)
        if rom is None:
            continue
        s, _ = engine.joint_dof_map[(midx, jname)]

        if dof == 4:
            qjt = q[s:s+4]
            n = float(np.linalg.norm(qjt))
            if n < 1e-12:
                continue
            angles = anatomical_angles(quat_to_matrix(qjt / n), rom)
        else:
            angles = np.zeros(3)
            angles[:dof] = q[s:s + min(dof, 3)]

        for (label, dof_idx, lo, hi) in rom['axes']:
            if dof_idx >= 3 or (dof == 1 and dof_idx > 0):
                continue
            th = angles[dof_idx]
            if th > hi:
                status.append((jname, label, np.degrees(th), np.degrees(hi),
                               np.degrees(th - hi)))
            elif th < lo:
                status.append((jname, label, np.degrees(th), np.degrees(lo),
                               np.degrees(th - lo)))

    return status
