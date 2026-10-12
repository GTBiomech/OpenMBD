# physics_joint_limits.py
# Citation: Tierney. OpenMBD: An Open-Source Multibody Dynamics Simulator for Biomechanics Research and Education. F1000Research, 2026.
# Version: 1.4
# Research Contact: Dr Gregory Tierney (g.tierney@ulster.ac.uk)
"""
Physiological joint range-of-motion (ROM) restraints.

Each joint has a soft limit implemented as a penalty spring-damper that
activates only when an anatomical joint angle leaves its physiological range.

Anatomical angles
-----------------
Angles are measured about the PARENT body axes, the same axes used for the
joint-angle inputs (see joint_conventions.py): anterior = +x, left = +y,
up = +z, right-handed.  The relative rotation of the child, with the
zero-angle offset removed,

    R_user = R_T1 @ R_j @ R_T1^T

is decomposed as an intrinsic Y-X-Z sequence

    R_user = Ry(a) @ Rx(b) @ Rz(c)

    a = rotation about Y  -> sagittal plane  (flexion/extension)
    b = rotation about X  -> frontal plane   (ab/adduction, lateral flexion,
                                              inversion/eversion)
    c = rotation about Z  -> segment long axis (axial rotation)

so flexion is resolved first and axial rotation last.  The singularity is at
b = +-90 deg, outside the range of every joint except the shoulder.  At small
angles a, b, c equal the Y, X and Z joint-angle inputs.

Sign conventions (right-handed about the parent axes):
  a > 0 : distal end of a downward segment moves POSTERIOR (knee flexion,
          hip/shoulder extension, elbow extension, ankle plantarflexion);
          top of an upward segment moves ANTERIOR (spine/neck/head flexion)
  b > 0 : distal end of a downward segment moves to the model's LEFT
          (left hip/shoulder abduction, right hip/shoulder adduction; left
          ankle eversion, right ankle inversion); top of an upward segment
          moves to the RIGHT (right lateral flexion)
  c > 0 : anticlockwise axial rotation seen from above

Two (a, b, c) triplets describe every rotation.  The one with the lower
penalty energy is used, so large flexion (knee 140 deg, hip 130 deg, shoulder
180 deg) is not misread as a small flexion plus 180 deg of abduction and
axial rotation.

Generalised forces
------------------
The penalty torques tau are conjugate to the angle RATES.  With
omega_u = E @ rates the angular velocity of R_user in its rotated frame,
virtual power gives the generalised force on omega_u as E^{-T} @ tau, which
keeps the elastic part of the limit exactly conservative.  E^{-T} is exact
for |b| < ~72 deg and uses a small Tikhonov damping beyond that, so it stays
bounded at b = +-90 deg.  The engine's spherical-joint velocity is the
relative angular velocity in child-body axes, omega_child, and
omega_u = R_T1 @ R_T2^T @ omega_child.

Restraint law (per anatomical axis, outside the range only):
    tau = -k * violation - c * (rate in the violating direction)

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
    # ── Hip joints (flexion is NEGATIVE: the thigh swings anterior) ──────
    'hipl': {
        'axes': [
            ('extension(+)/flexion(-)',     0, -130,  30),
            ('abduction(+)/adduction(-)',   1,  -30,  45),
            ('axial rotation',              2,  -50,  50),
        ],
        'stiffness': 60.0, 'damping': 3.0,
    },
    'hipr': {
        'axes': [
            ('extension(+)/flexion(-)',     0, -130,  30),
            ('adduction(+)/abduction(-)',   1,  -45,  30),
            ('axial rotation',              2,  -50,  50),
        ],
        'stiffness': 60.0, 'damping': 3.0,
    },

    # ── Knee joints (flexion is POSITIVE: the shank swings posterior) ────
    'kneel': {
        'axes': [
            ('flexion(+)/extension(-)',     0,    0, 140),
            ('varus/valgus',                1,  -10,  10),
            ('axial rotation',              2,  -10,  10),
        ],
        'stiffness': 120.0, 'damping': 5.0,
    },
    'kneer': {
        'axes': [
            ('flexion(+)/extension(-)',     0,    0, 140),
            ('varus/valgus',                1,  -10,  10),
            ('axial rotation',              2,  -10,  10),
        ],
        'stiffness': 120.0, 'damping': 5.0,
    },

    # ── Ankle joints ──────────────────────────────────────────────────────
    'anklel': {
        'axes': [
            ('plantarflexion(+)/dorsiflexion(-)', 0, -20, 45),
            ('eversion(+)/inversion(-)',          1, -35, 20),
            ('adduction/abduction',               2, -15, 15),
        ],
        'stiffness': 40.0, 'damping': 2.0,
    },
    'ankler': {
        'axes': [
            ('plantarflexion(+)/dorsiflexion(-)', 0, -20, 45),
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
            ('extension(+)/flexion(-)',     0, -180,  60),
            ('abduction(+)/adduction(-)',   1,  -30, 180),
            ('int/ext rotation',            2,  -90,  90),
        ],
        'stiffness': 25.0, 'damping': 1.5,
    },
    'shoulderr': {
        'axes': [
            ('extension(+)/flexion(-)',     0, -180,  60),
            ('adduction(+)/abduction(-)',   1, -180,  30),
            ('int/ext rotation',            2,  -90,  90),
        ],
        'stiffness': 25.0, 'damping': 1.5,
    },

    # ── Elbow joints (flexion is NEGATIVE: the forearm swings anterior) ──
    'elbow': {
        'axes': [
            ('extension(+)/flexion(-)',     0, -145,   0),
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
            ('flexion(+)/extension(-)',     0,  -25,  45),
            ('lateral flex R(+)/L(-)',      1,  -30,  30),
            ('axial rotation',              2,  -40,  40),
        ],
        'stiffness': 35.0, 'damping': 2.0,
    },
    # ── Thoracic spine (upperbackJnt) ─────────────────────────────────────
    'upperbackjnt': {
        'axes': [
            ('flexion(+)/extension(-)',     0,  -20,  30),
            ('lateral flex R(+)/L(-)',      1,  -25,  25),
            ('axial rotation',              2,  -35,  35),
        ],
        'stiffness': 35.0, 'damping': 2.0,
    },

    # ── Neck joints ───────────────────────────────────────────────────────
    'neck': {
        'axes': [
            ('flexion(+)/extension(-)',     0,  -60,  70),
            ('lateral flex R(+)/L(-)',      1,  -45,  45),
            ('axial rotation',              2,  -80,  80),
        ],
        'stiffness': 15.0, 'damping': 1.0,
    },
    'head': {
        'axes': [
            ('flexion(+)/extension(-)',     0,  -30,  30),
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
_INV_DAMPING = 0.15
_INV_DAMPING_COS = 0.3


def _rom_enabled(engine, key):
    r = getattr(engine, '_joint_restraints', {}).get(key)
    return True if r is None else bool(getattr(r, 'rom_limits', True))


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
    """Both intrinsic Y-X-Z solutions (a, b, c) of R = Ry(a) Rx(b) Rz(c)."""
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
    omega = E @ [a_dot, b_dot, c_dot] for R = Ry(a) Rx(b) Rz(c), with omega
    resolved in the rotated (child-side) frame.
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


def anatomical_angles(R_user, rom):
    """
    Anatomical (a, b, c) angles of a joint rotation about the parent axes,
    choosing whichever of the two Y-X-Z solutions has the lower penalty
    energy for this ROM entry.
    """
    s1, s2 = _yxz_candidates(R_user)
    v1, v2 = _violations(s1, rom), _violations(s2, rom)
    return s1 if float(v1 @ v1) <= float(v2 @ v2) else s2


def _joint_frames(jinfo):
    R1 = np.asarray(jinfo['T1'], dtype=float)[:3, :3]
    R2 = np.asarray(jinfo['T2'], dtype=float)[:3, :3]
    return R1, R2


def user_rotation(jinfo, R_joint):
    """R_user = R_T1 @ R_j @ R_T1^T (identity at zero input angle)."""
    R1, _ = _joint_frames(jinfo)
    return R1 @ R_joint @ R1.T


def compute_joint_limit_torques(engine) -> np.ndarray:
    """
    Generalised force vector Q_rom (size nq) of soft penalty torques for every
    joint outside its physiological range.

    Quaternion spherical joints (dof = 4)
        angles   : anatomical (a, b, c) of R_user
        rates    : E^{-1} @ omega_u,  omega_u = R_T1 @ R_T2^T @ qdot[s:s+3]
        torques  : tau = -k * violation - c * (rate in the violating direction)
        Q        : R_T2 @ R_T1^T @ E^{-T} @ tau

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
        if not _rom_enabled(engine, (midx, jname)):
            continue                    # switched off by a joint restraint

        s, _ = engine.joint_dof_map[(midx, jname)]
        k    = rom['stiffness']
        c    = rom['damping']

        if dof == 4:
            qjt = q[s:s+4]
            norm = float(np.linalg.norm(qjt))
            if norm < 1e-12:
                continue
            R1, R2 = _joint_frames(jinfo)
            R_user = R1 @ quat_to_matrix(qjt / norm) @ R1.T
            angles = anatomical_angles(R_user, rom)
            viol = _violations(angles, rom)
            if not np.any(viol):
                continue

            M = R1 @ R2.T                 # child-body axes -> rotated user frame
            omega_u = M @ qdot[s:s+3]
            E = _E_joint_yxz(angles)
            cb2 = np.cos(angles[1]) ** 2
            lam2 = _INV_DAMPING ** 2 * max(0.0, 1.0 - cb2 / _INV_DAMPING_COS ** 2)
            EtE = E.T @ E + lam2 * np.eye(3)
            rates = np.linalg.solve(EtE, E.T @ omega_u)          # ~ E^{-1} omega

            tau = np.zeros(3)
            for i in range(3):
                if viol[i] > 0.0:
                    tau[i] = -k * viol[i] - c * max(0.0, rates[i])
                elif viol[i] < 0.0:
                    tau[i] = -k * viol[i] - c * min(0.0, rates[i])

            Q_u = E @ np.linalg.solve(EtE, tau)                    # ~ E^{-T} tau
            Q[s:s+3] += M.T @ Q_u

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
            # --- Revolute joint: angle about the hinge axis ---
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
    Angles are about the parent-body axes.
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
        if not _rom_enabled(engine, (midx, jname)):
            continue                    # switched off by a joint restraint
        s, _ = engine.joint_dof_map[(midx, jname)]

        if dof == 4:
            qjt = q[s:s+4]
            n = float(np.linalg.norm(qjt))
            if n < 1e-12:
                continue
            angles = anatomical_angles(
                user_rotation(jinfo, quat_to_matrix(qjt / n)), rom)
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
