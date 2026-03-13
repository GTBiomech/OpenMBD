# physics_joint_limits.py
# Citation: Tierney. OpenMBD: An Open-Source Multibody Dynamics Simulator for Biomechanics Research and Education. F1000Research, 2026.
# Version: 1.1
# Research Contact: Dr Gregory Tierney (g.tierney@ulster.ac.uk)
"""
Physiologically realistic joint range-of-motion (ROM) enforcement.

Each joint has a soft limit implemented as a penalty spring-damper that
activates only when the joint angle exceeds its physiological range.  This is
numerically cleaner than hard constraints and avoids impulse discontinuities
during impact.

Angles are stored in radians in the engine state, so all limits below are
converted to radians on module load.

Convention for ZYX Euler angles [α, β, γ] (radians):
  α = rotation about Z (yaw / abduction-adduction depending on joint)
  β = rotation about Y (pitch / flexion-extension depending on joint)
  γ = rotation about X (roll / internal-external rotation depending on joint)

The mapping between Euler DOF and anatomical motion is joint-specific and
defined in the ROM table.  Limits are given as [min_deg, max_deg] per DOF.

Joint matching
--------------
``_find_rom`` performs a case-insensitive substring search of the joint name
against the ordered keys of ROM_RAD.  More-specific keys are listed before
shorter ones that could otherwise match multiple joints.

The table covers every *spherical* and *revolute* joint in openmbd_male.json:
  lowerbackJnt, upperbackJnt, neckJnt, headJnt,
  shoulderlJnt, shoulderrJnt, elbowlJnt, elbowrJnt,
  wristlJnt, wristrJnt, hiplJnt, hiprJnt,
  kneelJnt, kneerJnt, anklelJnt, anklerJnt

ROM sources:
  American Academy of Orthopaedic Surgeons (AAOS) normative values,
  supplemented with values from Kapandji (1987) and White & Panjabi (1990).
"""

import numpy as np

# ---------------------------------------------------------------------------
#  ROM TABLE  (degrees, later converted to radians)
#  Key: joint name substring used for matching (case-insensitive)
#  Value: {'axes': [('anatomical_name', dof_index, min_deg, max_deg), ...],
#           'stiffness': Nm/rad, 'damping': Nm·s/rad}
#
#  dof_index: 0=α(Z), 1=β(Y), 2=γ(X)  for spherical / free joints
#             0 only                    for revolute joints
# ---------------------------------------------------------------------------

_ROM_DEG = {
    # ── Hip joints ────────────────────────────────────────────────────────
    'hipl': {
        'axes': [
            ('abduction(+)/adduction(-)',  0, -30,  45),   # Z: abduction positive
            ('flexion(+)/extension(-)',    1, -30, 130),    # Y: flexion positive
            ('internal(+)/external(-) rot',2, -50,  50),   # X: internal rotation +
        ],
        'stiffness': 60.0, 'damping': 3.0,
    },
    'hipr': {
        'axes': [
            ('abduction(-)/adduction(+)',  0, -45,  30),
            ('flexion(+)/extension(-)',    1, -30, 130),
            ('internal(+)/external(-) rot',2, -50,  50),
        ],
        'stiffness': 60.0, 'damping': 3.0,
    },

    # ── Knee joints (primarily sagittal — flexion is positive Y) ──────────
    'kneel': {
        'axes': [
            ('valgus(+)/varus(-)',         0, -10,  10),   # small out-of-plane
            ('flexion(+)/extension(-)',    1,   0, 140),
            ('axial rotation',             2, -10,  10),
        ],
        'stiffness': 120.0, 'damping': 5.0,
    },
    'kneer': {
        'axes': [
            ('valgus(-)/varus(+)',         0, -10,  10),
            ('flexion(+)/extension(-)',    1,   0, 140),
            ('axial rotation',             2, -10,  10),
        ],
        'stiffness': 120.0, 'damping': 5.0,
    },

    # ── Ankle joints ──────────────────────────────────────────────────────
    'anklel': {
        'axes': [
            ('inversion(+)/eversion(-)',   0, -20,  35),
            ('plantarflex(+)/dorsiflex(-)',1, -20,  45),
            ('adduction/abduction',        2, -15,  15),
        ],
        'stiffness': 40.0, 'damping': 2.0,
    },
    'ankler': {
        'axes': [
            ('inversion(-)/eversion(+)',   0, -35,  20),
            ('plantarflex(+)/dorsiflex(-)',1, -20,  45),
            ('adduction/abduction',        2, -15,  15),
        ],
        'stiffness': 40.0, 'damping': 2.0,
    },

    # ── Shoulder joints ───────────────────────────────────────────────────
    'shoulderl': {
        'axes': [
            ('horizontal abd(+)/add(-)',   0, -130, 30),
            ('elevation(+)/depression(-)', 1,  -60, 180),
            ('int(+)/ext(-) rotation',     2,  -90,  90),
        ],
        'stiffness': 25.0, 'damping': 1.5,
    },
    'shoulderr': {
        'axes': [
            ('horizontal abd(-)/add(+)',   0,  -30, 130),
            ('elevation(+)/depression(-)', 1,  -60, 180),
            ('int(+)/ext(-) rotation',     2,  -90,  90),
        ],
        'stiffness': 25.0, 'damping': 1.5,
    },

    # ── Elbow joints (mainly sagittal) ────────────────────────────────────
    'elbow': {
        'axes': [
            ('valgus/varus',               0, -10,  10),
            ('flexion(+)/extension(-)',    1,   0, 145),
            ('pronation(+)/supination(-)', 2,  -90,  90),
        ],
        'stiffness': 80.0, 'damping': 3.0,
    },

    # ── Wrist joints ──────────────────────────────────────────────────────
    'wrist': {
        'axes': [
            ('radial(+)/ulnar(-) deviation',0, -30,  20),
            ('extension(+)/flexion(-)',     1,  -75,  75),
            ('pronation/supination',        2,  -20,  20),
        ],
        'stiffness': 20.0, 'damping': 1.0,
    },

    # ── Lumbar spine (lowerbackJnt) ───────────────────────────────────────
    'lowerback': {
        'axes': [
            ('lateral flex R(+)/L(-)',     0, -30,  30),
            ('extension(+)/flexion(-)',    1,  -45,  25),
            ('axial rotation',             2,  -40,  40),
        ],
        'stiffness': 35.0, 'damping': 2.0,
    },
    # ── Thoracic spine (upperbackJnt) ─────────────────────────────────────
    'upperbackjnt': {
        'axes': [
            ('lateral flex R(+)/L(-)',     0, -25,  25),
            ('extension(+)/flexion(-)',    1,  -30,  20),
            ('axial rotation',             2,  -35,  35),
        ],
        'stiffness': 35.0, 'damping': 2.0,
    },

    # ── Neck joints ───────────────────────────────────────────────────────
    'neck': {
        'axes': [
            ('lateral flex R(+)/L(-)',     0, -45,  45),
            ('extension(+)/flexion(-)',    1,  -70,  60),
            ('axial rotation',             2,  -80,  80),
        ],
        'stiffness': 15.0, 'damping': 1.0,
    },
    'head': {
        'axes': [
            ('lateral flex',               0, -20,  20),
            ('extension(+)/flexion(-)',    1,  -30,  30),
            ('axial rotation',             2,  -40,  40),
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


def _find_rom(joint_name: str):
    """Return the ROM entry for a joint by substring match (case-insensitive)."""
    jl = joint_name.lower()
    for key, entry in ROM_RAD.items():
        if key in jl:
            return entry
    return None


def compute_joint_limit_torques(engine) -> np.ndarray:
    """
    Compute a generalised force vector Q_rom (size nq) that applies soft
    penalty torques whenever a joint angle exceeds its physiological range.

    The penalty is a spring-damper activated only in the violation zone:

        τ_penalty(θ) = −k · max(0, θ − θ_max) − k · min(0, θ − θ_min)
                     − c · θ̇ · [1 if in violation else 0]

    This is continuous at the boundary (force = 0 exactly at limit) and
    increases linearly into the violation zone, preventing numerical
    instability from step-function switching.

    Returns Q_rom: generalised torques to be added to Q in assemble_A_and_B.
    """
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

        for (_, dof_idx, lo, hi) in rom['axes']:
            if dof_idx >= dof:
                continue
            theta  = q[s + dof_idx]
            thetad = qdot[s + dof_idx]

            if theta > hi:
                violation = theta - hi
                Q[s + dof_idx] -= k * violation + c * max(0.0, thetad)
            elif theta < lo:
                violation = lo - theta
                Q[s + dof_idx] += k * violation - c * min(0.0, thetad)

    return Q


def get_joint_limit_status(engine) -> list:
    """
    Return a list of (joint_name, dof_label, angle_deg, limit_deg, violation_deg)
    for every joint currently outside its physiological range.
    Useful for debugging and visualisation.
    """
    q       = engine.state[:engine.nq]
    status  = []

    for (midx, jname, jinfo, dof) in engine.joint_list:
        if jinfo.get('is_root_joint', False):
            continue
        rom = _find_rom(jname)
        if rom is None:
            continue
        s, _ = engine.joint_dof_map[(midx, jname)]

        for (label, dof_idx, lo, hi) in rom['axes']:
            if dof_idx >= dof:
                continue
            theta_deg = np.degrees(q[s + dof_idx])
            lo_deg    = np.degrees(lo)
            hi_deg    = np.degrees(hi)
            if theta_deg > hi_deg:
                status.append((jname, label, theta_deg, hi_deg, theta_deg - hi_deg))
            elif theta_deg < lo_deg:
                status.append((jname, label, theta_deg, lo_deg, theta_deg - lo_deg))

    return status