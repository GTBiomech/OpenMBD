# physics_joint_restraints.py
# Citation: Tierney. OpenMBD: An Open-Source Multibody Dynamics Simulator for Biomechanics Research and Education. F1000Research, 2026.
# Version: 1.2
# Research Contact: Dr Gregory Tierney (g.tierney@ulster.ac.uk)
"""
User-defined joint restraints: elastic, viscous damping and Coulomb friction.

Joints are free unless a restraint is defined.  All inputs use the same axes
as the joint angle, velocity and torque inputs (right-handed, about the
PARENT body axes, slots ordered [Z, Y, X]):

  elastic   tau_i = -k_i (phi_i - phi0_i)        or  -curve_i(phi_i - phi0_i)
            phi = angles of the child relative to the parent about the
            parent axes, decomposed Y -> X' -> Z'' (R = Ry Rx Rz).  Flexion
            about Y is taken first so it stays unique over +-180 deg (hip and
            knee flexion pass 90 deg); for a rotation about a single axis the
            angles equal the joint inputs.  The angle torques are mapped to a
            physical moment through E^-T, where omega_parent = E(phi) phi_dot.
  damping   T_i = -c_i omega_i                   omega about the parent axes
  friction  T_i = -f_i ramp(|omega_i|) sign(omega_i)
            ramp = 0 below ramp[0], 1 above ramp[1], linear in between.

Restraints are properties of the model and are read from the joint record
in the model file:
  {"name": "kneelJnt", ...,
   "stiffness":     [kZ, kY, kX]        N m/rad
   "neutral_angle": [phiZ, phiY, phiX]  deg
   "damping":       [cZ, cY, cX]        N m s/rad
   "friction":      [fZ, fY, fX]        N m
   "elastic_curve": [[deg, N m], ...]  or {"Z": [...], "Y": [...], "X": [...]}
   "rom_limits":    true                built-in range-of-motion limit on/off}
Scalars apply to all three axes.  An elastic curve replaces the linear
stiffness for its axis; a curve given only for non-negative angles is
mirrored (odd).  The curve gives the moment resisting a deflection, so the
applied torque is -curve(angle - neutral).

The engine applies everything implicitly through an equivalent stiffness K
and damping D (see assemble_A_and_B), so stiff or heavily damped restraints
remain stable at the usual time steps.
"""

import numpy as np

from joint_conventions import (zyx_slots_to_xyz, relative_rotation,
                               revolute_axis_parent, user_frame_rotation)
from physics_utils import quat_to_matrix

KEYS = ('stiffness', 'neutral_angle', 'damping', 'friction',
        'elastic_curve', 'rom_limits')

# Tikhonov regularisation of E^-1 near the X = +-90 deg singularity of the
# Y -> X' -> Z'' angles: exact while |cos X| >= _INV_COS, ramping to
# _INV_LAMBDA^2 at cos X = 0.
_INV_LAMBDA = 0.15
_INV_COS = 0.3


# ----------------------------------------------------------------------
# Input parsing
# ----------------------------------------------------------------------

def vec_xyz(value, clip_negative=True):
    """Scalar or [Z, Y, X] list -> (X, Y, Z) float array."""
    if value is None:
        return np.zeros(3)
    if np.isscalar(value):
        v = np.full(3, float(value))
    else:
        lst = [float(x) for x in list(value)[:3]]
        lst += [0.0] * (3 - len(lst))
        v = zyx_slots_to_xyz(lst)
    return np.maximum(v, 0.0) if clip_negative else v


class ElasticCurve:
    """Odd-extended, linearly extrapolated angle (deg) -> torque (N m) table."""

    def __init__(self, points):
        pts = np.asarray(points, dtype=float).reshape(-1, 2)
        pts = pts[np.argsort(pts[:, 0])]
        if pts[0, 0] >= 0.0:
            neg = -pts[::-1]
            if pts[0, 0] == 0.0:
                neg = neg[:-1]
            pts = np.vstack([neg, pts])
        if len(pts) < 2:
            raise ValueError("elastic_curve needs at least two points")
        self.x = np.radians(pts[:, 0])
        self.y = pts[:, 1]
        self.slopes = np.diff(self.y) / np.maximum(np.diff(self.x), 1e-12)

    def _segment(self, x):
        i = int(np.searchsorted(self.x, x) - 1)
        return min(max(i, 0), len(self.slopes) - 1)

    def __call__(self, x):
        """Restraint magnitude for deflection x (rad)."""
        i = self._segment(x)
        return float(self.y[i] + self.slopes[i] * (x - self.x[i]))

    def slope(self, x):
        return float(self.slopes[self._segment(x)])


def parse_curves(value):
    """elastic_curve entry -> [curve_X, curve_Y, curve_Z] (None where absent)."""
    if value is None:
        return [None, None, None]
    if isinstance(value, dict):
        out = []
        for ax in ('X', 'Y', 'Z'):
            pts = value.get(ax, value.get(ax.lower()))
            out.append(ElasticCurve(pts) if pts is not None else None)
        return out
    c = ElasticCurve(value)
    return [c, c, c]


class JointRestraint:
    """Resolved restraint for one joint; all vectors in (X, Y, Z) order."""

    def __init__(self, K, phi0, C, F, curves, rom_limits):
        self.K = K
        self.phi0 = phi0
        self.C = C
        self.F = F
        self.curves = curves
        self.rom_limits = bool(rom_limits)

    @property
    def has_elastic(self):
        return bool(np.any(self.K > 0.0)) or any(c is not None for c in self.curves)

    @property
    def has_viscous(self):
        return bool(np.any(self.C > 0.0)) or bool(np.any(self.F > 0.0))

    @property
    def active(self):
        return self.has_elastic or self.has_viscous or not self.rom_limits


def resolve(jinfo):
    """Restraint defined on a model-file joint record."""
    def pick(key, default=None):
        val = jinfo.get(key, default)
        return default if val is None else val

    return JointRestraint(
        K=vec_xyz(pick('stiffness')),
        phi0=np.radians(vec_xyz(pick('neutral_angle'), clip_negative=False)),
        C=vec_xyz(pick('damping')),
        F=vec_xyz(pick('friction')),
        curves=parse_curves(pick('elastic_curve')),
        rom_limits=pick('rom_limits', True),
    )


# ----------------------------------------------------------------------
# Kinematics helpers
# ----------------------------------------------------------------------

def restraint_angles(R):
    """
    Angles (X, Y, Z) in radians with R = Ry(phi_Y) Rx(phi_X) Rz(phi_Z):
    phi_Y in (-180, 180], phi_X in [-90, 90], phi_Z in (-180, 180].
    """
    sb = float(np.clip(-R[1, 2], -1.0, 1.0))
    b = np.arcsin(sb)
    if abs(np.cos(b)) > 1e-8:
        a = np.arctan2(R[0, 2], R[2, 2])
        c = np.arctan2(R[1, 0], R[1, 1])
    else:
        a = np.arctan2(-R[2, 0], R[0, 0])
        c = 0.0
    return np.array([b, a, c])


def restraint_E(phi):
    """omega_parent = E(phi) @ d(phi)/dt, phi = (X, Y, Z), R = Ry Rx Rz."""
    b, a, _ = phi
    ca, sa, cb, sb = np.cos(a), np.sin(a), np.cos(b), np.sin(b)
    return np.array([[ca, 0.0, sa * cb],
                     [0.0, 1.0, -sb],
                     [-sa, 0.0, ca * cb]])


def inverse_E(phi):
    """Regularised E^-1 (3x3)."""
    E = restraint_E(phi)
    cb2 = np.cos(phi[0]) ** 2
    lam2 = _INV_LAMBDA ** 2 * max(0.0, 1.0 - cb2 / _INV_COS ** 2)
    return np.linalg.solve(E.T @ E + lam2 * np.eye(3), E.T)


def _wrap(x):
    return (x + np.pi) % (2.0 * np.pi) - np.pi


def ramp(rate_abs, r1, r2):
    if rate_abs <= r1:
        return 0.0
    if rate_abs >= r2 or r2 <= r1:
        return 1.0
    return (rate_abs - r1) / (r2 - r1)


def _elastic_axis(r, i, dphi):
    """(torque, tangent stiffness) on one angle axis, torque = -f(dphi)."""
    c = r.curves[i]
    if c is not None:
        return -c(dphi), max(c.slope(dphi), 0.0)
    return -r.K[i] * dphi, r.K[i]


# ----------------------------------------------------------------------
# Assembly
# ----------------------------------------------------------------------

def assemble(engine, q, qdot, friction_ramp=(0.1, 1.0)):
    """
    Return (Q_e, K, D):
      Q_e : elastic generalised force at q            (nq,)
      K   : tangent elastic stiffness, Q_e(q + dq) ~ Q_e - K dq   (nq, nq)
      D   : viscous + friction equivalent damping, Q = -D qdot     (nq, nq)
    Friction uses the secant coefficient f ramp(|w|)/|w| per axis, which is
    exact at the current rate and, applied implicitly, cannot reverse a joint.
    """
    nq = engine.nq
    Qe = np.zeros(nq)
    K = np.zeros((nq, nq))
    D = np.zeros((nq, nq))
    restraints = getattr(engine, '_joint_restraints', None)
    if not restraints:
        return Qe, K, D
    r1, r2 = friction_ramp

    for key, r in restraints.items():
        if key not in engine.joint_dof_map:
            continue
        s, dof = engine.joint_dof_map[key]
        jinfo = engine._jinfo_by_key.get(key)

        if dof == 4:
            qj = q[s:s + 4]
            nj = np.linalg.norm(qj)
            qj = qj / nj if nj > 1e-12 else np.array([1., 0., 0., 0.])
            R_j = quat_to_matrix(qj)
            R_pc = relative_rotation(jinfo, R_j)

            if r.has_viscous:
                w_p = R_pc @ qdot[s:s + 3]
                c_eff = r.C.copy()
                for i in range(3):
                    if r.F[i] > 0.0:
                        a = abs(w_p[i])
                        if a > 1e-12:
                            c_eff[i] += r.F[i] * ramp(a, r1, r2) / a
                D[s:s + 3, s:s + 3] += R_pc.T @ (c_eff[:, None] * R_pc)

            if r.has_elastic:
                phi = restraint_angles(user_frame_rotation(jinfo, R_j))
                tau = np.zeros(3)
                k_t = np.zeros(3)
                for i in range(3):
                    tau[i], k_t[i] = _elastic_axis(r, i, _wrap(phi[i] - r.phi0[i]))
                Einv = inverse_E(phi)
                T_p = Einv.T @ tau                       # moment, parent axes
                Qe[s:s + 3] += R_pc.T @ T_p
                Kp = Einv.T @ (k_t[:, None] * Einv)      # parent axes
                K[s:s + 3, s:s + 3] += R_pc.T @ Kp @ R_pc

        elif dof == 1:
            ax = revolute_axis_parent(jinfo)
            if r.has_viscous:
                w_p = ax * qdot[s]
                c_eff = r.C.copy()
                for i in range(3):
                    if r.F[i] > 0.0:
                        a = abs(w_p[i])
                        if a > 1e-12:
                            c_eff[i] += r.F[i] * ramp(a, r1, r2) / a
                D[s, s] += float(np.sum(c_eff * ax * ax))
            if r.has_elastic:
                th = q[s]
                for i in range(3):
                    if abs(ax[i]) < 1e-12:
                        continue
                    tau_i, k_i = _elastic_axis(r, i, _wrap(ax[i] * th - r.phi0[i]))
                    Qe[s] += ax[i] * tau_i
                    K[s, s] += k_i * ax[i] * ax[i]

    return Qe, K, D


def load(engine, model_idx, config=None):
    """Populate engine._joint_restraints for one model from its model file."""
    if not hasattr(engine, '_joint_restraints'):
        engine._joint_restraints = {}
    for key in [k for k in engine._joint_restraints if k[0] == model_idx]:
        del engine._joint_restraints[key]
    for (midx, jname, jinfo, dof) in engine.joint_list:
        if midx != model_idx or dof >= 6:
            continue                      # free joint: nothing to restrain
        r = resolve(jinfo)
        if r.active:
            engine._joint_restraints[(midx, jname)] = r
