# physics_constraints.py
# Citation: Tierney. OpenMBD: An Open-Source Multibody Dynamics Simulator for Biomechanics Research and Education. F1000Research, 2026.
# Version: 2.0
# Research Contact: Dr Gregory Tierney (g.tierney@ulster.ac.uk)
"""
Elastic ellipsoid contact model.

Force law
---------
The contact force is a point force made of elastic, damping and friction
parts, applied at a single point P:

    F_e  = gamma(|lambda_dot|) * f_e(lambda)            elastic (+ hysteresis)
    F_d  = C_d * lambda_dot                             damping
    F_n  = max(0, F_e + F_d)                            normal, resistive only
    F_f  = mu * ramp(|v_t|) * F_n                       friction, opposes v_t

lambda is the penetration (minimum distance between parallel tangent planes),
lambda_dot its rate, gamma the dynamic amplification factor and v_t the
tangential relative velocity at P.  Damping adds to the elastic force while
the penetration increases and subtracts from it while it decreases; a contact
never pulls.

Force-penetration characteristics
---------------------------------
Every ellipsoid carries a loading curve f_l(x) (and optionally an unloading
curve f_u(x)), tabulated as [[x, F], ...] with f_l(0) = 0.  Between table
points the curve is linear; beyond the last point it is extrapolated
linearly with the slope of the last segment.

For contact between two ellipsoids the combined characteristic is obtained
by adding the penetrations of the two surfaces at equal force (the surfaces
act in series).  That requires both loading curves to be strictly
increasing.  If one surface has a non-monotonic curve (crush or fracture,
e.g. a bonnet or windscreen) only that surface's characteristic is used and
the other surface is treated as rigid; if both are non-monotonic the one
with the lower peak force is used.  For ellipsoid-ground contact the ground
is rigid and the ellipsoid characteristic is used.

Point of application
--------------------
P1 and P2 are the deepest points of the two surfaces along the contact
normal (P2 lies on the ground plane for ground contact).  With penetrations
x1 and x2 of the two surfaces at the current force,

    P = P1 + x1 / (x1 + x2) * (P2 - P1)

so P lies on the rigid surface when only one characteristic is used.

Hysteresis
----------
When an unloading curve is given, energy is dissipated as follows.  Loading
follows f_l up to the largest penetration reached so far, x_max.  Unloading
from x_max follows a straight line of slope s (the hysteresis slope) until it
meets f_u and then continues along f_u.  Reloading retraces f_u and then the
slope line back to (x_max, f_l(x_max)), after which loading continues along
f_l.  In closed form, for x < x_max:

    f(x) = min( f_l(x), max( f_u(x), f_l(x_max) - s * (x_max - x) ) )

The history of each contact pair is kept for the whole simulation, so a
surface that has been loaded and unloaded responds along its unloading
branch on the next contact.  Hysteresis is active only once x_max exceeds the
elastic limit.  Per-ellipsoid keys in the model file:

    "unload_curve"      : [[x, F], ...]  (omit for a purely elastic surface)
    "hysteresis_slope"  : N/m (default: steepest slope of the loading curve)
    "elastic_limit"     : m   (default: 0)
"""

import warnings

import numpy as np


# ──────────────────────────────────────────────────────────────────────────
# Piecewise-linear characteristics
# ──────────────────────────────────────────────────────────────────────────

_DEFAULT_CURVE = np.array([[0.0, 0.0], [0.1, 5000.0]])


def _as_curve(curve):
    if curve is None:
        return None
    c = np.asarray(curve, dtype=float)
    if c.ndim != 2 or c.shape[1] != 2 or len(c) < 2:
        return None
    return c


def interp_extrap(x, curve):
    """Linear interpolation of a [[x, y], ...] table with linear extrapolation."""
    xs, ys = curve[:, 0], curve[:, 1]
    if x >= xs[-1]:
        dx = xs[-1] - xs[-2]
        slope = (ys[-1] - ys[-2]) / dx if dx > 0.0 else 0.0
        return float(ys[-1] + slope * (x - xs[-1]))
    if x <= xs[0]:
        dx = xs[1] - xs[0]
        slope = (ys[1] - ys[0]) / dx if dx > 0.0 else 0.0
        return float(ys[0] + slope * (x - xs[0]))
    return float(np.interp(x, xs, ys))


def inverse_extrap(y, curve):
    """Inverse of a strictly increasing table: x such that curve(x) = y."""
    return interp_extrap(y, curve[:, ::-1])


def is_strictly_increasing(curve):
    return (curve is not None
            and bool(np.all(np.diff(curve[:, 0]) > 0.0))
            and bool(np.all(np.diff(curve[:, 1]) > 0.0)))


def max_slope(curve):
    dx = np.diff(curve[:, 0])
    dy = np.diff(curve[:, 1])
    ok = dx > 0.0
    if not np.any(ok):
        return 0.0
    return float(np.max(dy[ok] / dx[ok]))


def peak_force(curve):
    return float(np.max(curve[:, 1]))


def series_curve(curve_a, curve_b):
    """
    Combined characteristic of two strictly increasing curves acting in
    series: x(F) = x_a(F) + x_b(F).  Exact for piecewise-linear tables
    (breakpoints at the union of both force breakpoints), including the
    linear extrapolation beyond the last point.
    """
    f = np.unique(np.concatenate([curve_a[:, 1], curve_b[:, 1]]))
    f = f[f >= 0.0]
    if f[0] > 0.0:
        f = np.concatenate([[0.0], f])
    f = np.concatenate([f, [2.0 * f[-1] + 1.0]])
    x = np.array([max(0.0, inverse_extrap(fi, curve_a))
                  + max(0.0, inverse_extrap(fi, curve_b)) for fi in f])
    x[0] = 0.0
    return np.column_stack([x, f])


# ──────────────────────────────────────────────────────────────────────────
# Contact characteristic of an interacting pair
# ──────────────────────────────────────────────────────────────────────────

_warned_nonmonotonic = set()


def _surface_props(ell, energy_retention):
    """(loading, unloading, slope, elastic_limit) of one ellipsoid."""
    load = _as_curve(getattr(ell, 'force_curve', None)) if ell is not None else None
    if load is None:
        load = _DEFAULT_CURVE
    unload = _as_curve(getattr(ell, 'unload_curve', None)) if ell is not None else None
    if unload is None and energy_retention is not None and is_strictly_increasing(load):
        unload = load * np.array([1.0, float(energy_retention)])
    slope = getattr(ell, 'hysteresis_slope', None) if ell is not None else None
    xe = getattr(ell, 'elastic_limit', None) if ell is not None else None
    return load, unload, slope, (float(xe) if xe is not None else 0.0)


class ContactCharacteristic:
    """
    Resolved force-penetration behaviour of one interacting pair.

    weight(F) returns x_A / (x_A + x_B), the fraction of P1 -> P2 at which the
    force acts (0 = on surface A, 1 = on surface B).
    """

    def __init__(self, load, unload, slope, elastic_limit, comp_a, comp_b):
        self.load = load
        self.unload = unload
        self.slope = float(slope) if slope is not None else max_slope(load)
        self.elastic_limit = float(elastic_limit)
        self._comp_a = comp_a       # loading curve of A, or None if A rigid
        self._comp_b = comp_b       # loading curve of B, or None if B rigid

    def weight(self, force):
        if self._comp_a is None:
            return 0.0
        if self._comp_b is None:
            return 1.0
        if force <= 0.0:
            # Ratio of initial compliances.
            ca = self._comp_a[1, 0] / max(self._comp_a[1, 1], 1e-12)
            cb = self._comp_b[1, 0] / max(self._comp_b[1, 1], 1e-12)
        else:
            ca = max(0.0, inverse_extrap(force, self._comp_a))
            cb = max(0.0, inverse_extrap(force, self._comp_b))
        tot = ca + cb
        return ca / tot if tot > 1e-15 else 0.5

    def elastic_force(self, x, state):
        """
        Elastic contact force at penetration x, updating the hysteresis
        history in `state` ({'x_max': float}).  Must be called once per step.
        """
        if x <= 0.0:
            return 0.0
        f_load = max(0.0, interp_extrap(x, self.load))
        if self.unload is None:
            return f_load
        x_max = state.get('x_max', 0.0)
        if x >= x_max:
            state['x_max'] = x
            return f_load
        if x_max <= self.elastic_limit:
            return f_load
        f_max = max(0.0, interp_extrap(x_max, self.load))
        f_unl = max(0.0, interp_extrap(x, self.unload))
        f_slope = f_max - self.slope * (x_max - x)
        return max(0.0, min(f_load, max(f_unl, f_slope)))


_characteristic_cache = {}


def resolve_characteristic(ell_a, ell_b, energy_retention=None):
    """
    ContactCharacteristic for ellipsoid A against ellipsoid B (B = None for
    the ground plane).  Cached per surface pair; the cache holds references
    to both surfaces so object ids cannot be recycled while an entry exists.
    """
    key = (id(ell_a), id(ell_b), energy_retention)
    hit = _characteristic_cache.get(key)
    if hit is not None and hit[0] is ell_a and hit[1] is ell_b:
        return hit[2]

    la, ua, sa, xa = _surface_props(ell_a, energy_retention)
    if ell_b is None:
        char = ContactCharacteristic(la, ua, sa, xa, la, None)
    else:
        lb, ub, sb, xb = _surface_props(ell_b, energy_retention)
        inc_a, inc_b = is_strictly_increasing(la), is_strictly_increasing(lb)
        if inc_a and inc_b:
            load = series_curve(la, lb)
            unload = None
            if ua is not None or ub is not None:
                cu_a = ua if ua is not None else la
                cu_b = ub if ub is not None else lb
                if is_strictly_increasing(cu_a) and is_strictly_increasing(cu_b):
                    unload = series_curve(cu_a, cu_b)
                else:
                    unload = cu_a if ua is not None else cu_b
            slope = (1.0 / (1.0 / sa + 1.0 / sb)
                     if (sa is not None and sb is not None and sa > 0 and sb > 0)
                     else None)
            char = ContactCharacteristic(load, unload, slope, xa + xb, la, lb)
        else:
            if inc_a and not inc_b:
                use_a = False
            elif inc_b and not inc_a:
                use_a = True
            else:
                use_a = peak_force(la) <= peak_force(lb)
            tag = (getattr(ell_a, 'name', ''), getattr(ell_b, 'name', ''))
            if tag not in _warned_nonmonotonic:
                _warned_nonmonotonic.add(tag)
                used = tag[0] if use_a else tag[1]
                warnings.warn(
                    f"contact {tag[0]} / {tag[1]}: a loading curve is not "
                    f"strictly increasing, so the characteristics cannot be "
                    f"combined; using '{used}' only and treating the other "
                    f"surface as rigid", RuntimeWarning)
            if use_a:
                char = ContactCharacteristic(la, ua, sa, xa, la, None)
            else:
                char = ContactCharacteristic(lb, ub, sb, xb, None, lb)

    _characteristic_cache[key] = (ell_a, ell_b, char)
    return char


def clear_characteristic_cache():
    _characteristic_cache.clear()


# ──────────────────────────────────────────────────────────────────────────
# Contact-pair history
# ──────────────────────────────────────────────────────────────────────────

def contact_key(bodyA, ell_nameA, bodyB, ell_nameB):
    """Order-independent hashable key for a contact pair (bodyB may be None)."""
    kA = (str(id(bodyA)), str(ell_nameA))
    kB = ((str(id(bodyB)) if bodyB is not None else 'ground'), str(ell_nameB))
    return (kA, kB) if kA <= kB else (kB, kA)


# Kept for callers that import the former names.
_contact_key = contact_key
_default_state_cache = {}


def clear_contact_cache():
    _default_state_cache.clear()


def prune_contact_cache(active_keys):
    """Retained for compatibility; contact history is kept for the whole run."""
    return None


# ──────────────────────────────────────────────────────────────────────────
# Contact
# ──────────────────────────────────────────────────────────────────────────

class SimpleContact:
    """
    One active ellipsoid-ellipsoid or ellipsoid-ground contact.

    ``normal`` points INTO bodyA (from bodyB towards bodyA), so the contact
    force on bodyA is +F_n * normal and the penetration rate is
    lambda_dot = -(v_A - v_B) . normal.

    Parameters
    ----------
    bodyA, bodyB      : RigidBody (bodyB = None for the ground)
    point_a, point_b  : deepest points P1 (on A) and P2 (on B / the ground)
    normal            : unit normal pointing into bodyA
    penetration       : lambda >= 0
    friction          : Coulomb coefficient mu
    ellipsoidA/B      : EllipsoidGeometry (B = None for the ground)
    damping           : C_d (N s/m)
    energy_retention  : optional unloading/loading ratio used for surfaces
                        without an unloading curve (None = elastic)
    state_cache       : dict holding the per-pair hysteresis history
    v_slip_threshold  : friction ramp width (m/s)
    """

    FRICTION_RAMP_VEL = 0.01   # m/s

    def __init__(self, bodyA, bodyB, point_a, point_b, normal, penetration,
                 friction=0.5, ellipsoidA=None, ellipsoidB=None,
                 damping=0.0, energy_retention=None, state_cache=None,
                 v_slip_threshold=None, restitution=None):
        self.bodyA = bodyA
        self.bodyB = bodyB
        n = np.asarray(normal, dtype=float)
        n_mag = np.linalg.norm(n)
        self.normal = n / n_mag if n_mag > 1e-12 else np.array([0.0, 0.0, 1.0])
        self.penetration = max(0.0, float(penetration))
        self.friction = float(friction)
        self.ellipsoidA = ellipsoidA
        self.ellipsoidB = ellipsoidB
        self.damping = float(damping)
        self.eta = energy_retention
        self.v_slip_thr = (v_slip_threshold if v_slip_threshold is not None
                           else self.FRICTION_RAMP_VEL)
        self.point_a = np.asarray(point_a, dtype=float)
        self.point_b = np.asarray(point_b, dtype=float)

        nameA = ellipsoidA.name if ellipsoidA is not None else 'ground_A'
        nameB = ellipsoidB.name if ellipsoidB is not None else 'ground_B'
        self._cache_key = contact_key(bodyA, nameA, bodyB, nameB)
        cache = state_cache if state_cache is not None else _default_state_cache
        state = cache.get(self._cache_key)
        if state is None:
            state = {'x_max': 0.0}
            cache[self._cache_key] = state
        self._state = state

        self.characteristic = resolve_characteristic(ellipsoidA, ellipsoidB,
                                                     energy_retention)
        # Static elastic force (before dynamic amplification); updates the
        # hysteresis history exactly once per step.
        self.F_elastic = self.characteristic.elastic_force(self.penetration, state)
        state['loading'] = self.penetration >= state['x_max']

        w = self.characteristic.weight(self.F_elastic)
        self.point = self.point_a + w * (self.point_b - self.point_a)
        self.rA = self.point - bodyA.pos
        self.rB = (self.point - bodyB.pos) if bodyB is not None else None

        self._global_scale = 1.0
        self._magnitude_cache = None

    def get_contact_force(self, penetration, v_rel_n):
        """Normal force magnitude without dynamic amplification."""
        if penetration <= 0.0:
            return 0.0
        return max(0.0, self.F_elastic + self.damping * (-v_rel_n))

    def _friction_ramp(self, v_slip_mag):
        """Linear ramp 0 -> 1 over [0, v_slip_threshold]."""
        return min(1.0, v_slip_mag / (self.v_slip_thr + 1e-12))

    def to_dict(self):
        if self._magnitude_cache is not None:
            F_mag = self._magnitude_cache[0] * self._global_scale
        else:
            vA = self.bodyA.get_velocity_at_point(self.point)
            vB = (self.bodyB.get_velocity_at_point(self.point)
                  if self.bodyB is not None else np.zeros(3))
            F_mag = self.get_contact_force(self.penetration,
                                           float(np.dot(vA - vB, self.normal)))
        return {
            'bodyA':           self.bodyA.name,
            'bodyB':           self.bodyB.name if self.bodyB else 'ground',
            'ellipsoidA':      self.ellipsoidA.name if self.ellipsoidA is not None else '',
            'ellipsoidB':      self.ellipsoidB.name if self.ellipsoidB is not None else 'ground',
            'point':           self.point.tolist(),
            'normal':          self.normal.tolist(),
            'penetration':     round(self.penetration, 6),
            'force':           (self.normal * F_mag).tolist(),
            'force_magnitude': round(F_mag, 3),
            'elastic_force':   round(self.F_elastic, 3),
            'pen_max':         round(self._state['x_max'], 6),
            'loading':         self._state.get('loading', True),
            'friction':        self.friction,
            'damping':         self.damping,
        }


# ──────────────────────────────────────────────────────────────────────────
# Kept for import compatibility (not used by the engine)
# ──────────────────────────────────────────────────────────────────────────

class JointConstraint:
    def __init__(self, parent_body, child_body,
                 parent_anchor, child_anchor, joint_type='fixed'):
        self.parent_body          = parent_body
        self.child_body           = child_body
        self.parent_anchor_local  = parent_anchor
        self.child_anchor_local   = child_anchor
        self.joint_type           = joint_type
        self.parent_anchor_world  = np.zeros(3)
        self.child_anchor_world   = np.zeros(3)
        self.accumulated_impulse  = np.zeros(3)

    def update_anchors(self):
        if self.parent_body.inv_mass > 0:
            self.parent_anchor_world = (self.parent_body.pos
                                        + self.parent_body.R
                                        @ self.parent_anchor_local)
        else:
            self.parent_anchor_world = self.parent_anchor_local.copy()
        if self.child_body.inv_mass > 0:
            self.child_anchor_world = (self.child_body.pos
                                       + self.child_body.R
                                       @ self.child_anchor_local)
        else:
            self.child_anchor_world = self.child_anchor_local.copy()
