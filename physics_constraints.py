# physics_constraints.py  
# Citation: Tierney. OpenMBD: An Open-Source Multibody Dynamics Simulator for Biomechanics Research and Education. F1000Research, 2026.
# Version: 1.0 
# Research Contact: Dr Gregory Tierney (g.tierney@ulster.ac.uk)

import numpy as np


# ──────────────────────────────────────────────────────────────────────────
# Hysteresis state cache  (module-level, survives across PhysicsEngine steps)
# ──────────────────────────────────────────────────────────────────────────
# Key:   (id_bodyA, ell_nameA, id_bodyB, ell_nameB)  – unique contact pair
# Value: dict with keys:
#           'pen_max'   – maximum penetration reached so far (loading peak)
#           'loading'   – True if last step was loading, False if unloading
#           'pen_prev'  – penetration depth at previous step
_contact_state_cache: dict = {}


def clear_contact_cache():
    """Call this when resetting the simulation to initial conditions."""
    global _contact_state_cache
    _contact_state_cache.clear()


def _contact_key(bodyA, ell_nameA, bodyB, ell_nameB):
    """
    Unique, order-independent hashable key for a contact pair.

    Uses string representations so the comparison never tries to order
    int vs NoneType (which raises TypeError in Python 3 for ground contacts
    where bodyB is None).
    """
    idA = str(id(bodyA))
    idB = str(id(bodyB)) if bodyB is not None else 'ground'
    kA = (idA, str(ell_nameA))
    kB = (idB, str(ell_nameB))
    # Canonical order: sort so (A,B) and (B,A) produce the same key.
    if kA <= kB:
        return (kA, kB)
    return (kB, kA)


# ──────────────────────────────────────────────────────────────────────────
# hysteresis 
# ──────────────────────────────────────────────────────────────────────────

def _elastic_force_from_curve(curve, penetration):
    """
    Look up elastic force from a piecewise-linear force-penetration loading
    curve.  Returns 0 for penetration ≤ 0.

    Parameters
    ----------
    curve : np.ndarray, shape (N, 2)
        Columns: [penetration, force].  Must have curve[0,0] == 0.
    penetration : float
        Current contact depth (metres).
    """
    if penetration <= 0.0 or curve is None or len(curve) == 0:
        return 0.0
    return float(np.interp(penetration,
                           curve[:, 0], curve[:, 1],
                           left=0.0, right=float(curve[-1, 1])))


def _combined_curve(curve_a, curve_b):
    """

    When BOTH contacting ellipsoids have force-penetration characteristics the
    combined characteristic is obtained by adding the penetrations of the two
    surfaces at each force level (series compliance):

        x_combined(F)  =  x_A(F)  +  x_B(F)

    Equivalently, for a given total penetration λ the contact force F satisfies:

        λ  =  x_A(F)  +  x_B(F)

    This is solved numerically via root-finding over the force axis.

    The combined curve is returned as a sampled (penetration, force) array
    evaluated on a common force grid, matching the format of the input curves.
    Result is cached by object-id pair to avoid recomputing every timestep.

    Parameters
    ----------
    curve_a, curve_b : np.ndarray shape (N,2)  – (penetration, force) tables,
                       monotonically increasing in both columns.

    Returns
    -------
    np.ndarray shape (M,2) — combined (penetration, force) table, suitable for
    use in _hysteresis_force / _elastic_force_from_curve.
    """
    # Force grid: sample on the union of both curves' force ranges.
    # Use the LOWER of the two maximum forces as the combined ceiling
    # (neither surface can be forced beyond its own characteristic limit).
    f_max = min(float(curve_a[-1, 1]), float(curve_b[-1, 1]))
    if f_max <= 0.0:
        return np.array([[0.0, 0.0], [0.1, 5000.0]])

    # Sample force axis with denser points at low penetration where curves
    # are most nonlinear (log-spaced lower half + linear upper half).
    n = 200
    f_lo = np.logspace(np.log10(max(f_max * 1e-4, 1e-2)), np.log10(f_max * 0.5), n // 2)
    f_hi = np.linspace(f_max * 0.5, f_max, n // 2)
    f_grid = np.unique(np.concatenate([[0.0], f_lo, f_hi]))

    # Invert each curve: pen(F) via linear interpolation on (force, pen) axes.
    # np.interp requires x to be increasing; force axes are monotone increasing.
    pen_a = np.interp(f_grid, curve_a[:, 1], curve_a[:, 0],
                      left=0.0, right=float(curve_a[-1, 0]))
    pen_b = np.interp(f_grid, curve_b[:, 1], curve_b[:, 0],
                      left=0.0, right=float(curve_b[-1, 0]))

    pen_combined = pen_a + pen_b   # Series compliance: λ = λ_A + λ_B

    # Return as (penetration, force) table — drop duplicate penetration values
    combined = np.column_stack([pen_combined, f_grid])
    # Ensure strict monotonicity in penetration (required by np.interp)
    _, unique_idx = np.unique(combined[:, 0], return_index=True)
    return combined[unique_idx]


# Module-level cache: (id(curve_a), id(curve_b)) -> combined curve.
# Valid as long as curve arrays are not mutated (they are read-only numpy arrays
# built once at model load time).
_combined_curve_cache = {}


def _get_combined_curve(curve_a, curve_b):
    """Return cached combined curve for (curve_a, curve_b) pair."""
    key = (id(curve_a), id(curve_b))
    if key not in _combined_curve_cache:
        _combined_curve_cache[key] = _combined_curve(curve_a, curve_b)
    return _combined_curve_cache[key]


def _hysteresis_force(curve, penetration, pen_max, energy_retention=0.25):
    """
    Hysteresis Model:
      • Loading  (pen >= pen_max):  F = f_loading(pen)
      • Unloading (pen < pen_max):  F = η · f_loading(pen)
        where η = energy_retention ∈ [0, 1].
        Default η = 0.25 → 75% energy absorption per cycle,
        consistent with soft human tissue (Yamada 1973, PMHS data).

    Parameters
    ----------
    curve       : np.ndarray (N,2) – may be a plain or combined curve
    penetration : float  – current depth
    pen_max     : float  – historical maximum depth for this contact pair
    """
    F_load = _elastic_force_from_curve(curve, penetration)
    if penetration >= pen_max:          # loading branch
        return F_load
    else:                               # unloading branch
        return energy_retention * F_load


# ──────────────────────────────────────────────────────────────────────────
# Main contact class
# ──────────────────────────────────────────────────────────────────────────

class SimpleContact:
    """
    Ellipsoid-ellipsoid / ellipsoid-plane contact.

    Normal convention
    ----------------------------------------
    ``normal`` always points INTO bodyA (away from the contact surface /
    away from bodyB).  Equivalently, ``dot(vA - vB, normal) > 0`` means
    the bodies are *approaching* (loading); < 0 means separating (unloading).

    This single convention is used consistently by:
      • detect_contacts   – stores ``contact_normal = -geometric_normal``
      • _contact_magnitude – ``v_rel_n = dot(vA-vB, c.normal)``
      • get_applied_force  – repulsive on bodyA along ``+normal``
      • resolve_improved   – same sign for impulse application

    Parameters
    ----------
    bodyA, bodyB        : RigidBody objects (bodyB = None for ground contact)
    point               : np.ndarray (3,) – contact point in world frame
    normal              : np.ndarray (3,) – unit normal pointing INTO bodyA
    penetration         : float – current penetration depth (≥ 0)
    friction            : float – Coulomb friction coefficient μ
    restitution         : float – kept for API compatibility (not used in force path)
    ellipsoidA          : EllipsoidGeometry or None
    ellipsoidB          : EllipsoidGeometry or None
    damping             : float – C_d in N·s/m  (damping coefficient, see §9.1 Eq. 9.2)
    energy_retention    : float – η ∈ [0,1] for hysteresis unloading curve
    v_slip_threshold    : float – velocity (m/s) below which friction ramps to 0
    """

    # Default energy-retention ratio (η).
    # 0.25 → 75 % energy absorbed per cycle.  Validated against:
    # • Yamada (1973) soft tissue pendulum data: ≈ 70–80 % damping
    # • PMHS thorax impact tests (Kroell 1971): ≈ 65–80 %
    
    DEFAULT_ENERGY_RETENTION = 0.25

    # Friction velocity ramp threshold (m/s).
    # Below this the friction force linearly ramps to zero to prevent
    # stick-slip oscillations 
    FRICTION_RAMP_VEL = 0.01   # m/s

    def __init__(self,
                 bodyA,
                 bodyB,
                 point,
                 normal,
                 penetration,
                 friction=0.5,
                 restitution=0.1,
                 ellipsoidA=None,
                 ellipsoidB=None,
                 damping=500.0,
                 energy_retention=None,
                 v_slip_threshold=None):

        self.bodyA        = bodyA
        self.bodyB        = bodyB
        self.point        = np.asarray(point, dtype=float)
        n                 = np.asarray(normal, dtype=float)
        n_mag             = np.linalg.norm(n)
        self.normal       = n / n_mag if n_mag > 1e-12 else np.array([0., 0., 1.])
        self.penetration  = max(0.0, float(penetration))
        self.friction     = float(friction)
        self.restitution  = float(restitution)
        self.ellipsoidA   = ellipsoidA
        self.ellipsoidB   = ellipsoidB
        self.damping      = float(damping)          # C_d  N·s/m
        self.eta          = (energy_retention
                             if energy_retention is not None
                             else self.DEFAULT_ENERGY_RETENTION)
        self.v_slip_thr   = (v_slip_threshold
                             if v_slip_threshold is not None
                             else self.FRICTION_RAMP_VEL)

        # Lever arms from body CGs to contact point
        self.rA = self.point - bodyA.pos
        self.rB = (self.point - bodyB.pos) if bodyB is not None else None

        # Retrieve / create persistent hysteresis state for this pair
        nameA = ellipsoidA.name if ellipsoidA is not None else 'ground_A'
        nameB = ellipsoidB.name if ellipsoidB is not None else 'ground_B'
        self._cache_key = _contact_key(bodyA, nameA, bodyB, nameB)
        state = _contact_state_cache.get(self._cache_key)
        if state is None:
            state = {'pen_max': 0.0, 'loading': True, 'pen_prev': 0.0}
            _contact_state_cache[self._cache_key] = state
        self._state = state

        # Update hysteresis state
        pen_prev = state['pen_prev']
        if self.penetration >= pen_prev:         # loading (deepening)
            state['pen_max'] = max(state['pen_max'], self.penetration)
            state['loading'] = True
        else:                                    # unloading (separating)
            state['loading'] = False
            if self.penetration <= 0.0:
                state['pen_max'] = 0.0           # reset after full separation
        state['pen_prev'] = self.penetration

        # Accumulated impulse storage (kept for compatibility with
        # resolve_improved() if that path is ever called)
        self.accumulated_normal_impulse  = 0.0
        self.accumulated_tangent_impulse = np.zeros(2)

    # ─────────────────────────────────────────────────────────────────
    # Public API: get_contact_force  (used by PhysicsEngine force path)
    # ─────────────────────────────────────────────────────────────────

    def get_contact_force(self, penetration, v_rel_n):
        """
        elastic + damping contact force .

        Sign convention
        ───────────────
        • `normal` points FROM the contact surface INTO bodyA.
        • v_rel = v_A − v_B resolved along that normal.
        • The loading/unloading branch is determined from penetration
          history (`self._state['loading']`) rather than velocity sign.
          This avoids convention mismatches when normal orientation differs
          across contact generators.

        Returns
        -------
        float – total normal contact force magnitude (always ≥ 0).
        """
        if penetration <= 0.0:
            return 0.0

        # ── Pick force-penetration curve ──────────────
        #
        # Contact characteristics are combined when an ellipsoid contacts a plane or another ellipsoid:
        # OpenMBD assigns a force_curve to every ellipsoid, so whenever two
        # ellipsoids interact both curves are available.
        # When only one surface has a curve (e.g. ground contact).

        ca = (self.ellipsoidA.force_curve
              if self.ellipsoidA is not None and self.ellipsoidA.force_curve is not None
              else None)
        cb = (self.ellipsoidB.force_curve
              if self.ellipsoidB is not None and self.ellipsoidB.force_curve is not None
              else None)

        if ca is not None and cb is not None:
            curve = _get_combined_curve(ca, cb)
        elif ca is not None:
            curve = ca   
        elif cb is not None:
            curve = cb   
        else:
            # Fallback: linear spring, k ≈ 50 kN/m (soft tissue)
            curve = np.array([[0.0, 0.0], [0.1, 5000.0]])

        # ── Elastic force with hysteresis ────────────────────────────
        pen_max   = self._state['pen_max']
        F_elastic = _hysteresis_force(curve, penetration, pen_max, self.eta)

        # ── Damping force─────────────────────
        # F_d = C_d · |v_norm|  (always non-negative)
        F_damping = self.damping * abs(v_rel_n)

        # ── Combine loading/unloading rule ─────────────────
        is_loading = bool(self._state.get('loading', True))
        if is_loading:
            F_total = F_elastic + F_damping
        else:
            # Unloading: damping opposes elastic recovery
            F_total = max(0.0, F_elastic - F_damping)

        return max(0.0, F_total)

    # ─────────────────────────────────────────────────────────────────
    # Friction ramp 
    # ─────────────────────────────────────────────────────────────────

    def _friction_ramp(self, v_slip_mag):
        """
        Linear ramp from 0 to 1 over [0, v_slip_threshold].
        Prevents stick-slip vibration at near-zero sliding velocity.
        """
        return min(1.0, v_slip_mag / (self.v_slip_thr + 1e-12))

    # ─────────────────────────────────────────────────────────────────
    # resolve_improved  (impulse-based resolver, kept for compatibility)
    # ─────────────────────────────────────────────────────────────────

    def resolve_improved(self, dt, baumgarte_coef=0.2):
        """
        Sequential impulse contact resolver (Catto 2005 style).
        Used when physics_engine.PhysicsEngine drives the simulation
        (as opposed to PhysicsEngine which uses the force path above).

        Improvements:
          • Correct v_rel_n sign (approaching → positive).
          • loading/unloading damping rule.
          • Friction ramp.
        """
        vA = self.bodyA.get_velocity_at_point(self.point)
        vB = (self.bodyB.get_velocity_at_point(self.point)
              if self.bodyB is not None else np.zeros(3))
        v_rel   = vA - vB
        v_rel_n = np.dot(v_rel, self.normal)

        # Skip if no contact to resolve
        if v_rel_n <= -0.001 and self.penetration < 0.001:
            return

        force_n   = self.get_contact_force(self.penetration, v_rel_n)
        impulse_n = force_n * dt

        # Effective mass along normal
        rA        = self.point - self.bodyA.pos
        rA_x_n    = np.cross(rA, self.normal)
        inv_m_n   = self.bodyA.inv_mass
        if self.bodyA.inv_mass > 0:
            inv_m_n += rA_x_n @ self.bodyA.get_inv_world_inertia() @ rA_x_n
        if self.bodyB is not None:
            rB      = self.point - self.bodyB.pos
            rB_x_n  = np.cross(rB, self.normal)
            inv_m_n += self.bodyB.inv_mass
            if self.bodyB.inv_mass > 0:
                inv_m_n += rB_x_n @ self.bodyB.get_inv_world_inertia() @ rB_x_n

        if inv_m_n < 1e-10:
            return

        # Apply normal impulse
        n_imp = impulse_n * self.normal
        if self.bodyA.inv_mass > 0:
            self.bodyA.vel     += n_imp * self.bodyA.inv_mass
            self.bodyA.ang_vel += (self.bodyA.get_inv_world_inertia()
                                   @ np.cross(rA, n_imp))
        if self.bodyB is not None and self.bodyB.inv_mass > 0:
            self.bodyB.vel     -= n_imp * self.bodyB.inv_mass
            self.bodyB.ang_vel -= (self.bodyB.get_inv_world_inertia()
                                   @ np.cross(rB, n_imp))

        # Friction impulse 
        if self.friction > 0.0 and force_n > 0.0:
            vA2      = self.bodyA.get_velocity_at_point(self.point)
            vB2      = (self.bodyB.get_velocity_at_point(self.point)
                        if self.bodyB is not None else np.zeros(3))
            v_rel2   = vA2 - vB2
            v_rel_n2 = np.dot(v_rel2, self.normal)
            v_tang   = v_rel2 - v_rel_n2 * self.normal
            v_t_mag  = np.linalg.norm(v_tang)
            if v_t_mag > 1e-6:
                ramp       = self._friction_ramp(v_t_mag)
                f_dir      = -(v_tang / v_t_mag)
                f_imp_mag  = min(self.friction * impulse_n * ramp,
                                 v_t_mag / (inv_m_n + 1e-12))
                f_imp      = f_imp_mag * f_dir
                if self.bodyA.inv_mass > 0:
                    self.bodyA.vel     += f_imp * self.bodyA.inv_mass
                    self.bodyA.ang_vel += (self.bodyA.get_inv_world_inertia()
                                           @ np.cross(rA, f_imp))
                if self.bodyB is not None and self.bodyB.inv_mass > 0:
                    self.bodyB.vel     -= f_imp * self.bodyB.inv_mass
                    self.bodyB.ang_vel -= (self.bodyB.get_inv_world_inertia()
                                           @ np.cross(rB, f_imp))

    # ─────────────────────────────────────────────────────────────────
    # Serialisation (for CSV export)
    # ─────────────────────────────────────────────────────────────────

    def to_dict(self):
        vA = self.bodyA.get_velocity_at_point(self.point)
        vB = (self.bodyB.get_velocity_at_point(self.point)
              if self.bodyB is not None else np.zeros(3))
        v_rel_n = np.dot(vA - vB, self.normal)
        F_mag = self.get_contact_force(self.penetration, v_rel_n)
        return {
            'bodyA':           self.bodyA.name,
            'bodyB':           self.bodyB.name if self.bodyB else 'ground',
            'point':           self.point.tolist(),
            'normal':          self.normal.tolist(),
            'penetration':     round(self.penetration, 6),

            'force':           (self.normal * F_mag).tolist(),
            'force_magnitude': round(F_mag, 3),
            'pen_max':         round(self._state['pen_max'], 6),
            'loading':         self._state['loading'],
            'friction':        self.friction,
            'damping':         self.damping,
        }


# ──────────────────────────────────────────────────────────────────────────
# JointConstraint  (unchanged, kept for import compatibility)
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