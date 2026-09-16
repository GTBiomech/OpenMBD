# physics_engine.py  
# Citation: Tierney. OpenMBD: An Open-Source Multibody Dynamics Simulator for Biomechanics Research and Education. F1000Research, 2026.
# Version: 1.6 
# Research Contact: Dr Gregory Tierney (g.tierney@ulster.ac.uk)

import numpy as np
from physics_constraints import SimpleContact, clear_contact_cache, prune_contact_cache
from physics_joint_limits import compute_joint_limit_torques as _jl_compute_joint_limit_torques
from physics_utils import quat_to_matrix, matrix_to_quat, skew


class PhysicsEngine:
    """
    Multibody dynamics engine based on the Principle of Virtual Power.
    Uses analytic ZYX-Euler Jacobians and Recursive Newton-Euler for bias.
    """

    def __init__(self):
        self.models = []
        self.model_configs = []
        self.bodies = []
        self.joint_list = []        # (model_idx, j_name, j_info, dof)
        self.nq = 0
        self.state = None           # [q ; qdot]

        self.parent_idx = []
        self.joint_type = []
        self.joint_axis_local = []
        self.joint_T1 = []
        self.joint_T2 = []
        self.joint_start_idx = []
        self.joint_dof = []

        self.gravity = np.array([0.0, 0.0, -9.81])

        # ── Timestep ──────────────────────────────────────────────────

        self.dt = 0.00001                             
        self.record_every = 10   # record one state per this many integration steps

        self.time = 0.0

        # ── Friction ──────────────────────────────────────────────────
 
        self.friction_coef = 0.4                     

        # ── Contact penetration slop ──────────────────────────────────

        self.contact_penetration_slop = 0.001        

        # ── Contact damping ───────────────────────────────────────────

        self.contact_damping = 150.0

        # ── Prescribed joint torques ──────────────────────────────────
        # Structure: list of dicts, one per torque entry:
        #   { 'model_idx': int, 'joint_name': str,
        #     'torque': np.ndarray (dof,),   # N·m, ZYX order
        #     't_start': float,              # s
        #     'duration': float }            # s (0 = single step)
        self.prescribed_torques = []

        # ── Contact evaluation mode ────────────────────
        # When one ellipsoid simultaneously contacts multiple surfaces the
        # total stiffness is artificially multiplied by the contact count.
        # 
        #
        #   'continuous' (default) – scale every simultaneous contact force
        #       by  Fe_max / ΣFe  so the total elastic force never exceeds
        #       the single largest interaction.  Recommended for body-body
        #       contacts where ellipsoids overlap by design.
        #
        #   'discrete' – apply only the contact with the largest elastic
        #       force; all others are suppressed.  Recommended where a single
        #       well-defined contact surface exists (e.g. leg vs. bonnet).
        #
        #   'none' – legacy behaviour: all contacts fire independently
        #       (physically correct only when ellipsoids cannot overlap).
        #
        # The grouping unit is per-body-pair: contacts between the same
        # (bodyA, bodyB) pair are evaluated together; contacts between
        # different body pairs are always independent
        self.contact_evaluation_mode = 'continuous'

        # ── Hysteresis energy retention  ──
        self.contact_energy_retention = 0.25

        # ── Dynamic Amplification Factor ──

        self.dyn_amp_C1 = 1.0   # static multiplier  (1.0 = no amplification)
        self.dyn_amp_C2 = 0.0   # velocity coefficient 
        self.dyn_amp_C3 = 1.0   # reference velocity (m/s)
        self.dyn_amp_C4 = 1.0   # exponent
        # DAF parameters: γ = C1 + C2·(|ε̇|/C3)^C4
        # Restored from 0.0.  C2=0 disabled rate stiffening
        # To prevent DAF-induced instability the factor is evaluated at the
        # PREVIOUS step's approach velocity (see _contact_magnitude); this
        # gives an explicit (non-feedback) evaluation that is unconditionally
        # stable for the Symplectic-Euler integrator used here.

        self.state_history = []
        self.contact_history = []
        self.recording = True
        self.step_count = 0
        self.contacts = []
        self.joint_constraints = []   # alias kept for export_csv compatibility


        self.enable_self_contact = False

    # ------------------------------------------------------------------
    # ZYX Euler helpers  (verified against numerical FD)
    # ------------------------------------------------------------------

    @staticmethod
    def _E_body_zyx(angles):
        """
        Angular velocity Jacobian in the child body LOCAL frame for ZYX Euler
        angles [alpha(Z), beta(Y), gamma(X)] stored in radians.

        omega_local = E_body @ [alpha_dot, beta_dot, gamma_dot]

        Derivation: E_body = R_rel^T @ E_world where
          E_world = [[0, -sa, ca*cb], [0, ca, sa*cb], [1, 0, -sb]].
        The result is independent of alpha (cancels in R^T @ E_world).
        """
        beta, gamma = angles[1], angles[2]
        cb, sb = np.cos(beta),  np.sin(beta)
        cg, sg = np.cos(gamma), np.sin(gamma)
        return np.array([
            [-sb,    0.0, 1.0],
            [cb*sg,  cg,  0.0],
            [cb*cg, -sg,  0.0]
        ])

    @staticmethod
    def _bias_local_zyx(angles, qdot_joint):
        """
        Analytic angular acceleration bias in child body LOCAL frame for a
        ZYX spherical joint.  Equals E_body_dot @ qdot_joint (qddot = 0).
        Reference: Wittenberg p112-114.
        """
        beta, gamma = angles[1], angles[2]
        ad, bd, gd = qdot_joint[0], qdot_joint[1], qdot_joint[2]
        cb, sb = np.cos(beta),  np.sin(beta)
        cg, sg = np.cos(gamma), np.sin(gamma)
        bx = -cb * bd * ad
        by = (-sb*sg*bd + cb*cg*gd)*ad  +  (-sg*gd)*bd
        bz = (-sb*cg*bd - cb*sg*gd)*ad  +  (-cg*gd)*bd
        return np.array([bx, by, bz])

    @staticmethod
    def _E_world_zyx(angles):
        """
        Angular velocity Jacobian in WORLD frame for ZYX Euler angles.
        omega_world = E_world @ [alpha_dot, beta_dot, gamma_dot]

        """
        alpha, beta = angles[0], angles[1]
        # Clamp β away from ±π/2 singularity (same limit as initialisation)
        _BETA_LIM = np.radians(85.0)
        beta = float(np.clip(beta, -_BETA_LIM, _BETA_LIM))
        ca, sa = np.cos(alpha), np.sin(alpha)
        cb, sb = np.cos(beta),  np.sin(beta)
        return np.array([
            [0.0, -sa,  ca*cb],
            [0.0,  ca,  sa*cb],
            [1.0,  0.0, -sb  ]
        ])

    # ------------------------------------------------------------------
    # Physiological joint limits  (penalty-based viscoelastic stops)
    # ------------------------------------------------------------------
    # ROM values, per-joint stiffness, and damping are defined in
    # physics_joint_limits.py (AAOS + Kapandji sources).  That module is
    # the single source of truth; there is no duplicate table here.
    # ------------------------------------------------------------------

    # Contact-normal refinement (see _ellipsoid_pair_contact).
    CONTACT_NORMAL_ITERS = 12
    CONTACT_NORMAL_TOL   = 1e-7   # m, tangential gradient norm

    JOINT_PASSIVE_D = 40.0   # N*m*s / rad – within-ROM passive damping

    # Pulse width substituted when a prescribed torque is given duration=0.
    # Keeps the delivered angular impulse independent of the timestep.
    MIN_TORQUE_PULSE = 1e-3   # s

    @staticmethod
    def _joint_R2T(jinfo):
        """
        Transpose of the rotation block of a joint's T2 (child-side joint
        frame).  Maps a vector resolved in the CHILD body frame into the
        rotating joint frame in which the joint quaternion is defined:
            omega_joint = R_T2^T @ omega_child.
        Cached on the joint-info dict.
        """
        R2T = jinfo.get('_R2T')
        if R2T is None:
            R2T = np.asarray(jinfo['T2'], dtype=float)[:3, :3].T.copy()
            jinfo['_R2T'] = R2T
        return R2T

    def _compute_passive_damping_torques(self, qdot: np.ndarray) -> np.ndarray:
        """
        Return generalised force vector Q_passive (size nq) that applies
        JOINT_PASSIVE_D viscous damping to every non-root joint DOF.

        Q_passive[s:s+dof] = −JOINT_PASSIVE_D · qdot[s:s+dof]

        This is equivalent to adding a physical dashpot across each joint
        that resists relative angular (or linear) velocity.  Root DOFs are
        excluded so translational free-fall and whole-body orientation
        remain governed solely by gravity and contact forces.

        NOTE: this explicit form is retained only for diagnostics and
        backwards compatibility.  assemble_A_and_B applies the same dashpot
        IMPLICITLY via (A + dt·D)·qddot = B − D·qdot, which is stable for any
        D and dt.  The explicit form is NOT:  qdot += dt·(A^{-1}·B) requires
        |1 − D·dt/A_ii| < 1, i.e. D < 2·A_ii/dt.  The binding DOFs are the
        wrist roll axes (A_ii ≈ 8.45e-4 kg·m²), not the elbow — at
        dt = 1e-4 s that gives D_max ≈ 16.9 N·m·s/rad, so the current
        JOINT_PASSIVE_D = 40.0 diverges by a factor of ~2.4.
        """
        return -self._passive_damping_diag() * qdot

    def _passive_damping_diag(self) -> np.ndarray:
        """
        Return the (nq,) diagonal of the passive joint-damping matrix D.

        Root DOFs are zero (free-floating root takes no passive drag) and, for
        quaternion spherical joints (dof=4), only the 3 omega DOFs are damped —
        never the quaternion-norm constraint slot at index s+3.

        Cached: depends only on topology, which is rebuilt by add_model().
        """
        cached = getattr(self, '_D_diag_cache', None)
        if cached is not None and cached.shape[0] == self.nq:
            return cached
        D = np.zeros(self.nq)
        for (midx, jname, jinfo, dof) in self.joint_list:
            if jinfo.get('is_root_joint', False):
                continue
            s, _ = self.joint_dof_map[(midx, jname)]
            active_dof = 3 if dof == 4 else dof
            D[s:s + active_dof] = self.JOINT_PASSIVE_D
        self._D_diag_cache = D
        return D

    def compute_joint_limit_torques(self, q, qdot):
        """Delegate to physics_joint_limits — single source of truth for ROM."""
        return _jl_compute_joint_limit_torques(self)

    # ------------------------------------------------------------------
    # Model management
    # ------------------------------------------------------------------

    def add_model(self, model, rigid_bodies, config):
        model_idx = len(self.models)
        self.models.append(model)
        self.model_configs.append(config)

        for body in rigid_bodies:
            body.model_idx = model_idx
            self.bodies.append(body)

        # Deduplicate by object identity (root_joint aliases rootJoint).
        seen_ids = set()
        for j_name, j_info in model.joint_infos.items():
            if id(j_info) in seen_ids:
                continue
            seen_ids.add(id(j_info))

            jtype = j_info.get('type', 'fixed')
            if j_info.get('is_root_joint', False) and jtype == 'spherical':
                jtype = 'free'
                j_info['type'] = 'free'

            if jtype == 'spherical':
                # Store as unit quaternion [qw,qx,qy,qz] — no Euler singularity.
                # dof=4: qdot[s:s+3]=omega_local (3 true DOFs), qdot[s+3]=0 (norm slot).
                self.joint_list.append((model_idx, j_name, j_info, 4))
            elif jtype == 'revolute':
                self.joint_list.append((model_idx, j_name, j_info, 1))
            elif jtype == 'free':
                # Root DOF: 3 translation + 4 quaternion (no Euler singularity)
                self.joint_list.append((model_idx, j_name, j_info, 7))
            elif jtype == 'fixed' and j_info.get('is_root_joint', False):
                # Weld to inertial space: zero DOFs, constant world pose.
                self.joint_list.append((model_idx, j_name, j_info, 0))

        new_nq = sum(d for (_, _, _, d) in self.joint_list)

        old_nq = self.nq
        new_state = np.zeros(2 * new_nq)
        if old_nq > 0:
            new_state[:old_nq]              = self.state[:old_nq]
            new_state[new_nq:new_nq+old_nq] = self.state[old_nq:2*old_nq]
        self.nq    = new_nq
        self.state = new_state

        self.joint_dof_map = {}
        start = 0
        for (midx, jname, jinfo, dof) in self.joint_list:
            self.joint_dof_map[(midx, jname)] = (start, dof)
            start += dof

        self._build_kinematic_tree()
        # Invalidate adjacency exclusion cache — topology changed
        if hasattr(self, '_adj_exclusions'):
            del self._adj_exclusions
        self._D_diag_cache = None   # nq changed — rebuild damping diagonal
        self._initialize_state_from_config(model_idx, config)

        # Snapshot AFTER config so reset_to_initial() restores correctly.
        self.initial_state = self.state.copy()

        # Record initial penetration offsets for same-model ellipsoid pairs
        # and ground contacts so phantom forces are suppressed from t=0.
        # Must come AFTER initial_state snapshot and kinematic init.
        self._record_initial_overlaps()

    def reset_to_initial(self):
        if not hasattr(self, 'initial_state'):
            return
        self.state       = self.initial_state.copy()
        self.time        = 0.0
        self.step_count  = 0
        self.contacts    = []
        self.state_history   = []
        self.contact_history = []
        # clear hysteresis cache so pen_max resets to 0
        clear_contact_cache()
        # Rebuild adjacency exclusion set in case topology changed
        if hasattr(self, '_adj_exclusions'):
            del self._adj_exclusions
        self.update_kinematics_from_q(self.state[:self.nq])
        self._update_body_velocities_from_qdot(self.state[:self.nq],
                                               self.state[self.nq:])
        # Re-record initial overlaps for the fresh starting pose
        self._record_initial_overlaps()

    def _initialize_state_from_config(self, model_idx, config):
        """
        Write initial q (radians) and qdot (m/s, rad/s) from ModelConfig.
        Joint angles in config are stored in degrees (UI convention).

        """
        for (midx, jname, jinfo, dof) in self.joint_list:
            if midx != model_idx:
                continue
            s, _ = self.joint_dof_map[(midx, jname)]

            if jinfo.get('is_root_joint', False):
                pos = np.array([float(x) for x in config.pos_str.split()])
                if dof == 7:
                    # q[s:s+3]  = translation (world frame)
                    # q[s+3:s+7] = quaternion [qw, qx, qy, qz] (unit, no singularity)
                    self.state[s:s+3] = pos
                    ang_deg = (config.joints.get('root_joint')
                               or config.joints.get(jname, [0, 0, 0]))
                    ang_rad = np.radians(ang_deg[:3])
                    # Build rotation matrix from ZYX Euler, convert to quaternion.
                    # This accepts any angles without clamping or singularity.
                    a, b, g = ang_rad
                    ca, sa = np.cos(a), np.sin(a)
                    cb, sb = np.cos(b), np.sin(b)
                    cg, sg = np.cos(g), np.sin(g)
                    R = np.array([
                        [ca*cb, ca*sb*sg - sa*cg, ca*sb*cg + sa*sg],
                        [sa*cb, sa*sb*sg + ca*cg, sa*sb*cg - ca*sg],
                        [-sb,   cb*sg,             cb*cg           ]
                    ])
                    self.state[s+3:s+7] = matrix_to_quat(R)   # [qw,qx,qy,qz]
                    try:
                        vel = np.array([float(x) for x in config.vel_str.split()])
                    except Exception:
                        vel = np.zeros(3)
                    self.state[self.nq+s : self.nq+s+3] = vel
                    # Root angular velocity stored as world-frame omega (rad/s).
                    # Read from joint_vels (same mechanism as all other joints).
                    # joint_vels entries are supplied in the UI's [wZ,wY,wX]
                    # label order (see ModelConfig.joint_vels docstring and
                    # the identical convention used for joint_torques below),
                    # but qdot[s+3:s+6] is consumed everywhere else in this
                    # file (see _update_body_velocities_from_qdot, rnea,
                    # compute_a1_a2_analytic, and the quaternion ODE in
                    # step()) as a literal (omega_x, omega_y, omega_z) world
                    # Cartesian vector. Reorder [wZ,wY,wX] -> (x,y,z) here so
                    # the "Z"-labeled slider genuinely drives world-Z spin
                    # instead of silently driving world-X (Y is unaffected
                    # either way, which is why this went unnoticed by any
                    # test that only ever drove the middle/Y component).
                    # Same precedence as the root ANGLES above ('root_joint'
                    # first, then the model's real root-joint name) -- the two
                    # lookups used opposite orders.  ModelConfig.ang_vel_str
                    # (the "ang_vel" field of a saved configuration, also
                    # [wZ,wY,wX]) was never read at all; it is now honoured
                    # when neither joint_vels key is present.
                    joint_vels = getattr(config, 'joint_vels', {})
                    if 'root_joint' in joint_vels:
                        vel_src = joint_vels['root_joint']
                    elif jname in joint_vels:
                        vel_src = joint_vels[jname]
                    else:
                        try:
                            vel_src = [float(x) for x in
                                       getattr(config, 'ang_vel_str', '').split()]
                        except ValueError:
                            vel_src = []
                    vel_zyx = np.zeros(3)
                    vel_list = list(vel_src)[:3]
                    vel_zyx[:len(vel_list)] = np.asarray(vel_list, dtype=float)
                    ang_vel_rad = vel_zyx[::-1]   # [wZ,wY,wX] -> (wX,wY,wZ)
                    self.state[self.nq+s+3 : self.nq+s+6] = ang_vel_rad
                    # qdot[s+6] = 0  (quaternion norm constraint — not a DOF)
                elif dof == 3:
                    ang_deg = config.joints.get(jname, [0,0,0])
                    self.state[s:s+3] = np.radians(ang_deg[:3])
                elif dof == 1:
                    # Revolute joint to GROUND (a limb pinned to inertial
                    # space, e.g. the hip joint of the MADYMO ball-kick leg).
                    # Angle and rate are scalars about the joint axis; the
                    # config supplies them in the same ZYX slot convention as
                    # a non-root revolute.
                    ang_deg = (config.joints.get(jname)
                               or config.joints.get('root_joint', [0.0, 0.0, 0.0]))
                    ax = None
                    for k_b, body_k in enumerate(self.bodies):
                        if body_k.model_idx == model_idx and self.bodies[k_b].name.endswith(
                                jinfo.get('child_name', '')):
                            ax = self.joint_axis_local[k_b]
                            break
                    # ZYX slot convention: slot 0 = Z, slot 1 = Y, slot 2 = X,
                    # so an axis vector's component index i maps to slot 2 - i.
                    # The previous expression used the component index directly,
                    # which is correct only for a Y axis and transposed Z with X
                    # -- a revolute joint built on a +Z axis in the model editor
                    # took its angle from the X-labelled field and discarded the
                    # value entered in the Z field.
                    slot = (2 - int(np.argmax(np.abs(ax)))) if ax is not None else 0
                    val = float(ang_deg[slot]) if slot < len(ang_deg) else 0.0
                    self.state[s] = np.radians(val)
                    jv = getattr(config, 'joint_vels', {})
                    vl = jv.get(jname, jv.get('root_joint', [0.0, 0.0, 0.0]))
                    self.state[self.nq + s] = (float(vl[slot])
                                               if slot < len(vl) else 0.0)
            else:
                ang_deg = config.joints.get(jname, [0,0,0])
                if dof == 4:
                    # Spherical joint stored as quaternion [qw,qx,qy,qz].
                    # GUI provides ZYX Euler degrees; convert to quaternion.
                    ang_rad = np.radians(ang_deg[:3])
                    a, b, g = ang_rad
                    ca, sa = np.cos(a), np.sin(a)
                    cb, sb = np.cos(b), np.sin(b)
                    cg, sg = np.cos(g), np.sin(g)
                    R = np.array([
                        [ca*cb, ca*sb*sg - sa*cg, ca*sb*cg + sa*sg],
                        [sa*cb, sa*sb*sg + ca*cg, sa*sb*cg - ca*sg],
                        [-sb,   cb*sg,             cb*cg           ]
                    ])
                    self.state[s:s+4] = matrix_to_quat(R)   # [qw,qx,qy,qz]
                    # Per-joint angular velocity (rad/s) in LOCAL frame.
                    # Same [wZ,wY,wX] -> (x,y,z) reorder as the root case
                    # above: qdot[s:s+3] is consumed as a literal local-frame
                    # (omega_x,omega_y,omega_z) vector (e.g.
                    # `omega_rel = body.R @ qd_j[:3]` in
                    # _update_body_velocities_from_qdot / rnea), so it must
                    # not be left in the UI's Z-first label order.
                    joint_vels = getattr(config, 'joint_vels', {})
                    vel_zyx = joint_vels.get(jname, [0.0, 0.0, 0.0])
                    vel_xyz = [float(v) for v in vel_zyx[:3]][::-1]
                    self.state[self.nq+s : self.nq+s+3] = vel_xyz
                    # qdot[s+3] = 0 (quaternion norm constraint slot)
                elif dof == 1:
                    # Revolute joint: the single DOF is the rotation angle
                    # about the joint axis.  The UI supplies ang_deg as a
                    # three-element ZYX Euler vector; we must project it onto
                    # the actual joint axis stored in joint_axis_local.
                    #
                    # Find the body index that owns this joint so we can look
                    # up the pre-computed child-frame axis.
                    angle_rad = 0.0
                    body_found = False
                    for k_b, body_k in enumerate(self.bodies):
                        if body_k.model_idx != model_idx:
                            continue
                        vis_k = body_k.name.split('_', 1)[1]
                        vis_obj_k = self.models[model_idx].bodies.get(vis_k)
                        if (vis_obj_k is not None and
                                vis_obj_k.joint_name_to_parent == jname):
                            ax = self.joint_axis_local[k_b]
                            if ax is not None:
                                # Dominant axis component selects ZYX Euler slot
                                # Slot 0 = Z-rotation = ang_deg[0]
                                # Slot 1 = Y-rotation = ang_deg[1]
                                # Slot 2 = X-rotation = ang_deg[2]
                                # Component index i therefore maps to slot 2 - i;
                                # using i directly transposed Z and X (see the
                                # matching comment in the dof == 1 root branch).
                                slot = 2 - int(np.argmax(np.abs(ax)))
                                deg_val = (float(ang_deg[slot])
                                           if slot < len(ang_deg) else 0.0)
                                angle_rad = np.radians(deg_val)
                            else:
                                angle_rad = np.radians(float(ang_deg[0])
                                                       if len(ang_deg) > 0 else 0.0)
                            body_found = True
                            break
                    if not body_found:
                        # Fallback: use the first element (Z-rotation)
                        angle_rad = np.radians(float(ang_deg[0])
                                               if len(ang_deg) > 0 else 0.0)
                    self.state[s] = angle_rad
                    # Per-joint angular velocity (rad/s) — single scalar.
                    # Read from the SAME ZYX slot the angle was read from.
                    # This used to be hard-coded to vel_list[0], so a joint
                    # whose axis selected slot 1 or 2 took its angle from that
                    # slot but its rate from slot 0 -- the initial angle and
                    # the initial velocity referred to different axes.
                    joint_vels = getattr(config, 'joint_vels', {})
                    vel_list = joint_vels.get(jname, [0.0, 0.0, 0.0])
                    self.state[self.nq + s] = (float(vel_list[slot])
                                               if slot < len(vel_list) else 0.0)
                else:
                    n = min(dof, len(ang_deg))
                    self.state[s:s+n] = np.radians(ang_deg[:n])
                    # Per-joint angular velocity (rad/s)
                    joint_vels = getattr(config, 'joint_vels', {})
                    vel_rad = joint_vels.get(jname, [0.0] * dof)
                    nv = min(dof, len(vel_rad))
                    self.state[self.nq+s : self.nq+s+nv] = [
                        float(v) for v in vel_rad[:nv]
                    ]

        # Load any prescribed torques defined for this model
        self._load_prescribed_torques_from_config(model_idx, config)

    def _load_prescribed_torques_from_config(self, model_idx, config):
        """
        Populate self.prescribed_torques from ModelConfig.joint_torques.

        ModelConfig.joint_torques format:
            { joint_name: {'torque': [τZ, τY, τX],
                           't_start': float,
                           'duration': float}, ... }

        Existing entries for this model_idx are replaced so that
        rebuild_physics() always produces a clean state.
        """
        # Remove any stale entries for this model
        self.prescribed_torques = [
            e for e in self.prescribed_torques
            if e['model_idx'] != model_idx
        ]

        # Resolve this model's REAL root-joint name (e.g. 'rootJoint',
        # 'car_free_jnt', 'bike_free_jnt', ...). The config/UI layer always
        # keys the root entry with the canonical alias 'root_joint' (see
        # _initialize_state_from_config's root branch, which accepts either
        # 'root_joint' or the real name) but self.joint_dof_map is only ever
        # populated with the joint's real underlying name -- it never
        # contains the literal string 'root_joint'. Without this resolution
        # step, `key = (model_idx, 'root_joint')` would never be found in
        # joint_dof_map inside _compute_prescribed_torques, so every root
        # torque entry was silently skipped every step -- prescribed torques
        # on the root never reached the equations of motion at all.
        real_root_name = None
        for (m, jn, ji, dof) in self.joint_list:
            if m == model_idx and ji.get('is_root_joint', False):
                real_root_name = jn
                break

        jt = getattr(config, 'joint_torques', {})
        for jname, spec in jt.items():
            trq      = spec.get('torque',   [0.0, 0.0, 0.0])
            t_start  = float(spec.get('t_start',  0.0))
            duration = float(spec.get('duration', 0.0))
            resolved_name = jname
            if real_root_name is not None and (
                    jname == 'root_joint' or jname.startswith('root_joint_')):
                resolved_name = real_root_name
            # Only register entries where at least one axis has a non-zero torque
            if any(abs(float(v)) > 1e-9 for v in trq):
                self.prescribed_torques.append({
                    'model_idx':  model_idx,
                    'joint_name': resolved_name,
                    'torque':     np.array([float(v) for v in trq]),
                    't_start':    t_start,
                    'duration':   duration,
                })


    def _root_revolute_frame(self, i, ji):
        """
        World-frame rotation axis and joint point for a revolute joint that
        connects a body directly to inertial space (parent = GROUND).

        Returns (axis_world, point_world).
        """
        ax = self.bodies[i].R @ self.joint_axis_local[i]
        n = np.linalg.norm(ax)
        ax = ax / n if n > 1e-12 else np.array([0.0, 0.0, 1.0])
        return ax, np.asarray(ji['T1'])[:3, 3]

    def _build_kinematic_tree(self):
        n = len(self.bodies)
        self.parent_idx      = [-1]    * n
        self.joint_type      = ['fixed']* n
        self.joint_axis_local= [None]  * n
        self.joint_T1        = [None]  * n
        self.joint_T2        = [None]  * n
        self.joint_T2_inv    = [None]  * n
        self.joint_start_idx = [-1]    * n
        self.joint_dof       = [0]     * n

        name_to_idx = {b.name: i for i, b in enumerate(self.bodies)}

        for i, body in enumerate(self.bodies):
            model = self.models[body.model_idx]
            vis_name = body.name.split('_', 1)[1]
            vis_body = model.bodies.get(vis_name)
            if vis_body is None:
                continue
            if vis_body.joint_name_to_parent == "None":
                # A body whose parent is GROUND never gets joint_name_to_parent
                # set (children_map only records body->body joints), so a
                # REVOLUTE root joint would be invisible here and its axis
                # would never be registered.  Look it up directly.
                rj = next((ji_ for jn_, ji_ in model.joint_infos.items()
                           if ji_.get('is_root_joint', False)
                           and ji_.get('type') == 'revolute'
                           and ji_.get('child_name') == vis_name), None)
                if rj is None:
                    continue
                T2r = np.asarray(rj['T2'])
                axc = T2r[:3, 2].astype(float).copy()
                nrm = np.linalg.norm(axc)
                self.joint_axis_local[i] = (axc / nrm if nrm > 1e-10
                                            else np.array([0.0, 0.0, 1.0]))
                self.joint_type[i] = 'revolute'
                self.joint_T1[i] = np.asarray(rj['T1']).copy()
                self.joint_T2[i] = T2r.copy()
                self.joint_T2_inv[i] = np.asarray(rj['T2_inv']).copy()
                k0 = (body.model_idx, rj['name'])
                if k0 in self.joint_dof_map:
                    self.joint_start_idx[i], self.joint_dof[i] = self.joint_dof_map[k0]
                continue
            ji = model.joint_infos.get(vis_body.joint_name_to_parent)
            if ji is None:
                continue
            if ji['parent'] is None:
                # Root joint (parent = GROUND).  A FREE root needs nothing
                # here, but a REVOLUTE root (a limb pinned to inertial space,
                # e.g. the hip of the MADYMO ball-kick leg) still needs its
                # axis and DOF slot recorded -- otherwise joint_axis_local[i]
                # stays None and the joint silently contributes no motion.
                if ji.get('type') == 'revolute':
                    T2r = np.asarray(ji['T2'])
                    axc = T2r[:3, 2].astype(float).copy()
                    nrm = np.linalg.norm(axc)
                    self.joint_axis_local[i] = (axc / nrm if nrm > 1e-10
                                                else np.array([0.0, 0.0, 1.0]))
                    self.joint_type[i] = 'revolute'
                    self.joint_T1[i] = np.asarray(ji['T1']).copy()
                    self.joint_T2[i] = T2r.copy()
                    self.joint_T2_inv[i] = np.asarray(ji['T2_inv']).copy()
                    key0 = (body.model_idx, vis_body.joint_name_to_parent)
                    if key0 in self.joint_dof_map:
                        self.joint_start_idx[i], self.joint_dof[i] = self.joint_dof_map[key0]
                continue
            # Bodies are named "<config id>_<body>" by the caller, and the
            # config id is NOT the same as model_idx (the load order) once a
            # model has been removed in the Setup tab: deleting Model A leaves
            # Model B with id 1 but model_idx 0.  Building the parent name
            # from model_idx then failed to find ANY parent, so every segment
            # became a disconnected root -- the whole model fell apart with
            # most bodies unintegrated.  Use the body's own name prefix.
            prefix = body.name.split('_', 1)[0]
            parent_name = f"{prefix}_{ji['parent'].name}"
            if parent_name not in name_to_idx:
                continue
            self.parent_idx[i]  = name_to_idx[parent_name]
            self.joint_T1[i]    = ji['T1'].copy()
            self.joint_T2[i]    = ji['T2'].copy()
            self.joint_T2_inv[i]= ji['T2_inv'].copy()
            jt = ji.get('type', 'fixed')
            self.joint_type[i]  = jt
            if jt == 'revolute':
                # Convention: the revolute axis is the Z-axis OF THE JOINT
                # FRAME.  T1 places the joint frame in the parent body, so
                # T1[:3,2] is that axis expressed in PARENT coordinates -- and
                # in the joint frame itself the axis is, by construction,
                # exactly (0,0,1).
                #
                # The dynamics need it in CHILD coordinates, because every
                # consumer computes  ax = child_body.R @ joint_axis_local[i].
                # T2 places the joint frame in the child body, so
                #     axis_child = R_T2 @ (0,0,1) = T2[:3,2].
                #
                # The previous code used  T2[:3,:3].T @ T1[:3,2], which is
                # wrong twice over: it fed a PARENT-frame vector into a
                # joint->child map, and it used R_T2 transposed (that maps
                # child->joint, the opposite of what is needed).  Both errors
                # vanish when T1 and T2 have no rotation, which is why this
                # went unnoticed -- every bundled model's revolute joints (of
                # which there are none) and every identity-framed joint agree.
                T2 = ji['T2']
                axis_child_local = T2[:3, 2].astype(float).copy()
                norm = np.linalg.norm(axis_child_local)
                if norm > 1e-10:
                    axis_child_local = axis_child_local / norm
                else:
                    axis_child_local = np.array([0.0, 0.0, 1.0])
                self.joint_axis_local[i] = axis_child_local
            key = (body.model_idx, vis_body.joint_name_to_parent)
            if key in self.joint_dof_map:
                self.joint_start_idx[i], self.joint_dof[i] = self.joint_dof_map[key]

        self.children = [[] for _ in range(n)]
        for i, p in enumerate(self.parent_idx):
            if p != -1:
                self.children[p].append(i)

    # ------------------------------------------------------------------
    # BF-B: Update body.vel / body.ang_vel from qdot
    # ------------------------------------------------------------------

    def _update_body_velocities_from_qdot(self, q, qdot):
        """
        Run the RNEA forward-pass velocity recursion and store results in
        body.vel and body.ang_vel so that get_velocity_at_point() returns
        physically correct, state-consistent values for contact forces.

        Must be called AFTER update_kinematics_from_q(q) (needs body.R).
        """
        nb = len(self.bodies)
        v     = [np.zeros(3) for _ in range(nb)]
        omega = [np.zeros(3) for _ in range(nb)]

        # BFS topological order
        order, visited = [], [False]*nb
        stack = [i for i, p in enumerate(self.parent_idx) if p == -1]
        while stack:
            i = stack.pop()
            if visited[i]: continue
            visited[i] = True; order.append(i)
            stack.extend(self.children[i])

        for i in order:
            body = self.bodies[i]
            if self.parent_idx[i] == -1:
                midx = body.model_idx
                for (m, jn, ji, dof) in self.joint_list:
                    if m == midx and ji.get('is_root_joint', False):
                        s, _ = self.joint_dof_map[(m, jn)]
                        qd = qdot[s:s+dof]
                        if dof == 7:
                            # q[s:s+3]=pos, q[s+3:s+7]=quat; qdot[s:s+3]=vel, qdot[s+3:s+6]=omega_world
                            omega[i] = qd[3:6]
                            r_jcg = body.pos - q[s:s+3]
                            v[i] = qd[:3] + np.cross(omega[i], r_jcg)
                        elif dof == 3:
                            ang = q[s:s+3]
                            omega[i] = self._E_world_zyx(ang) @ qd
                            r_jcg = body.pos
                            v[i] = np.cross(omega[i], r_jcg)
                        elif dof == 1:
                            axw, pj = self._root_revolute_frame(i, ji)
                            omega[i] = axw * qd[0]
                            v[i] = np.cross(omega[i], body.pos - pj)
                        elif dof == 0:
                            omega[i] = np.zeros(3); v[i] = np.zeros(3)
                        break
            else:
                pi   = self.parent_idx[i]
                pb   = self.bodies[pi]
                jtype = self.joint_type[i]
                s    = self.joint_start_idx[i]
                dof  = self.joint_dof[i]
                qd_j = qdot[s:s+dof] if s != -1 else np.zeros(max(dof, 1))

                T1, T2   = self.joint_T1[i], self.joint_T2[i]
                pj_world = pb.pos + pb.R @ (T1[:3,3] - pb.cg_local)
                cj_world = body.R @ (T2[:3,3] - body.cg_local)
                r_pj     = pj_world - pb.pos
                v_par_jnt = v[pi] + np.cross(omega[pi], r_pj)

                if jtype == 'revolute' and dof == 1 and s != -1:
                    ax = body.R @ self.joint_axis_local[i]
                    omega[i] = omega[pi] + ax * qd_j[0]
                    # v_CG = v_joint + omega x r_jcg, r_jcg = -cj_world
                    v[i] = v_par_jnt + np.cross(omega[i], -cj_world)
                elif jtype == 'spherical' and dof == 4 and s != -1:
                    # qdot[s:s+3] = omega_local (body frame); qdot[s+3] = 0 (norm slot)
                    omega_rel = body.R @ qd_j[:3]
                    omega[i] = omega[pi] + omega_rel
                    v[i] = v_par_jnt + np.cross(omega[i], -cj_world)
                elif jtype == 'spherical' and dof == 3 and s != -1:
                    ang = q[s:s+3]
                    E = self._E_body_zyx(ang)
                    omega_rel = body.R @ (E @ qd_j)
                    omega[i] = omega[pi] + omega_rel
                    # v_CG = v_joint + omega x r_jcg, r_jcg = -cj_world
                    v[i] = v_par_jnt + np.cross(omega[i], -cj_world)
                else:  # fixed
                    # r_pc = parent CG -> child CG.  cj_world is child CG ->
                    # joint, so joint -> child CG is -cj_world and therefore
                    # r_pc = r_pj - cj_world.  (Was "+", which disagreed with
                    # compute_a1_a2_analytic's r_pc = cb.pos - pb.pos and made
                    # the mass matrix inconsistent with the bias forces for
                    # every fixed-joint body and its descendants.)
                    omega[i] = omega[pi]
                    v[i] = v[pi] + np.cross(omega[pi], r_pj - cj_world)

            # Write to body object so get_velocity_at_point() is correct
            body.vel     = v[i].copy()
            body.ang_vel = omega[i].copy()

    # ------------------------------------------------------------------
    # Contact detection 
    # ------------------------------------------------------------------

    @staticmethod
    def _ellipsoid_surface_normal(d, R, r):
        """Gradient-based outward surface normal (more accurate than d/|d|)."""
        g = R @ (R.T @ d / (r ** 2 + 1e-20))
        n = np.linalg.norm(g)
        if n < 1e-12:
            return d / (np.linalg.norm(d) + 1e-12)
        return g / n

    @staticmethod
    def _ellipsoid_radial_extent(semi_axes, R, normal):
        """
         Effective radius of an ellipsoid along the contact normal.

        
            r_eff = ‖ semi_axes * (R^T · n̂) ‖



        Parameters
        ----------
        semi_axes : (3,) array  – ellipsoid semi-axes [a, b, c]
        R         : (3,3) array – ellipsoid orientation matrix (world frame)
        normal    : (3,) array  – unit contact normal (world frame)
        """
        local_n = R.T @ normal          # normal in ellipsoid local frame
        scaled  = semi_axes * local_n   # element-wise: (a·nx, b·ny, c·nz)
        return float(np.linalg.norm(scaled))

    @classmethod
    def _ellipsoid_pair_contact(cls, p1, R1, r1, p2, R2, r2, slop):
        """
        Penetration depth, contact normal and contact point for a pair of
        ellipsoids (centre p, orientation R, semi-axes r).

        The overlap of two convex bodies along a unit direction n is
            f(n) = h1(n) + h2(n) - d.n,      d = p2 - p1,
        where h(n) = |diag(r) R^T n| is the ellipsoid support function.  The
        penetration depth (minimum translation distance) is min_n f(n), and
        the minimising n is the contact normal.

        The previous implementation evaluated f at a single heuristic
        direction (the sum of the two gradient normals at the centre offset).
        That is exact for spheres, but because f(n) >= min f for every n it
        always OVER-estimates the depth, and for elongated ellipsoids at
        oblique relative orientation the error is large.  On 300 random
        near-contact pairs (semi-axes 30-200 mm) the heuristic reported
        penetration 2 mm too deep at the median and up to 45 mm too deep
        (i.e. a contact force between bodies that were not touching), with
        a normal error of 7 deg median and up to 34 deg.  A misdirected
        normal feeds straight into the moment arm of every contact force,
        so it biases whole-body rotation after impact.

        The heuristic direction is kept as the starting point and refined by
        projected gradient descent on the unit sphere with a backtracking
        step (monotone in f), which reduces the errors on the same test set
        to 0.01 mm / 0.7 deg (90th percentile).  Because the start value is
        an upper bound on the true depth, pairs whose heuristic depth is
        already below -slop are rejected without refinement, so the extra
        cost is paid only for pairs that are genuinely in or near contact.

        Returns (pen, normal, cp) with `normal` the geometric direction from
        ellipsoid 1 towards ellipsoid 2, or None if pen <= -slop.  `cp` is
        the midpoint of the two support points, which at the optimum lie on
        a common line along `normal` a distance `pen` apart.
        """
        d = p2 - p1
        n1 = cls._ellipsoid_surface_normal(d, R1, r1)
        n2 = -cls._ellipsoid_surface_normal(-d, R2, r2)
        n = n1 + n2
        nlen = np.linalg.norm(n)
        n = n / nlen if nlen > 1e-12 else n1

        M1 = (R1 * (r1 * r1)) @ R1.T
        M2 = (R2 * (r2 * r2)) @ R2.T

        def _eval(nv):
            m1 = M1 @ nv
            m2 = M2 @ nv
            h1 = np.sqrt(max(float(nv @ m1), 1e-30))
            h2 = np.sqrt(max(float(nv @ m2), 1e-30))
            return h1 + h2 - float(d @ nv), m1 / h1 + m2 / h2 - d, h1, h2, m1, m2

        f, g, h1, h2, m1, m2 = _eval(n)
        if f <= -slop:
            return None

        L1 = float(np.max(r1)) ** 2
        L2 = float(np.max(r2)) ** 2
        for _ in range(cls.CONTACT_NORMAL_ITERS):
            gt = g - float(g @ n) * n
            if float(gt @ gt) < cls.CONTACT_NORMAL_TOL ** 2:
                break
            eta = 1.0 / (L1 / h1 + L2 / h2 + abs(float(d @ n)))
            accepted = False
            for _bt in range(6):
                nn = n - eta * gt
                nn = nn / np.sqrt(float(nn @ nn))
                fn, gn, hn1, hn2, mn1, mn2 = _eval(nn)
                if fn <= f:
                    accepted = True
                    break
                eta *= 0.5
            if not accepted:
                break
            n, f, g, h1, h2, m1, m2 = nn, fn, gn, hn1, hn2, mn1, mn2

        if f <= -slop:
            return None
        s1 = p1 + m1 / h1          # support point of ellipsoid 1 along +n
        s2 = p2 - m2 / h2          # support point of ellipsoid 2 along -n
        return f, n, 0.5 * (s1 + s2)

    def _build_adjacency_exclusions(self):
        """
        Build a set of (i, j) body-index pairs that should NOT generate
        contact forces because they are kinematically adjacent (parent–child
        or grandparent–grandchild).  
        """
        excluded = set()
        n = len(self.bodies)

        def ancestors(i, hops):
            """Return all body indices within `hops` steps up the tree."""
            result, cur = set(), i
            for _ in range(hops):
                p = self.parent_idx[cur]
                if p == -1:
                    break
                result.add(p)
                cur = p
            return result

        for i in range(n):
            # Direct parent/child
            p = self.parent_idx[i]
            if p != -1:
                excluded.add((min(i, p), max(i, p)))
            # Grandparent
            for anc in ancestors(i, 2):
                excluded.add((min(i, anc), max(i, anc)))
            # Siblings (share the same parent) — avoid torso-segment pairs
            if p != -1:
                for sib in self.children[p]:
                    if sib != i:
                        excluded.add((min(i, sib), max(i, sib)))

        self._adj_exclusions = excluded

    # ------------------------------------------------------------------
    # Initial-overlap offset  (permanent, same-model only)
    # ------------------------------------------------------------------


    def _record_initial_overlaps(self):
        """
        Scan the current pose and record initial penetration depths for
        all same-model ellipsoid pairs and ground contacts.
        """
        from physics_constraints import _contact_key
        self._initial_pen_offsets    = {}   # cache_key -> pen_at_t0
        self._initial_ground_offsets = {}   # (body.name, ell.name) -> pen_at_t0

        # Ensure kinematics are current
        self.update_kinematics_from_q(self.state[:self.nq])
        if not hasattr(self, '_adj_exclusions'):
            self._build_adjacency_exclusions()

        nb = len(self.bodies)
        for i in range(nb):
            b1 = self.bodies[i]
            for j in range(i + 1, nb):
                b2 = self.bodies[j]
                # Only offset same-model pairs — inter-model starts separated
                if b1.model_id != b2.model_id:
                    continue
                # Adjacency-excluded pairs are already suppressed; skip them
                if (i, j) in self._adj_exclusions:
                    continue
                for e1 in b1.ellipsoids:
                    T1_w = b1.get_body_transform() @ e1.local_T
                    p1, R1, r1 = T1_w[:3, 3], T1_w[:3, :3], e1.dims
                    r1_max = float(np.max(r1))
                    for e2 in b2.ellipsoids:
                        T2_w = b2.get_body_transform() @ e2.local_T
                        p2, R2, r2 = T2_w[:3, 3], T2_w[:3, :3], e2.dims
                        d = p2 - p1
                        dist = np.linalg.norm(d)
                        if dist < 1e-10:
                            continue
                        if dist > r1_max + float(np.max(r2)) + self.contact_penetration_slop:
                            continue
                        hit = self._ellipsoid_pair_contact(p1, R1, r1, p2, R2, r2,
                                                           0.0)
                        if hit is not None and hit[0] > 0.0:
                            pen = hit[0]
                            key = _contact_key(b1, e1.name, b2, e2.name)
                            self._initial_pen_offsets[key] = max(
                                self._initial_pen_offsets.get(key, 0.0), pen)

        # Ground offsets — use the EXACT quadratic z-extent formula
        for body in self.bodies:
            for ell in body.ellipsoids:
                T   = body.get_body_transform() @ ell.local_T
                pos, Rm, r = T[:3, 3], T[:3, :3], ell.dims
                z_ext = float(np.sqrt(
                    (Rm[2, 0] * r[0]) ** 2 +
                    (Rm[2, 1] * r[1]) ** 2 +
                    (Rm[2, 2] * r[2]) ** 2))
                pen = -(pos[2] - z_ext)
                if pen > 0.0:
                    gkey = (body.name, ell.name)
                    self._initial_ground_offsets[gkey] = max(
                        self._initial_ground_offsets.get(gkey, 0.0), pen)

    def detect_contacts(self):
        contacts = []

        # Build adjacency exclusion set once per topology (cached).
        if not hasattr(self, '_adj_exclusions'):
            self._build_adjacency_exclusions()

        # ── All body-body contacts (inter-model AND intra-model) ──────
        # OLD: skipped all same-model pairs → head could pass through leg.
        # NEW: skip only kinematically adjacent pairs (≤ 2 hops in tree).
        nb = len(self.bodies)

        # Each body's world transform is constant for the whole sweep, but was
        # being recomputed inside the j- and ellipsoid-loops -- ~2000 calls per
        # step for 42 bodies.  Compute once up front.
        _bt = [b.get_body_transform() for b in self.bodies]

        for i in range(nb):
            b1 = self.bodies[i]
            _bt1 = _bt[i]
            for j in range(i + 1, nb):
                b2 = self.bodies[j]

                same_model = (b1.model_id == b2.model_id)

                # Default behaviour for the bundled human models: disable
                # intra-model self-contact to avoid forces from designed
                # geometric overlap between neighbouring body ellipsoids.
                if same_model and not self.enable_self_contact:
                    continue

                # Optional self-contact mode keeps adjacency filtering.
                if same_model and (i, j) in self._adj_exclusions:
                    continue

                _bt2 = _bt[j]
                for e1 in b1.ellipsoids:
                    T1_w = _bt1 @ e1.local_T
                    p1, R1, r1 = T1_w[:3,3], T1_w[:3,:3], e1.dims
                    r1_bound   = float(np.max(r1))

                    for e2 in b2.ellipsoids:
                        T2_w = _bt2 @ e2.local_T
                        p2, R2, r2 = T2_w[:3,3], T2_w[:3,:3], e2.dims

                        d = p2 - p1
                        dist = np.linalg.norm(d)
                        if dist < 1e-10:
                            continue

                        # Fast bounding-sphere cull
                        r2_bound = float(np.max(r2))
                        if dist > r1_bound + r2_bound + self.contact_penetration_slop:
                            continue

                        hit = self._ellipsoid_pair_contact(
                            p1, R1, r1, p2, R2, r2, self.contact_penetration_slop)
                        if hit is not None:
                            pen, normal, cp = hit
                            # `normal` is the geometric direction FROM ellipsoid A
                            # TOWARD ellipsoid B.  The contact object stores the
                            # REPULSION normal for bodyA, which points from the
                            # contact surface INTO bodyA, i.e. the opposite.
                            # (A former sanity check compared this with the
                            # body-CG offset and could warn spuriously for
                            # multi-ellipsoid bodies; for the ellipsoid centres
                            # force_normal . (p1 - p2) > 0 holds by construction.)
                            force_normal = -normal
                            # Contact point: midpoint of the two support points,
                            # which lie on a common line along the normal.  For
                            # two spheres this is on the line of centres, so the
                            # normal force exerts no torque about either centre.
                            # For same-model pairs: subtract the permanent
                            # initial-overlap offset so resting body-region
                            # ellipsoids (which overlap by design) generate no
                            # force at rest, but genuine new contacts during
                            # motion (arm hitting head, etc.) are still resolved.
                            pen_eff = pen
                            if b1.model_id == b2.model_id:
                                from physics_constraints import _contact_key
                                ikey = _contact_key(b1, e1.name, b2, e2.name)
                                pen0 = (self._initial_pen_offsets.get(ikey, 0.0)
                                        if hasattr(self, '_initial_pen_offsets')
                                        else 0.0)
                                pen_eff = max(0.0, pen - pen0)
                            contacts.append(SimpleContact(
                                b1, b2, cp, force_normal, pen_eff,
                                friction=self.friction_coef, restitution=0.1,
                                ellipsoidA=e1, ellipsoidB=e2,
                                damping=self.contact_damping,
                                energy_retention=self.contact_energy_retention))

        # ── Ground contacts (z = 0 plane) ─────────────────────────────
        for _bi, body in enumerate(self.bodies):
            for ell in body.ellipsoids:
                T  = _bt[_bi] @ ell.local_T
                pos, Rm, r = T[:3,3], T[:3,:3], ell.dims

                z_ext = float(np.sqrt(
                    (Rm[2, 0] * r[0]) ** 2 +
                    (Rm[2, 1] * r[1]) ** 2 +
                    (Rm[2, 2] * r[2]) ** 2))
                pen = -(pos[2] - z_ext)
                if pen > -self.contact_penetration_slop:
                    # Lowest point of the ellipsoid: its support point along
                    # -z, i.e. pos - R diag(r^2) R^T e_z / z_ext.  The previous
                    # point (pos.x, pos.y, pos.z - z_ext) is correct only when a
                    # principal axis is vertical; for a tilted elongated
                    # ellipsoid (foot, thigh, torso of a falling body) it put
                    # the ground reaction under the centre instead of under the
                    # heel/toe, removing the restoring moment it should exert.
                    col = (Rm * (r * r)) @ Rm[2, :]
                    cp = pos - col / max(z_ext, 1e-12)

                    gkey = (body.name, ell.name)
                    pen0 = (self._initial_ground_offsets.get(gkey, 0.0)
                            if hasattr(self, '_initial_ground_offsets') else 0.0)
                    pen_eff = max(0.0, pen - pen0)
                    contacts.append(SimpleContact(
                        body, None, cp, np.array([0., 0., 1.]),
                        pen_eff,
                        friction=self.friction_coef, restitution=0.1,
                        ellipsoidA=ell, ellipsoidB=None,
                        damping=self.contact_damping,
                        energy_retention=self.contact_energy_retention))
        self.contacts = contacts
        prune_contact_cache({c._cache_key for c in contacts})
        self._build_contact_index()
        return contacts

    def _build_contact_index(self):
        """
        Map id(body) -> list of contacts touching that body.

        get_contacts_for_body() is called twice per body per RNEA backward pass
        (once from get_applied_force, once from get_applied_moment) plus again
        in _accumulate_contact_forces_on_bodies.  Scanning self.contacts every
        time is O(n_bodies * n_contacts) per step; with 42 bodies that scan
        dominated the step cost.
        """
        idx = {}
        for c in self.contacts:
            idx.setdefault(id(c.bodyA), []).append(c)
            if c.bodyB is not None:
                idx.setdefault(id(c.bodyB), []).append(c)
        self._contact_index = idx

    def get_contacts_for_body(self, body):
        idx = getattr(self, '_contact_index', None)
        if idx is None:
            return [c for c in self.contacts if c.bodyA is body or c.bodyB is body]
        return idx.get(id(body), [])

    # ------------------------------------------------------------------
    # Contact evaluation scaling  (Fix 4: Newton's 3rd law symmetric)
    # ------------------------------------------------------------------

    def _compute_all_contact_scales(self):
        """
        Compute and store a single symmetric scale factor on every active
        contact object (``c._global_scale``) so that Newton's 3rd law is
        exactly preserved for all evaluation modes.

        **Root cause of the previous bug**
        -----------------------------------
        The old ``_contact_evaluation_scales(body)`` computed a scale from
        *each body's own perspective* independently.  When body A had N>1
        contacts its scale was Fe_max/ΣFe < 1, but each opponent body B
        computed its scale as 1.0 (singleton group).  Action ≠ Reaction.

        **Fix**
        --------
        Scales are computed once per step at the *contact* level, not the
        body level.  For each body that acts as a shared receiving surface
        (i.e. is touched by N>1 opponents simultaneously) the scale
        Fe_max/ΣFe is computed and then written to every contact in that
        group.  If the same contact appears in two receiver groups (both
        endpoints are shared surfaces) the *minimum* (most restrictive)
        scale is kept.  Because ``c._global_scale`` is read by both
        ``get_applied_force(bodyA)`` and ``get_applied_force(bodyB)`` for
        the same contact, both bodies see the identical scale and
        Newton's 3rd law is satisfied exactly.

        Ground contacts (bodyB=None) are always singleton groups: the
        infinite ground plane cannot be a shared surface in the same sense.

        Must be called once per step after ``detect_contacts()`` and before
        any force evaluation.
        """
        # Initialise all contacts to scale 1.0 and clear the per-step
        # magnitude cache (Bug 6 fix: _contact_magnitude must be evaluated
        # once per contact per step — see _contact_magnitude docstring).
        for c in self.contacts:
            c._global_scale = 1.0
            c._magnitude_cache = None   # cleared here, populated on first call

        mode = self.contact_evaluation_mode
        if mode == 'none' or not self.contacts:
            return

        from physics_constraints import _hysteresis_force, _get_combined_curve

        def _fe(c):
            """Elastic force for a contact (no damping — scale defined on Fe)."""
            ca = (c.ellipsoidA.force_curve
                  if c.ellipsoidA is not None and c.ellipsoidA.force_curve is not None
                  else None)
            cb = (c.ellipsoidB.force_curve
                  if c.ellipsoidB is not None and c.ellipsoidB.force_curve is not None
                  else None)
            ua = getattr(c.ellipsoidA, 'unload_curve', None)
            ub = getattr(c.ellipsoidB, 'unload_curve', None) if c.ellipsoidB is not None else None
            if ca is not None and cb is not None:
                curve = _get_combined_curve(ca, cb)
                unload_curve = (_get_combined_curve(ua, ub)
                                if ua is not None and ub is not None
                                else (ua if ua is not None else ub))
            elif ca is not None:
                curve = ca
                unload_curve = ua
            elif cb is not None:
                curve = cb
                unload_curve = ub
            else:
                curve = np.array([[0.0, 0.0], [0.1, 5000.0]])
                unload_curve = None
            return max(0.0, _hysteresis_force(curve, c.penetration,
                                              c._state['pen_max'], c.eta,
                                              unload_curve=unload_curve))

        # ── Group by BODY PAIR, not by body ──────────────────────────
        # The class docstring defines the grouping unit explicitly: "contacts
        # between the same (bodyA, bodyB) pair are evaluated together;
        # contacts between different body pairs are always independent".
        # The implementation grouped by single body instead, which lumped
        # every opponent of a body into one group.  Two consequences:
        #
        #   1. Independent body pairs were coupled.  A head resting against a
        #      pelvis had its force reduced because the same head also touched
        #      a thigh -- two genuinely separate contact surfaces, which the
        #      documented design says must not interact.
        #   2. A contact could belong to two groups, so the min() needed to
        #      keep Newton's 3rd law symmetric could zero EVERY contact of a
        #      pair (the pair's own winner losing in the other group).  In
        #      'discrete' mode that annihilated 67 % of active body pairs on
        #      testimpact1.json -- those bodies transmitted no force at all
        #      and passed through each other.
        #
        # Grouping by pair fixes both, and is strictly simpler: each contact
        # now belongs to exactly one group, so no min() reconciliation is
        # needed and both endpoints trivially read the same scale.
        pair_groups = {}
        for c in self.contacts:
            if c.bodyB is None:
                continue          # ground: always its own singleton group
            key = (min(id(c.bodyA), id(c.bodyB)), max(id(c.bodyA), id(c.bodyB)))
            pair_groups.setdefault(key, []).append(c)

        for grp in pair_groups.values():
            if len(grp) < 2:
                continue          # singleton pair: scale stays 1.0
            fe_vals = [_fe(c) for c in grp]
            fe_max  = max(fe_vals)
            fe_sum  = sum(fe_vals)
            if fe_max <= 0.0:
                continue          # no force to transmit yet

            if mode == 'discrete':
                # Exactly one contact per pair survives: the strongest.
                # argmax breaks ties deterministically, so a pair can never
                # be left with nothing.
                k = int(np.argmax(fe_vals))
                for idx, c in enumerate(grp):
                    c._global_scale = 1.0 if idx == k else 0.0
            else:  # 'continuous'
                if fe_sum > 1e-12:
                    s = min(1.0, fe_max / fe_sum)
                    for c in grp:
                        c._global_scale = s

    # ------------------------------------------------------------------
    # Applied forces & moments (BF-B + BF-C + BF-6 + BF-7)
    # ------------------------------------------------------------------

    def _dynamic_amplification(self, v_norm_abs):
        """
        Dynamic Amplification Factor (Appendix C, Form 3):
            γ = C1 + C2 · (|ε̇| / C3)^C4

        Scales the elastic contact force by the approach velocity to model
        rate-dependent soft-tissue stiffening.  


        Parameters
        ----------
        v_norm_abs : float  – |v_norm|, approach speed in m/s (≥ 0)
        """
        C1, C2, C3, C4 = (self.dyn_amp_C1, self.dyn_amp_C2,
                           self.dyn_amp_C3, self.dyn_amp_C4)
        return C1 + C2 * (v_norm_abs / (C3 + 1e-12)) ** C4

    def _contact_magnitude(self, c):
        """
        Total normal contact force for contact `c`:
          F_total = γ(|v_norm_prev|) · F_elastic(λ, hysteresis) ± F_damping
        where γ is the dynamic amplification factor (Appendix C Form 3).

        Returns (magnitude, v_rel, v_rel_n).

        Per-step caching (Bug 6 fix)
        ----------------------------
        This method is called up to 6 times per contact per step (twice each
        for bodyA and bodyB inside rnea, and twice more in
        _accumulate_contact_forces_on_bodies).  Without caching the DAF state
        ``v_rel_n_prev`` is overwritten on the first call, so calls 2–6 would
        evaluate γ at the *current* velocity rather than the lagged one —
        causing inconsistency between the equations of motion and the recorded
        forces when C2 > 0.

        The cache ``c._magnitude_cache`` is set to None by
        ``_compute_all_contact_scales()`` at the start of every step (after
        detect_contacts but before force evaluation).  The first call within
        each step computes the result and caches it; all subsequent calls
        return the cached tuple directly.
        """
        if c._magnitude_cache is not None:
            return c._magnitude_cache
        vA = c.bodyA.get_velocity_at_point(c.point)
        vB = (c.bodyB.get_velocity_at_point(c.point)
              if c.bodyB is not None else np.zeros(3))
        v_rel   = vA - vB
        v_rel_n = np.dot(v_rel, c.normal)

        v_rel_n_prev = c._state.get('v_rel_n_prev', v_rel_n)
        c._state['v_rel_n_prev'] = v_rel_n   # update for next step

        # Dynamic amplification scales the elastic force only (not damping)
        gamma = self._dynamic_amplification(abs(v_rel_n_prev))

        # Amplified elastic + hysteresis force
        pen_max  = c._state['pen_max']
        from physics_constraints import _hysteresis_force, _get_combined_curve
        ca = (c.ellipsoidA.force_curve
              if c.ellipsoidA is not None and c.ellipsoidA.force_curve is not None
              else None)
        cb = (c.ellipsoidB.force_curve
              if c.ellipsoidB is not None and c.ellipsoidB.force_curve is not None
              else None)
        ua = getattr(c.ellipsoidA, 'unload_curve', None)
        ub = getattr(c.ellipsoidB, 'unload_curve', None) if c.ellipsoidB is not None else None
        if ca is not None and cb is not None:
            curve = _get_combined_curve(ca, cb)
            unload_curve = (_get_combined_curve(ua, ub)
                            if ua is not None and ub is not None
                            else (ua if ua is not None else ub))
        elif ca is not None:
            curve = ca
            unload_curve = ua
        elif cb is not None:
            curve = cb
            unload_curve = ub
        else:
            curve = np.array([[0.0, 0.0], [0.1, 5000.0]])
            unload_curve = None

        F_elastic = gamma * _hysteresis_force(
            curve, c.penetration, pen_max, c.eta,
            unload_curve=unload_curve)

        # ── Damping (not amplified) ───────────────────────────────────
        # Kelvin-Voigt with a unilateral (non-adhesive) clamp:
        #     F = max(0, F_elastic + c_d * lambda_dot)
        # c.normal points from the contact surface INTO bodyA, so
        # v_rel_n = (v_A - v_B).n is NEGATIVE while the bodies approach, and
        # the penetration rate is lambda_dot = -v_rel_n.
        #
        # The previous form used abs(v_rel_n) and took the sign from the
        # penetration-history flag c._state['loading'].  Those two disagree
        # whenever the flag lags the actual normal velocity -- at every
        # loading/unloading reversal, and any step where penetration deepens
        # while v_rel_n is momentarily positive (or vice versa) -- and the
        # damping force then pushed the wrong way.  Taking the sign straight
        # from lambda_dot removes the ambiguity; the history flag keeps its
        # proper job, which is selecting the elastic hysteresis branch.
        pen_rate  = -v_rel_n
        F_damping = c.damping * pen_rate

        # max(0, ...) enforces the unilateral condition: a contact may push
        # but never pull, so a fast-separating contact simply releases.
        F_total = F_elastic + F_damping

        result = max(0.0, F_total), v_rel, v_rel_n
        c._magnitude_cache = result   # reused by all subsequent calls this step
        return result

    def get_applied_force(self, body, t):
        """
        External forces: gravity + normal contact force + Coulomb friction.
        Contact forces are scaled by ``c._global_scale`` set by
        ``_compute_all_contact_scales()`` — the same value is used for both
        bodyA and bodyB of each contact so Newton's 3rd law is satisfied.
        """
        F = body.mass * self.gravity
        for c in self.get_contacts_for_body(body):
            scale = getattr(c, '_global_scale', 1.0)
            if scale == 0.0:
                continue
            magnitude, v_rel, v_rel_n = self._contact_magnitude(c)
            if magnitude <= 0.0:
                continue

            sign = +1.0 if body is c.bodyA else -1.0
            F += sign * scale * magnitude * c.normal

            # Coulomb friction with velocity ramp
            if c.friction > 0.0:
                v_t     = v_rel - v_rel_n * c.normal
                v_t_mag = np.linalg.norm(v_t)
                if v_t_mag > 1e-6:
                    ramp = c._friction_ramp(v_t_mag)
                    F -= sign * scale * c.friction * magnitude * ramp * (v_t / v_t_mag)

        return F

    def get_applied_moment(self, body, t):
        """
        External moments from contacts (normal + friction).
        Uses ``c._global_scale`` for Newton's 3rd law symmetry.
        """
        M = np.zeros(3)
        for c in self.get_contacts_for_body(body):
            scale = getattr(c, '_global_scale', 1.0)
            if scale == 0.0:
                continue
            magnitude, v_rel, v_rel_n = self._contact_magnitude(c)
            if magnitude <= 0.0:
                continue

            sign = +1.0 if body is c.bodyA else -1.0
            r    = c.point - body.pos
            M += np.cross(r, sign * scale * magnitude * c.normal)

            if c.friction > 0.0:
                v_t     = v_rel - v_rel_n * c.normal
                v_t_mag = np.linalg.norm(v_t)
                if v_t_mag > 1e-6:
                    ramp = c._friction_ramp(v_t_mag)
                    F_f  = -(sign * scale * c.friction * magnitude * ramp
                             * (v_t / v_t_mag))
                    M   += np.cross(r, F_f)

        return M

    # ------------------------------------------------------------------
    # Kinematics update
    # ------------------------------------------------------------------

    def update_kinematics_from_q(self, q):
        """
        Propagate generalised coordinates q to all body poses (pos, R, quat).
        Angles in q are RADIANS; the visual model expects DEGREES.
        Root translation is read from q (not from static pos_str).
        """
        root_pos   = [None] * len(self.models)
        root_ang   = [np.zeros(3)] * len(self.models)
        jstates    = [{} for _ in range(len(self.models))]
        idx = 0

        for (midx, jname, jinfo, dof) in self.joint_list:
            if jinfo.get('is_root_joint', False):
                if dof == 7:
                    root_pos[midx] = q[idx:idx+3]
                    root_ang[midx] = q[idx+3:idx+7]   # quaternion [qw,qx,qy,qz]
                elif dof == 3:
                    root_ang[midx] = q[idx:idx+3]
                idx += dof
            else:
                jt = jinfo.get('type', 'fixed')
                if jt == 'spherical' and dof == 4:
                    # Quaternion spherical joint — convert to degrees for visual model
                    qjt = q[idx:idx+4]
                    norm = np.linalg.norm(qjt)
                    qjt = qjt / norm if norm > 1e-12 else np.array([1.,0.,0.,0.])
                    R = quat_to_matrix(qjt)
                    sb = float(np.clip(-R[2, 0], -1.0, 1.0))
                    b  = np.arcsin(sb)
                    cb = np.cos(b)
                    if abs(cb) > 1e-6:
                        a = np.arctan2(R[1, 0] / cb, R[0, 0] / cb)
                        g = np.arctan2(R[2, 1] / cb, R[2, 2] / cb)
                    else:
                        a = np.arctan2(-R[0, 1], R[1, 1])
                        g = 0.0
                    jstates[midx][jname] = np.array([a, b, g])   # radians, visual model converts
                    idx += 4
                elif jt == 'spherical' and dof == 3:
                    jstates[midx][jname] = q[idx:idx+3]
                    idx += 3
                elif jt == 'revolute' and dof == 1:
                    # Map the single revolute DOF to the correct ZYX Euler slot
                    # (Z=slot 0, Y=slot 1, X=slot 2) so that the visual model's
                    # get_joint_rotation_matrix rotates about the right axis.
                    euler_angles = np.zeros(3)
                    # Look up the pre-computed child-frame axis for this joint
                    axis_slot = 0  # default: Z-rotation
                    for k_body, body_k in enumerate(self.bodies):
                        if body_k.model_idx != midx:
                            continue
                        vis_k = body_k.name.split('_', 1)[1]
                        vis_obj_k = self.models[midx].bodies.get(vis_k)
                        if (vis_obj_k is not None and
                                vis_obj_k.joint_name_to_parent == jname):
                            # Always ZYX slot 0. The visual model composes
                            # parent @ T1 @ R(euler) @ T2_inv, the same product
                            # the dynamics poses use, where the joint rotation
                            # is Rz(theta) in the joint frame by construction --
                            # T1 already carries the physical axis direction, so
                            # R(euler) must be a pure Z rotation whatever that
                            # direction is. Selecting the slot from the axis
                            # vector rotated the rendered body about the joint
                            # frame's X or Y axis instead, so the setup-tab
                            # preview showed a deflected revolute joint bent in
                            # the wrong plane relative to what was simulated.
                            axis_slot = 0
                            break
                    euler_angles[axis_slot] = q[idx]
                    jstates[midx][jname] = euler_angles
                    idx += 1
                else:
                    idx += dof

        for midx, model in enumerate(self.models):
            cfg  = self.model_configs[midx]
            rpos = (root_pos[midx] if root_pos[midx] is not None
                    else np.array([float(x) for x in cfg.pos_str.split()]))
            ra = root_ang[midx]
            if len(ra) == 4:
                # Quaternion root: convert to ZYX Euler degrees for visual model
                R = quat_to_matrix(ra)
                # Extract ZYX Euler - clamp cb to [-1,1] to avoid NaN in arcsin
                sb = float(np.clip(-R[2, 0], -1.0, 1.0))
                b  = np.arcsin(sb)
                cb = np.cos(b)
                if abs(cb) > 1e-6:
                    a = np.arctan2(R[1, 0] / cb, R[0, 0] / cb)
                    g = np.arctan2(R[2, 1] / cb, R[2, 2] / cb)
                else:
                    a = np.arctan2(-R[0, 1], R[1, 1])
                    g = 0.0
                rang_deg = np.degrees([a, b, g])
            else:
                rang_deg = np.degrees(ra)
            jstates_deg = {k: np.degrees(v) for k, v in jstates[midx].items()}
            model.update_kinematics(jstates_deg, rpos, rang_deg)

        # Compute body.pos/R/quat directly from the quaternion (and, for
        # revolute joints, an exact single-axis rotation) state -- see
        # _compute_body_poses_from_state for why this must not go through
        # the Euler-angle round trip used above to drive the visual model.
        self._compute_body_poses_from_state(q)

    @staticmethod
    def _rodrigues_rotation(axis, angle):
        """
        Rotation matrix for a rotation of `angle` radians about a unit
        `axis` (Rodrigues' formula). A single-axis rotation has no
        gimbal-lock singularity for ANY axis direction (unlike a 3-angle
        ZYX Euler composition), so this is the exact, safe way to build a
        revolute joint's rotation regardless of how its axis is oriented.
        """
        axis = np.asarray(axis, dtype=float)
        n = np.linalg.norm(axis)
        if n < 1e-12:
            return np.eye(3)
        axis = axis / n
        K = skew(axis)
        return np.eye(3) + np.sin(angle) * K + (1.0 - np.cos(angle)) * (K @ K)

    def _compute_body_poses_from_state(self, q):
        """
        Forward kinematics computed DIRECTLY from the generalised
        coordinates -- sets body.pos, body.R, body.quat for every body,
        without ever routing a quaternion or revolute DOF through Euler
        angles.

        Why this exists (separate from the block above that drives the
        visual MultibodyHumanModel for the setup-tab preview):
        `update_kinematics_from_q` used to finish by extracting ZYX Euler
        angles from every joint's quaternion, feeding those degrees into
        the visual model's Rz@Ry@Rx forward-kinematics chain, and copying
        THAT rotation matrix into body.R/pos via
        body.set_state_from_transform(vis.global_transform). Even though
        the state `q` that gets time-integrated is quaternion-based and
        never itself hits a singularity, body.R/pos -- which is what
        actually drives every subsequent dynamics computation (mass
        matrix via get_world_inertia, contact detection, gravity/contact
        force application, get_velocity_at_point, ...) -- was being
        silently rebuilt through that lossy, gimbal-lock-prone Euler round
        trip on EVERY step, for EVERY body. Any joint whose relative pitch
        passed through +-90 deg (e.g. testimpact1.json's second model,
        whose root_joint is initialised at pitch=90 deg exactly) would see
        the extracted alpha/gamma flip discontinuously as cos(beta)
        drifted across the `abs(cb) > 1e-6` branch threshold on floating-
        point noise alone -- producing a real, visible jolt (an
        oscillating, sign-flipping spurious angular velocity) even though
        the underlying quaternion state was perfectly smooth throughout.

        This method instead composes each body's world transform directly:
        quaternion -> matrix for root/spherical joints, and an exact
        single-axis Rodrigues rotation for revolute joints -- both
        singularity-free for any orientation/axis, so body.R/pos stay
        smooth no matter what the body's pitch is.
        """
        n = len(self.bodies)
        T_origin = [None] * n   # 4x4 world transform of each body's ORIGIN frame

        order, visited = [], [False] * n
        stack = [i for i, p in enumerate(self.parent_idx) if p == -1]
        while stack:
            i = stack.pop()
            if visited[i]:
                continue
            visited[i] = True
            order.append(i)
            stack.extend(self.children[i])

        for i in order:
            body = self.bodies[i]

            if self.parent_idx[i] == -1:
                midx = body.model_idx
                root_entry = next(
                    ((jn, ji, dof) for (m, jn, ji, dof) in self.joint_list
                     if m == midx and ji.get('is_root_joint', False)), None)
                if root_entry is None:
                    continue
                jn, ji, dof = root_entry
                s, _ = self.joint_dof_map[(midx, jn)]
                if dof == 7:
                    pos_origin = q[s:s + 3]
                    quat = q[s + 3:s + 7]
                    norm = np.linalg.norm(quat)
                    quat = quat / norm if norm > 1e-12 else np.array([1., 0., 0., 0.])
                    R = quat_to_matrix(quat)
                elif dof == 3:
                    # Legacy Euler-angle root DOF (not used by any bundled
                    # model). This representation is itself Euler-based --
                    # there's no quaternion to fall back on, so it retains
                    # the original gimbal-lock-prone behaviour, same as
                    # before, for this otherwise-unused code path only.
                    a, b, g = q[s:s + 3]
                    ca, sa = np.cos(a), np.sin(a)
                    cb, sb = np.cos(b), np.sin(b)
                    cg, sg = np.cos(g), np.sin(g)
                    R = np.array([
                        [ca*cb, ca*sb*sg - sa*cg, ca*sb*cg + sa*sg],
                        [sa*cb, sa*sb*sg + ca*cg, sa*sb*cg - ca*sg],
                        [-sb,   cb*sg,             cb*cg           ]
                    ])
                    cfg = self.model_configs[midx]
                    pos_origin = np.array([float(x) for x in cfg.pos_str.split()])
                elif dof == 0:
                    # Welded to inertial space: pose is constant.
                    T = np.asarray(ji['T1']) @ np.asarray(ji['T2_inv'])
                    T_origin[i] = T
                    Rg = body.orthonormalize_rotation(T[:3, :3])
                    body.R = Rg
                    body.quat = matrix_to_quat(Rg)
                    body.pos = T[:3, 3] + Rg @ body.cg_local
                    continue
                elif dof == 1:
                    # Revolute to ground: the "parent" is inertial space, so
                    # the world transform is simply T1 @ Rz(q) @ T2_inv.
                    T1r, T2ir = ji['T1'], ji['T2_inv']
                    R4 = np.eye(4)
                    R4[:3, :3] = self._rodrigues_rotation(
                        np.array([0.0, 0.0, 1.0]), q[s])
                    T_origin[i] = T1r @ R4 @ T2ir
                    T = T_origin[i]
                    Rg = body.orthonormalize_rotation(T[:3, :3])
                    body.R = Rg
                    body.quat = matrix_to_quat(Rg)
                    body.pos = T[:3, 3] + Rg @ body.cg_local
                    continue
                else:
                    continue
                T = np.eye(4)
                T[:3, :3] = R
                T[:3, 3] = pos_origin
                T_origin[i] = T
            else:
                pi = self.parent_idx[i]
                Tp = T_origin[pi]
                if Tp is None:
                    continue
                T1  = self.joint_T1[i]
                T2i = self.joint_T2_inv[i]
                jtype = self.joint_type[i]
                s, dof = self.joint_start_idx[i], self.joint_dof[i]

                R_joint = np.eye(3)
                if jtype == 'spherical' and dof == 4 and s != -1:
                    quat = q[s:s + 4]
                    norm = np.linalg.norm(quat)
                    quat = quat / norm if norm > 1e-12 else np.array([1., 0., 0., 0.])
                    R_joint = quat_to_matrix(quat)
                elif jtype == 'spherical' and dof == 3 and s != -1:
                    # Legacy Euler spherical joint (dead code path in the
                    # current bundled models -- add_model always assigns
                    # dof=4/quaternion to 'spherical' joints).
                    a, b, g = q[s:s + 3]
                    ca, sa = np.cos(a), np.sin(a)
                    cb, sb = np.cos(b), np.sin(b)
                    cg, sg = np.cos(g), np.sin(g)
                    R_joint = np.array([
                        [ca*cb, ca*sb*sg - sa*cg, ca*sb*cg + sa*sg],
                        [sa*cb, sa*sb*sg + ca*cg, sa*sb*cg - ca*sg],
                        [-sb,   cb*sg,             cb*cg           ]
                    ])
                elif jtype == 'revolute' and dof == 1 and s != -1:
                    # R_joint sits BETWEEN T1 and T2_inv, so it acts in the
                    # JOINT frame -- where the rotation axis is (0,0,1) by
                    # construction (T1's Z column is what defines the axis).
                    #
                    # The previous code passed T1[:3,2], which is that same
                    # axis expressed in PARENT coordinates, into this
                    # joint-frame slot.  Whenever T1 carried any rotation the
                    # body then turned about the wrong axis entirely -- e.g.
                    # an axis of [0,1,0] requested in openmbd_model_editor.html
                    # produced rotation about world -X.
                    #
                    # (Equivalently one may keep Rodrigues(T1[:3,2], q) and
                    # move it BEFORE T1, since T1 @ Rz(q) == Rodrigues(T1[:,2], q) @ T1.)
                    R_joint = self._rodrigues_rotation(
                        np.array([0.0, 0.0, 1.0]), q[s])
                # 'fixed' (or anything else): R_joint stays identity

                R4 = np.eye(4)
                R4[:3, :3] = R_joint
                T_origin[i] = Tp @ T1 @ R4 @ T2i

            T = T_origin[i]
            if T is None:
                continue
            R = body.orthonormalize_rotation(T[:3, :3])
            body.R = R
            body.quat = matrix_to_quat(R)
            body.pos = T[:3, 3] + R @ body.cg_local

    # ------------------------------------------------------------------
    # Jacobians  A1 (linear vel) and A2 (angular vel)
    # ------------------------------------------------------------------

    def compute_a1_a2_analytic(self, q, qdot):
        """
        Virtual-power Jacobians (world frame).
          A1[i]  is the 3 x nq matrix  dv_CG_i / dqdot_j
          A2[i]  is the 3 x nq matrix  domega_i / dqdot_j
        """
        nb, nq = len(self.bodies), self.nq
        A1 = np.zeros((nb, 3, nq))
        A2 = np.zeros((nb, 3, nq))
        roots = [i for i, p in enumerate(self.parent_idx) if p == -1]

        def recurse(i):
            pi = self.parent_idx[i]
            pb = self.bodies[pi]
            cb = self.bodies[i]
            T1, T2  = self.joint_T1[i], self.joint_T2[i]
            jtype   = self.joint_type[i]
            s, dof  = self.joint_start_idx[i], self.joint_dof[i]


            cj_world = cb.R @ (T2[:3, 3] - cb.cg_local)   # joint pos rel to child CG, world frame
            r_pc     = cb.pos - pb.pos                      # parent CG → child CG (world frame)

            # Propagate parent Jacobian columns (single BLAS call)
            A1[i] = A1[pi] - skew(r_pc) @ A2[pi]
            A2[i] = A2[pi].copy()

            if jtype == 'revolute' and dof == 1 and s != -1:
                ax = cb.R @ self.joint_axis_local[i]
                A2[i][:, s] += ax

                A1[i][:, s] -= np.cross(ax, cj_world)

            elif jtype == 'spherical' and dof == 4 and s != -1:
                # qdot[s:s+3] = omega in child LOCAL frame.
                # omega_world = R_child @ omega_local  =>  Jacobian cols = R_child columns
                E_world = cb.R                          # 3x3: maps local omega to world
                A2[i][:, s:s + 3] += E_world
                A1[i][:, s:s + 3] += skew(cj_world) @ E_world
                # slot s+3 is the quaternion norm constraint — zero column (already zero)

            elif jtype == 'spherical' and dof == 3 and s != -1:
                E_world = cb.R @ self._E_body_zyx(q[s:s + 3])
                A2[i][:, s:s + 3] += E_world
                A1[i][:, s:s + 3] += skew(cj_world) @ E_world

            for ch in self.children[i]:
                recurse(ch)

        for r in roots:
            midx  = self.bodies[r].model_idx
            rkey  = next(((m, jn) for (m, jn, ji, _) in self.joint_list
                          if m == midx and ji.get('is_root_joint', False)), None)

            if rkey is not None:
                s, dof = self.joint_dof_map[rkey]

                if dof == 7:
                    r_cg_world = self.bodies[r].pos - q[s:s + 3]
                    A1[r, :, s:s + 3] = np.eye(3)
                    # omega_world = qdot[s+3:s+6] directly (no E matrix needed)
                    A2[r, :, s + 3:s + 6]  = np.eye(3)
                    A1[r, :, s + 3:s + 6] -= skew(r_cg_world)

                elif dof == 3:
                    r_cg_world = self.bodies[r].pos               # joint at origin
                    Ew = self._E_world_zyx(q[s:s + 3])
                    A2[r, :, s:s + 3]  = Ew
                    A1[r, :, s:s + 3] -= skew(r_cg_world) @ Ew

                elif dof == 1:
                    ji_r = next(ji for (m, jn, ji, _) in self.joint_list
                                if m == midx and ji.get('is_root_joint', False))
                    axw, pj = self._root_revolute_frame(r, ji_r)
                    A2[r, :, s] = axw
                    A1[r, :, s] = np.cross(axw, self.bodies[r].pos - pj)

            for ch in self.children[r]:
                recurse(ch)

        return A1, A2

    # ------------------------------------------------------------------
    # Recursive Newton-Euler (bias torques, qddot = 0)
    # ------------------------------------------------------------------

    def rnea(self, q, qdot):
        """
        Bias generalised forces: gravity + contacts (normal + friction) + Coriolis.
        Assumes update_kinematics_from_q(q) and
        _update_body_velocities_from_qdot(q, qdot) have already been called.
        """
        nb = len(self.bodies)
        v     = [np.zeros(3) for _ in range(nb)]
        omega = [np.zeros(3) for _ in range(nb)]
        a     = [np.zeros(3) for _ in range(nb)]
        alpha = [np.zeros(3) for _ in range(nb)]

        order, visited = [], [False]*nb
        stack = [i for i, p in enumerate(self.parent_idx) if p == -1]
        while stack:
            i = stack.pop()
            if visited[i]: continue
            visited[i] = True; order.append(i)
            stack.extend(self.children[i])

        # ── Forward pass ──────────────────────────────────────────────
        for i in order:
            body = self.bodies[i]
            if self.parent_idx[i] == -1:
                midx = body.model_idx
                for (m, jn, ji, dof) in self.joint_list:
                    if m == midx and ji.get('is_root_joint', False):
                        s, _ = self.joint_dof_map[(m, jn)]
                        qd = qdot[s:s+dof]
                        if dof == 7:
                            omega[i] = qd[3:6]
                            r_jcg = body.pos - q[s:s+3]
                            v[i] = qd[:3] + np.cross(omega[i], r_jcg)
                            a[i] = np.cross(omega[i], np.cross(omega[i], r_jcg))
                        elif dof == 3:
                            ang = q[s:s+3]
                            omega[i] = self._E_world_zyx(ang) @ qd
                            # joint at world origin; r = body.pos
                            r_jcg = body.pos
                            v[i] = np.cross(omega[i], r_jcg)
                            a[i] = np.cross(omega[i], np.cross(omega[i], r_jcg))
                        elif dof == 1:
                            axw, pj = self._root_revolute_frame(i, ji)
                            omega[i] = axw * qd[0]
                            r_jcg = body.pos - pj
                            v[i] = np.cross(omega[i], r_jcg)
                            a[i] = np.cross(omega[i], np.cross(omega[i], r_jcg))
                        elif dof == 0:
                            omega[i] = np.zeros(3); v[i] = np.zeros(3); a[i] = np.zeros(3)
                        alpha[i] = np.zeros(3)
                        break
                continue

            pi   = self.parent_idx[i]
            pb   = self.bodies[pi]
            jtype = self.joint_type[i]
            s    = self.joint_start_idx[i]
            dof  = self.joint_dof[i]
            qd_j = qdot[s:s+dof] if s != -1 else np.zeros(max(dof, 1))

            T1, T2  = self.joint_T1[i], self.joint_T2[i]
            pj_world = pb.pos + pb.R @ (T1[:3,3] - pb.cg_local)
            r_pj     = pj_world - pb.pos
            cj_world = body.R @ (T2[:3,3] - body.cg_local)
            v_par_jnt = v[pi] + np.cross(omega[pi], r_pj)

            if jtype == 'revolute' and dof == 1:
                ax = body.R @ self.joint_axis_local[i]
                omega[i] = omega[pi] + ax * qd_j[0]
               
                v[i]     = v_par_jnt + np.cross(omega[i], -cj_world)
                a_pj     = (a[pi]
                            + np.cross(alpha[pi], r_pj)
                            + np.cross(omega[pi], np.cross(omega[pi], r_pj)))
                # alpha_i = alpha_parent + omega_parent × (ax * qdot)
                #           [Coriolis: axis rotates with parent frame]
                alpha[i] = alpha[pi] + np.cross(omega[pi], ax * qd_j[0])
                a[i]     = (a_pj
                            + np.cross(alpha[i], -cj_world)
                            + np.cross(omega[i], np.cross(omega[i], -cj_world)))

            elif jtype == 'spherical' and dof == 4:
                # qdot[s:s+3] = omega_local (body frame)
                omega_rel = body.R @ qd_j[:3]
                omega[i]  = omega[pi] + omega_rel
                v[i]      = v_par_jnt + np.cross(omega[i], -cj_world)

                # Coriolis bias: alpha_rel = omega_parent x omega_rel (transport term)
                alpha_rel  = np.cross(omega[pi], omega_rel)

                a_pj = (a[pi]
                        + np.cross(alpha[pi], r_pj)
                        + np.cross(omega[pi], np.cross(omega[pi], r_pj)))
                alpha[i] = alpha[pi] + alpha_rel
                a[i]     = (a_pj
                            + np.cross(alpha[i], -cj_world)
                            + np.cross(omega[i], np.cross(omega[i], -cj_world)))

            elif jtype == 'spherical' and dof == 3:
                ang = q[s:s+3]
                E   = self._E_body_zyx(ang)
                omega_rel = body.R @ (E @ qd_j)
                omega[i]  = omega[pi] + omega_rel
                
                v[i]      = v_par_jnt + np.cross(omega[i], -cj_world)

                bias_local = self._bias_local_zyx(ang, qd_j)
                alpha_rel  = body.R @ bias_local

                a_pj = (a[pi]
                        + np.cross(alpha[pi], r_pj)
                        + np.cross(omega[pi], np.cross(omega[pi], r_pj)))
                alpha[i] = (alpha[pi]
                            + alpha_rel
                            + np.cross(omega[pi], omega_rel))
                a[i]     = (a_pj
                            + np.cross(alpha[i], -cj_world)
                            + np.cross(omega[i], np.cross(omega[i], -cj_world)))

            else:  # fixed
                # See _update_body_velocities_from_qdot: r_pc = r_pj - cj_world.
                r_tot    = r_pj - cj_world
                omega[i] = omega[pi]
                v[i]     = v[pi] + np.cross(omega[pi], r_tot)
                alpha[i] = alpha[pi]
                a[i]     = (a[pi]
                            + np.cross(alpha[pi], r_tot)
                            + np.cross(omega[pi], np.cross(omega[pi], r_tot)))

        # Keep the velocity-product (qddot = 0) accelerations: the TOTAL body
        # acceleration is  a_i = A1[i] @ qddot + a_bias[i]  (likewise alpha),
        # which _compute_body_accelerations needs.
        self._bias_lin_acc = a
        self._bias_ang_acc = alpha

        # ── Backward pass ─────────────────────────────────────────────
        f   = [np.zeros(3) for _ in range(nb)]
        tau = [np.zeros(3) for _ in range(nb)]
        for i in reversed(order):
            body     = self.bodies[i]
            f_star   = body.mass * a[i]
            Iw       = body.get_world_inertia()
            tau_star = Iw @ alpha[i] + np.cross(omega[i], Iw @ omega[i])

            # get_applied_force includes gravity + normal + friction 
            f[i]   = self.get_applied_force(body, self.time)  - f_star
            tau[i] = self.get_applied_moment(body, self.time) - tau_star


            for ch in self.children[i]:
                r = self.bodies[ch].pos - body.pos   # child CG − parent CG (world frame)
                f[i]   += f[ch]
                tau[i] += tau[ch] + np.cross(r, f[ch])

        # ── Project to generalised torques ────────────────────────────
        tau_joint = np.zeros(self.nq)
        for i in range(nb):
            if self.parent_idx[i] == -1:
                continue
            s   = self.joint_start_idx[i]
            dof = self.joint_dof[i]
            if s == -1:
                continue
            body = self.bodies[i]
            pb   = self.bodies[self.parent_idx[i]]
            T1   = self.joint_T1[i]
            pj_l = T1[:3,3] - pb.cg_local
            pj_w = pb.pos + pb.R @ pj_l
            M_jnt = tau[i] + np.cross(body.pos - pj_w, f[i])

            if self.joint_type[i] == 'revolute':
                ax = body.R @ self.joint_axis_local[i]
                tau_joint[s] = np.dot(M_jnt, ax)

            elif self.joint_type[i] == 'spherical' and dof == 4:
                # omega_local is the true DOF; project torque to local frame.
                # Generalised torque = R^T @ M_jnt  (inverse of R @ omega_local)
                tau_joint[s:s+3] = body.R.T @ M_jnt
                # slot s+3 stays zero (norm constraint)

            elif self.joint_type[i] == 'spherical' and dof == 3:
                E = self._E_body_zyx(q[s:s+3])
                M_local = body.R.T @ M_jnt
                tau_joint[s:s+3] = E.T @ M_local

            elif self.joint_type[i] == 'free' and dof == 7:
                tau_joint[s:s+3] = f[i]
                tau_joint[s+3:s+6] = M_jnt   # omega is direct DOF, projection = identity

        # ── Project root-body generalised forces  ─────────────

        for i in order:
            if self.parent_idx[i] != -1:
                continue                # non-root: already handled above
            midx = self.bodies[i].model_idx
            for (m, jn, ji, dof) in self.joint_list:
                if m != midx or not ji.get('is_root_joint', False):
                    continue
                s, _ = self.joint_dof_map[(m, jn)]
                body  = self.bodies[i]
                if dof == 7:
                    # Translational: generalised force = world force on root body
                    tau_joint[s:s+3] = f[i]
                    # Rotational: omega_world is a direct DOF, so projection is identity
                    M_root = tau[i] + np.cross(body.pos - q[s:s+3], f[i])
                    tau_joint[s+3:s+6] = M_root
                    # slot s+6 is the quaternion-norm constraint row — leave zero
                elif dof == 3:
                    ang = q[s:s+3]
                    Ew  = self._E_world_zyx(ang)
                    tau_joint[s:s+3] = Ew.T @ tau[i]
                elif dof == 1:
                    axw, pj = self._root_revolute_frame(i, ji)
                    M_jnt = tau[i] + np.cross(body.pos - pj, f[i])
                    tau_joint[s] = np.dot(M_jnt, axw)
                break   # only one root joint per model

        return tau_joint

    # ------------------------------------------------------------------
    # Prescribed joint torques
    # ------------------------------------------------------------------

    def _compute_prescribed_torques(self, t):
        """
        Return a generalised-force vector contribution from all active
        prescribed torque entries at simulation time t.

        Each entry in self.prescribed_torques is a dict:
            model_idx  : int        – which loaded model this applies to
            joint_name : str        – must match a key in joint_dof_map
            torque     : ndarray    – (dof,) N·m, ZYX axis order
            t_start    : float      – simulation time to begin applying (s)
            duration   : float      – how long to apply (s); 0 = one step

        The torque is added directly to the matching slice of the
        generalised force vector B.  For spherical joints the torque
        vector is projected through E^T (same as joint-limit torques),
        so the units remain N·m in generalised coordinates.
        """
        tau = np.zeros(self.nq)
        for entry in self.prescribed_torques:
            t_start  = entry['t_start']
            duration = float(entry['duration'])
            # A duration of 0 used to mean "one integration step", so the
            # delivered angular impulse was tau*dt -- it silently changed by a
            # factor of 10 when the user changed dt from 1e-4 to 1e-5, which
            # makes any result using it irreproducible.  Substitute a fixed,
            # dt-independent minimum pulse width instead.
            if duration <= 0.0:
                duration = self.MIN_TORQUE_PULSE
                if not getattr(self, '_warned_zero_duration', False):
                    import warnings
                    warnings.warn(
                        f"prescribed torque on '{entry['joint_name']}' has "
                        f"duration=0; using MIN_TORQUE_PULSE="
                        f"{self.MIN_TORQUE_PULSE} s so the delivered impulse "
                        f"does not depend on dt", RuntimeWarning)
                    self._warned_zero_duration = True
            t_end = t_start + duration
            if not (t_start <= t < t_end):
                continue

            key = (entry['model_idx'], entry['joint_name'])
            if key not in self.joint_dof_map:
                continue
            s, dof = self.joint_dof_map[key]

            trq = np.asarray(entry['torque'], dtype=float)
            # Half-sine envelope: the specified magnitude is the PEAK.
            #   scale = sin(pi * (t - t_start) / duration)
            # Rises smoothly 0 -> peak at mid-pulse -> 0, avoiding the
            # integrator transients caused by a rectangular step.
            # NOTE: the delivered angular impulse is therefore
            #   (2/pi) * peak * duration  ~=  0.6366 * peak * duration,
            # not peak*duration.  Size `torque` accordingly.
            phase = (t - t_start) / duration
            scale = np.sin(np.pi * phase)

            if dof == 1:
                # Revolute joint: the user supplies a ZYX torque vector
                # [τZ, τY, τX].  The generalised force is the scalar projection
                # of that torque onto the joint axis.  The axis stored in
                # joint_axis_local[k] is in child-body local frame; we need
                # it in world frame to dot with the world-frame torque vector.
                # Find the matching body index to look up the axis.
                tau_scalar = float(trq[0])   # fallback: Z-component only
                midx = entry['model_idx']
                jname = entry['joint_name']
                for k_b, body_k in enumerate(self.bodies):
                    if body_k.model_idx != midx:
                        continue
                    vis_k = body_k.name.split('_', 1)[1]
                    vis_obj_k = self.models[midx].bodies.get(vis_k)
                    if (vis_obj_k is not None and
                            vis_obj_k.joint_name_to_parent == jname):
                        ax_local = self.joint_axis_local[k_b]
                        if ax_local is not None:
                            ax_world = body_k.R @ ax_local
                            # Build a ZYX Euler basis from the torque slots
                            # so the dot product is axis-aligned correctly:
                            #   trq = [τZ, τY, τX]  →  τ_world = [0,0,τZ] + [0,τY,0] + [τX,0,0]
                            #   but we just project: τ_gen = ax_world · trq_world
                            # trq_world assumes the torque is already in world axes:
                            # trq[0] acts about world Z, trq[1] about world Y, trq[2] about X
                            trq_world = np.array([
                                float(trq[2]) if len(trq) > 2 else 0.0,  # X
                                float(trq[1]) if len(trq) > 1 else 0.0,  # Y
                                float(trq[0]) if len(trq) > 0 else 0.0,  # Z
                            ])
                            tau_scalar = float(np.dot(ax_world, trq_world))
                        break
                tau[s] += tau_scalar * scale
            elif dof == 4:
                # Quaternion spherical joint: qdot[s:s+3] are the 3 real
                # angular-velocity DOFs (omega_local, child LOCAL x,y,z
                # axes); slot s+3 is the quaternion-norm constraint and
                # carries no generalised force. `trq` is supplied in the
                # same [tauZ,tauY,tauX] label order as joint_vels/angles
                # (see the revolute-joint case above, and the reorder
                # applied to joint_vels in _initialize_state_from_config),
                # so reorder to (X,Y,Z) before it lands on the
                # (omega_x,omega_y,omega_z) generalised-force slots.
                trq_zyx = trq[:3] if len(trq) >= 3 else np.pad(trq, (0, 3 - len(trq)))
                trq_xyz = trq_zyx[::-1]
                tau[s:s + 3] += trq_xyz * scale
            elif dof == 7:
                # Root free joint: qdot[s:s+3] is LINEAR velocity (conjugate
                # generalised force = a literal FORCE); qdot[s+3:s+6] is the
                # ABSOLUTE angular velocity in fixed world axes (conjugate
                # generalised force = a TORQUE); qdot[s+6] is the
                # quaternion-norm constraint slot (unused). A prescribed
                # "torque" must land entirely on the angular slots
                # (s+3:s+6) -- previously it fell into the generic branch
                # below and was written starting at tau[s], i.e. applied as
                # a translational FORCE instead of a rotational TORQUE (this
                # was verified experimentally: a 300 N*m pulse produced a
                # large linear root velocity of order 1 m/s instead of
                # driving the intended spin). It also needs the same
                # [tauZ,tauY,tauX] -> (X,Y,Z) reorder used for the spherical
                # case just above and for the root's initial joint_vels in
                # _initialize_state_from_config.
                trq_zyx = trq[:3] if len(trq) >= 3 else np.pad(trq, (0, 3 - len(trq)))
                trq_xyz = trq_zyx[::-1]
                tau[s + 3:s + 6] += trq_xyz * scale
            else:
                n = min(dof, len(trq))
                tau[s:s + n] += trq[:n] * scale

        return tau

    # ------------------------------------------------------------------
    # Mass matrix + generalised force assembly
    # ------------------------------------------------------------------

    def assemble_A_and_B(self, q, qdot, t):
        nq = self.nq
        A1, A2 = self.compute_a1_a2_analytic(q, qdot)
        self._last_jacobians = (A1, A2)   # reused by _compute_body_accelerations

        A = np.zeros((nq, nq))
        for i, body in enumerate(self.bodies):
            Ai1 = A1[i]
            Ai2 = A2[i]
            Iw  = body.get_world_inertia()
            A += body.mass * (Ai1.T @ Ai1)
            A += Ai2.T @ (Iw @ Ai2)

        # Tikhonov regularisation.  1e-4 was chosen for the legacy dof=3 Euler
        # root (det(E_world) ~ 0.087 at beta = 85 deg), a code path no bundled
        # model uses.  With quaternion roots/joints A is well conditioned, and
        # 1e-4 is ~12 % of the smallest true diagonal (wrist roll,
        # A_ii ~ 8.4e-4 kg m^2), which visibly distorts the distal segments.
        A += 1e-6 * np.eye(nq)

        # ── Passive joint damping, applied IMPLICITLY ─────────────────
        # Explicit damping (B += -D q̇, then q̇ += dt A⁻¹B) is only stable for
        # D < 2 A_ii / dt.  At dt = 1e-4 the wrist roll DOFs give
        # D_max = 16.9 N·m·s/rad, well under JOINT_PASSIVE_D = 40, so the
        # dashpot itself diverged and spun the whole model up in free fall.
        # Substituting q̇_new = q̇ + dt·q̈ into A q̈ = B_other - D q̇_new gives
        #     (A + dt·D) q̈ = B_other - D q̇
        # which is unconditionally stable for any D and dt.
        D = self._passive_damping_diag()
        A[np.diag_indices(nq)] += self.dt * D

        B  = self.rnea(q, qdot)
        B += self.compute_joint_limit_torques(q, qdot)
        B += -D * qdot
        B += self._compute_prescribed_torques(self.time)
        return A, B

    # ------------------------------------------------------------------
    # Integration  (Symplectic Euler)
    # ------------------------------------------------------------------

    def step(self):
        """
        Advance one dt using Symplectic (semi-implicit) Euler.

        Step order:
          1. update_kinematics_from_q(q)              body.pos, body.R
          2. _update_body_velocities_from_qdot(q, qd)  body.vel, body.ang_vel [BF-B]
          3. detect_contacts()
          4. _compute_all_contact_scales()             Fix 4: symmetric N3L scales
          5. assemble_A_and_B  (uses body.vel via get_applied_force)
          6. qddot = A^-1 B
          7. _compute_body_accelerations               Fix 5: uses pre-integration q/qddot
          8. _accumulate_contact_forces_on_bodies      Fix 5: uses pre-integration body.vel
          9. record_state (if due)
         10. Symplectic Euler: qdot_new = qdot + dt*qddot
                               q_new   = q    + dt*qdot_new
         11. update kinematics & velocities for display
        """
        nq = self.nq
        q    = self.state[:nq].copy()
        qdot = self.state[nq:].copy()

        # Steps 1–3: kinematics and contact detection at time t
        self.update_kinematics_from_q(q)
        self._update_body_velocities_from_qdot(q, qdot)   # BF-B: body.vel = v(t)
        self.detect_contacts()

        # Step 4: compute symmetric contact scales once (Fix 4)
        self._compute_all_contact_scales()

        # Steps 5–6: equations of motion
        A, B = self.assemble_A_and_B(q, qdot, self.time)
        try:
            qddot = np.linalg.solve(A, B)
        except np.linalg.LinAlgError:
            qddot = np.linalg.pinv(A) @ B

        qddot = np.clip(qddot, -1e5, 1e5)

        # Steps 7–8: record forces using pre-integration state (Fix 5)
        # body.vel is still v(t) here — temporally consistent with the forces
        # that were passed to assemble_A_and_B above.
        self._compute_body_accelerations(q, qddot)
        self._accumulate_contact_forces_on_bodies()

        # Step 9: record state if due (uses the freshly accumulated forces)
        if self.recording and self.step_count % self.record_every == 0:
            self.record_state()

        # Step 10: Symplectic Euler integration
        qdot_new = qdot + self.dt * qddot

        q_new = self._integrate_positions(q, qdot_new, self.dt)

        self.state     = np.concatenate([q_new, qdot_new])
        self.time     += self.dt
        self.step_count += 1

        # Step 11: update display state to t+dt (body.vel = v(t+dt))
        self.update_kinematics_from_q(q_new)
        self._update_body_velocities_from_qdot(q_new, qdot_new)

    def _integrate_positions(self, q, qdot_new, dt):
        """
        Position update of Symplectic Euler: q_new = q + dt * qdot_new, with
        the quaternion DOFs (free root, spherical joints) advanced on SO(3)
        and re-normalised.  Factored out of step() so the kinematic mapping
        can be verified independently of the dynamics.
        """
        q_new = q.copy()
        for (midx, jname, jinfo, dof) in self.joint_list:
            s, _ = self.joint_dof_map[(midx, jname)]
            if jinfo.get('is_root_joint', False) and dof == 7:
                # Translation: Euler as normal
                q_new[s:s+3] = q[s:s+3] + dt * qdot_new[s:s+3]
                # ROOT quaternion kinematics: qdot[s+3:s+6] is an ABSOLUTE
                # angular velocity resolved in the fixed GLOBAL/world axes
                # (see _update_body_velocities_from_qdot, rnea, and
                # compute_a1_a2_analytic, which all consume it directly as
                # omega_world with no rotation applied -- A2[:,s+3:s+6] is
                # literally the identity Jacobian). The correct ODE for a
                # quaternion driven by a WORLD-frame angular velocity is the
                # LEFT quaternion product:
                #     dq/dt = 0.5 * Omega(omega_world) (x) q
                # i.e. omega as the pure quaternion (0,ox,oy,oz) multiplied
                # on the LEFT of q. The previous code used the RIGHT product
                # (0.5 * q (x) omega) instead -- that formula is only valid
                # for a BODY-LOCAL angular velocity (it's the correct one
                # used a few lines below for CHILD spherical joints, whose
                # omega is genuinely local/relative). Using the body-frame
                # formula here effectively resolved the root's "world-frame"
                # spin through the body's own current (rotating) axes, so a
                # fixed-world-axis spin command produced completely
                # different motion depending on the body's initial/current
                # orientation (verified with testspin1-4.json: identical
                # world-Y omega=12 rad/s decayed to ~3 rad/s and leaked into
                # X/Z for yaw=+/-90 deg initial poses, but was preserved for
                # yaw=0/180 deg -- a clear orientation-dependent artifact of
                # the wrong multiplication order, not real dynamics).
                qw, qx, qy, qz = q[s+3:s+7]
                ox, oy, oz = qdot_new[s+3:s+6]
                dqw = 0.5 * (-ox*qx - oy*qy - oz*qz)
                dqx = 0.5 * ( ox*qw + oy*qz - oz*qy)
                dqy = 0.5 * ( oy*qw + oz*qx - ox*qz)
                dqz = 0.5 * ( oz*qw + ox*qy - oy*qx)
                new_q = np.array([qw + dt*dqw,
                                  qx + dt*dqx,
                                  qy + dt*dqy,
                                  qz + dt*dqz])
                q_new[s+3:s+7] = new_q / np.linalg.norm(new_q)   # re-normalise
            elif dof == 4 and not jinfo.get('is_root_joint', False):
                # Spherical joint stored as quaternion [qw,qx,qy,qz].
                # qdot[s:s+3] = omega in child LOCAL frame; qdot[s+3] = 0 (norm slot).
                qw, qx, qy, qz = q[s:s+4]
                # qdot[s:s+3] is the RELATIVE angular velocity resolved in the
                # CHILD body frame (the dynamics use omega_rel_world =
                # R_child @ qdot[s:s+3]; see _update_body_velocities_from_qdot,
                # rnea and compute_a1_a2_analytic).  The stored quaternion,
                # however, is the JOINT rotation R_j in
                #     R_child = R_parent @ R_T1 @ R_j @ R_T2^T,
                # whose own body-frame rate omega_j satisfies
                #     R_parent @ R_T1 @ R_j @ omega_j = R_child @ R_T2 @ omega_j.
                # Hence omega_j = R_T2^T @ omega_child.  Feeding omega_child
                # straight into dq/dt = 0.5 q (x) omega is only valid when T2
                # carries no rotation.  Every joint in the bundled human models
                # has R_T2 = diag(1,-1,-1), so the Y and Z components were
                # integrated with the WRONG SIGN: the pose rotated opposite to
                # the velocity the equations of motion were solving for.
                ox, oy, oz = self._joint_R2T(jinfo) @ qdot_new[s:s+3]
                dqw = 0.5 * (-qx*ox - qy*oy - qz*oz)
                dqx = 0.5 * ( qw*ox - qz*oy + qy*oz)
                dqy = 0.5 * ( qz*ox + qw*oy - qx*oz)
                dqz = 0.5 * (-qy*ox + qx*oy + qw*oz)
                new_q = np.array([qw + dt*dqw,
                                  qx + dt*dqx,
                                  qy + dt*dqy,
                                  qz + dt*dqz])
                q_new[s:s+4] = new_q / np.linalg.norm(new_q)   # re-normalise
            else:
                s, dof_j = self.joint_dof_map[(midx, jname)]
                q_new[s:s+dof_j] = q[s:s+dof_j] + dt * qdot_new[s:s+dof_j]

        return q_new

    # ------------------------------------------------------------------
    # Per-body acceleration and contact force helpers
    # ------------------------------------------------------------------

    def _compute_body_accelerations(self, q, qddot):
        """
        Map generalised accelerations qddot -> per-body Cartesian accelerations.

          a_CG_i  = A1[i] @ qddot + a_bias_i
          alpha_i = A2[i] @ qddot + alpha_bias_i

        where the bias terms are the velocity-product (centripetal, Coriolis
        and joint-transport) accelerations from the RNEA forward pass with
        qddot = 0.  They were previously omitted, so recorded linear and
        angular accelerations were wrong whenever segments rotated -- on a
        free-flying model with ~3 rad/s joint rates the recorded values
        differed from the finite-difference acceleration by 84 m/s^2 and
        133 rad/s^2.  Head acceleration output is exactly where this matters.

        Must be called after assemble_A_and_B() for the same (q, qdot).
        Results are stored in body.lin_accel and body.ang_accel (world frame).
        """
        jac = getattr(self, '_last_jacobians', None)
        if jac is None:
            jac = self.compute_a1_a2_analytic(q, self.state[self.nq:])
        A1, A2 = jac
        a_b = getattr(self, '_bias_lin_acc', None)
        al_b = getattr(self, '_bias_ang_acc', None)
        for i, body in enumerate(self.bodies):
            body.lin_accel = A1[i] @ qddot
            body.ang_accel = A2[i] @ qddot
            if a_b is not None:
                body.lin_accel = body.lin_accel + a_b[i]
                body.ang_accel = body.ang_accel + al_b[i]

    def _accumulate_contact_forces_on_bodies(self):
        """
        Compute total contact force and torque acting on each body from all
        active contacts and store in ``body.contact_force`` /
        ``body.contact_torque`` (world frame, relative to body CG).

        Uses ``c._global_scale`` — the same symmetric scale factor used in
        ``get_applied_force()`` — so recorded forces are consistent with the
        forces that drove the equations of motion.  Gravity is excluded.

        Called *before* symplectic-Euler integration (Fix 5) so that
        ``body.vel`` reflects the pre-integration velocities that were
        actually used to assemble the equations of motion, eliminating the
        temporal inconsistency between recorded and applied forces.
        """
        for body in self.bodies:
            body.contact_force  = np.zeros(3)
            body.contact_torque = np.zeros(3)

        for body in self.bodies:
            for c in self.get_contacts_for_body(body):
                scale = getattr(c, '_global_scale', 1.0)
                if scale == 0.0:
                    continue
                magnitude, v_rel, v_rel_n = self._contact_magnitude(c)
                if magnitude <= 0.0:
                    continue

                sign = +1.0 if body is c.bodyA else -1.0
                F_n = sign * scale * magnitude * c.normal

                # Coulomb friction
                F_f = np.zeros(3)
                if c.friction > 0.0:
                    v_t     = v_rel - v_rel_n * c.normal
                    v_t_mag = np.linalg.norm(v_t)
                    if v_t_mag > 1e-6:
                        ramp = c._friction_ramp(v_t_mag)
                        F_f = -(sign * scale * c.friction * magnitude * ramp
                                * (v_t / v_t_mag))

                F_total = F_n + F_f
                r = c.point - body.pos
                body.contact_force  += F_total
                body.contact_torque += np.cross(r, F_total)

    # ------------------------------------------------------------------
    # History
    # ------------------------------------------------------------------

    def record_state(self):
        """
        Snapshot the current step into state_history for analysis/export.

        Position and orientation (quat) are recorded in WORLD axes/frame,
        as usual -- position is a point (not a free vector) and quat is the
        very definition of the body's local axes relative to world, so
        neither has a meaningful separate "local-axis projection".

        Every other recorded quantity -- linear velocity, linear
        acceleration, angular velocity, angular acceleration, contact
        force, and contact torque/moment -- is recorded as an ABSOLUTE
        (total, world-composed) value but resolved onto each body's OWN
        current LOCAL axes: x_local = R_body^T @ x_world. This matches the
        absolute/relative and world/local convention used for the live
        simulation state (root = absolute values in fixed world axes;
        child = relative values composed with the parent's world value via
        R_body @ x_relative -- see _update_body_velocities_from_qdot) while
        making the *recorded* output meaningful in each body's own frame,
        e.g. "HeadVelX"/"HeadForceX" is the head's own local-X component,
        not an arbitrary world-X component that depends on how the head
        happens to be tilted. Vector magnitudes (|Vel|, |Force|, etc.) are
        unaffected either way, since R_body is orthonormal.

        Only the OUTPUT here is reframed -- body.vel, body.ang_vel,
        body.lin_accel, body.ang_accel, body.contact_force and
        body.contact_torque themselves are left untouched in WORLD axes,
        since the rest of the engine (contact point velocities,
        inertia-tensor rotation, the quaternion ODE, force/torque
        accumulation, etc.) requires them in that frame to be correct.
        """
        body_states = []
        for b in self.bodies:
            Rt = b.R.T
            vel_world        = b.vel
            ang_vel_world    = b.ang_vel
            lin_accel_world  = getattr(b, 'lin_accel',     np.zeros(3))
            ang_accel_world  = getattr(b, 'ang_accel',     np.zeros(3))
            force_world      = getattr(b, 'contact_force',  np.zeros(3))
            torque_world     = getattr(b, 'contact_torque', np.zeros(3))

            body_states.append({
                'name': b.name, 'model_id': b.model_id,
                'pos':  b.pos.copy(),
                'quat': b.quat.copy(),
                'vel':        Rt @ vel_world,
                'ang_vel':    Rt @ ang_vel_world,
                'lin_accel':  Rt @ lin_accel_world,
                'ang_accel':  Rt @ ang_accel_world,
                'force':      Rt @ force_world,
                'torque':     Rt @ torque_world,
            })
        self.state_history.append({'time': self.time, 'body_states': body_states})
        self.contact_history.append([c.to_dict() for c in self.contacts])

    def clear_history(self):
        self.state_history  = []
        self.contact_history = []
