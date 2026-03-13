# physics_engine.py  
# Citation: Tierney. OpenMBD: An Open-Source Multibody Dynamics Simulator for Biomechanics Research and Education. F1000Research, 2026.
# Version: 1.0 
# Research Contact: Dr Gregory Tierney (g.tierney@ulster.ac.uk)

import numpy as np
from physics_constraints import SimpleContact, clear_contact_cache
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

        self.dt = 0.0001                             
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

    JOINT_PASSIVE_D = 40.0   # N*m*s / rad – within-ROM passive damping

    def _compute_passive_damping_torques(self, qdot: np.ndarray) -> np.ndarray:
        """
        Return generalised force vector Q_passive (size nq) that applies
        JOINT_PASSIVE_D viscous damping to every non-root joint DOF.

        Q_passive[s:s+dof] = −JOINT_PASSIVE_D · qdot[s:s+dof]

        This is equivalent to adding a physical dashpot across each joint
        that resists relative angular (or linear) velocity.  Root DOFs are
        excluded so translational free-fall and whole-body orientation
        remain governed solely by gravity and contact forces.

        Stability argument
        ------------------
        The explicit-Euler update  qdot += dt·(A^{-1}·B)  is stable for a
        damped oscillator only when  |1 − c·dt/A_ii| < 1, i.e.
        c < 2·A_ii / dt.  For the elbow (A_ii ≈ 0.003465 kg·m²,
        dt = 0.0001 s) this gives c_max ≈ 69.3 Nm·s/rad.
        JOINT_PASSIVE_D = 25 satisfies c < c_max with a 64 % safety margin.
        """
        tau = np.zeros(self.nq)
        for (midx, jname, jinfo, dof) in self.joint_list:
            if jinfo.get('is_root_joint', False):
                continue                     # free-floating root: no passive drag
            s, _ = self.joint_dof_map[(midx, jname)]
            tau[s:s + dof] -= self.JOINT_PASSIVE_D * qdot[s:s + dof]
        return tau

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
                self.joint_list.append((model_idx, j_name, j_info, 3))
            elif jtype == 'revolute':
                self.joint_list.append((model_idx, j_name, j_info, 1))
            elif jtype == 'free':
                # Root DOF: 3 translation + 4 quaternion (no Euler singularity)
                self.joint_list.append((model_idx, j_name, j_info, 7))

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
                    # Root angular velocity stored as world-frame omega (rad/s)
                    try:
                        ang_vel_rad = np.array([float(x) for x in
                                                getattr(config, 'ang_vel_str', '0 0 0').split()])
                    except Exception:
                        ang_vel_rad = np.zeros(3)
                    self.state[self.nq+s+3 : self.nq+s+6] = ang_vel_rad[:3]
                    # qdot[s+6] = 0  (quaternion norm constraint — not a DOF)
                elif dof == 3:
                    ang_deg = config.joints.get(jname, [0,0,0])
                    self.state[s:s+3] = np.radians(ang_deg[:3])
            else:
                ang_deg = config.joints.get(jname, [0,0,0])
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
        jt = getattr(config, 'joint_torques', {})
        for jname, spec in jt.items():
            trq      = spec.get('torque',   [0.0, 0.0, 0.0])
            t_start  = float(spec.get('t_start',  0.0))
            duration = float(spec.get('duration', 0.0))
            # Only register entries where at least one axis has a non-zero torque
            if any(abs(float(v)) > 1e-9 for v in trq):
                self.prescribed_torques.append({
                    'model_idx':  model_idx,
                    'joint_name': jname,
                    'torque':     np.array([float(v) for v in trq]),
                    't_start':    t_start,
                    'duration':   duration,
                })

    def _build_kinematic_tree(self):
        n = len(self.bodies)
        self.parent_idx      = [-1]    * n
        self.joint_type      = ['fixed']* n
        self.joint_axis_local= [None]  * n
        self.joint_T1        = [None]  * n
        self.joint_T2        = [None]  * n
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
                continue
            ji = model.joint_infos.get(vis_body.joint_name_to_parent)
            if ji is None or ji['parent'] is None:
                continue
            parent_name = f"{body.model_idx}_{ji['parent'].name}"
            if parent_name not in name_to_idx:
                continue
            self.parent_idx[i]  = name_to_idx[parent_name]
            self.joint_T1[i]    = ji['T1'].copy()
            self.joint_T2[i]    = ji['T2'].copy()
            jt = ji.get('type', 'fixed')
            self.joint_type[i]  = jt
            if jt == 'revolute':
                self.joint_axis_local[i] = np.array([0, 0, 1])
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
                elif jtype == 'spherical' and dof == 3 and s != -1:
                    ang = q[s:s+3]
                    E = self._E_body_zyx(ang)
                    omega_rel = body.R @ (E @ qd_j)
                    omega[i] = omega[pi] + omega_rel
                    # v_CG = v_joint + omega x r_jcg, r_jcg = -cj_world
                    v[i] = v_par_jnt + np.cross(omega[i], -cj_world)
                else:  # fixed
                    omega[i] = omega[pi]
                    v[i] = v[pi] + np.cross(omega[pi], r_pj + cj_world)

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
                        n1 =  self._ellipsoid_surface_normal( d, R1, r1)
                        n2 = -self._ellipsoid_surface_normal(-d, R2, r2)
                        normal = n1 + n2
                        nlen = np.linalg.norm(normal)
                        normal = normal / nlen if nlen > 1e-12 else n1
                        r1_eff = self._ellipsoid_radial_extent(r1, R1, normal)
                        r2_eff = self._ellipsoid_radial_extent(r2, R2, normal)
                        pen = (r1_eff + r2_eff) - np.dot(d, normal)
                        if pen > 0.0:
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
        for i in range(nb):
            b1 = self.bodies[i]
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

                for e1 in b1.ellipsoids:
                    T1_w = b1.get_body_transform() @ e1.local_T
                    p1, R1, r1 = T1_w[:3,3], T1_w[:3,:3], e1.dims
                    r1_bound   = float(np.max(r1))

                    for e2 in b2.ellipsoids:
                        T2_w = b2.get_body_transform() @ e2.local_T
                        p2, R2, r2 = T2_w[:3,3], T2_w[:3,:3], e2.dims

                        d = p2 - p1
                        dist = np.linalg.norm(d)
                        if dist < 1e-10:
                            continue

                        # Fast bounding-sphere cull
                        r2_bound = float(np.max(r2))
                        if dist > r1_bound + r2_bound + self.contact_penetration_slop:
                            continue

                        # Gradient-based surface normals
                        n1 =  self._ellipsoid_surface_normal( d, R1, r1)
                        n2 = -self._ellipsoid_surface_normal(-d, R2, r2)
                        normal = n1 + n2
                        nlen = np.linalg.norm(normal)
                        normal = normal / nlen if nlen > 1e-12 else n1

                        # `normal` above is the geometric direction FROM body A
                        # TOWARD body B (for spheres this is +d/|d|).
                        # The contact object stores the REPULSION normal for bodyA,
                        # which must point FROM the contact surface INTO bodyA,
                        # i.e. the OPPOSITE of the A→B geometric direction.
                        force_normal = -normal   # points: B→A (into bodyA, repulsive)
                        # Sanity check: force_normal should point roughly from
                        # bodyB centroid toward bodyA centroid.
                        _d_AB = b1.pos - b2.pos  # vector A←B
                        assert np.dot(force_normal, _d_AB) >= -0.5, (
                            f"force_normal points toward bodyB, not bodyA "
                            f"(dot={np.dot(force_normal, _d_AB):.3f}). "
                            f"Bodies: {b1.name}, {b2.name}"
                        )

                        # BF-4: correct directional effective radii
                        r1_eff = self._ellipsoid_radial_extent(r1, R1, normal)
                        r2_eff = self._ellipsoid_radial_extent(r2, R2, normal)
                        pen = (r1_eff + r2_eff) - np.dot(d, normal)

                        if pen > -self.contact_penetration_slop:
                            cp = p1 + normal * r1_eff
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
        for body in self.bodies:
            for ell in body.ellipsoids:
                T  = body.get_body_transform() @ ell.local_T
                pos, Rm, r = T[:3,3], T[:3,:3], ell.dims

                z_ext = float(np.sqrt(
                    (Rm[2, 0] * r[0]) ** 2 +
                    (Rm[2, 1] * r[1]) ** 2 +
                    (Rm[2, 2] * r[2]) ** 2))
                pen = -(pos[2] - z_ext)
                if pen > -self.contact_penetration_slop:
                    cp = np.array([pos[0], pos[1], pos[2] - z_ext])

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
        return contacts

    def get_contacts_for_body(self, body):
        return [c for c in self.contacts if c.bodyA is body or c.bodyB is body]

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
            if ca is not None and cb is not None:
                curve = _get_combined_curve(ca, cb)
            elif ca is not None:
                curve = ca
            elif cb is not None:
                curve = cb
            else:
                curve = np.array([[0.0, 0.0], [0.1, 5000.0]])
            return max(0.0, _hysteresis_force(curve, c.penetration,
                                              c._state['pen_max'], c.eta))

        # For every body that has 2+ non-ground contacts, form one receiver group
        # and compute the scale that prevents force summation on that surface.
        for body in self.bodies:
            # Collect non-ground contacts where this body is an endpoint
            grp = [c for c in self.contacts
                   if c.bodyB is not None and (c.bodyA is body or c.bodyB is body)]
            if len(grp) < 2:
                continue

            fe_vals = [_fe(c) for c in grp]
            fe_max  = max(fe_vals)
            fe_sum  = sum(fe_vals)

            if mode == 'discrete':
                for c, fe in zip(grp, fe_vals):
                    new_s = 1.0 if fe == fe_max else 0.0
                    # Take minimum so the most restrictive group wins
                    c._global_scale = min(c._global_scale, new_s)
            else:  # 'continuous'
                if fe_sum > 1e-12:
                    s = min(1.0, fe_max / fe_sum)
                    for c in grp:
                        c._global_scale = min(c._global_scale, s)
                # else all scales stay at 1.0 (no forces yet)

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
        if ca is not None and cb is not None:
            curve = _get_combined_curve(ca, cb)
        elif ca is not None:
            curve = ca
        elif cb is not None:
            curve = cb
        else:
            curve = np.array([[0.0, 0.0], [0.1, 5000.0]])

        F_elastic = gamma * _hysteresis_force(
            curve, c.penetration, pen_max, c.eta)

        # Damping  (not amplified)
        F_damping = c.damping * abs(v_rel_n)

        is_loading = bool(c._state.get('loading', True))
        if is_loading:
            F_total = F_elastic + F_damping
        else:
            F_total = max(0.0, F_elastic - F_damping)

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
                if jt == 'spherical' and dof == 3:
                    jstates[midx][jname] = q[idx:idx+3]
                    idx += 3
                elif jt == 'revolute' and dof == 1:
                    jstates[midx][jname] = np.array([q[idx], 0.0, 0.0])
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

        for body in self.bodies:
            vis = self.models[body.model_idx].bodies.get(
                      body.name.split('_', 1)[1])
            if vis is not None:
                body.set_state_from_transform(vis.global_transform)

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
                alpha[i] = alpha[pi] + np.cross(omega[pi], ax * qd_j[0])
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
                r_tot    = r_pj + cj_world
                omega[i] = omega[pi]
                v[i]     = v[pi] + np.cross(omega[pi], r_tot)
                alpha[i] = alpha[pi]
                a[i]     = (a[pi]
                            + np.cross(alpha[pi], r_tot)
                            + np.cross(omega[pi], np.cross(omega[pi], r_tot)))

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
            duration = entry['duration']
            t_end    = t_start + duration if duration > 0.0 else t_start + self.dt
            if not (t_start <= t < t_end):
                continue

            key = (entry['model_idx'], entry['joint_name'])
            if key not in self.joint_dof_map:
                continue
            s, dof = self.joint_dof_map[key]

            trq = np.asarray(entry['torque'], dtype=float)
            n   = min(dof, len(trq))
            # Half-sine envelope: specified magnitude is the peak.
            # scale = sin(π · (t − t_start) / duration)
            # Rises smoothly 0 → peak at mid-pulse → 0, avoiding
            # the integrator transients caused by a rectangular step.
            phase = (t - t_start) / duration if duration > 0.0 else 0.5
            scale = np.sin(np.pi * phase)
            tau[s:s + n] += trq[:n] * scale

        return tau

    # ------------------------------------------------------------------
    # Mass matrix + generalised force assembly
    # ------------------------------------------------------------------

    def assemble_A_and_B(self, q, qdot, t):
        nq = self.nq
        A1, A2 = self.compute_a1_a2_analytic(q, qdot)

        A = np.zeros((nq, nq))
        for i, body in enumerate(self.bodies):
            Ai1 = A1[i]
            Ai2 = A2[i]
            Iw  = body.get_world_inertia()
            A += body.mass * (Ai1.T @ Ai1)
            A += Ai2.T @ (Iw @ Ai2)

        A += 1e-4 * np.eye(nq)   # Tikhonov regularisation 
        # 1e-6 is insufficient when β≈85° (det(E_world)≈0.087); 1e-4 keeps
        # the relative error in qddot below 0.1% while robustly handling
        # near-singular poses without distorting well-conditioned steps.
        B  = self.rnea(q, qdot)
        B += self.compute_joint_limit_torques(q, qdot)
        B += self._compute_passive_damping_torques(qdot)
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

        # Integrate q — quaternion root DOFs need special treatment
        q_new = q.copy()
        for (midx, jname, jinfo, dof) in self.joint_list:
            if jinfo.get('is_root_joint', False) and dof == 7:
                s, _ = self.joint_dof_map[(midx, jname)]
                # Translation: Euler as normal
                q_new[s:s+3] = q[s:s+3] + self.dt * qdot_new[s:s+3]
                # Quaternion: qdot_quat = 0.5 * Q(q) * omega_world
                qw, qx, qy, qz = q[s+3:s+7]
                ox, oy, oz = qdot_new[s+3:s+6]
                dqw = 0.5 * (-qx*ox - qy*oy - qz*oz)
                dqx = 0.5 * ( qw*ox + qy*oz - qz*oy)
                dqy = 0.5 * ( qw*oy + qz*ox - qx*oz)
                dqz = 0.5 * ( qw*oz + qx*oy - qy*ox)
                new_q = np.array([qw + self.dt*dqw,
                                  qx + self.dt*dqx,
                                  qy + self.dt*dqy,
                                  qz + self.dt*dqz])
                q_new[s+3:s+7] = new_q / np.linalg.norm(new_q)   # re-normalise
            else:
                s, dof_j = self.joint_dof_map[(midx, jname)]
                q_new[s:s+dof_j] = q[s:s+dof_j] + self.dt * qdot_new[s:s+dof_j]

        self.state     = np.concatenate([q_new, qdot_new])
        self.time     += self.dt
        self.step_count += 1

        # Step 11: update display state to t+dt (body.vel = v(t+dt))
        self.update_kinematics_from_q(q_new)
        self._update_body_velocities_from_qdot(q_new, qdot_new)

    # ------------------------------------------------------------------
    # Per-body acceleration and contact force helpers
    # ------------------------------------------------------------------

    def _compute_body_accelerations(self, q, qddot):
        """
        Map generalised accelerations qddot -> per-body Cartesian accelerations.

        Uses the velocity Jacobians A1, A2:
          a_CG_i  = A1[i] @ qddot   (linear  acceleration, world frame)
          alpha_i = A2[i] @ qddot   (angular acceleration, world frame)

        Results are stored in body.lin_accel and body.ang_accel.
        These are total accelerations (gravity + inertial + contact).
        """
        A1, A2 = self.compute_a1_a2_analytic(q, self.state[self.nq:])
        for i, body in enumerate(self.bodies):
            body.lin_accel = A1[i] @ qddot
            body.ang_accel = A2[i] @ qddot

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
        body_states = [{
            'name': b.name, 'model_id': b.model_id,
            'pos': b.pos.copy(), 'quat': b.quat.copy(),
            'vel': b.vel.copy(), 'ang_vel': b.ang_vel.copy(),
            'lin_accel': getattr(b, 'lin_accel', np.zeros(3)).copy(),
            'ang_accel': getattr(b, 'ang_accel', np.zeros(3)).copy(),
            'force':  getattr(b, 'contact_force',  np.zeros(3)).copy(),
            'torque': getattr(b, 'contact_torque', np.zeros(3)).copy(),
        } for b in self.bodies]
        self.state_history.append({'time': self.time, 'body_states': body_states})
        self.contact_history.append([c.to_dict() for c in self.contacts])

    def clear_history(self):
        self.state_history  = []
        self.contact_history = []