# ui_setup_tab.py
# Citation: Tierney. OpenMBD: An Open-Source Multibody Dynamics Simulator for Biomechanics Research and Education. F1000Research, 2026.
# Version: 1.0 
# Research Contact: Dr Gregory Tierney (g.tierney@ulster.ac.uk)

import tkinter as tk
from tkinter import ttk, filedialog, messagebox
import json
import os
import numpy as np
import matplotlib
matplotlib.use("TkAgg")
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure

from models_parser import JSONModelParser, get_text_as_array
from models_multibody import MultibodyHumanModel
from models_config import ModelConfig
from geometry_mesh_generator import generate_ellipsoid_mesh
from ui_controls import AngleControlFrame

# Labels and default colours for up to 3 models
_MODEL_LABELS  = ["Model A", "Model B", "Model C"]
_MODEL_COLOURS = ["cyan", "magenta", "yellow"]
_MAX_MODELS    = 3


class SimulationCreatorTab(ttk.Frame):
    def __init__(self, notebook, main_app):
        super().__init__(notebook)
        self.main_app = main_app
        self.ax = None

        # Radio-button variable – holds the currently selected model id (int)
        self._radio_var = tk.IntVar(value=0)

        self._init_layout()

    # ------------------------------------------------------------------ #
    #  Layout                                                              #
    # ------------------------------------------------------------------ #

    def _init_layout(self):
        pane = ttk.PanedWindow(self, orient=tk.HORIZONTAL)
        pane.pack(fill=tk.BOTH, expand=True)

        left = ttk.Frame(pane, padding=10)
        pane.add(left, weight=1)

        # ── Simulation Parameters ──────────────────────────────────────
        gf = ttk.LabelFrame(left, text="Simulation Parameters")
        gf.pack(fill=tk.X, pady=5)

        f_dt_dur = ttk.Frame(gf)
        f_dt_dur.pack(fill=tk.X)
        ttk.Label(f_dt_dur, text="Sim Dt (s):").pack(side=tk.LEFT)
        self.ent_dt = ttk.Entry(f_dt_dur, width=8)
        self.ent_dt.insert(0, "0.0001")
        self.ent_dt.pack(side=tk.LEFT)
        ttk.Label(f_dt_dur, text="Output Dt (s):").pack(side=tk.LEFT, padx=(8, 0))
        self.ent_out_dt = ttk.Entry(f_dt_dur, width=8)
        self.ent_out_dt.insert(0, "0.001")
        self.ent_out_dt.pack(side=tk.LEFT)
        ttk.Label(f_dt_dur, text="Dur (s):").pack(side=tk.LEFT, padx=(8, 0))
        self.ent_dur = ttk.Entry(f_dt_dur, width=8)
        self.ent_dur.insert(0, "0.03")
        self.ent_dur.pack(side=tk.LEFT)

        f_cof = ttk.Frame(gf)
        f_cof.pack(fill=tk.X)
        ttk.Label(f_cof, text="CoF (Friction):").pack(side=tk.LEFT)
        self.ent_cof = ttk.Entry(f_cof, width=8)
        self.ent_cof.insert(0, "0.4")
        self.ent_cof.bind('<Return>', lambda e: self.update_cof())
        self.ent_cof.pack(side=tk.LEFT)

        f_damp = ttk.Frame(gf)
        f_damp.pack(fill=tk.X)
        ttk.Label(f_damp, text="Contact Damping (N·s/m):").pack(side=tk.LEFT)
        self.ent_damping = ttk.Entry(f_damp, width=8)
        self.ent_damping.insert(0, "150")
        self.ent_damping.bind('<Return>', lambda e: self.update_damping())
        self.ent_damping.pack(side=tk.LEFT)

        # ── Model Configuration ────────────────────────────────────────
        ms = ttk.LabelFrame(left, text="Model Configuration")
        ms.pack(fill=tk.X, pady=5)

        # Row 1 – radio buttons (rebuilt dynamically) + Add / Remove
        self._radio_frame = ttk.Frame(ms)
        self._radio_frame.pack(fill=tk.X, pady=(4, 0))

        # Radio buttons are stored here so we can rebuild them
        self._radio_buttons = {}   # mid -> Radiobutton widget

        btn_row = ttk.Frame(ms)
        btn_row.pack(fill=tk.X, pady=(2, 4))
        self._btn_add = ttk.Button(btn_row, text="+ Add Model",
                                   command=self._add_model)
        self._btn_add.pack(side=tk.LEFT, padx=(0, 4))
        self._btn_remove = ttk.Button(btn_row, text="− Remove Selected",
                                      command=self._remove_model)
        self._btn_remove.pack(side=tk.LEFT)

        # Model-path row
        self.xml_frame = ttk.Frame(ms)
        self.xml_frame.pack(fill=tk.X, pady=5)
        ttk.Label(self.xml_frame, text="Model Path:").pack(side=tk.LEFT)
        self.ent_xml_path = ttk.Entry(self.xml_frame)
        self.ent_xml_path.pack(side=tk.LEFT, fill=tk.X, expand=True)
        ttk.Button(self.xml_frame, text="Browse",
                   command=self.browse_xml).pack(side=tk.RIGHT)
        ttk.Button(self.xml_frame, text="Load",
                   command=self.load_xml).pack(side=tk.RIGHT)

        # ── Joint Configuration ────────────────────────────────────────
        self.cf = ttk.LabelFrame(left, text="Joint Configuration")
        self.cf.pack(fill=tk.BOTH, expand=True, pady=5)

        ttk.Label(self.cf, text="Root Pos (X Y Z):").pack(anchor='w')
        self.ent_pos = ttk.Entry(self.cf)
        self.ent_pos.pack(fill=tk.X)
        self.ent_pos.bind('<Return>', lambda e: self.update_model_data())

        ttk.Label(self.cf, text="Root Vel (X Y Z) m/s:").pack(anchor='w')
        self.ent_vel = ttk.Entry(self.cf)
        self.ent_vel.pack(fill=tk.X)
        self.ent_vel.bind('<Return>', lambda e: self.update_model_data())

        ttk.Label(self.cf, text="Root Ang Vel (Z Y X) rad/s:").pack(anchor='w')
        self.ent_ang_vel = ttk.Entry(self.cf)
        self.ent_ang_vel.pack(fill=tk.X)
        self.ent_ang_vel.insert(0, "0.0 0.0 0.0")
        self.ent_ang_vel.bind('<Return>', lambda e: self.update_model_data())

        ttk.Label(self.cf, text="Joint:").pack(anchor='w', pady=(10, 0))
        self.cmb_joints = ttk.Combobox(self.cf, state="readonly")
        self.cmb_joints.pack(fill=tk.X)
        self.cmb_joints.bind("<<ComboboxSelected>>", self.on_joint_sel)

        self.angle_controls = []
        for angle_name in ['Yaw (Z)', 'Pitch (Y)', 'Roll (X)']:
            ctrl = AngleControlFrame(self.cf, angle_name,
                                     command=self.on_angle_change)
            ctrl.pack(fill=tk.X, pady=2)
            self.angle_controls.append(ctrl)

        ttk.Label(self.cf, text="Joint Angular Vel (Z Y X) rad/s:",
                  foreground='gray').pack(anchor='w', pady=(6, 0))
        self.joint_vel_controls = []
        for vel_name in ['ωZ (rad/s)', 'ωY (rad/s)', 'ωX (rad/s)']:
            ctrl = AngleControlFrame(self.cf, vel_name, from_=-12, to=12,
                                     command=self.on_joint_vel_change)
            ctrl.pack(fill=tk.X, pady=1)
            self.joint_vel_controls.append(ctrl)

        # ── Torque pulse controls ──────────────────────────────────────
        ttk.Separator(self.cf, orient='horizontal').pack(fill=tk.X,
                                                         pady=(8, 2))
        ttk.Label(self.cf, text="Joint Torque Pulse (N·m):",
                  foreground='gray').pack(anchor='w')
        self.joint_torque_controls = []
        for trq_name in ['τZ (N·m)', 'τY (N·m)', 'τX (N·m)']:
            ctrl = AngleControlFrame(self.cf, trq_name, from_=-500, to=500,
                                     command=self.on_joint_torque_change)
            ctrl.pack(fill=tk.X, pady=1)
            self.joint_torque_controls.append(ctrl)

        f_torque_time = ttk.Frame(self.cf)
        f_torque_time.pack(fill=tk.X, pady=2)
        ttk.Label(f_torque_time, text="Start (s):", width=10).pack(side=tk.LEFT)
        self.ent_torque_start = ttk.Entry(f_torque_time, width=7)
        self.ent_torque_start.insert(0, "0.0")
        self.ent_torque_start.pack(side=tk.LEFT, padx=(0, 8))
        self.ent_torque_start.bind('<Return>',
                                   lambda e: self.on_joint_torque_change())
        self.ent_torque_start.bind('<FocusOut>',
                                   lambda e: self.on_joint_torque_change())
        ttk.Label(f_torque_time, text="Dur (s):", width=8).pack(side=tk.LEFT)
        self.ent_torque_dur = ttk.Entry(f_torque_time, width=7)
        self.ent_torque_dur.insert(0, "0.1")
        self.ent_torque_dur.pack(side=tk.LEFT)
        self.ent_torque_dur.bind('<Return>',
                                 lambda e: self.on_joint_torque_change())
        self.ent_torque_dur.bind('<FocusOut>',
                                 lambda e: self.on_joint_torque_change())

        # ── Bottom buttons ─────────────────────────────────────────────
        f_config_btns = ttk.Frame(left)
        f_config_btns.pack(fill=tk.X, pady=5)
        ttk.Button(f_config_btns, text="Save Config",
                   command=self.save_config).pack(side=tk.LEFT,
                                                  fill=tk.X, expand=True)
        ttk.Button(f_config_btns, text="Load Config",
                   command=self.load_config).pack(side=tk.LEFT,
                                                  fill=tk.X, expand=True)

        ttk.Button(left, text="Finalize Setup & Switch to Run",
                   command=self.generate).pack(fill=tk.X, pady=5)

        # ── 3-D preview (right pane) ───────────────────────────────────
        right = ttk.Frame(pane)
        pane.add(right, weight=3)
        self.fig = Figure(figsize=(5, 5))
        self.ax = self.fig.add_subplot(111, projection='3d')
        self.canvas = FigureCanvasTkAgg(self.fig, master=right)
        self.canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
        self.canvas.mpl_connect("motion_notify_event", self.sync_cam)

        # Build the initial radio buttons for however many models exist
        self._rebuild_radio_buttons()

    # ------------------------------------------------------------------ #
    #  Dynamic model add / remove                                          #
    # ------------------------------------------------------------------ #

    @property
    def current_model_id(self):
        return self._radio_var.get()

    def _rebuild_radio_buttons(self):
        """Destroy and recreate radio buttons to match current model set."""
        for w in self._radio_buttons.values():
            w.destroy()
        self._radio_buttons.clear()

        for mid in sorted(self.main_app.models.keys()):
            label = _MODEL_LABELS[mid] if mid < len(_MODEL_LABELS) \
                else f"Model {mid}"
            rb = ttk.Radiobutton(
                self._radio_frame,
                text=label,
                variable=self._radio_var,
                value=mid,
                command=lambda m=mid: self.select_model(m),
            )
            rb.pack(side=tk.LEFT, padx=(0, 8))
            self._radio_buttons[mid] = rb

        # Make sure the currently selected id is still valid
        if self._radio_var.get() not in self.main_app.models:
            first = sorted(self.main_app.models.keys())[0]
            self._radio_var.set(first)

        # Grey-out Add when at max; Remove when only one model left
        n = len(self.main_app.models)
        self._btn_add.config(
            state=tk.NORMAL if n < _MAX_MODELS else tk.DISABLED)
        self._btn_remove.config(
            state=tk.NORMAL if n > 1 else tk.DISABLED)

    def _next_free_id(self):
        """Return the smallest non-negative integer not already in models."""
        existing = set(self.main_app.models.keys())
        for i in range(_MAX_MODELS):
            if i not in existing:
                return i
        return None

    def _add_model(self):
        if len(self.main_app.models) >= _MAX_MODELS:
            messagebox.showinfo("Limit reached",
                                f"Maximum {_MAX_MODELS} models supported.")
            return

        mid = self._next_free_id()
        if mid is None:
            return

        cfg = ModelConfig(mid)
        cfg.color = _MODEL_COLOURS[mid] if mid < len(_MODEL_COLOURS) \
            else "yellow"
        self.main_app.models[mid] = cfg

        self._rebuild_radio_buttons()

        # Switch focus to the new model so the user can configure it
        self._radio_var.set(mid)
        self.select_model(mid)

        messagebox.showinfo(
            "Model Added",
            f"{_MODEL_LABELS[mid] if mid < len(_MODEL_LABELS) else f'Model {mid}'} added.\n"
            "Browse and Load a model file to configure it."
        )

    def _remove_model(self):
        if len(self.main_app.models) <= 1:
            messagebox.showinfo("Cannot remove",
                                "At least one model is required.")
            return

        mid = self.current_model_id
        label = _MODEL_LABELS[mid] if mid < len(_MODEL_LABELS) \
            else f"Model {mid}"

        if not messagebox.askyesno("Remove model",
                                   f"Remove {label} from the simulation?"):
            return

        del self.main_app.models[mid]

        # Switch selection to the first remaining model
        first = sorted(self.main_app.models.keys())[0]
        self._radio_var.set(first)

        self._rebuild_radio_buttons()
        self.main_app.rebuild_physics()
        self.select_model(first)

    # ------------------------------------------------------------------ #
    #  Model selection / UI refresh                                        #
    # ------------------------------------------------------------------ #

    def select_model(self, model_id):
        self._radio_var.set(model_id)
        self.update_model_ui()
        self.update_kinematics()

    def update_cof(self):
        try:
            cof_value = float(self.ent_cof.get())
            if self.main_app.engine is not None:
                self.main_app.engine.friction_coef = cof_value
        except ValueError:
            messagebox.showerror("Input Error",
                                 "Coefficient of Friction must be a number.")

    def update_damping(self):
        try:
            damp_value = float(self.ent_damping.get())
            if damp_value < 0:
                raise ValueError("negative")
            if self.main_app.engine is not None:
                self.main_app.engine.contact_damping = damp_value
        except ValueError:
            messagebox.showerror("Input Error",
                                 "Contact Damping must be a non-negative number (N·s/m).")

    def browse_xml(self):
        m = self.main_app.models[self.current_model_id]
        path = filedialog.askopenfilename(
            defaultextension=".json",
            filetypes=[("JSON files", "*.json"), ("All files", "*.*")]
        )
        if path:
            m.path = path
            self.ent_xml_path.delete(0, tk.END)
            self.ent_xml_path.insert(0, path)

    def load_xml(self):
        m = self.main_app.models[self.current_model_id]
        m.path = self.ent_xml_path.get()
        self.refresh_model(m, m.pos_str, m.vel_str)
        self.update_model_ui()

    # ------------------------------------------------------------------ #
    #  Initial load                                                        #
    # ------------------------------------------------------------------ #

    def initial_load(self, config_path=None):
        if not os.path.exists('openmbd_male.json'):
            print("⚠ No openmbd_male.json found. Please run the model "
                  "updater first.")
            return

        if config_path is None and os.path.exists('testsim.json'):
            config_path = 'testsim.json'

        if config_path and os.path.exists(config_path):
            print(f"Auto-loading config: {config_path}")
            self._load_config_file(config_path)
            first = sorted(self.main_app.models.keys())[0]
            self._radio_var.set(first)
            self._rebuild_radio_buttons()
            self.update_model_ui()
            return

        # Default: one model only
        self.main_app.models = {0: ModelConfig(0)}
        self.main_app.models[0].color = _MODEL_COLOURS[0]
        self.main_app.models[0].path  = 'openmbd_male.json'

        self.ent_dt.delete(0, tk.END)
        self.ent_dt.insert(0, "0.0001")

        self.refresh_model(self.main_app.models[0], "0.0 0.0 1.05",
                           "0.0 0.0 0.0")

        self._radio_var.set(0)
        self._rebuild_radio_buttons()
        self.update_model_ui()

    # ------------------------------------------------------------------ #
    #  Model loading helpers                                               #
    # ------------------------------------------------------------------ #

    def refresh_model(self, model, def_pos=None, def_vel=None,
                      skip_rebuild=False):
        try:
            model.parser = JSONModelParser(model.path)
            model.multibody_model = MultibodyHumanModel(model.path)

            if def_pos is not None:
                model.pos_str = def_pos
            if def_vel is not None:
                model.vel_str = def_vel

            model.is_loaded = True

            for j_name, joint_info in \
                    model.multibody_model.joint_infos.items():
                if j_name == 'root_joint' or \
                        joint_info.get('is_root_joint', False):
                    if j_name not in model.joints:
                        model.joints[j_name] = [0.0, 0.0, 0.0]
                elif joint_info.get('type') != 'fixed':
                    if j_name not in model.joints:
                        model.joints[j_name] = [0.0, 0.0, 0.0]

            for j_name, angles in model.joints.items():
                model.multibody_model.joint_states[j_name] = angles

            if not skip_rebuild:
                self.main_app.rebuild_physics()
                self.update_kinematics()

        except Exception as e:
            import traceback
            traceback.print_exc()
            messagebox.showerror("Model Load Error",
                                 f"Could not load model file:\n{e}")

    def update_model_ui(self):
        mid = self.current_model_id
        if mid not in self.main_app.models:
            return
        m = self.main_app.models[mid]

        self.ent_xml_path.delete(0, tk.END)
        self.ent_xml_path.insert(0, m.path if m.path else "")

        self.ent_pos.delete(0, tk.END)
        self.ent_pos.insert(0, m.pos_str if m.pos_str else "0.0 0.0 0.0")
        self.ent_vel.delete(0, tk.END)
        self.ent_vel.insert(0, m.vel_str if m.vel_str else "0.0 0.0 0.0")
        self.ent_ang_vel.delete(0, tk.END)
        self.ent_ang_vel.insert(0, m.ang_vel_str if m.ang_vel_str else "0.0 0.0 0.0")

        if not m.is_loaded or m.multibody_model is None:
            self.cmb_joints['values'] = []
            self.cmb_joints.set('')
            for ctrl in self.angle_controls:
                ctrl.set_value(0.0)
            for ctrl in self.joint_vel_controls:
                ctrl.set_value(0.0)
            for ctrl in self.joint_torque_controls:
                ctrl.set_value(0.0)
            return

        display_joints = []
        for j_name, joint_info in m.multibody_model.joint_infos.items():
            joint_type = joint_info.get('type', 'fixed')
            if j_name == 'root_joint' or \
                    joint_info.get('is_root_joint', False):
                display_joints.append(j_name)
            elif joint_type == 'fixed':
                continue
            else:
                display_joints.append(j_name)

        display_joints = list(dict.fromkeys(display_joints))

        filtered = []
        for jn in display_joints:
            if jn == 'root_joint' or jn.startswith('root_joint_'):
                filtered.append(jn)
            else:
                low = jn.lower()
                if any(kw in low for kw in [
                        'head', 'ankle', 'upperback', 'lowerback', 'elbow', 'hip', 'knee',
                        'lumbar', 'neck', 'shoulder', 'wrist']):
                    filtered.append(jn)

        filtered = sorted(filtered,
                          key=lambda x: (x != 'root_joint', x))
        self.cmb_joints['values'] = filtered

        if filtered:
            self.cmb_joints.set(filtered[0])
            self._load_joint_angles(m, filtered[0])
            self._load_joint_vels(m, filtered[0])
            self._load_joint_torques(m, filtered[0])
        else:
            self.cmb_joints.set('')
            for ctrl in self.angle_controls:
                ctrl.set_value(0.0)
            for ctrl in self.joint_vel_controls:
                ctrl.set_value(0.0)
            for ctrl in self.joint_torque_controls:
                ctrl.set_value(0.0)

    def _load_joint_angles(self, model, joint_name):
        if joint_name in model.joints:
            for i, angle in enumerate(model.joints[joint_name]):
                if i < len(self.angle_controls):
                    self.angle_controls[i].set_value(angle)
        else:
            for ctrl in self.angle_controls:
                ctrl.set_value(0.0)

    def _load_joint_vels(self, model, joint_name):
        """Populate the joint velocity controls for the selected joint."""
        vels = model.joint_vels.get(joint_name, [0.0, 0.0, 0.0])
        for i, ctrl in enumerate(self.joint_vel_controls):
            ctrl.set_value(vels[i] if i < len(vels) else 0.0)

    def _load_joint_torques(self, model, joint_name):
        """Populate the torque pulse controls for the selected joint."""
        spec = model.joint_torques.get(joint_name,
                                       {'torque': [0.0, 0.0, 0.0],
                                        't_start': 0.0,
                                        'duration': 0.1})
        trq = spec.get('torque', [0.0, 0.0, 0.0])
        for i, ctrl in enumerate(self.joint_torque_controls):
            ctrl.set_value(trq[i] if i < len(trq) else 0.0)
        self.ent_torque_start.delete(0, tk.END)
        self.ent_torque_start.insert(0, str(spec.get('t_start', 0.0)))
        self.ent_torque_dur.delete(0, tk.END)
        self.ent_torque_dur.insert(0, str(spec.get('duration', 0.1)))

    def update_model_data(self):
        mid = self.current_model_id
        if mid not in self.main_app.models:
            return
        m = self.main_app.models[mid]
        m.pos_str     = self.ent_pos.get()
        m.vel_str     = self.ent_vel.get()
        m.ang_vel_str = self.ent_ang_vel.get()
        # Flush current joint vel controls into joint_vels dict
        jn = self.cmb_joints.get()
        if jn:
            m.joint_vels[jn] = [ctrl.get_value()
                                 for ctrl in self.joint_vel_controls]
            # Flush torque pulse controls
            try:
                t_start  = float(self.ent_torque_start.get())
            except ValueError:
                t_start  = 0.0
            try:
                duration = float(self.ent_torque_dur.get())
            except ValueError:
                duration = 0.1
            m.joint_torques[jn] = {
                'torque':   [ctrl.get_value()
                             for ctrl in self.joint_torque_controls],
                't_start':  t_start,
                'duration': duration,
            }

        if m.is_loaded:
            self.main_app.rebuild_physics()
        self.update_kinematics()

    def on_joint_sel(self, e):
        mid = self.current_model_id
        if mid not in self.main_app.models:
            return
        m = self.main_app.models[mid]
        jn = self.cmb_joints.get()
        if not jn:
            for ctrl in self.angle_controls:
                ctrl.set_value(0.0)
            for ctrl in self.joint_vel_controls:
                ctrl.set_value(0.0)
            for ctrl in self.joint_torque_controls:
                ctrl.set_value(0.0)
            self.ent_torque_start.delete(0, tk.END)
            self.ent_torque_start.insert(0, "0.0")
            self.ent_torque_dur.delete(0, tk.END)
            self.ent_torque_dur.insert(0, "0.1")
            return
        self._load_joint_angles(m, jn)
        self._load_joint_vels(m, jn)
        self._load_joint_torques(m, jn)

    def on_angle_change(self):
        mid = self.current_model_id
        if mid not in self.main_app.models:
            return
        m = self.main_app.models[mid]
        jn = self.cmb_joints.get()
        if jn:
            angles = [ctrl.get_value() for ctrl in self.angle_controls]
            m.joints[jn] = angles
            if m.multibody_model is not None:
                m.multibody_model.joint_states[jn] = angles
            # Rebuild so the new angles are committed into the engine state
            # vector via _initialize_state_from_config.
            self.main_app.rebuild_physics()
            self.update_kinematics()

    def on_joint_vel_change(self):
        """Store joint velocity from controls into model.joint_vels and rebuild."""
        mid = self.current_model_id
        if mid not in self.main_app.models:
            return
        m = self.main_app.models[mid]
        jn = self.cmb_joints.get()
        if jn:
            vels = [ctrl.get_value() for ctrl in self.joint_vel_controls]
            m.joint_vels[jn] = vels
            self.main_app.rebuild_physics()

    def on_joint_torque_change(self):
        """Store torque pulse spec into model.joint_torques and rebuild."""
        mid = self.current_model_id
        if mid not in self.main_app.models:
            return
        m = self.main_app.models[mid]
        jn = self.cmb_joints.get()
        if not jn:
            return
        trq = [ctrl.get_value() for ctrl in self.joint_torque_controls]
        try:
            t_start  = float(self.ent_torque_start.get())
        except ValueError:
            t_start  = 0.0
        try:
            duration = float(self.ent_torque_dur.get())
        except ValueError:
            duration = 0.1
        m.joint_torques[jn] = {
            'torque':   trq,
            't_start':  t_start,
            'duration': duration,
        }
        self.main_app.rebuild_physics()

    # ------------------------------------------------------------------ #
    #  Kinematics / drawing                                                #
    # ------------------------------------------------------------------ #

    def update_kinematics(self):
        engine = self.main_app.engine
        if engine is None:
            return
        for mid, model in self.main_app.models.items():
            if not model.is_loaded or model.multibody_model is None:
                continue
            mb = model.multibody_model
            root_pos = get_text_as_array(model.pos_str)
            root_angles = model.joints.get('root_joint', [0.0, 0.0, 0.0])
            for j_name, angles in model.joints.items():
                if j_name == 'root_joint' or \
                        j_name.startswith('root_joint'):
                    continue
                mb.joint_states[j_name] = angles
            mb.update_kinematics(mb.joint_states, root_pos, root_angles)
            for body_name, body_visual in mb.bodies.items():
                for eb in engine.bodies:
                    if eb.model_id == mid:
                        eb_name = eb.name
                        if '_' in eb_name and eb_name[0].isdigit():
                            eb_name = eb_name.split('_', 1)[1]
                        if eb_name == body_name:
                            eb.set_state_from_transform(
                                body_visual.global_transform)
                            break
        self.draw()

    def draw(self):
        if not self.ax:
            return
        self.ax.clear()
        self.ax.set_box_aspect([1, 1, 1])
        self.ax.set_xlabel('X')
        self.ax.set_ylabel('Y')
        self.ax.set_zlabel('Z')
        xx, yy = np.meshgrid(np.linspace(-1.5, 1.5, 2),
                              np.linspace(-1.5, 1.5, 2))
        self.ax.plot_surface(xx, yy, np.zeros_like(xx),
                             color='green', alpha=0.1)
        for mid, model in self.main_app.models.items():
            if not model.is_loaded or model.multibody_model is None:
                continue
            color = model.color
            for body_name, body in model.multibody_model.bodies.items():
                for ell in body.ellipsoids:
                    T_total = body.global_transform @ ell['local_T']
                    X, Y, Z = generate_ellipsoid_mesh(
                        ell['dims'], T_total, resolution=12)
                    try:
                        self.ax.plot_surface(
                            X, Y, Z, color=color, alpha=0.6,
                            edgecolor='k', linewidth=0.1,
                            antialiased=False)
                    except Exception:
                        self.ax.plot_wireframe(
                            X, Y, Z, color=color, alpha=0.8,
                            linewidths=1.0)
        if hasattr(self.main_app, 'cam'):
            self.ax.view_init(elev=self.main_app.cam[0],
                              azim=self.main_app.cam[1])
        self.ax.set_xlim(-1.5, 1.5)
        self.ax.set_ylim(-1.5, 1.5)
        self.ax.set_zlim(0, 2.5)
        self.canvas.draw_idle()

    def sync_cam(self, e):
        if self.ax:
            self.main_app.cam = (self.ax.elev, self.ax.azim)

    # ------------------------------------------------------------------ #
    #  Save / Load config                                                  #
    # ------------------------------------------------------------------ #

    def save_config(self):
        f = filedialog.asksaveasfilename(
            defaultextension=".json",
            filetypes=[("JSON", "*.json")])
        if f:
            d = {
                "dt":     self.ent_dt.get(),
                "out_dt": self.ent_out_dt.get(),
                "dur":    self.ent_dur.get(),
                "cof":    self.ent_cof.get(),
                "contact_damping": self.ent_damping.get(),
                "models": [m.to_dict()
                           for k, m in
                           sorted(self.main_app.models.items())],
            }
            try:
                with open(f, 'w') as o:
                    json.dump(d, o, indent=4)
                messagebox.showinfo("Config Saved",
                                    f"Configuration saved to {f}")
            except Exception as e:
                messagebox.showerror("Save Error",
                                     f"Could not save config file:\n{e}")

    def _load_config_file(self, f):
        try:
            with open(f, 'r') as fp:
                data = json.load(fp)

            self.ent_dt.delete(0, tk.END)
            self.ent_dt.insert(0, data.get("dt", "0.0001"))
            self.ent_out_dt.delete(0, tk.END)
            self.ent_out_dt.insert(0, data.get("out_dt", "0.001"))
            self.ent_dur.delete(0, tk.END)
            self.ent_dur.insert(0, data.get("dur", "0.03"))
            self.ent_cof.delete(0, tk.END)
            self.ent_cof.insert(0, data.get("cof", "0.4"))
            self.update_cof()
            self.ent_damping.delete(0, tk.END)
            self.ent_damping.insert(0, data.get("contact_damping", "150"))
            self.update_damping()

            if "models" in data:
                # Rebuild models dict from file (supports 1-3 models)
                self.main_app.models = {}
                for d_model in data["models"]:
                    mid = d_model.get("id")
                    if mid is None or mid >= _MAX_MODELS:
                        continue
                    cfg = ModelConfig(mid)
                    cfg.color = (_MODEL_COLOURS[mid]
                                 if mid < len(_MODEL_COLOURS) else "yellow")
                    cfg.path    = d_model.get("path", cfg.path)
                    cfg.pos_str     = d_model.get("pos",       cfg.pos_str)
                    cfg.vel_str     = d_model.get("vel",       cfg.vel_str)
                    cfg.ang_vel_str = d_model.get("ang_vel",   "0.0 0.0 0.0")
                    cfg.joints      = dict(d_model.get("joints", {}))
                    cfg.joint_vels  = dict(d_model.get("joint_vels", {}))
                    cfg.joint_torques = dict(d_model.get("joint_torques", {}))
                    self.main_app.models[mid] = cfg
                    if cfg.path and os.path.exists(cfg.path):
                        self.refresh_model(cfg, cfg.pos_str, cfg.vel_str,
                                           skip_rebuild=True)

            self._rebuild_radio_buttons()
            self.main_app.rebuild_physics()
            self.update_model_ui()
            self.update_kinematics()
            return True

        except Exception as e:
            import traceback
            traceback.print_exc()
            messagebox.showerror("Load Error",
                                 f"Could not load config file:\n{e}")
            return False

    def load_config(self):
        f = filedialog.askopenfilename(
            defaultextension=".json",
            filetypes=[("JSON", "*.json")])
        if f:
            if self._load_config_file(f):
                messagebox.showinfo("Config Loaded",
                                    f"Configuration loaded from {f}")

    # ------------------------------------------------------------------ #
    #  Finalize                                                            #
    # ------------------------------------------------------------------ #

    def generate(self):
        try:
            dt_value     = float(self.ent_dt.get())
            out_dt_value = float(self.ent_out_dt.get())
            self.main_app.duration  = float(self.ent_dur.get())
        except ValueError:
            messagebox.showerror("Input Error",
                                 "Sim Dt, Output Dt, and Duration must be numbers.")
            return

        if out_dt_value < dt_value:
            messagebox.showerror("Input Error",
                                 "Output Dt must be ≥ Sim Dt.")
            return

        self.update_cof()
        self.update_damping()

        # Flush any unsaved pos/vel/ang_vel text box and joint vel edits
        for mid, model in self.main_app.models.items():
            if mid == self.current_model_id:
                model.pos_str     = self.ent_pos.get()
                model.vel_str     = self.ent_vel.get()
                model.ang_vel_str = self.ent_ang_vel.get()
                jn = self.cmb_joints.get()
                if jn:
                    model.joint_vels[jn] = [ctrl.get_value()
                                            for ctrl in self.joint_vel_controls]
                    try:
                        t_start  = float(self.ent_torque_start.get())
                    except ValueError:
                        t_start  = 0.0
                    try:
                        duration = float(self.ent_torque_dur.get())
                    except ValueError:
                        duration = 0.1
                    model.joint_torques[jn] = {
                        'torque':   [ctrl.get_value()
                                     for ctrl in self.joint_torque_controls],
                        't_start':  t_start,
                        'duration': duration,
                    }


        self.main_app.rebuild_physics()
        self.main_app.engine.dt = dt_value
        record_every = max(1, round(out_dt_value / dt_value))
        self.main_app.engine.record_every = record_every

        self.update_kinematics()

        self.main_app.nb.select(self.main_app.run_tab)
        self.main_app.run_tab.update_display()