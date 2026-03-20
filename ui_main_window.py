# ui_main_window.py
# Citation: Tierney. OpenMBD: An Open-Source Multibody Dynamics Simulator for Biomechanics Research and Education. F1000Research, 2026.
# Version: 1.0 
# Research Contact: Dr Gregory Tierney (g.tierney@ulster.ac.uk)

import tkinter as tk
from tkinter import ttk
import os
import numpy as np

from physics_engine import PhysicsEngine
from physics_rigid_body import RigidBody
from models_config import ModelConfig
from ui_setup_tab import SimulationCreatorTab
from ui_run_tab import RunnerTab
from models_parser import get_text_as_array
from ui_analysis_tab import AnalysisTab


class SimulatorGUI:
    def __init__(self, root, config_path=None):
        self.root = root
        self.root.title("OpenMBD - Multibody Dynamics Simulator")
        self.models = {0: ModelConfig(0)}   # start with one model; user can add up to 3
        self.engine = None

        self.duration = 0.5               
        self.cam = (20, -45)

        def on_closing():
            if hasattr(self, 'run_tab'):
                self.run_tab.is_running = False
                if (hasattr(self.run_tab, 'simulation_thread')
                        and self.run_tab.simulation_thread):
                    self.run_tab.simulation_thread.join(timeout=1.0)
            self.root.destroy()
            os._exit(0)

        self.root.protocol("WM_DELETE_WINDOW", on_closing)

        self.nb = ttk.Notebook(root)
        self.nb.pack(fill=tk.BOTH, expand=True)
        self.create_tab   = SimulationCreatorTab(self.nb, self)
        self.run_tab      = RunnerTab(self.nb, self)
        self.analysis_tab = AnalysisTab(self.nb, self)
        self.nb.add(self.create_tab,   text="1. Setup")
        self.nb.add(self.run_tab,      text="2. Run")
        self.nb.add(self.analysis_tab, text="3. Analysis")

        print("=" * 60)
        print("OpenMBD Multibody Dynamics Simulator")
        print("=" * 60)

        self.create_tab.initial_load(config_path=config_path)

    def rebuild_physics(self):
        try:
            dt_value = float(self.create_tab.ent_dt.get())
        except Exception:
            dt_value = 0.0002   # fallback matches PhysicsEngine default

        try:
            self.duration = float(self.create_tab.ent_dur.get())
        except Exception:
            pass  # keep previous value

        self.engine = PhysicsEngine()
        self.engine.dt = dt_value

        try:
            cof_value = float(self.create_tab.ent_cof.get())
            self.engine.friction_coef = cof_value
        except Exception:
            pass

        for mid, cfg in self.models.items():
            if not cfg.is_loaded or cfg.parser is None:
                continue

            model_bodies = []
            for name, data in cfg.parser.bodies.items():
                newname = f"{mid}_{name}"
                rb = RigidBody(newname, data, mid)
                model_bodies.append(rb)


            self.engine.add_model(cfg.multibody_model, model_bodies, cfg)

        if hasattr(self, 'analysis_tab'):
            self.analysis_tab.refresh_body_list()