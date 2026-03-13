# openmbd_simulator.py
# Citation: Tierney. OpenMBD: An Open-Source Multibody Dynamics Simulator for Biomechanics Research and Education. F1000Research, 2026.
# Version: 1.0 
# Research Contact: Dr Gregory Tierney (g.tierney@ulster.ac.uk)

import sys
sys.dont_write_bytecode = True

import tkinter as tk
from ui_main_window import SimulatorGUI

if __name__ == "__main__":
    # Optional: pass a sim config JSON as the first command-line argument.
    config_path = sys.argv[1] if len(sys.argv) > 1 else None

    root = tk.Tk()
    root.geometry("1200x900")
    app = SimulatorGUI(root, config_path=config_path)
    root.mainloop()