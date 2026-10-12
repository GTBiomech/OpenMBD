# ui_controls.py
# Citation: Tierney. OpenMBD: An Open-Source Multibody Dynamics Simulator for Biomechanics Research and Education. F1000Research, 2026.
# Version: 1.2
# Research Contact: Dr Gregory Tierney (g.tierney@ulster.ac.uk)

import tkinter as tk
from tkinter import ttk

class AngleControlFrame(ttk.Frame):
    """
    Labelled entry + slider for one scalar input (angle, angular velocity or
    torque).  Values typed into the entry are limited to the slider range
    [from_, to]; `unit` is shown after the value.
    """

    def __init__(self, parent, label_text, from_=-180, to=180, command=None,
                 unit="°", decimals=1):
        super().__init__(parent)
        self.decimals = int(decimals)
        self.command = command
        self.lo = float(min(from_, to))
        self.hi = float(max(from_, to))
        self.unit = unit

        ttk.Label(self, text=label_text, width=10).pack(side=tk.LEFT)

        self.entry = ttk.Entry(self, width=8)
        self.entry.pack(side=tk.LEFT, padx=5)
        self.entry.insert(0, "0.0")
        self.entry.bind('<Return>', self.on_entry_change)
        self.entry.bind('<FocusOut>', self.on_entry_change)

        # Slider steps match the displayed precision: a coarser slider would
        # round typed values when set_value() moves it.
        resolution = 10.0 ** (-self.decimals)
        self.slider = tk.Scale(self, from_=from_, to=to, orient=tk.HORIZONTAL,
                               length=150, showvalue=0, resolution=resolution,
                               command=self.on_slider_move)
        self.slider.set(0.0)
        self.slider.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=5)

        self.display_label = ttk.Label(self, text=self._fmt(0.0), width=12)
        self.display_label.pack(side=tk.RIGHT)

        self.value_var = tk.StringVar()
        self.value_var.set("0.0")
        self.entry.config(textvariable=self.value_var)

    def _fmt(self, value):
        sep = "" if self.unit == "°" else " "
        return f"{value:.{self.decimals}f}{sep}{self.unit}"

    def _clamp(self, value):
        return max(self.lo, min(self.hi, value))

    def _show(self, value):
        self.entry.delete(0, tk.END)
        self.entry.insert(0, f"{value:.{self.decimals}f}")
        self.value_var.set(f"{value:.{self.decimals}f}")
        self.display_label.config(text=self._fmt(value))

    def on_slider_move(self, value):
        """Handle slider movement"""
        try:
            v = float(value)
        except ValueError:
            return
        self._show(v)
        if self.command:
            self.command()

    def on_entry_change(self, event=None):
        """Handle entry box change"""
        try:
            v = self._clamp(float(self.entry.get()))
        except ValueError:
            self._show(float(self.slider.get()))
            return
        self.slider.set(v)
        self._show(v)
        if self.command:
            self.command()

    def get_value(self):
        """Get current value as float"""
        try:
            return float(self.entry.get())
        except ValueError:
            return 0.0

    def set_value(self, value):
        """Set value from external source"""
        try:
            v = self._clamp(float(value))
        except (TypeError, ValueError):
            v = 0.0
        self.slider.set(v)
        self._show(v)
