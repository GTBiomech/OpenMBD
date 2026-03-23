# ui_controls.py
# Citation: Tierney. OpenMBD: An Open-Source Multibody Dynamics Simulator for Biomechanics Research and Education. F1000Research, 2026.
# Version: 1.0 
# Research Contact: Dr Gregory Tierney (g.tierney@ulster.ac.uk)

import tkinter as tk
from tkinter import ttk

class AngleControlFrame(ttk.Frame):
    def __init__(self, parent, label_text, from_=-180, to=180, command=None):
        super().__init__(parent)
        self.command = command
        
        ttk.Label(self, text=label_text, width=10).pack(side=tk.LEFT)
        
        # Create entry widget WITHOUT StringVar initially
        self.entry = ttk.Entry(self, width=8)
        self.entry.pack(side=tk.LEFT, padx=5)
        self.entry.insert(0, "0.0")  # Manually set initial value
        self.entry.bind('<Return>', self.on_entry_change)
        self.entry.bind('<FocusOut>', self.on_entry_change)
        
        # Create slider widget
        self.slider = tk.Scale(self, from_=from_, to=to, orient=tk.HORIZONTAL,
                              length=150, showvalue=0, command=self.on_slider_move)
        self.slider.set(0.0)
        self.slider.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=5)
        
        # Create value display label
        self.display_label = ttk.Label(self, text="0.0°", width=8)
        self.display_label.pack(side=tk.RIGHT)
        
        # Create a StringVar for external binding if needed
        self.value_var = tk.StringVar()
        self.value_var.set("0.0")
        # Now bind it to the entry
        self.entry.config(textvariable=self.value_var)
    
    def on_slider_move(self, value):
        """Handle slider movement"""
        try:
            angle = float(value)
            # Update entry directly
            self.entry.delete(0, tk.END)
            self.entry.insert(0, f"{angle:.1f}")
            # Also update StringVar
            self.value_var.set(f"{angle:.1f}")
            # Update label
            self.display_label.config(text=f"{angle:.1f}°")
            if self.command:
                self.command()
        except ValueError:
            pass
    
    def on_entry_change(self, event=None):
        """Handle entry box change"""
        try:
            # Get value from entry
            text = self.entry.get()
            angle = float(text)
            # Clamp to valid range
            angle = max(-180, min(180, angle))
            # Update slider
            self.slider.set(angle)
            # Update entry with formatted value
            self.entry.delete(0, tk.END)
            self.entry.insert(0, f"{angle:.1f}")
            # Update StringVar
            self.value_var.set(f"{angle:.1f}")
            # Update label
            self.display_label.config(text=f"{angle:.1f}°")
            if self.command:
                self.command()
        except ValueError:
            # Reset to slider value if invalid
            angle = self.slider.get()
            self.entry.delete(0, tk.END)
            self.entry.insert(0, f"{angle:.1f}")
            self.value_var.set(f"{angle:.1f}")
            self.display_label.config(text=f"{angle:.1f}°")
    
    def get_value(self):
        """Get current value as float"""
        try:
            return float(self.entry.get())
        except ValueError:
            return 0.0
    
    def set_value(self, value):
        """Set value from external source"""
        try:
            value = float(value)
        except ValueError:
            value = 0.0
        
        # Clamp value
        value = max(-180, min(180, value))
        
        # Update slider
        self.slider.set(value)
        
        # Update entry directly
        self.entry.delete(0, tk.END)
        self.entry.insert(0, f"{value:.1f}")
        
        # Update StringVar
        self.value_var.set(f"{value:.1f}")
        
        # Update label
        self.display_label.config(text=f"{value:.1f}°")