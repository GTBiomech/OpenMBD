# ui_analysis_tab.py
# Citation: Tierney. OpenMBD: An Open-Source Multibody Dynamics Simulator for Biomechanics Research and Education. F1000Research, 2026.
# Version: 1.0 
# Research Contact: Dr Gregory Tierney (g.tierney@ulster.ac.uk)

"""
Tab 3 – Analysis
Plots kinematics and dynamics of any rigid body over the simulation history.

Layout
──────
Left panel  : body selector (combobox), channel tree, plot button, clear button,
              colour legend, export subplot PNG button.
Right panel : matplotlib figure with up to 6 stacked subplots sharing the time axis.

"""

import tkinter as tk
from tkinter import ttk, filedialog, messagebox
import numpy as np
import matplotlib
matplotlib.use("TkAgg")
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure
import matplotlib.gridspec as gridspec

# ── palette: one colour per plotted line, cycles if more than 10 ──────────────
_COLOURS = [
    '#00e5ff', '#ff4081', '#69ff47', '#ffea00', '#ff6d00',
    '#d500f9', '#00bcd4', '#f50057', '#76ff03', '#ff9100',
]

# ── channel definitions ────────────────────────────────────────────────────────
# Each entry: (display_name, group, extractor_fn(body_state) -> float)
def _mag(v): return float(np.linalg.norm(v))

_CHANNELS = [
    # Position
    ("Pos X",          "Position",         lambda bs: bs['pos'][0]),
    ("Pos Y",          "Position",         lambda bs: bs['pos'][1]),
    ("Pos Z",          "Position",         lambda bs: bs['pos'][2]),
    # Velocity
    ("Vel X",          "Velocity",         lambda bs: bs['vel'][0]),
    ("Vel Y",          "Velocity",         lambda bs: bs['vel'][1]),
    ("Vel Z",          "Velocity",         lambda bs: bs['vel'][2]),
    ("|Vel|",          "Velocity",         lambda bs: _mag(bs['vel'])),
    # Angular velocity
    ("AngVel X",       "Ang Velocity",     lambda bs: bs['ang_vel'][0]),
    ("AngVel Y",       "Ang Velocity",     lambda bs: bs['ang_vel'][1]),
    ("AngVel Z",       "Ang Velocity",     lambda bs: bs['ang_vel'][2]),
    ("|AngVel|",       "Ang Velocity",     lambda bs: _mag(bs['ang_vel'])),
    # Linear acceleration
    ("LinAcc X",       "Lin Acceleration", lambda bs: bs.get('lin_accel', [0,0,0])[0]),
    ("LinAcc Y",       "Lin Acceleration", lambda bs: bs.get('lin_accel', [0,0,0])[1]),
    ("LinAcc Z",       "Lin Acceleration", lambda bs: bs.get('lin_accel', [0,0,0])[2]),
    ("|LinAcc|",       "Lin Acceleration", lambda bs: _mag(bs.get('lin_accel', [0,0,0]))),
    # Angular acceleration
    ("AngAcc X",       "Ang Acceleration", lambda bs: bs.get('ang_accel', [0,0,0])[0]),
    ("AngAcc Y",       "Ang Acceleration", lambda bs: bs.get('ang_accel', [0,0,0])[1]),
    ("AngAcc Z",       "Ang Acceleration", lambda bs: bs.get('ang_accel', [0,0,0])[2]),
    ("|AngAcc|",       "Ang Acceleration", lambda bs: _mag(bs.get('ang_accel', [0,0,0]))),
    # Orientation quaternion
    ("Quat W",         "Orientation",      lambda bs: bs['quat'][0]),
    ("Quat X",         "Orientation",      lambda bs: bs['quat'][1]),
    ("Quat Y",         "Orientation",      lambda bs: bs['quat'][2]),
    ("Quat Z",         "Orientation",      lambda bs: bs['quat'][3]),
    # Applied force
    ("Force X",        "Applied Force",    lambda bs: bs['force'][0]),
    ("Force Y",        "Applied Force",    lambda bs: bs['force'][1]),
    ("Force Z",        "Applied Force",    lambda bs: bs['force'][2]),
    ("|Force|",        "Applied Force",    lambda bs: _mag(bs['force'])),
    # Applied torque
    ("Torque X",       "Applied Torque",   lambda bs: bs['torque'][0]),
    ("Torque Y",       "Applied Torque",   lambda bs: bs['torque'][1]),
    ("Torque Z",       "Applied Torque",   lambda bs: bs['torque'][2]),
    ("|Torque|",       "Applied Torque",   lambda bs: _mag(bs['torque'])),
]

# Contact channels removed to ensure GUI consistency
_CONTACT_CHANNELS = []

# Y-axis units per group
_UNITS = {
    "Position":          "m",
    "Velocity":          "m/s",
    "Ang Velocity":      "rad/s",
    "Lin Acceleration":  "m/s²",
    "Ang Acceleration":  "rad/s²",
    "Orientation":       "–",
    "Applied Force":     "N",
    "Applied Torque":    "N·m",
}


class AnalysisTab(ttk.Frame):
    def __init__(self, notebook, main_app):
        super().__init__(notebook)
        self.main_app = main_app
        # list of (body_name, channel_label, colour, extractor_or_tag)
        self._active_plots: list = []
        self._colour_idx = 0
        self._build_layout()

    # ──────────────────────────────────────────────────────────────────────────
    #  Layout
    # ──────────────────────────────────────────────────────────────────────────

    def _build_layout(self):
        # Dark background for the whole tab
        style = ttk.Style()
        style.configure("Analysis.TFrame", background="#0d1117")
        style.configure("Analysis.TLabel",
                        background="#0d1117", foreground="#c9d1d9",
                        font=("Consolas", 9))
        style.configure("Analysis.TLabelframe",
                        background="#0d1117", foreground="#58a6ff")
        style.configure("Analysis.TLabelframe.Label",
                        background="#0d1117", foreground="#58a6ff",
                        font=("Consolas", 9, "bold"))

        self.configure(style="Analysis.TFrame")

        pane = ttk.PanedWindow(self, orient=tk.HORIZONTAL)
        pane.pack(fill=tk.BOTH, expand=True)

        # ── Left control panel ─────────────────────────────────────────────
        left = ttk.Frame(pane, padding=8, style="Analysis.TFrame")
        pane.add(left, weight=0)

        # Body selector
        body_frame = ttk.LabelFrame(left, text="  Body  ",
                                    style="Analysis.TLabelframe", padding=6)
        body_frame.pack(fill=tk.X, pady=(0, 6))

        ttk.Label(body_frame, text="Select body:",
                  style="Analysis.TLabel").pack(anchor='w')
        self.cmb_body = ttk.Combobox(body_frame, state="readonly", width=26,
                                     font=("Consolas", 9))
        self.cmb_body.pack(fill=tk.X, pady=(2, 0))
        self.cmb_body.bind("<<ComboboxSelected>>", lambda e: None)

        # Channel tree
        ch_frame = ttk.LabelFrame(left, text="  Channels  ",
                                  style="Analysis.TLabelframe", padding=4)
        ch_frame.pack(fill=tk.BOTH, expand=True, pady=(0, 6))

        tree_scroll = ttk.Scrollbar(ch_frame, orient=tk.VERTICAL)
        self.channel_tree = ttk.Treeview(
            ch_frame,
            columns=("unit",),
            show="tree headings",
            selectmode="extended",
            yscrollcommand=tree_scroll.set,
            height=18,
        )
        tree_scroll.config(command=self.channel_tree.yview)
        self.channel_tree.heading("#0",    text="Channel")
        self.channel_tree.heading("unit",  text="Unit")
        self.channel_tree.column("#0",     width=130, stretch=True)
        self.channel_tree.column("unit",   width=55,  stretch=False)
        self.channel_tree.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        tree_scroll.pack(side=tk.RIGHT, fill=tk.Y)
        self._populate_channel_tree()

        # Add / remove / clear buttons
        btn_frame = ttk.Frame(left, style="Analysis.TFrame")
        btn_frame.pack(fill=tk.X, pady=(0, 6))

        tk.Button(btn_frame, text="＋ Add to Plot",
                  bg="#238636", fg="white",
                  font=("Consolas", 9, "bold"),
                  relief=tk.FLAT, cursor="hand2",
                  command=self._add_selected).pack(fill=tk.X, pady=2)

        tk.Button(btn_frame, text="✕ Remove Selected",
                  bg="#3a1c1c", fg="#f85149",
                  font=("Consolas", 9),
                  relief=tk.FLAT, cursor="hand2",
                  command=self._remove_selected).pack(fill=tk.X, pady=2)

        tk.Button(btn_frame, text="⟳ Refresh / Replot",
                  bg="#1f3a5f", fg="#58a6ff",
                  font=("Consolas", 9),
                  relief=tk.FLAT, cursor="hand2",
                  command=self.replot).pack(fill=tk.X, pady=2)

        tk.Button(btn_frame, text="✕ Clear All",
                  bg="#21262d", fg="#8b949e",
                  font=("Consolas", 9),
                  relief=tk.FLAT, cursor="hand2",
                  command=self._clear_all).pack(fill=tk.X, pady=2)

        tk.Button(btn_frame, text="⬇ Export PNG",
                  bg="#21262d", fg="#8b949e",
                  font=("Consolas", 9),
                  relief=tk.FLAT, cursor="hand2",
                  command=self._export_png).pack(fill=tk.X, pady=2)

        # Active-series legend
        legend_frame = ttk.LabelFrame(left, text="  Active Series  ",
                                      style="Analysis.TLabelframe", padding=4)
        legend_frame.pack(fill=tk.BOTH, expand=True, pady=(0, 0))

        self.legend_list = tk.Listbox(
            legend_frame,
            bg="#0d1117", fg="#c9d1d9",
            selectbackground="#1f3a5f",
            font=("Consolas", 8),
            relief=tk.FLAT, height=8,
        )
        self.legend_list.pack(fill=tk.BOTH, expand=True)

        # ── Right plot panel ───────────────────────────────────────────────
        right = ttk.Frame(pane, style="Analysis.TFrame")
        pane.add(right, weight=1)

        self.fig = Figure(facecolor="#0d1117")
        self.canvas = FigureCanvasTkAgg(self.fig, master=right)
        self.canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)

        self._draw_empty_state()

    def _populate_channel_tree(self):
        """Build the grouped treeview of available channels."""
        groups = {}
        for (name, group, _fn) in _CHANNELS:
            groups.setdefault(group, []).append(name)
        for (name, group) in _CONTACT_CHANNELS:
            groups.setdefault(group, []).append(name)

        for group, names in groups.items():
            unit = _UNITS.get(group, "")
            parent = self.channel_tree.insert(
                "", "end", text=group, values=(unit,), open=False)
            for n in names:
                self.channel_tree.insert(parent, "end", text=n, values=(unit,))

    # ──────────────────────────────────────────────────────────────────────────
    #  Public entry-points called from the run tab / main window
    # ──────────────────────────────────────────────────────────────────────────

    def refresh_body_list(self):
        """Rebuild the body combobox from current engine bodies."""
        eng = self.main_app.engine
        if eng is None:
            return
        names = [b.name for b in eng.bodies]
        self.cmb_body['values'] = names
        if names and not self.cmb_body.get():
            self.cmb_body.set(names[0])

    def on_simulation_complete(self):
        """Called when the run-tab loop finishes; refreshes plot if series active."""
        self.refresh_body_list()
        if self._active_plots:
            self.replot()

    # ──────────────────────────────────────────────────────────────────────────
    #  Add / remove series
    # ──────────────────────────────────────────────────────────────────────────

    def _add_selected(self):
        body_name = self.cmb_body.get()
        if not body_name:
            messagebox.showwarning("No body", "Select a body first.")
            return

        selections = self.channel_tree.selection()
        if not selections:
            messagebox.showwarning("No channel", "Select at least one channel.")
            return

        added = 0
        for item_id in selections:
            ch_label = self.channel_tree.item(item_id, "text")
            # Skip group-level rows (they have children)
            if self.channel_tree.get_children(item_id):
                continue

            key = (body_name, ch_label)
            # Avoid duplicates
            if any(p[0] == body_name and p[1] == ch_label
                   for p in self._active_plots):
                continue

            colour = _COLOURS[self._colour_idx % len(_COLOURS)]
            self._colour_idx += 1
            self._active_plots.append((body_name, ch_label, colour))
            added += 1

        if added:
            self._rebuild_legend()
            self.replot()

    def _remove_selected(self):
        sel = self.legend_list.curselection()
        if not sel:
            return
        # Remove in reverse so indices stay valid
        for i in reversed(sel):
            del self._active_plots[i]
        self._rebuild_legend()
        self.replot()

    def _clear_all(self):
        self._active_plots.clear()
        self._colour_idx = 0
        self._rebuild_legend()
        self._draw_empty_state()

    def _rebuild_legend(self):
        self.legend_list.delete(0, tk.END)
        for body_name, ch_label, colour in self._active_plots:
            self.legend_list.insert(tk.END, f"● {body_name}  ·  {ch_label}")
            idx = self.legend_list.size() - 1
            self.legend_list.itemconfig(idx, fg=colour)

    # ──────────────────────────────────────────────────────────────────────────
    #  Plotting
    # ──────────────────────────────────────────────────────────────────────────

    def _draw_empty_state(self):
        self.fig.clear()
        ax = self.fig.add_subplot(111)
        ax.set_facecolor("#0d1117")
        ax.text(0.5, 0.5,
                "Add channels from the left panel\nthen run the simulation",
                ha='center', va='center', transform=ax.transAxes,
                color='#484f58', fontsize=12, fontfamily='Consolas')
        for spine in ax.spines.values():
            spine.set_edgecolor('#21262d')
        ax.tick_params(colors='#484f58')
        self.canvas.draw_idle()

    def replot(self):
        """Extract data from state_history and redraw all active series."""
        eng = self.main_app.engine
        if eng is None or not eng.state_history:
            self._draw_empty_state()
            return
        if not self._active_plots:
            self._draw_empty_state()
            return

        history    = eng.state_history
        c_history  = eng.contact_history
        times      = np.array([s['time'] for s in history])

        # Build a lookup: body_name -> index in body_states list
        if history:
            body_index = {bs['name']: i
                          for i, bs in enumerate(history[0]['body_states'])}
        else:
            body_index = {}

        # Group active plots by channel so we can stack like-channel types
        # together in the same subplot.  Each subplot = one y-axis unit family.
        n = min(len(self._active_plots), 6)
        if n == 0:
            self._draw_empty_state()
            return

        self.fig.clear()
        bg = "#0d1117"
        ax_color = "#161b22"
        grid_color = "#21262d"
        tick_color = "#8b949e"
        label_color = "#c9d1d9"

        # Decide subplot grouping: group by unit when > 4 series
        if len(self._active_plots) <= 4:
            # One subplot per series
            subplot_map = {i: i for i in range(len(self._active_plots))}
            n_subplots = len(self._active_plots)
        else:
            # Group by channel unit
            unit_to_subplot = {}
            subplot_map = {}
            idx = 0
            for i, (_, ch_label, _) in enumerate(self._active_plots):
                unit = self._channel_unit(ch_label)
                if unit not in unit_to_subplot:
                    if idx >= 6:
                        idx = 5   # cap at 6 subplots
                    unit_to_subplot[unit] = idx
                    idx += 1
                subplot_map[i] = unit_to_subplot[unit]
            n_subplots = len(unit_to_subplot)

        gs = gridspec.GridSpec(n_subplots, 1, figure=self.fig,
                               hspace=0.08, left=0.10, right=0.97,
                               top=0.95, bottom=0.07)
        axes = [self.fig.add_subplot(gs[i]) for i in range(n_subplots)]

        # Style all axes
        for ax in axes:
            ax.set_facecolor(ax_color)
            ax.grid(True, color=grid_color, linewidth=0.6, linestyle='--')
            ax.tick_params(colors=tick_color, labelsize=7)
            for spine in ax.spines.values():
                spine.set_edgecolor(grid_color)
            ax.yaxis.label.set_color(label_color)
            ax.xaxis.label.set_color(tick_color)

        # Plot each series
        legend_entries = {i: [] for i in range(n_subplots)}

        for series_i, (body_name, ch_label, colour) in \
                enumerate(self._active_plots):
            sp_idx = subplot_map.get(series_i, 0)
            ax = axes[sp_idx]

            data = self._extract_series(
                ch_label, body_name, body_index, history, c_history, times)

            if data is None:
                continue

            line, = ax.plot(times[:len(data)], data,
                            color=colour, linewidth=1.2,
                            label=f"{body_name} · {ch_label}")
            legend_entries[sp_idx].append(line)

        # Add legends and y-labels
        for sp_idx, ax in enumerate(axes):
            lines = legend_entries[sp_idx]
            if lines:
                leg = ax.legend(handles=lines, loc='upper right',
                                fontsize=7, framealpha=0.3,
                                labelcolor='linecolor',
                                facecolor='#161b22',
                                edgecolor='#30363d')
            # Y-label from unit
            unit = self._unit_for_subplot(sp_idx, subplot_map)
            ax.set_ylabel(unit, fontsize=7, color=tick_color)
            # Only bottom axis shows x-label
            if sp_idx < n_subplots - 1:
                ax.set_xticklabels([])
            else:
                ax.set_xlabel("Time (s)", fontsize=8)

        # Shared x-range
        if len(times) > 1:
            for ax in axes:
                ax.set_xlim(times[0], times[-1])

        self.canvas.draw_idle()

    def _extract_series(self, ch_label, body_name,
                        body_index, history, c_history, times):
        """Return a 1-D numpy array of values for the requested channel."""

        # Kinematic / dynamic channels — look up the extractor fn
        fn = None
        for (name, _group, extractor) in _CHANNELS:
            if name == ch_label:
                fn = extractor
                break
        if fn is None:
            return None

        bi = body_index.get(body_name)
        if bi is None:
            return None

        vals = []
        for frame in history:
            bs = frame['body_states'][bi]
            try:
                vals.append(fn(bs))
            except Exception:
                vals.append(0.0)
        return np.array(vals)

    # ──────────────────────────────────────────────────────────────────────────
    #  Helpers
    # ──────────────────────────────────────────────────────────────────────────

    def _channel_unit(self, ch_label):
        for (name, group, _) in _CHANNELS:
            if name == ch_label:
                return _UNITS.get(group, "?")
        return "?"

    def _unit_for_subplot(self, sp_idx, subplot_map):
        for series_i, mapped_sp in subplot_map.items():
            if mapped_sp == sp_idx and series_i < len(self._active_plots):
                _, ch_label, _ = self._active_plots[series_i]
                return self._channel_unit(ch_label)
        return ""

    def _export_png(self):
        if not self._active_plots:
            messagebox.showwarning("Nothing to export", "No active plots.")
            return
        path = filedialog.asksaveasfilename(
            defaultextension=".png",
            filetypes=[("PNG image", "*.png"), ("All files", "*.*")],
            title="Export plot as PNG",
        )
        if path:
            self.fig.savefig(path, dpi=150, bbox_inches='tight',
                             facecolor=self.fig.get_facecolor())
            messagebox.showinfo("Exported", f"Saved to {path}")