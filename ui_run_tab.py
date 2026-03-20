# ui_run_tab.py
# Citation: Tierney. OpenMBD: An Open-Source Multibody Dynamics Simulator for Biomechanics Research and Education. F1000Research, 2026.
# Version: 1.0 
# Research Contact: Dr Gregory Tierney (g.tierney@ulster.ac.uk)

import tkinter as tk
from tkinter import ttk, filedialog, messagebox
import numpy as np
import matplotlib
matplotlib.use("TkAgg")
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure
import threading
import time
import csv
import datetime
import queue

from geometry_mesh_generator import generate_ellipsoid_mesh

class RunnerTab(ttk.Frame):
    def __init__(self, nb, app):
        super().__init__(nb)
        self.app = app
        self.is_running = False
        self.simulation_thread = None
        self._gif_frames = []       # raw RGBA bytes, one per display update
        self._gif_frame_wh = (1, 1) # set on first capture
        self._layout()

    def _layout(self):
        f = ttk.Frame(self)
        f.pack(side=tk.LEFT, fill=tk.Y, padx=5, pady=5)
        ttk.Button(f, text="Run", command=self.toggle).pack(fill=tk.X, pady=5)
        ttk.Button(f, text="Reset", command=self.reset).pack(fill=tk.X, pady=5)

        self.contact_info = ttk.Label(f, text="Contacts: 0")
        self.contact_info.pack(pady=5)

        self.lbl = ttk.Label(f, text="Ready")
        self.lbl.pack()

        self.fig = Figure(figsize=(5, 5))
        self.ax = self.fig.add_subplot(111, projection='3d')
        self.canvas = FigureCanvasTkAgg(self.fig, master=self)
        self.canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)

        self.update_queue = queue.Queue()
        self.check_queue()

    def check_queue(self):
        try:
            while True:
                func = self.update_queue.get_nowait()
                func()
        except queue.Empty:
            pass
        finally:
            if self.winfo_exists():
                self.after(30, self.check_queue)

    def update_display(self):
        self.ax.clear()
        self.ax.set_box_aspect([1, 1, 1])
        self.ax.set_xlabel('X')
        self.ax.set_ylabel('Y')
        self.ax.set_zlabel('Z')

        xx, yy = np.meshgrid(np.linspace(-1.5, 1.5, 2), np.linspace(-1.5, 1.5, 2))
        self.ax.plot_surface(xx, yy, np.zeros_like(xx), color='green', alpha=0.1)

        for b in self.app.engine.bodies:
            color = self.app.models[b.model_id].color
            T_origin = np.eye(4)
            T_origin[:3, :3] = b.R
            T_origin[:3, 3]  = b.pos - b.R @ b.cg_local
            for ell in b.ellipsoids:
                T_ell = T_origin @ ell.local_T
                X, Y, Z = generate_ellipsoid_mesh(ell.dims, T_ell, resolution=12)
                try:
                    self.ax.plot_surface(X, Y, Z, color=color, alpha=0.6,
                                         edgecolor='k', linewidth=0.1, antialiased=False)
                except Exception:
                    self.ax.plot_wireframe(X, Y, Z, color=color, alpha=0.8, linewidths=1.0)

        if hasattr(self.app.engine, 'contact_history') and self.app.engine.contact_history:
            self.contact_info.config(text=f"Contacts: {len(self.app.engine.contact_history[-1])}")
        elif hasattr(self.app.engine, 'contacts'):
            self.contact_info.config(text=f"Contacts: {len(self.app.engine.contacts)}")

        if hasattr(self.app, 'cam'):
            self.ax.view_init(elev=self.app.cam[0], azim=self.app.cam[1])
        else:
            self.ax.set_xlim(-1.5, 1.5)
            self.ax.set_ylim(-1.5, 1.5)
            self.ax.set_zlim(0, 3)
        self.canvas.draw()

    def toggle(self):
        if self.is_running:
            self.is_running = False
            self.lbl.config(text="Paused")
            if self.simulation_thread and self.simulation_thread.is_alive():
                self.simulation_thread.join(timeout=1.0)
        else:
            # Always commit the current UI values to the engine before starting,
            # so the user never has to press Finalize first.
            try:
                self.app.engine.dt = float(self.app.create_tab.ent_dt.get())
            except Exception:
                pass
            try:
                self.app.duration = float(self.app.create_tab.ent_dur.get())
            except Exception:
                pass

            self.is_running = True
            if self.simulation_thread and self.simulation_thread.is_alive():
                self.simulation_thread.join(timeout=0.5)
            self.simulation_thread = threading.Thread(target=self.loop, daemon=True)
            self.simulation_thread.start()

    def loop(self):
        eng = self.app.engine
        self.update_queue.put(lambda: self.lbl.config(text="Running..."))
        eng.clear_history()
        self.update_queue.put(lambda: self._gif_frames.clear())

        # Use integer step count to avoid floating-point accumulation drift.
        # eng.time += dt repeated N times undershoots the target slightly, causing
        # the float comparison (eng.time < duration) to fire one extra batch.
        total_steps = max(1, round(self.app.duration / eng.dt))
        steps_per_frame = max(1, round(0.02 / eng.dt))
        step_count = 0

        try:
            while self.is_running and step_count < total_steps:
                batch = min(steps_per_frame, total_steps - step_count)
                try:
                    for _ in range(batch):
                        if not self.is_running:
                            break
                        eng.step()
                        # BF-E: do NOT call eng.record_state() here.
                        # step() records internally every 10 steps automatically.
                        step_count += 1
                except ZeroDivisionError:
                    self.is_running = False
                    self.update_queue.put(lambda: messagebox.showerror(
                        "Error", "Dt cannot be zero."))
                    self.update_queue.put(lambda: self.lbl.config(text="Error"))
                    return
                except Exception as e:
                    print(f"Simulation step error: {e}")
                    import traceback; traceback.print_exc()

                if self.is_running:
                    self.update_queue.put(self._display_and_capture)
                    time.sleep(0.02)

        except Exception as e:
            print(f"Simulation loop error: {e}")
            import traceback; traceback.print_exc()

        self.is_running = False
        if step_count >= total_steps:
            self.update_queue.put(lambda: self.lbl.config(text="Finished."))
            self.export_csv(auto=True)
            self.update_queue.put(self.export_gif_auto)
        else:
            self.update_queue.put(lambda: self.lbl.config(text="Paused."))

    def reset(self):
        self.is_running = False
        if self.simulation_thread and self.simulation_thread.is_alive():
            self.simulation_thread.join(timeout=1.0)
        self.app.engine.reset_to_initial()
        self.lbl.config(text="Ready")
        self.update_display()

    def _display_and_capture(self):
        """Called on main thread each display update: grab canvas pixels directly.
        Using buffer_rgba() reads exactly what is rendered -- no PNG round-trip,
        no timestamp arithmetic, guaranteed uniform frame spacing."""
        self.update_display()
        buf = bytes(self.fig.canvas.buffer_rgba())
        w, h = self.fig.canvas.get_width_height()
        self._gif_frames.append(buf)
        self._gif_frame_wh = (w, h)

    def export_gif_auto(self):
        """Stitch live-captured frames into a GIF at a fixed 20 fps frame rate."""
        import os, datetime
        if not self._gif_frames:
            self.lbl.config(text='Finished. (no frames for GIF)')
            return
        try:
            from PIL import Image as PILImage
        except ImportError:
            self.lbl.config(text='Finished. (install Pillow for GIF)')
            return

        timestamp = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
        gif_path = f'simulation_{timestamp}.gif'
        self.lbl.config(text='Saving GIF...')
        self.update()
        try:
            w, h = self._gif_frame_wh
            # Fixed 500 ms per frame 
            imgs = [PILImage.frombytes('RGBA', (w, h), b) for b in self._gif_frames]
            imgs[0].save(
                gif_path, save_all=True, append_images=imgs[1:],
                optimize=False, duration=500, loop=0)
            self.lbl.config(text=f'Finished. GIF: {os.path.basename(gif_path)}')
            print(f'GIF saved: {gif_path}  ({len(imgs)} frames)')
        except Exception as exc:
            import traceback; traceback.print_exc()
            self.lbl.config(text=f'Finished. GIF error: {exc}')
    def export_csv(self, auto=False):
        if not self.app.engine.state_history:
            if not auto:
                messagebox.showwarning("No Data", "No simulation data to export.")
            return

        if auto:
            timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
            base_name = f"simulation_{timestamp}"
        else:
            base_name = filedialog.asksaveasfilename(
                defaultextension=".csv",
                filetypes=[("CSV files", "*.csv"), ("All files", "*.*")],
                title="Save simulation data")
            if not base_name:
                return
            if base_name.endswith('.csv'):
                base_name = base_name[:-4]

        kinematics_file = f"{base_name}_output.csv"
        summary_file    = f"{base_name}_summary.txt"

        try:
            with open(kinematics_file, 'w', newline='') as f:
                writer = csv.writer(f)
                header = ['Time']
                for b in self.app.engine.bodies:
                    header.extend([
                        f'{b.name}_PosX', f'{b.name}_PosY', f'{b.name}_PosZ',
                        f'{b.name}_QuatW', f'{b.name}_QuatX', f'{b.name}_QuatY', f'{b.name}_QuatZ',
                        f'{b.name}_VelX', f'{b.name}_VelY', f'{b.name}_VelZ',
                        f'{b.name}_AngVelX', f'{b.name}_AngVelY', f'{b.name}_AngVelZ',
                        f'{b.name}_LinAccX', f'{b.name}_LinAccY', f'{b.name}_LinAccZ',
                        f'{b.name}_AngAccX', f'{b.name}_AngAccY', f'{b.name}_AngAccZ',
                        f'{b.name}_ForceX', f'{b.name}_ForceY', f'{b.name}_ForceZ',
                        f'{b.name}_TorqueX', f'{b.name}_TorqueY', f'{b.name}_TorqueZ',
                    ])
                writer.writerow(header)
                for state in self.app.engine.state_history:
                    row = [state['time']]
                    for bs in state['body_states']:
                        la = bs.get('lin_accel', [0.0, 0.0, 0.0])
                        aa = bs.get('ang_accel', [0.0, 0.0, 0.0])
                        row.extend([
                            bs['pos'][0], bs['pos'][1], bs['pos'][2],
                            bs['quat'][0], bs['quat'][1], bs['quat'][2], bs['quat'][3],
                            bs['vel'][0], bs['vel'][1], bs['vel'][2],
                            bs['ang_vel'][0], bs['ang_vel'][1], bs['ang_vel'][2],
                            la[0], la[1], la[2],
                            aa[0], aa[1], aa[2],
                            bs['force'][0], bs['force'][1], bs['force'][2],
                            bs['torque'][0], bs['torque'][1], bs['torque'][2],
                        ])
                    writer.writerow(row)

            with open(summary_file, 'w') as f:
                f.write(f"Simulation Summary - {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
                f.write("=" * 50 + "\n\n")
                f.write(f"Duration: {self.app.duration} seconds\n")
                f.write(f"Time step (dt): {self.app.engine.dt} seconds\n")
                f.write(f"Friction coefficient: {self.app.engine.friction_coef}\n")
                f.write(f"Contact damping: {self.app.engine.contact_damping} N*s/m\n")
                f.write(f"Total simulation steps: {len(self.app.engine.state_history)}\n")
                f.write(f"Total contacts recorded: {sum(len(c) for c in self.app.engine.contact_history)}\n")
                f.write(f"Total joint constraints: {len(self.app.engine.joint_constraints)}\n\n")
                f.write("Models:\n")
                for mid, model in self.app.models.items():
                    f.write(f"  Model {mid}: {model.path}\n")
                    f.write(f"    Position: {model.pos_str}\n")
                    f.write(f"    Velocity: {model.vel_str}\n")
                    f.write(f"    Joints configured: {len(model.joints)}\n\n")
                f.write("Files generated:\n")
                f.write(f"  Output data:  {kinematics_file}\n")
                f.write(f"  This summary: {summary_file}\n")

            if not auto:
                self.lbl.config(text="Data exported.")
                messagebox.showinfo("Export Complete",
                    f"Data exported:\n• {kinematics_file}\n• {summary_file}")
            else:
                print(f"Auto-exported:\n  {kinematics_file}\n  {summary_file}")

        except Exception as e:
            self.lbl.config(text="Export error")
            messagebox.showerror("Export Error", f"Could not export data: {str(e)}")