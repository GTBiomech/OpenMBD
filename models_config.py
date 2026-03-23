# models_config.py
# Citation: Tierney. OpenMBD: An Open-Source Multibody Dynamics Simulator for Biomechanics Research and Education. F1000Research, 2026.
# Version: 1.0 
# Research Contact: Dr Gregory Tierney (g.tierney@ulster.ac.uk)

class ModelConfig:
    def __init__(self, mid):
        self.id = mid
        self.path = "openmbd_male.json"
        self.pos_str = "0.0 0.0 0.95"
        self.vel_str = "0.0 0.0 0.0"
        self.ang_vel_str = "0.0 0.0 0.0"   # root angular velocity (rad/s, ZYX order)
        self.joints = {}
        self.joint_vels    = {}            # joint_name -> [ωZ, ωY, ωX] rad/s
        self.joint_torques = {}            # joint_name -> [τZ, τY, τX] N·m, t_start, duration
        self.hierarchy = {}
        self.parser = None
        self.multibody_model = None
        self.is_loaded = False
        self.color = 'cyan' if mid == 0 else ('magenta' if mid == 1 else 'yellow')

    def to_dict(self):
        return {
            "id": self.id,
            "path": self.path,
            "pos": self.pos_str,
            "vel": self.vel_str,
            "ang_vel": self.ang_vel_str,
            "joints": self.joints,
            "joint_vels":    self.joint_vels,
            "joint_torques": self.joint_torques,
        }

    def from_dict(self, d):
        self.path        = d.get("path", "")
        self.pos_str     = d.get("pos", "0 0 0")
        self.vel_str     = d.get("vel", "0 0 0")
        self.ang_vel_str = d.get("ang_vel", "0 0 0")
        self.joints      = d.get("joints", {})
        self.joint_vels    = d.get("joint_vels",    {})
        self.joint_torques = d.get("joint_torques", {})