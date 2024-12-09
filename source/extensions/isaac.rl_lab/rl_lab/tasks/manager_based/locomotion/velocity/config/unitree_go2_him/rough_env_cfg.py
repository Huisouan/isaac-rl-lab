import glob

from omni.isaac.lab.managers import EventTermCfg as EventTerm
from omni.isaac.lab.managers import ObservationTermCfg as ObsTerm
from omni.isaac.lab.utils import configclass

from ... import mdp
from .env.events import reset_amp
from ...velocity_env_cfg import LocomotionVelocityRoughEnvCfg, create_obsgroup_class

##
# Pre-defined configs
##
# use cloud assets
# from omni.isaac.lab_assets.unitree import UNITREE_GO2_CFG  # isort: skip
# use local assets
from rl_lab.assets.go2_model import UNITREE_GO2_CFG  # isort: skip


@configclass
class UnitreeA1HimRoughEnvCfg(LocomotionVelocityRoughEnvCfg):
    

    def __post_init__(self):
        # post init of parent
        super().__post_init__()
        self._run_disable_zero_weight_rewards = True
        # ------------------------------Sence------------------------------
        # switch robot to unitree-a1
        self.scene.robot = UNITREE_GO2_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")
        self.scene.height_scanner.prim_path = "{ENV_REGEX_NS}/Robot/base"
        # scale down the terrains because the robot is small
        self.scene.terrain.terrain_generator.sub_terrains["boxes"].grid_height_range = (0.025, 0.1)
        self.scene.terrain.terrain_generator.sub_terrains["random_rough"].noise_range = (0.01, 0.06)
        self.scene.terrain.terrain_generator.sub_terrains["random_rough"].noise_step = 0.01
        # no terrain curriculum
        self.curriculum.terrain_levels = None
        # ------------------------------Observations------------------------------
        self.observations.policy.base_ang_vel.scale = 0.25
        self.observations.policy.joint_pos.scale = 1.0
        self.observations.policy.joint_vel.scale = 0.05
        #self.observations.policy.base_lin_vel = None
        #self.observations.policy.base_ang_vel = None
        #self.observations.policy.height_scan = None
        self.observations.AMP = create_obsgroup_class('AMPCfg',{
            'base_pos_z': ObsTerm(func=mdp.base_pos_z),
            'base_lin_vel': ObsTerm(func=mdp.base_lin_vel),
            'base_ang_vel': ObsTerm(func=mdp.base_ang_vel),
            'joint_pos': ObsTerm(func=mdp.joint_pos),
            'joint_vel': ObsTerm(func=mdp.joint_vel),
        }, enable_corruption=True, concatenate_terms=True)()


        # ------------------------------Actions------------------------------
        # reduce action scale
        self.actions.joint_pos.scale = 0.25
        self.actions.joint_pos.clip = {".*": (-100, 100)}

        # ------------------------------Events------------------------------
        self.events.physics_material = None
        self.events.randomize_actuator_gains = None
        self.events.randomize_joint_parameters = None
        #self.events.push_robot = None   
        self.events.add_base_mass.params["mass_distribution_params"] = (-1.0, 3.0)
        self.events.add_base_mass.params["asset_cfg"].body_names = "base"
        self.events.base_external_force_torque.params["asset_cfg"].body_names = "base"
        self.events.reset_robot_joints.params["position_range"] = (1.0, 1.0)
        self.events.reset_base.params = {
            "pose_range": {"x": (-0.5, 0.5), "y": (-0.5, 0.5), "yaw": (-3.14, 3.14)},
            "velocity_range": {
                "x": (0.0, 0.0),
                "y": (0.0, 0.0),
                "z": (0.0, 0.0),
                "roll": (0.0, 0.0),
                "pitch": (0.0, 0.0),
                "yaw": (0.0, 0.0),
            },
        }


        # ------------------------------Rewards------------------------------
        # General
        # UNUESD self.rewards.is_alive.weight = 0
        self.rewards.is_terminated.weight = 0
        # Root penalties 
        self.rewards.lin_vel_z_l2.weight = -2.0
        self.rewards.ang_vel_xy_l2.weight = -0.05
        self.rewards.flat_orientation_l2.weight = -0.2
        self.rewards.base_height_l2.weight = 0
        self.rewards.body_lin_acc_l2.weight = 0

        # Joint penaltie
        self.rewards.joint_torques_l2.weight = 0
        # UNUESD self.rewards.joint_vel_l1.weight = 0.0
        self.rewards.joint_vel_l2.weight = 0
        self.rewards.joint_acc_l2.weight = -2.5e-7
        self.rewards.joint_power.weight = -2e-5
        self.rewards.joint_pos_limits.weight = 0
        self.rewards.joint_vel_limits.weight = 0
        self.rewards.base_height_l2.weight = -1.0
        self.rewards.base_height_l2.params["target_height"] = 0.4
        # Action penalties
        self.rewards.action_rate_l2.weight = -0.01
        # UNUESD self.rewards.action_l2.weight = 0.0

        # Contact sensor
        self.rewards.undesired_contacts.weight = 0
        self.rewards.contact_forces.weight = 0

        # Velocity-tracking rewards
        self.rewards.track_lin_vel_xy_exp.weight = 1
        self.rewards.track_ang_vel_z_exp.weight = 0.5

        # Others
        self.rewards.feet_air_time.params["sensor_cfg"].body_names = ".*_foot"
        self.rewards.feet_air_time.weight = 0
        self.rewards.foot_contact.weight = 0
        self.rewards.base_height_rough_l2.weight = 0
        self.rewards.feet_slide.weight = 0
        self.rewards.joint_power.weight = 0
        self.rewards.stand_still_when_zero_command.weight = 0.0

        # If the weight of rewards is 0, set rewards to None
        if self._run_disable_zero_weight_rewards:
            self.disable_zero_weight_rewards()

        # ------------------------------Terminations------------------------------
        self.terminations.illegal_contact.params["sensor_cfg"].body_names = "base"

        # ------------------------------Commands------------------------------
        self.commands.base_velocity.ranges.lin_vel_x = (-1.0, 3.0)
        self.commands.base_velocity.ranges.lin_vel_y = (-1.0, 1.0)
        self.commands.base_velocity.ranges.ang_vel_z = (-3.14, 3.14)
        self.commands.base_velocity.ranges.heading = (-3.14, 3.14)
        # ------------------------------AMP------------------------------
        self.urdf_path = "datasets/go2_description/urdf/go2_description.urdf"
        self.ee_names = ["FL_foot", "FR_foot", "RL_foot", "RR_foot"]
        self.base_name = "base"
        self.reference_state_initialization = False
        self.amp_motion_files = "datasets/mocap_motions_go2"
        self.amp_replay_buffer_size = 100000

        # ------------------------------Him------------------------------
        self.num_one_step_observations = 45
        self.encoder_steps = 6
        self.num_observations = self.num_one_step_observations * self.encoder_steps
        self.num_one_step_privileged_obs = 45 + 3 + 6 + 187  # additional: base_lin_vel, external_forces, scan_dots
        self.critict_steps = 1
        self.num_privileged_obs = self.num_one_step_privileged_obs * self.critict_steps  # if not None a privileged_obs_buf will be returned by step() (critic obs for asymmetric training). None is returned otherwise
        self.num_actions = 12
        self.env_spacing = 3.  # not used with heightfields/trimeshes
        self.send_timeouts = True  # send timeout information to the algorithm
        self.episode_length_s = 20  # episode length in seconds
