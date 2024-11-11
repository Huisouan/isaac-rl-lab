import numpy as np
from amp_utils import AMP_UTILS_DIR

VISUALIZE_RETARGETING = True

URDF_FILENAME = f"exts/robot_lab/robot_lab/third_party/amp_utils/models/a1/urdf/a1.urdf"
OUTPUT_DIR = f"exts/robot_lab/robot_lab/third_party/amp_utils/motion_files/mocap_motions_a1/"

REF_POS_SCALE = 0.825 # 缩放系数,如果遇到关节限位异常，尝试将此数变小
INIT_POS = np.array([0, 0, 0.32]) # a1
INIT_ROT = np.array([0, 0, 0, 1.0])

SIM_TOE_JOINT_IDS = [6, 11, 16, 21]
SIM_HIP_JOINT_IDS = [2, 7, 12, 17]
SIM_ROOT_OFFSET = np.array([0, 0, -0.04])
SIM_TOE_OFFSET_LOCAL = [
    np.array([0.0, -0.06, 0.0]),
    np.array([0.0, 0.06, 0.0]),
    np.array([0.0, -0.06, 0.0]),
    np.array([0.0, 0.06, 0.0])
]
TOE_HEIGHT_OFFSET = 0.02

DEFAULT_JOINT_POSE = np.array([0.1, 0.8, -1.5, 
                               0.1, 1.0, -1.5, 
                               -0.1, 0.8, -1.5, 
                               -0.10, 1.0, -1.5])
JOINT_DAMPING = [0.1, 0.05, 0.01,
                 0.1, 0.05, 0.01,
                 0.1, 0.05, 0.01,
                 0.1, 0.05, 0.01]

FORWARD_DIR_OFFSET = np.array([0, 0, 0])

FR_FOOT_NAME = "FR_foot"
FL_FOOT_NAME = "FL_foot"
HR_FOOT_NAME = "RR_foot"
HL_FOOT_NAME = "RL_foot"

MOCAP_MOTIONS = [
    # Output motion name, input file, frame start, frame end, motion weight.
    ["pace0", f"exts/robot_lab/robot_lab/third_party/amp_utils/datasets/keypoint_datasets/ai4animation/dog_walk00_joint_pos.txt", 162, 201, 1],
    ["pace1", f"exts/robot_lab/robot_lab/third_party/amp_utils/datasets/keypoint_datasets/ai4animation/dog_walk00_joint_pos.txt", 201, 400, 1],
    ["pace2", f"exts/robot_lab/robot_lab/third_party/amp_utils/datasets/keypoint_datasets/ai4animation/dog_walk00_joint_pos.txt", 400, 600, 1],
    ["trot0", f"exts/robot_lab/robot_lab/third_party/amp_utils/datasets/keypoint_datasets/ai4animation/dog_walk03_joint_pos.txt", 448, 481, 1],
    ["trot1", f"exts/robot_lab/robot_lab/third_party/amp_utils/datasets/keypoint_datasets/ai4animation/dog_walk03_joint_pos.txt", 400, 600, 1],
    ["trot2", f"exts/robot_lab/robot_lab/third_party/amp_utils/datasets/keypoint_datasets/ai4animation/dog_run04_joint_pos.txt", 480, 663, 1],
    ["canter0", f"exts/robot_lab/robot_lab/third_party/amp_utils/datasets/keypoint_datasets/ai4animation/dog_run00_joint_pos.txt", 430, 480, 1],
    ["canter1", f"exts/robot_lab/robot_lab/third_party/amp_utils/datasets/keypoint_datasets/ai4animation/dog_run00_joint_pos.txt", 380, 430, 1],
    ["canter2", f"exts/robot_lab/robot_lab/third_party/amp_utils/datasets/keypoint_datasets/ai4animation/dog_run00_joint_pos.txt", 480, 566, 1],
    ["right_turn0", f"exts/robot_lab/robot_lab/third_party/amp_utils/datasets/keypoint_datasets/ai4animation/dog_walk09_joint_pos.txt", 1085, 1124, 1.5],
    ["right_turn1", f"exts/robot_lab/robot_lab/third_party/amp_utils/datasets/keypoint_datasets/ai4animation/dog_walk09_joint_pos.txt", 560, 670, 1.5],
    ["left_turn0", f"exts/robot_lab/robot_lab/third_party/amp_utils/datasets/keypoint_datasets/ai4animation/dog_walk09_joint_pos.txt", 2404, 2450, 1.5],
    ["left_turn1", f"exts/robot_lab/robot_lab/third_party/amp_utils/datasets/keypoint_datasets/ai4animation/dog_walk09_joint_pos.txt", 120, 220, 1.5]
]
"""
Joint 0:
  Name: floating_base
  Type: FIXED
  Axis: (0.0, 0.0, 0.0)
  Parent Index: -1
  Lower Limit: 0.0
  Upper Limit: -1.0

Joint 1:
  Name: imu_joint
  Type: FIXED
  Axis: (0.0, 0.0, 0.0)
  Parent Index: 0
  Lower Limit: 0.0
  Upper Limit: -1.0

Joint 2:
  Name: FR_hip_joint
  Type: REVOLUTE
  Axis: (1.0, 0.0, 0.0)
  Parent Index: 0
  Lower Limit: -0.802851455917
  Upper Limit: 0.802851455917

Joint 3:
  Name: FR_hip_fixed
  Type: FIXED
  Axis: (0.0, 0.0, 0.0)
  Parent Index: 2
  Lower Limit: 0.0
  Upper Limit: -1.0

Joint 4:
  Name: FR_thigh_joint
  Type: REVOLUTE
  Axis: (0.0, 1.0, 0.0)
  Parent Index: 2
  Lower Limit: -1.0471975512
  Upper Limit: 4.18879020479

Joint 5:
  Name: FR_calf_joint
  Type: REVOLUTE
  Axis: (0.0, 1.0, 0.0)
  Parent Index: 4
  Lower Limit: -2.69653369433
  Upper Limit: -0.916297857297

Joint 6:
  Name: FR_foot_fixed
  Type: FIXED
  Axis: (0.0, 0.0, 0.0)
  Parent Index: 5
  Lower Limit: 0.0
  Upper Limit: -1.0

Joint 7:
  Name: FL_hip_joint
  Type: REVOLUTE
  Axis: (1.0, 0.0, 0.0)
  Parent Index: 0
  Lower Limit: -0.802851455917
  Upper Limit: 0.802851455917

Joint 8:
  Name: FL_hip_fixed
  Type: FIXED
  Axis: (0.0, 0.0, 0.0)
  Parent Index: 7
  Lower Limit: 0.0
  Upper Limit: -1.0

Joint 9:
  Name: FL_thigh_joint
  Type: REVOLUTE
  Axis: (0.0, 1.0, 0.0)
  Parent Index: 7
  Lower Limit: -1.0471975512
  Upper Limit: 4.18879020479

Joint 10:
  Name: FL_calf_joint
  Type: REVOLUTE
  Axis: (0.0, 1.0, 0.0)
  Parent Index: 9
  Lower Limit: -2.69653369433
  Upper Limit: -0.916297857297

Joint 11:
  Name: FL_foot_fixed
  Type: FIXED
  Axis: (0.0, 0.0, 0.0)
  Parent Index: 10
  Lower Limit: 0.0
  Upper Limit: -1.0

Joint 12:
  Name: RR_hip_joint
  Type: REVOLUTE
  Axis: (1.0, 0.0, 0.0)
  Parent Index: 0
  Lower Limit: -0.802851455917
  Upper Limit: 0.802851455917

Joint 13:
  Name: RR_hip_fixed
  Type: FIXED
  Axis: (0.0, 0.0, 0.0)
  Parent Index: 12
  Lower Limit: 0.0
  Upper Limit: -1.0

Joint 14:
  Name: RR_thigh_joint
  Type: REVOLUTE
  Axis: (0.0, 1.0, 0.0)
  Parent Index: 12
  Lower Limit: -1.0471975512
  Upper Limit: 4.18879020479

Joint 15:
  Name: RR_calf_joint
  Type: REVOLUTE
  Axis: (0.0, 1.0, 0.0)
  Parent Index: 14
  Lower Limit: -2.69653369433
  Upper Limit: -0.916297857297

Joint 16:
  Name: RR_foot_fixed
  Type: FIXED
  Axis: (0.0, 0.0, 0.0)
  Parent Index: 15
  Lower Limit: 0.0
  Upper Limit: -1.0

Joint 17:
  Name: RL_hip_joint
  Type: REVOLUTE
  Axis: (1.0, 0.0, 0.0)
  Parent Index: 0
  Lower Limit: -0.802851455917
  Upper Limit: 0.802851455917

Joint 18:
  Name: RL_hip_fixed
  Type: FIXED
  Axis: (0.0, 0.0, 0.0)
  Parent Index: 17
  Lower Limit: 0.0
  Upper Limit: -1.0

Joint 19:
  Name: RL_thigh_joint
  Type: REVOLUTE
  Axis: (0.0, 1.0, 0.0)
  Parent Index: 17
  Lower Limit: -1.0471975512
  Upper Limit: 4.18879020479

Joint 20:
  Name: RL_calf_joint
  Type: REVOLUTE
  Axis: (0.0, 1.0, 0.0)
  Parent Index: 19
  Lower Limit: -2.69653369433
  Upper Limit: -0.916297857297

Joint 21:
  Name: RL_foot_fixed
  Type: FIXED
  Axis: (0.0, 0.0, 0.0)
  Parent Index: 20
  Lower Limit: 0.0
  Upper Limit: -1.0



"""