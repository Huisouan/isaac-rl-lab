import numpy as np
from amp_utils import AMP_UTILS_DIR

VISUALIZE_RETARGETING = True

URDF_FILENAME = f"exts/robot_lab/robot_lab/third_party/amp_utils/models/go2_description/urdf/go2_description.urdf"
OUTPUT_DIR = f"exts/robot_lab/robot_lab/third_party/amp_utils/motion_files/mocap_motions_go2/"

REF_POS_SCALE = 0.825 # 缩放系数,如果遇到关节限位异常，尝试将此数变小
INIT_POS = np.array([0, 0, 0.32]) # a1
INIT_ROT = np.array([0, 0, 0, 1.0])

SIM_TOE_JOINT_IDS = [7,13,19,25]
SIM_HIP_JOINT_IDS = [2,8,14,20]
SIM_ROOT_OFFSET = np.array([0, 0, -0.04])
SIM_TOE_OFFSET_LOCAL = [
    np.array([0.0, 0.06, 0.0]),
    np.array([0.0, -0.06, 0.0]),
    np.array([0.0, 0.06, 0.0]),
    np.array([0.0, -0.06, 0.0])
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

MOCAP_DIR = "exts/robot_lab/amp_utils/datasets/lssp_keypoints"
"""
Joint 0:
  Name: Head_upper_joint
  Type: FIXED
  Axis: (0.0, 0.0, 0.0)
  Parent Index: -1
  Lower Limit: 0.0
  Upper Limit: -1.0

Joint 1:
  Name: Head_lower_joint
  Type: FIXED
  Axis: (0.0, 0.0, 0.0)
  Parent Index: 0
  Lower Limit: 0.0
  Upper Limit: -1.0

Joint 2:
  Name: FL_hip_joint
  Type: REVOLUTE
  Axis: (1.0, 0.0, 0.0)
  Parent Index: -1
  Lower Limit: -1.0472
  Upper Limit: 1.0472

Joint 3:
  Name: FL_thigh_joint
  Type: REVOLUTE
  Axis: (0.0, 1.0, 0.0)
  Parent Index: 2
  Lower Limit: -1.5708
  Upper Limit: 3.4907

Joint 4:
  Name: FL_calf_joint
  Type: REVOLUTE
  Axis: (0.0, 1.0, 0.0)
  Parent Index: 3
  Lower Limit: -2.7227
  Upper Limit: -0.83776

Joint 5:
  Name: FL_calflower_joint
  Type: FIXED
  Axis: (0.0, 0.0, 0.0)
  Parent Index: 4
  Lower Limit: 0.0
  Upper Limit: -1.0
RL
Joint 6:
  Name: FL_calflower1_joint
  Type: FIXED
  Axis: (0.0, 0.0, 0.0)
  Parent Index: 5
  Lower Limit: 0.0
  Upper Limit: -1.0

Joint 7:
  Name: FL_foot_joint
  Type: FIXED
  Axis: (0.0, 0.0, 0.0)
  Parent Index: 4
  Lower Limit: 0.0
  Upper Limit: -1.0

Joint 8:
  Name: FR_hip_joint
  Type: REVOLUTE
  Axis: (1.0, 0.0, 0.0)
  Parent Index: -1
  Lower Limit: -1.0472
  Upper Limit: 1.0472

Joint 9:
  Name: FR_thigh_joint
  Type: REVOLUTE
  Axis: (0.0, 1.0, 0.0)
  Parent Index: 8
  Lower Limit: -1.5708
  Upper Limit: 3.4907

Joint 10:
  Name: FR_calf_joint
  Type: REVOLUTE
  Axis: (0.0, 1.0, 0.0)
  Parent Index: 9
  Lower Limit: -2.7227
  Upper Limit: -0.83776

Joint 11:
  Name: FR_calflower_joint
  Type: FIXED
  Axis: (0.0, 0.0, 0.0)
  Parent Index: 10
  Lower Limit: 0.0
  Upper Limit: -1.0

Joint 12:
  Name: FR_calflower1_joint
  Type: FIXED
  Axis: (0.0, 0.0, 0.0)
  Parent Index: 11
  Lower Limit: 0.0
  Upper Limit: -1.0

Joint 13:
  Name: FR_foot_joint
  Type: FIXED
  Axis: (0.0, 0.0, 0.0)
  Parent Index: 10
  Lower Limit: 0.0
  Upper Limit: -1.0

Joint 14:
  Name: RL_hip_joint
  Type: REVOLUTE
  Axis: (1.0, 0.0, 0.0)
  Parent Index: -1
  Lower Limit: -1.0472
  Upper Limit: 1.0472

Joint 15:
  Name: RL_thigh_joint
  Type: REVOLUTE
  Axis: (0.0, 1.0, 0.0)
  Parent Index: 14
  Lower Limit: -0.5236
  Upper Limit: 4.5379

Joint 16:
  Name: RL_calf_joint
  Type: REVOLUTE
  Axis: (0.0, 1.0, 0.0)
  Parent Index: 15
  Lower Limit: -2.7227
  Upper Limit: -0.83776

Joint 17:
  Name: RL_calflower_joint
  Type: FIXED
  Axis: (0.0, 0.0, 0.0)
  Parent Index: 16
  Lower Limit: 0.0
  Upper Limit: -1.0

Joint 18:
  Name: RL_calflower1_joint
  Type: FIXED
  Axis: (0.0, 0.0, 0.0)
  Parent Index: 17
  Lower Limit: 0.0
  Upper Limit: -1.0

Joint 19:
  Name: RL_foot_joint
  Type: FIXED
  Axis: (0.0, 0.0, 0.0)
  Parent Index: 16
  Lower Limit: 0.0
  Upper Limit: -1.0

Joint 20:
  Name: RR_hip_joint
  Type: REVOLUTE
  Axis: (1.0, 0.0, 0.0)
  Parent Index: -1
  Lower Limit: -1.0472
  Upper Limit: 1.0472

Joint 21:
  Name: RR_thigh_joint
  Type: REVOLUTE
  Axis: (0.0, 1.0, 0.0)
  Parent Index: 20
  Lower Limit: -0.5236
  Upper Limit: 4.5379

Joint 22:
  Name: RR_calf_joint
  Type: REVOLUTE
  Axis: (0.0, 1.0, 0.0)
  Parent Index: 21
  Lower Limit: -2.7227
  Upper Limit: -0.83776

Joint 23:
  Name: RR_calflower_joint
  Type: FIXED
  Axis: (0.0, 0.0, 0.0)
  Parent Index: 22
  Lower Limit: 0.0
  Upper Limit: -1.0

Joint 24:
  Name: RR_calflower1_joint
  Type: FIXED
  Axis: (0.0, 0.0, 0.0)
  Parent Index: 23
  Lower Limit: 0.0
  Upper Limit: -1.0

Joint 25:
  Name: RR_foot_joint
  Type: FIXED
  Axis: (0.0, 0.0, 0.0)
  Parent Index: 22
  Lower Limit: 0.0
  Upper Limit: -1.0

Joint 26:
  Name: imu_joint
  Type: FIXED
  Axis: (0.0, 0.0, 0.0)
  Parent Index: -1
  Lower Limit: 0.0
  Upper Limit: -1.0

Joint 27:
  Name: radar_joint
  Type: FIXED
  Axis: (0.0, 0.0, 0.0)
  Parent Index: -1
  Lower Limit: 0.0
  Upper Limit: -1.0





"""