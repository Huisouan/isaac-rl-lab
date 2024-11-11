import numpy as np

VISUALIZE_RETARGETING = True

URDF_FILENAME = "datasets/go2_description/urdf/go2_description.urdf"
OUTPUT_DIR = f"datasets/mocap_motions_go2/"

REF_POS_SCALE = 0.825 # 缩放系数,如果遇到关节限位异常，尝试将此数变小
INIT_POS = np.array([0, 0, 0.32]) # a1
INIT_ROT = np.array([0, 0, 0, 1.0])

"""
Joint Index: 0, Joint Name: Head_upper_joint, Joint Type: 4
Joint Index: 1, Joint Name: Head_lower_joint, Joint Type: 4
Joint Index: 2, Joint Name: FL_hip_joint, Joint Type: 0
Joint Index: 3, Joint Name: FL_thigh_joint, Joint Type: 0
Joint Index: 4, Joint Name: FL_calf_joint, Joint Type: 0
Joint Index: 5, Joint Name: FL_calflower_joint, Joint Type: 4
Joint Index: 6, Joint Name: FL_calflower1_joint, Joint Type: 4
Joint Index: 7, Joint Name: FL_foot_joint, Joint Type: 4
Joint Index: 8, Joint Name: FL_calf_rotor_joint, Joint Type: 4
Joint Index: 9, Joint Name: FL_thigh_rotor_joint, Joint Type: 4
Joint Index: 10, Joint Name: FL_hip_rotor_joint, Joint Type: 4
Joint Index: 11, Joint Name: FR_hip_joint, Joint Type: 0
Joint Index: 12, Joint Name: FR_thigh_joint, Joint Type: 0
Joint Index: 13, Joint Name: FR_calf_joint, Joint Type: 0
Joint Index: 14, Joint Name: FR_calflower_joint, Joint Type: 4
Joint Index: 15, Joint Name: FR_calflower1_joint, Joint Type: 4
Joint Index: 16, Joint Name: FR_foot_joint, Joint Type: 4
Joint Index: 17, Joint Name: FR_calf_rotor_joint, Joint Type: 4
Joint Index: 18, Joint Name: FR_thigh_rotor_joint, Joint Type: 4
Joint Index: 19, Joint Name: FR_hip_rotor_joint, Joint Type: 4
Joint Index: 20, Joint Name: RL_hip_joint, Joint Type: 0
Joint Index: 21, Joint Name: RL_thigh_joint, Joint Type: 0
Joint Index: 22, Joint Name: RL_calf_joint, Joint Type: 0
Joint Index: 23, Joint Name: RL_calflower_joint, Joint Type: 4
Joint Index: 24, Joint Name: RL_calflower1_joint, Joint Type: 4
Joint Index: 25, Joint Name: RL_foot_joint, Joint Type: 4
Joint Index: 26, Joint Name: RL_calf_rotor_joint, Joint Type: 4
Joint Index: 27, Joint Name: RL_thigh_rotor_joint, Joint Type: 4
Joint Index: 28, Joint Name: RL_hip_rotor_joint, Joint Type: 4
Joint Index: 29, Joint Name: RR_hip_joint, Joint Type: 0
Joint Index: 30, Joint Name: RR_thigh_joint, Joint Type: 0
Joint Index: 31, Joint Name: RR_calf_joint, Joint Type: 0
Joint Index: 32, Joint Name: RR_calflower_joint, Joint Type: 4
Joint Index: 33, Joint Name: RR_calflower1_joint, Joint Type: 4
Joint Index: 34, Joint Name: RR_foot_joint, Joint Type: 4
Joint Index: 35, Joint Name: RR_calf_rotor_joint, Joint Type: 4
Joint Index: 36, Joint Name: RR_thigh_rotor_joint, Joint Type: 4
Joint Index: 37, Joint Name: RR_hip_rotor_joint, Joint Type: 4
Joint Index: 38, Joint Name: imu_joint, Joint Type: 4
Joint Index: 39, Joint Name: radar_joint, Joint Type: 4
    """

SIM_TOE_JOINT_IDS = [16, 7, 34, 25]
SIM_HIP_JOINT_IDS = [11, 2, 29, 20]
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
  ["pace", "datasets/keypoint_datasets/ai4animation/motions/dog_walk00_joint_pos.txt",162,201,1],
  ["trot", "datasets/keypoint_datasets/ai4animation/motions/dog_walk03_joint_pos.txt",448,481 ,1],
  ["trot2", "datasets/keypoint_datasets/ai4animation/motions/dog_run04_joint_pos.txt",630,663 ,1],
  ["canter", "datasets/keypoint_datasets/ai4animation/motions/dog_run00_joint_pos.txt", 430, 459,1],
  ["left turn0", "datasets/keypoint_datasets/ai4animation/motions/dog_walk09_joint_pos.txt",1085,1124 ,1.5],
  ["right turn0", "datasets/keypoint_datasets/ai4animation/motions/dog_walk09_joint_pos.txt", 2404,2450,1.5],
  
]

