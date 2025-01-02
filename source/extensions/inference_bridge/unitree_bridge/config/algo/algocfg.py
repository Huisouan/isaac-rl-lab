class Algo_def_cfg:
    model_path = None
    
    policy_observation_dim = None

    policy_action_dim = None

    joint_order = None
    
    default_jointpos_bias = None

class A2CConfig_ge(Algo_def_cfg):
    model_path = "weights/Go2ActorCritic/policy.pt"
    
    policy_observation_dim = 45

    policy_action_dim = 12

    joint_order = (
    "FR_hip",   # hip
    "FR_thigh", # thigh
    "FR_calf",  # calf
    "FL_hip",   # hip
    "FL_thigh", # thigh
    "FL_calf",  # calf
    "RR_hip",   # hip
    "RR_thigh", # thigh
    "RR_calf",  # calf
    "RL_hip",   # hip
    "RL_thigh", # thigh
    "RL_calf"   # calf
)
    
    default_jointpos_bias = [
        0.0,0.8,-1.5,
        0.0,0.8,-1.5,
        0.0,1.0,-1.5,
        0.0,1.0,-1.5,
    ]    
    
    base_ang_vel_scale = 0.1
    joint_pos_scale = 1.0
    joint_vel_scale = 0.05
    actions_scale = 0.25

class HIMConfig_lab(Algo_def_cfg):
    model_path = "weights/unitree_go2_him_rough/2024-12-16_17-33-52/exported/policy.pt"
    
    policy_observation_dim = 270
    policy_action_dim = 12
    
    joint_order = (
    "FL_hip",
    "FR_hip",
    "RL_hip",
    "RR_hip",
    "FL_thigh",
    "FR_thigh",
    "RL_thigh",
    "RR_thigh",
    "FL_calf",
    "FR_calf",
    "RL_calf",
    "RR_calf",
)
    
    default_jointpos_bias = [ 
        0.1000, -0.1000,  0.1000, -0.1000,  
        0.8000,  0.8000,  1.0000,  1.0000,
        -1.5000, -1.5000, -1.5000, -1.5000]   
    
    base_ang_vel_scale = 0.25/1.5
    joint_pos_scale = 1.0
    joint_vel_scale = 0.05
    
    actions_joint_pos_scale = 0.25