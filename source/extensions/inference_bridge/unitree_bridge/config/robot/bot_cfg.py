class Bot_def_cfg:
    action_space = None
    joint_order = None
    

class GO2(Bot_def_cfg):
    
    kp = 20.0
    kd = 0.5
    
    
    action_space = 12
    
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
    #

