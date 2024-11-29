from dataclasses import MISSING
from typing import Literal
from omni.isaac.lab.utils import configclass

@configclass
class SpaceCfg:
    iscontinuous:bool = True
    mu_activation:str = "none"
    mu_init:str = 'default'
    
    sigma_activation:str = "none"
    sigma_init:str = 'const_initializer'
    sigma_val:float = -2.9
    
    fixed_sigma:bool = True
    learn_sigma:bool = False
    
    


@configclass
class ASENetcfg:
    
    name:str = 'ase'
    separate_disc:bool = True
    
    Spacecfg:SpaceCfg = SpaceCfg()
    
    mlp_units:list = [1024, 1024, 512]
    disc_units:list = [1024, 1024, 512]
    enc_units:list = [1024, 512]
    enc_separate:bool = False
    initializer:str = 'default'
    activation:str = 'relu'
    
    pass

@configclass
class ASECfg:
    # 获取ASE潜在形状
    class_name: str = "ASEagent"
    normalize_value:bool = True
    normalize_input:bool = True    

    ase_latent_shape:int = 64
    latent_steps_min:int =  1
    latent_steps_max:int =  150    
    
    disc_reward_scale:int = 2
    enc_reward_scale:int = 1
    enc_coef:int = 5
    enc_weight_decay:float = 0.0000
    enc_reward_scale:int = 1
    enc_grad_penalty:int = 0
    
    task_reward_w:float = 0.0
    disc_reward_w:float = 0.5
    enc_reward_w:float = 0.5
    
    bounds_loss_coef:int = 10
    
    amp_diversity_tar:float = 1.0
    
    normalize_amp_input:bool = True

    disc_logit_reg:float =  0.01
    disc_grad_penalty:float =  5
    disc_reward_scale:float = 2
    disc_weight_decay:float = 0.0001
    
    disc_coef:float = 5.0
    enc_coef:float = 5.0
    
    amp_diversity_bonus:float = 0.01

