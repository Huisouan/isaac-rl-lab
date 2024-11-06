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
class AMPCfg:
    # 获取AMP潜在形状
    ase_latent_shape:int = 64

@configclass
class AMPNetcfg:
    name:str = 'amp'
    separate_disc:bool = True
    
    Spacecfg:SpaceCfg = SpaceCfg
    
    mlp_units:list = [1024, 1024, 512]
    disc_units:list = [1024, 1024, 512]
    enc_units:list = [1024, 512]
    enc_separate: False
    initializer:str = 'default'
    activation:str = 'relu'
    
    pass
