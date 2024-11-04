from dataclasses import MISSING
from typing import Literal
from omni.isaac.lab.utils import configclass

class SpaceCfg:
    iscontinuous:bool = True
    mu_activation:str = "none"
    mu_init:str = 'default'
    
    sigma_activation:str = "none"
    sigma_init:str = 'const_initializer'
    sigma_val:float = -2.9
    
    fixed_sigma:bool = True
    learn_sigma:bool = False

class ASECfg:
    # 获取ASE潜在形状
    ase_latent_shape:int = 64

class ASENetcfg:
    name:str = 'ase'
    separate_disc:bool = True
    
    Spacecfg:SpaceCfg = SpaceCfg
    
    mlp_units:list = [1024, 1024, 512]
    disc_units:list = [1024, 1024, 512]
    enc_units:list = [1024, 512]
    
    initializer:str = 'default'
    activation:str = 'relu'
    
    pass