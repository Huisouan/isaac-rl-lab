from dataclasses import MISSING
from typing import Literal

from omni.isaac.lab.utils import configclass

@configclass
class Z_settings:
    
    z_length: int = 32
    """the dimention of the vector of the embedding space"""
    
    num_embeddings: int = 256
    """the num of the embeding vector of the codebook"""
    
    norm_z:bool = False
    bot_neck_z_embed_size:int = 32
    bot_neck_prop_embed_size:int = 64