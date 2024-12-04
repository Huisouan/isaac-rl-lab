import torch
from .vecenv_wrapper import RslRlVecEnvWrapper
from omni.isaac.lab.envs import DirectRLEnv, ManagerBasedRLEnv

class RslRlVecEnvWrapperextra(RslRlVecEnvWrapper):
    def __init__(self, env: ManagerBasedRLEnv | DirectRLEnv):
        super().__init__(env)
    
    def get_extra_obs(self) -> tuple[torch.Tensor, dict]:
        """Returns the current observations of the environment."""
        if hasattr(self.unwrapped, "observation_manager"):
            obs_dict = self.unwrapped.
        else:
            obs_dict = self.unwrapped._get_observations()
        return obs_dict["policy"], {"observations": obs_dict}