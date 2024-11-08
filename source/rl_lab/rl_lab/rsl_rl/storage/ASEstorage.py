from __future__ import annotations

import torch

from rsl_rl.utils import split_and_pad_trajectories

from .rollout_storage import RolloutStorage

class ASERolloutStorage(RolloutStorage):
    class ASETransition(RolloutStorage.Transition):
        def __init__(self):
            super().__init__()
            self.ase_latent = None
    
    def __init__(self, num_envs, num_transitions_per_env, obs_shape, privileged_obs_shape, actions_shape, device="cpu"):
        super().__init__(num_envs, num_transitions_per_env, obs_shape, privileged_obs_shape, actions_shape, device)
        self.ase_latent = torch.zeros(num_transitions_per_env, num_envs, device=self.device)