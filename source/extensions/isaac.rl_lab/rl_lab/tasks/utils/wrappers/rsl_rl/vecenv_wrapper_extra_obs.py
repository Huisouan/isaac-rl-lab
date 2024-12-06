import torch
from .vecenv_wrapper import RslRlVecEnvWrapper
from omni.isaac.lab.envs import DirectRLEnv, ManagerBasedRLEnv

class RslRlVecEnvWrapperextra(RslRlVecEnvWrapper):
    def __init__(self, env: ManagerBasedRLEnv | DirectRLEnv):
        """Initializes the wrapper.

        Note:
            The wrapper calls :meth:`reset` at the start since the RSL-RL runner does not call reset.

        Args:
            env: The environment to wrap around.

        Raises:
            ValueError: When the environment is not an instance of :class:`ManagerBasedRLEnv` or :class:`DirectRLEnv`.
        """
        # check that input is valid
        if not isinstance(env.unwrapped, ManagerBasedRLEnv) and not isinstance(env.unwrapped, DirectRLEnv):
            raise ValueError(
                "The environment must be inherited from ManagerBasedRLEnv or DirectRLEnv. Environment type:"
                f" {type(env)}"
            )
        # initialize the wrapper
        self.env = env
        # store information required by wrapper
        self.num_envs = self.unwrapped.num_envs
        self.device = self.unwrapped.device
        self.max_episode_length = self.unwrapped.max_episode_length
        if hasattr(self.unwrapped, "action_manager"):
            self.num_actions = self.unwrapped.action_manager.total_action_dim
        else:
            self.num_actions = self.unwrapped.num_actions
        if hasattr(self.unwrapped, "observation_manager"):
            if hasattr(self.unwrapped, "compute_observations"):
                self.num_obs = self.unwrapped.num_observations
            else:
                self.num_obs = self.unwrapped.observation_manager.group_obs_dim["policy"][0]
        else:
            self.num_obs = self.unwrapped.num_observations
        # -- privileged observations
        if (
            hasattr(self.unwrapped, "observation_manager")
            and "critic" in self.unwrapped.observation_manager.group_obs_dim
        ):
            self.num_privileged_obs = self.unwrapped.observation_manager.group_obs_dim["critic"][0]
        elif hasattr(self.unwrapped, "compute_observations"):
            self.num_privileged_obs = self.unwrapped.privileged_obs_dim     
        elif hasattr(self.unwrapped, "num_states"):
            self.num_privileged_obs = self.unwrapped.num_states
        else:
            self.num_privileged_obs = 0
        # reset at the start since the RSL-RL runner does not call reset
        self.env.reset()
    
    def get_observations(self) -> tuple[torch.Tensor, dict]:
        """Returns the current observations of the environment."""
        if hasattr(self.unwrapped, "observation_manager"):
            if hasattr(self.unwrapped, "compute_observations"):
                obs_dict = self.unwrapped.compute_observations()
            else:
                obs_dict = self.unwrapped.observation_manager.compute()
        else:
            obs_dict = self.unwrapped._get_observations()
        return obs_dict["policy"], {"observations": obs_dict}