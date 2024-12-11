#  Copyright 2021 ETH Zurich, NVIDIA CORPORATION
#  SPDX-License-Identifier: BSD-3-Clause

"""Implementation of different RL agents."""

from .pmcppo import PMCPPO
from .epmcppo import EPMCPPO
from .aseppo import ASEPPO
from .aseppov1 import ASEPPOV1
from .amp_discriminator import AMPDiscriminator
from .amp_ppo import AMPPPO
from .cvqvaeppo import CVQVAEPPO
from .ppo import PPO
from .him_ppo import HIMPPO

__all__ = [ "PPO"
            "PMCPPO", "EPMCPPO","ASEPPO","ASEPPOV1",
            "AMPDiscriminator","AMPPPO","CVQVAEPPO",
            "HIMPPO",
           ]
