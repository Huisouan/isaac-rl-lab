#  Copyright 2021 ETH Zurich, NVIDIA CORPORATION
#  SPDX-License-Identifier: BSD-3-Clause

"""Implementation of different RL agents."""

from .pmcppo import PMCPPO
from .epmcppo import EPMCPPO
__all__ = ["PMCPPO", "EPMCPPO"]
