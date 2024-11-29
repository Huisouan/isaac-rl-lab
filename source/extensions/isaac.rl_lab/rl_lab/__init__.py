#
# SPDX-License-Identifier: BSD-3-Clause

"""Package containing asset and sensor configurations."""

import os
import toml

RL_LAB_EXT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "../"))
"""Path to the extension source directory."""

RL_LAB_DATA_DIR = os.path.join(RL_LAB_EXT_DIR, "data")
"""Path to the extension data directory."""

RL_LAB_METADATA = toml.load(os.path.join(RL_LAB_EXT_DIR, "config", "extension.toml"))
"""Extension metadata dictionary parsed from the extension.toml file."""

##
# Configuration for different assets.
##
