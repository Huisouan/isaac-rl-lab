# 版权所有 (c) 2022-2024，Isaac Lab 项目开发者。
# 保留所有权利。
#
# SPDX 许可标识符: BSD-3-Clause

"""包含针对各种机器人环境的任务实现的包。"""

import os
import toml

# 通过相对路径方便地指向其他模块目录
ISAACLAB_TASKS_EXT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "../"))
"""扩展源目录的路径。"""

ISAACLAB_TASKS_METADATA = toml.load(os.path.join(ISAACLAB_TASKS_EXT_DIR, "config", "extension.toml"))
"""从 extension.toml 文件解析出的扩展元数据字典。"""

# 配置模块级别的变量
__version__ = ISAACLAB_TASKS_METADATA["package"]["version"]

##
# 注册 Gym 环境。
##

from omni.isaac.lab_tasks.utils import import_packages

# 黑名单用于防止从子包导入配置
_BLACKLIST_PKGS = ["utils"]
# 导入此包中的所有配置
import_packages(__name__, _BLACKLIST_PKGS)