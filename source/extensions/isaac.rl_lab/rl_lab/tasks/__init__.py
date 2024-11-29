# 版权所有 (c) 2022-2024，Isaac Lab 项目开发者。
# 保留所有权利。
#
# SPDX 许可标识符: BSD-3-Clause

"""包含针对各种机器人环境的任务实现的包。"""

from omni.isaac.lab_tasks.utils import import_packages

# 黑名单用于防止从子包导入配置
_BLACKLIST_PKGS = ["utils"]
# 导入此包中的所有配置
import_packages(__name__, _BLACKLIST_PKGS)