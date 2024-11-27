# Copyright (c) 2022-2024, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Script to play a checkpoint if an RL agent from RSL-RL."""

"""Launch Isaac Sim Simulator first."""

# 导入所需的时间模块
import time
# 导入系统模块，用于处理命令行参数
import sys
import argparse

from omni.isaac.lab.app import AppLauncher

# local imports
import cli_args  # isort: skip

# add argparse arguments
parser = argparse.ArgumentParser(description="Train an RL agent with RSL-RL.")
parser.add_argument("--video", action="store_true", default=False, help="Record videos during training.")
parser.add_argument("--video_length", type=int, default=200, help="Length of the recorded video (in steps).")
parser.add_argument(
    "--disable_fabric", action="store_true", default=False, help="Disable fabric and use USD I/O operations."
)
parser.add_argument("--num_envs", type=int, default='1', help="Number of environments to simulate.")
parser.add_argument("--task", type=str, default="Isaac-Amp-Unitree-go2-v0", help="Name of the task.")
# append RSL-RL cli arguments
cli_args.add_rsl_rl_args(parser)
# append AppLauncher cli args
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()
# always enable cameras to record video
if args_cli.video:
    args_cli.enable_cameras = True

# launch omniverse app
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

"""Rest everything follows."""

import gymnasium as gym
import os
import torch

from rl_lab.rsl_rl.runners import AmpOnPolicyRunner

from omni.isaac.lab.envs import DirectMARLEnv, multi_agent_to_single_agent
from omni.isaac.lab.utils.dict import print_dict
import rl_lab.tasks
import omni.isaac.lab_tasks  # noqa: F401
from omni.isaac.lab_tasks.utils import get_checkpoint_path, parse_env_cfg
from omni.isaac.lab_tasks.utils.wrappers.rsl_rl import (
    RslRlOnPolicyRunnerCfg,
    RslRlVecEnvWrapper,
    export_policy_as_jit,
    export_policy_as_onnx,
)
from unitree_sdk2_python.example.go2.low_level.go2_pd_control import Go2_PD_Control

from omni.isaac.lab.devices import Se2Gamepad
# 从unitree_sdk2py核心模块导入ChannelPublisher类，用于发布消息
from unitree_sdk2_python.unitree_sdk2py.core.channel import  ChannelFactoryInitialize
# 从unitree_sdk2py核心模块导入ChannelSubscriber类，用于订阅消息
from unitree_sdk2_python.unitree_sdk2py.core.channel import  ChannelFactoryInitialize

# 默认网络接口名称
default_network = 'enp0s31f6'

def main():
    """Play with RSL-RL agent."""
    # parse configuration
    env_cfg = parse_env_cfg(
        args_cli.task, device=args_cli.device, num_envs=args_cli.num_envs, use_fabric=not args_cli.disable_fabric
    )
    
    if args_cli.task == "Isaac-Amp-Unitree-go2-v0":
        print("[INFO] Using AmpOnPolicyRunner")
        env_cfg.amp_num_preload_transitions = 10
    
    agent_cfg: RslRlOnPolicyRunnerCfg = cli_args.parse_rsl_rl_cfg(args_cli.task, args_cli)


    # specify directory for logging experiments
    log_root_path = os.path.join("logs", "rsl_rl", agent_cfg.experiment_name)
    log_root_path = os.path.abspath(log_root_path)
    print(f"[INFO] Loading experiment from directory: {log_root_path}")
    resume_path = get_checkpoint_path(log_root_path, agent_cfg.load_run, agent_cfg.load_checkpoint)
    log_dir = os.path.dirname(resume_path)

    # create isaac environment
    env = gym.make(args_cli.task, cfg=env_cfg, render_mode="rgb_array" if args_cli.video else None)
    # wrap for video recording
    if args_cli.video:
        video_kwargs = {
            "video_folder": os.path.join(log_dir, "videos", "play"),
            "step_trigger": lambda step: step == 0,
            "video_length": args_cli.video_length,
            "disable_logger": True,
        }
        print("[INFO] Recording videos during training.")
        print_dict(video_kwargs, nesting=4)
        env = gym.wrappers.RecordVideo(env, **video_kwargs)

    # convert to single-agent instance if required by the RL algorithm
    if isinstance(env.unwrapped, DirectMARLEnv):
        env = multi_agent_to_single_agent(env)

    # wrap around environment for rsl-rl
    env = RslRlVecEnvWrapper(env)

    print(f"[INFO]: Loading model checkpoint from: {resume_path}")
    # load previously trained model
    ppo_runner = AmpOnPolicyRunner(env, agent_cfg.to_dict(), log_dir=None, device=agent_cfg.device)
    ppo_runner.load(resume_path)

    # obtain the trained policy for inference
    policy = ppo_runner.get_inference_policy(device=env.unwrapped.device)

    # export policy to onnx/jit
    export_model_dir = os.path.join(os.path.dirname(resume_path), "exported")
   # export_policy_as_jit(
   #     ppo_runner.alg.actor_critic, ppo_runner.obs_normalizer, path=export_model_dir, filename="policy.pt"
    #)
    #export_policy_as_onnx(
    #    ppo_runner.alg.actor_critic, normalizer=ppo_runner.obs_normalizer, path=export_model_dir, filename="policy.onnx"
    #)

    # reset environment
    obs, _ = env.get_observations()
    timestep = 0
    
    joystick = Se2Gamepad()
    
    # simulate environment
    while simulation_app.is_running():
        # run everything in inference mode
        with torch.inference_mode():
            # agent stepping
            actions = policy(obs)
            # env stepping
            obs, _, _, _ = env.step(actions)
            
            readings = joystick.advance()
            if readings[0] < 0:
                readings[0] = 0.5 * readings[0]
            else :
                readings[0] = 1*readings[0]
            readings[1] = 0.15 * readings[1]
            readings[2] = -1.5*readings[2]
            readings_tensor = torch.from_numpy(readings*2).cuda()  # 将 NumPy 数组转换为 PyTorch 张量，并移动到 GPU
            obs[0][4:7] = readings_tensor.float()
            print(obs[0][4:7].tolist())
            
            
        if args_cli.video:
            timestep += 1
            # Exit the play loop after recording one video
            if timestep == args_cli.video_length:
                break

    # close the simulator
    env.close()


if __name__ == "__main__":
    # run the main function
    # 警告提示
    print("WARNING: Please ensure there are no obstacles around the robot while running this example.")
    # 等待用户确认
    input("Press Enter to continue...")

    # 初始化通道工厂
    if len(sys.argv)>1:
        ChannelFactoryInitialize(0, sys.argv[1])
    else:
        ChannelFactoryInitialize(0,default_network)

    # 创建Custom对象
    custom = Go2_PD_Control()
    # 初始化Custom对象
    custom.Init()
    # 启动Custom对象
    custom.Start()    
    main()
    time.sleep(1)
    print("Done!")
    sys.exit(-1) 
    # close sim app
    simulation_app.close()