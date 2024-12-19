# Copyright (c) 2022-2024, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Script to play a checkpoint if an RL agent from RSL-RL."""

"""Launch Isaac Sim Simulator first."""

import argparse
import pandas as pd
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
parser.add_argument("--task", type=str, default="Isaac-Rough-Him-Unitree-go2-v0-play", help="Name of the task.")
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

from rl_lab.rsl_rl.runners import PmcOnPolicyRunner,AmpOnPolicyRunner,CvqvaeOnPolicyRunner,ASEOnPolicyRunner,HIMOnPolicyRunner

from omni.isaac.lab.envs import DirectMARLEnv, multi_agent_to_single_agent
from omni.isaac.lab.utils.dict import print_dict
import rl_lab.tasks
import omni.isaac.lab_tasks  # noqa: F401
from omni.isaac.lab_tasks.utils import get_checkpoint_path, parse_env_cfg
from omni.isaac.lab_tasks.utils.wrappers.rsl_rl import (
    RslRlOnPolicyRunnerCfg,

    export_policy_as_jit,
    export_policy_as_onnx,
)
from rl_lab.tasks.utils.wrappers.rsl_rl import RslRlVecEnvWrapperextra

from omni.isaac.lab.devices import Se2Gamepad


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
    log_root_path = os.path.join("weights", agent_cfg.experiment_name)
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
    env = RslRlVecEnvWrapperextra(env)

    if args_cli.task == "Isaac-Amp-Unitree-go2-v0":
        print("[INFO] Using AmpOnPolicyRunner")
        ppo_runner = AmpOnPolicyRunner(env, agent_cfg.to_dict(), log_dir=log_dir, device=agent_cfg.device)
    
    elif args_cli.task == "Isaac-Him-Unitree-go2-v0" or args_cli.task =="Isaac-Rough-Him-Unitree-go2-v0" or args_cli.task=="Isaac-Rough-Him-Unitree-go2-v0-play":
        print("[INFO] Using HimOnPolicyRunner")
        ppo_runner = HIMOnPolicyRunner(env, agent_cfg.to_dict(), log_dir=log_dir, device=agent_cfg.device)        
    
    elif args_cli.task == "Isaac-Ase-Unitree-go2-v0":
        print("[INFO] Using AseOnPolicyRunner")
        ppo_runner = ASEOnPolicyRunner(env, agent_cfg.to_dict(), log_dir=log_dir, device=agent_cfg.device)

    elif args_cli.task == "Isaac-go2-pmc-Direct-v0":
        print("[INFO] Using PmcOnPolicyRunner")
        ppo_runner = PmcOnPolicyRunner(env, agent_cfg.to_dict(), log_dir=log_dir, device=agent_cfg.device)
    
    elif args_cli.task == "Isaac-go2-cvqvae-Direct-v0":
        ppo_runner = CvqvaeOnPolicyRunner(env, agent_cfg.to_dict(), log_dir=log_dir, device=agent_cfg.device)
    else:
        raise NotImplementedError
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

    # 创建目录用于保存数据
    data_dir = "recorddata"
    if not os.path.exists(data_dir):
        os.makedirs(data_dir)
    
    # 记录文件路径
    record_file_path = os.path.join("algo_obs1.csv")
    first_write = True  # 用于判断是否第一次写入数据    
    
    joystick = Se2Gamepad()
    print_count = 0
    # simulate environment
    while simulation_app.is_running():
        # run everything in inference mode
        
        with torch.inference_mode():
            # agent stepping
            
            actions = policy(obs)

            # env stepping
            obs, _, _, _ = env.step(actions)
            
            if print_count % 200 == 0:
                # 限制打印的精度到小数点后三位
                print([f"{x:.3f}" for x in actions[0].tolist()])
                #print("pos:",env.unwrapped.action_manager._terms['joint_pos']._processed_actions)
                print_count = 0
            else :
                print_count += 1            

            
            readings = joystick.advance()
            if readings[0] < 0:
                readings[0] =  readings[0]
            else :
                readings[0] = 2*readings[0]
            readings[1] = 0.15 * readings[1]
            readings[2] = -1.5*readings[2]
            readings_tensor = torch.from_numpy(readings).cuda()  # 将 NumPy 数组转换为 PyTorch 张量，并移动到 GPU
            for i in range(6):
                obs[0][6+i*45:9+i*45] = readings_tensor
            
            
            
            print(obs[0][6:9].tolist())
            first_write = record_algo_obs(first_write,record_file_path, obs[0][:45])            
            
        if args_cli.video:
            timestep += 1
            # Exit the play loop after recording one video
            if timestep == args_cli.video_length:
                break

    # close the simulator
    env.close()

def record_algo_obs(first_write, record_file_path, single_obs):
    """记录 algo_obs 数据到 CSV 文件"""
    # 根据观察值的具体含义定义表头
    observation_names = [
        "base_ang_vel_x", "base_ang_vel_y", "base_ang_vel_z",
        "projected_gravity_x", "projected_gravity_y", "projected_gravity_z",
        "velocity_command_x", "velocity_command_y", "velocity_command_z",
        "joint_pos_0", "joint_pos_1", "joint_pos_2", "joint_pos_3", "joint_pos_4", "joint_pos_5",
        "joint_pos_6", "joint_pos_7", "joint_pos_8", "joint_pos_9", "joint_pos_10", "joint_pos_11",
        "joint_vel_0", "joint_vel_1", "joint_vel_2", "joint_vel_3", "joint_vel_4", "joint_vel_5",
        "joint_vel_6", "joint_vel_7", "joint_vel_8", "joint_vel_9", "joint_vel_10", "joint_vel_11",
        "action_0", "action_1", "action_2", "action_3", "action_4", "action_5",
        "action_6", "action_7", "action_8", "action_9", "action_10", "action_11"
    ]
    
    data_dict = {name: [value] for name, value in zip(observation_names, single_obs.tolist())}
    
    df = pd.DataFrame(data_dict)
    
    if first_write:
        df.to_csv(record_file_path, mode='w', index=False)
        first_write = False
    else:
        df.to_csv(record_file_path, mode='a', header=False, index=False)
    return first_write



if __name__ == "__main__":
    # run the main function
    main()
    # close sim app
    simulation_app.close()
