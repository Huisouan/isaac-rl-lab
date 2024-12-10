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

import threading

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

from omni.isaac.lab.devices import Se2Gamepad
from unitree_sdk2py.core.channel import  ChannelFactoryInitialize

from unitree_sdk2py.go2.low_level.go2_pd_control import Go2_PD_Control,get_key,process_key
# 默认网络接口名称
default_network = 'enp0s31f6'

model_joint_order = (
    "FL_hip",
    "FR_hip",
    "RL_hip",
    "RR_hip",
    "FL_thigh",
    "FR_thigh",
    "RL_thigh",
    "RR_thigh",
    "FL_calf",
    "FR_calf",
    "RL_calf",
    "RR_calf",
)
go2_joint_current_order = (
    "FR_hip",   # hip
    "FR_thigh", # thigh
    "FR_calf",  # calf
    "FL_hip",   # hip
    "FL_thigh", # thigh
    "FL_calf",  # calf
    "RR_hip",   # hip
    "RR_thigh", # thigh
    "RR_calf",  # calf
    "RL_hip",   # hip
    "RL_thigh", # thigh
    "RL_calf"   # calf
)

def joint_reorder(tensor, src_order, tgt_order):
    assert len(src_order) == len(tgt_order), f"src_order and tgt_order must have same length."
    index_map = [src_order.index(joint) for joint in tgt_order]
    tensor_new = tensor[index_map]
    return tensor_new

def go2_obs_process(imu_state,motor_state):

    quaternion = torch.zeros(4)
    for i in range(4):
        quaternion[i] = imu_state.quaternion[i]

    joint_pos = torch.zeros(12)
    joint_vel = torch.zeros(12)
    for i in range(12):
        joint_pos[i] = motor_state[i].q
        joint_vel[i] = motor_state[i].dq
    # 对关节角度进行重排序
    joint_pos =joint_reorder(joint_pos,go2_joint_current_order,model_joint_order)
    joint_vel = joint_reorder(joint_vel,go2_joint_current_order,model_joint_order)
    return quaternion,joint_pos,joint_vel


def main(go2):
    """Play with RSL-RL agent."""
    # parse configuration
    env_cfg = parse_env_cfg(
        args_cli.task, device=args_cli.device, num_envs=args_cli.num_envs, use_fabric=not args_cli.disable_fabric
    )
    
    if args_cli.task == "Isaac-Amp-Unitree-go2-v0":
        print("[INFO] Using AmpOnPolicyRunner")
        env_cfg.amp_num_preload_transitions = 1
    
    agent_cfg: RslRlOnPolicyRunnerCfg = cli_args.parse_rsl_rl_cfg(args_cli.task, args_cli)


    # 指定实验日志的目录路径
    log_root_path = os.path.join("weights", agent_cfg.experiment_name)
    # 将路径转换为绝对路径
    log_root_path = os.path.abspath(log_root_path)
    # 打印日志目录路径信息
    print(f"[INFO] Loading experiment from directory: {log_root_path}")
    # 获取检查点路径，根据配置加载特定的运行和检查点
    resume_path = get_checkpoint_path(log_root_path, agent_cfg.load_run, agent_cfg.load_checkpoint)
    # 获取检查点路径的父目录，作为日志目录
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
    effort_limit = torch.tensor(23.5).to(env.device).unsqueeze(0).repeat(env.num_actions)
    action_scale = torch.tensor(0.25).to(env.device).unsqueeze(0).repeat(env.num_actions)
    joint_vel_scale = torch.tensor(0.05).to(env.device).unsqueeze(0).repeat(env.num_actions)
    epoch_time_target = 0.02
    
    obs, _ = env.get_observations()
    timestep = 0
    imu_state,motor_state = go2.return_obs()  
    
    quat,joint_pos,joint_vel = go2_obs_process(imu_state,motor_state)
    quat = quat.to(env.device)
    joint_pos = joint_pos.to(env.device)
    joint_vel = joint_vel.to(env.device)
    
    
    obs[0][:31] = torch.cat([quat, torch.tensor([0, 0, 0],device=env.device), joint_pos, joint_vel])
    
    
    
    joystick = Se2Gamepad()
    
    offset = torch.tensor([ 0.1000, -0.1000,  0.1000, -0.1000,  0.8000,  0.8000,  1.0000,  1.0000,
         -1.5000, -1.5000, -1.5000, -1.5000], device='cuda:0').to(env.device)
    
    
    
    
    print_count = 0
    timestamp = time.time()
    
    while simulation_app.is_running():
        # run everything in inference mode
        
        with torch.inference_mode():
            # agent stepping
            actions = policy(obs)
            #venvobs, _, _, _ = env.step(actions)
            processed_actions = actions[0] * action_scale + offset
            # env stepping

            # 将 ctrl_kd 和 ctrl_kp 调整为 [action_dim]
            ctrl_kd = torch.tensor(go2.ctrl_kd).to(env.device).unsqueeze(0).repeat(env.num_actions)
            ctrl_kp = torch.tensor(go2.ctrl_kp).to(env.device).unsqueeze(0).repeat(env.num_actions)

            # 计算 action_max 和 action_min
            action_max = joint_pos + (effort_limit + ctrl_kd * joint_vel) / ctrl_kp
            action_min = joint_pos + (-effort_limit + ctrl_kd * joint_vel) / ctrl_kp
            
            
            clipped_actions = torch.clamp(processed_actions, min=action_min, max=action_max)
            
            # 对动作进行重排序
            actions_reordered = joint_reorder(clipped_actions,model_joint_order,go2_joint_current_order)
            #将动作输出给机器人
            
            
            time_remain = epoch_time_target - (time.time() - timestamp)
            
            go2.extent_targetPos = actions_reordered.cpu().numpy()
            
            timestamp = time.time()
            time.sleep(max(time_remain,0))
            
            #从机器人读取状态数据
            imu_state,motor_state = go2.return_obs()     
            quat,joint_pos,joint_vel = go2_obs_process(imu_state,motor_state)
            quat = quat.to(env.device)
            joint_pos = joint_pos.to(env.device)
            joint_vel = joint_vel.to(env.device)
            
            joint_vel = joint_vel * joint_vel_scale
            
            env.unwrapped.robot_twin(quat,joint_pos,joint_vel)
            
            readings = joystick.advance()
            if readings[0] < 0:
                readings[0] = 0.6 * readings[0]
            else :
                readings[0] = 2*readings[0]
            readings[1] = 0.3 * readings[1]
            readings[2] = -2*readings[2]
            readings_tensor = torch.from_numpy(readings).to(env.device)  # 将 NumPy 数组转换为 PyTorch 张量，并移动到 GPU
            
            obs[0] = torch.cat([quat, readings_tensor.float(), joint_pos, joint_vel,actions[0]])
            
            if print_count % 50 == 0:
                # 限制打印的精度到小数点后三位
                # 确保 actions_reordered 是一个张量
                if isinstance(actions_reordered, torch.Tensor):
                    
                    actions_list = actions_reordered.tolist()
                    if isinstance(actions_list, float):
                        print(f"actions:{actions_list:.3f}\n")
                        print(f"time:{time_remain:.3f}\n")
                    else:
                        print([f"{x:.3f}" for x in actions_list], "\n")
                else:
                    print("actions_reordered is not a tensor\n")
                
                print("command:", obs[0][4:7].tolist(), "\n")
                print_count = 0
            print_count += 1            
        if args_cli.video:
            timestep += 1
            # Exit the play loop after recording one video
            if timestep == args_cli.video_length:
                break

    # close the simulator
    env.close()

def control_loop(go2):
    while True:
        key = get_key()
        if key is not None:
            if key == 'z':
                go2.ctrl_kp += 0.01
                print('KP:',go2.ctrl_kp)
                print('state:',go2.control_mode)
            elif key == 'x':
                go2.ctrl_kp -= 0.01
                print('KP:',go2.ctrl_kp)
                print('state:',go2.control_mode)
            elif key == 'q':
                time.sleep(1)
                print("Done!")
                sys.exit(-1)
            else:
                go2.control_mode, reset_mode = process_key(key)
                print('state:',go2.control_mode)
                if reset_mode:
                    go2.reset()
                

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
    go2 = Go2_PD_Control()
    # 初始化Custom对象
    go2.Init()
    # 启动Custom对象
    go2.Start()    

    # 创建线程
    control_thread = threading.Thread(target=control_loop, args=(go2,))
    # 启动线程
    control_thread.start()    
    
    main(go2)
    
    time.sleep(1)
    print("Done!")
    sys.exit(-1) 
    # close sim app
    simulation_app.close()
