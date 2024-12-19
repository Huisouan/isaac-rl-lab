import time
import sys
import pygame

from unitree_sdk2py.core.channel import ChannelFactoryInitialize
from unitree_sdk2py.go2.low_level.go2_pd_sim2sim import Go2_SIM2SIM
from unitree_bridge.himloco import Himloco
from unitree_bridge.config.algo.algocfg import HIMConfig
from unitree_bridge.config.robot.bot_cfg import GO2
from unitree_bridge.process.joystick import *
# 默认网络接口名称
default_network = "lo"

def handle_joystick_events():
    velocity_commands = [0, 0, 0]
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            return 'quit', velocity_commands
        elif event.type == pygame.JOYAXISMOTION:
            # 假设摇杆0控制x方向，摇杆1控制y方向，摇杆2控制z方向
            if event.axis == 0:  # 左右摇杆
                velocity_commands[0] = event.value
            elif event.axis == 1:  # 上下摇杆
                velocity_commands[1] = event.value
            elif event.axis == 2:  # 另一个摇杆（如果有）
                velocity_commands[2] = event.value
    return None, velocity_commands

def main(go2):
    Algocfg = HIMConfig
    Botcfg = GO2
    model = Himloco(Algocfg, Botcfg)
    target_interval = 50
    while True:
        start_time = time.time()  # 记录循环开始时间

        imu_state, motor_state = go2.return_obs()
        result, velocity_commands = handle_joystick_events()
        if result == 'quit':
            break

        action = model.forward(imu_state, motor_state, velocity_commands)
        end_time = time.time()  # 记录循环结束时间
        elapsed_time = end_time - start_time  # 计算实际花费的时间

        sleep_time = max(0, target_interval - elapsed_time)  # 计算需要睡眠的时间
        time.sleep(sleep_time)  # 休眠
        # 赋值
        go2.extent_targetPos = action

if __name__ == "__main__":
    ChannelFactoryInitialize(1, default_network)

    # 创建Go2_SIM2SIM对象
    go2 = Go2_SIM2SIM()
    # 启动Go2_SIM2SIM对象
    go2.Start()
    
    # 初始化pygame
    pygame.init()
    pygame.joystick.init()

    # 检查是否有手柄连接
    if pygame.joystick.get_count() == 0:
        print("No joystick found.")
        sys.exit(-1)

    # 初始化第一个手柄
    joystick = pygame.joystick.Joystick(0)
    joystick.init()    

    main(go2)