import time
import mujoco
import mujoco.viewer
from threading import Thread
import threading

from unitree_sdk2py.core.channel import ChannelFactoryInitialize
from unitree_sdk2py_bridge import UnitreeSdk2Bridge, ElasticBand

import config


locker = threading.Lock()

mj_model = mujoco.MjModel.from_xml_path(config.ROBOT_SCENE)
mj_data = mujoco.MjData(mj_model)


if config.ENABLE_ELASTIC_BAND:
    elastic_band = ElasticBand()
    if config.ROBOT == "h1" or config.ROBOT == "g1":
        band_attached_link = mj_model.body("torso_link").id
    else:
        band_attached_link = mj_model.body("base_link").id
    viewer = mujoco.viewer.launch_passive(
        mj_model, mj_data, key_callback=elastic_band.MujuocoKeyCallback
    )
else:
    viewer = mujoco.viewer.launch_passive(mj_model, mj_data)

mj_model.opt.timestep = config.SIMULATE_DT
num_motor_ = mj_model.nu
dim_motor_sensor_ = 3 * num_motor_

time.sleep(0.2)


def SimulationThread():
    # 全局变量
    global mj_data, mj_model

    # 初始化ChannelFactory
    ChannelFactoryInitialize(config.DOMAIN_ID, config.INTERFACE)

    # 创建UnitreeSdk2Bridge实例
    unitree = UnitreeSdk2Bridge(mj_model, mj_data)

    # 如果启用了游戏手柄
    if config.USE_JOYSTICK:
        # 设置游戏手柄
        unitree.SetupJoystick(device_id=0, js_type=config.JOYSTICK_TYPE)

    # 如果启用了打印场景信息
    if config.PRINT_SCENE_INFORMATION:
        # 打印场景信息
        unitree.PrintSceneInformation()

    # 模拟循环
    while viewer.is_running():
        step_start = time.perf_counter()

        # 锁定
        locker.acquire()

        # 如果启用了弹性带
        if config.ENABLE_ELASTIC_BAND:
            if elastic_band.enable:
                # 应用弹性带力
                mj_data.xfrc_applied[band_attached_link, :3] = elastic_band.Advance(
                    mj_data.qpos[:3], mj_data.qvel[:3]
                )
        # 执行MuJoCo步进
        mujoco.mj_step(mj_model, mj_data)

        # 解锁
        locker.release()

        # 计算到下一步的时间间隔
        time_until_next_step = mj_model.opt.timestep - (
            time.perf_counter() - step_start
        )
        # 如果还有时间，则休眠
        if time_until_next_step > 0:
            time.sleep(time_until_next_step)


def PhysicsViewerThread():
    while viewer.is_running():
        locker.acquire()
        viewer.sync()
        locker.release()
        time.sleep(config.VIEWER_DT)


if __name__ == "__main__":
    viewer_thread = Thread(target=PhysicsViewerThread)
    sim_thread = Thread(target=SimulationThread)

    viewer_thread.start()
    sim_thread.start()
