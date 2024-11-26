# 导入所需的时间模块
import time
# 导入系统模块，用于处理命令行参数
import sys

# 从unitree_sdk2py核心模块导入ChannelPublisher类，用于发布消息
from unitree_sdk2py.core.channel import ChannelPublisher, ChannelFactoryInitialize
# 从unitree_sdk2py核心模块导入ChannelSubscriber类，用于订阅消息
from unitree_sdk2py.core.channel import ChannelSubscriber, ChannelFactoryInitialize
# 从默认的IDL模块导入LowCmd消息定义
from unitree_sdk2py.idl.default import unitree_go_msg_dds__LowCmd_
# 从默认的IDL模块导入LowState消息定义
from unitree_sdk2py.idl.default import unitree_go_msg_dds__LowState_
# 从unitree_go的消息定义模块导入LowCmd消息定义
from unitree_sdk2py.idl.unitree_go.msg.dds_ import LowCmd_
# 从unitree_go的消息定义模块导入LowState消息定义
from unitree_sdk2py.idl.unitree_go.msg.dds_ import LowState_
# 从utils模块导入CRC校验工具
from unitree_sdk2py.utils.crc import CRC
# 从utils模块导入周期线程工具
from unitree_sdk2py.utils.thread import RecurrentThread
# 导入常量模块
import unitree_legged_const as go2
# 从motion_switcher客户端模块导入MotionSwitcherClient类
from unitree_sdk2py.comm.motion_switcher.motion_switcher_client import MotionSwitcherClient
# 从sport客户端模块导入SportClient类
from unitree_sdk2py.go2.sport.sport_client import SportClient

# 默认网络接口名称
default_network = 'enp0s31f6'

# 定义Custom类
class Custom:
    def __init__(self):
        # 初始化PID控制器参数
        self.Kp = 60.0
        self.Kd = 5.0
        # 初始化时间消耗计数器
        self.time_consume = 0
        # 初始化速率计数器
        self.rate_count = 0
        # 初始化正弦波计数器
        self.sin_count = 0
        # 初始化运动时间计数器
        self.motiontime = 0
        # 设置控制循环周期
        self.dt = 0.002  # 0.001~0.01

        # 初始化低级命令对象
        self.low_cmd = unitree_go_msg_dds__LowCmd_()  
        # 初始化低级状态对象
        self.low_state = None  

        # 定义目标位置1
        self._targetPos_1 = [0.0, 1.36, -2.65, 0.0, 1.36, -2.65,
                             -0.2, 1.36, -2.65, 0.2, 1.36, -2.65]
        # 定义目标位置2
        self._targetPos_2 = [0.0, 0.67, -1.3, 0.0, 0.67, -1.3,
                             0.0, 0.67, -1.3, 0.0, 0.67, -1.3]
        # 定义目标位置3
        self._targetPos_3 = [-0.35, 1.36, -2.65, 0.35, 1.36, -2.65,
                             -0.5, 1.36, -2.65, 0.5, 1.36, -2.65]

        # 初始化起始位置
        self.startPos = [0.0] * 12
        # 定义各阶段持续时间
        self.duration_1 = 500
        self.duration_2 = 500
        self.duration_3 = 1000
        self.duration_4 = 900
        # 初始化各阶段完成百分比
        self.percent_1 = 0
        self.percent_2 = 0
        self.percent_3 = 0
        self.percent_4 = 0

        # 标记是否第一次运行
        self.firstRun = True
        # 标记是否完成
        self.done = False

        # 线程处理
        self.lowCmdWriteThreadPtr = None

        # 初始化CRC校验工具
        self.crc = CRC()

    # 公共方法
    def Init(self):
        # 初始化低级命令
        self.InitLowCmd()

        # 创建低级命令发布者
        self.lowcmd_publisher = ChannelPublisher("rt/lowcmd", LowCmd_)
        self.lowcmd_publisher.Init()

        # 创建低级状态订阅者
        self.lowstate_subscriber = ChannelSubscriber("rt/lowstate", LowState_)
        self.lowstate_subscriber.Init(self.LowStateMessageHandler, 10)

        # 初始化运动客户端
        self.sc = SportClient()  
        self.sc.SetTimeout(5.0)
        self.sc.Init()

        # 初始化动作切换客户端
        self.msc = MotionSwitcherClient()
        self.msc.SetTimeout(5.0)
        self.msc.Init()

        # 检查模式并释放模式
        status, result = self.msc.CheckMode()
        while result['name']:
            self.sc.StandDown()
            self.msc.ReleaseMode()
            status, result = self.msc.CheckMode()
            time.sleep(1)

    def Start(self):
        # 启动低级命令写入线程
        self.lowCmdWriteThreadPtr = RecurrentThread(
            interval=0.002, target=self.LowCmdWrite, name="writebasiccmd"
        )
        self.lowCmdWriteThreadPtr.Start()

    # 私有方法
    def InitLowCmd(self):
        # 初始化低级命令头
        self.low_cmd.head[0]=0xFE
        self.low_cmd.head[1]=0xEF
        # 设置级别标志
        self.low_cmd.level_flag = 0xFF
        # 设置GPIO
        self.low_cmd.gpio = 0
        # 初始化电机命令
        for i in range(20):
            self.low_cmd.motor_cmd[i].mode = 0x01  # (PMSM) mode
            self.low_cmd.motor_cmd[i].q= go2.PosStopF
            self.low_cmd.motor_cmd[i].kp = 0
            self.low_cmd.motor_cmd[i].dq = go2.VelStopF
            self.low_cmd.motor_cmd[i].kd = 0
            self.low_cmd.motor_cmd[i].tau = 0

    def LowStateMessageHandler(self, msg: LowState_):
        # 更新低级状态
        self.low_state = msg
        # 打印电机状态
        # print("FR_0 motor state: ", msg.motor_state[go2.LegID["FR_0"]])
        # 打印IMU状态
        # print("IMU state: ", msg.imu_state)
        # 打印电池状态
        # print("Battery state: voltage: ", msg.power_v, "current: ", msg.power_a)

    def LowCmdWrite(self):
        # 第一次运行时记录起始位置
        if self.firstRun:
            for i in range(12):
                self.startPos[i] = self.low_state.motor_state[i].q
            self.firstRun = False

        # 计算第一阶段完成百分比
        self.percent_1 += 1.0 / self.duration_1
        self.percent_1 = min(self.percent_1, 1)
        if self.percent_1 < 1:
            for i in range(12):
                self.low_cmd.motor_cmd[i].q = (1 - self.percent_1) * self.startPos[i] + self.percent_1 * self._targetPos_1[i]
                self.low_cmd.motor_cmd[i].dq = 0
                self.low_cmd.motor_cmd[i].kp = self.Kp
                self.low_cmd.motor_cmd[i].kd = self.Kd
                self.low_cmd.motor_cmd[i].tau = 0

        # 计算第二阶段完成百分比
        if (self.percent_1 == 1) and (self.percent_2 <= 1):
            self.percent_2 += 1.0 / self.duration_2
            self.percent_2 = min(self.percent_2, 1)
            for i in range(12):
                self.low_cmd.motor_cmd[i].q = (1 - self.percent_2) * self._targetPos_1[i] + self.percent_2 * self._targetPos_2[i]
                self.low_cmd.motor_cmd[i].dq = 0
                self.low_cmd.motor_cmd[i].kp = self.Kp
                self.low_cmd.motor_cmd[i].kd = self.Kd
                self.low_cmd.motor_cmd[i].tau = 0

        # 计算第三阶段完成百分比
        if (self.percent_1 == 1) and (self.percent_2 == 1) and (self.percent_3 < 1):
            self.percent_3 += 1.0 / self.duration_3
            self.percent_3 = min(self.percent_3, 1)
            for i in range(12):
                self.low_cmd.motor_cmd[i].q = self._targetPos_2[i] 
                self.low_cmd.motor_cmd[i].dq = 0
                self.low_cmd.motor_cmd[i].kp = self.Kp
                self.low_cmd.motor_cmd[i].kd = self.Kd
                self.low_cmd.motor_cmd[i].tau = 0

        # 计算第四阶段完成百分比
        if (self.percent_1 == 1) and (self.percent_2 == 1) and (self.percent_3 == 1) and (self.percent_4 <= 1):
            self.percent_4 += 1.0 / self.duration_4
            self.percent_4 = min(self.percent_4, 1)
            for i in range(12):
                self.low_cmd.motor_cmd[i].q = (1 - self.percent_4) * self._targetPos_2[i] + self.percent_4 * self._targetPos_3[i]
                self.low_cmd.motor_cmd[i].dq = 0
                self.low_cmd.motor_cmd[i].kp = self.Kp
                self.low_cmd.motor_cmd[i].kd = self.Kd
                self.low_cmd.motor_cmd[i].tau = 0

        # 计算CRC校验值
        self.low_cmd.crc = self.crc.Crc(self.low_cmd)
        # 发布低级命令
        self.lowcmd_publisher.Write(self.low_cmd)

if __name__ == '__main__':
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
    custom = Custom()
    # 初始化Custom对象
    custom.Init()
    # 启动Custom对象
    custom.Start()

    # 主循环
    while True:        
        if custom.percent_4 == 1.0: 
           time.sleep(1)
           print("Done!")
           sys.exit(-1)     
        time.sleep(1)