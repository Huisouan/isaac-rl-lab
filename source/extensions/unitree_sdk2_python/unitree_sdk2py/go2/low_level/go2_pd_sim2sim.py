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
from unitree_sdk2py.go2.low_level import unitree_legged_const as go2
# 从motion_switcher客户端模块导入MotionSwitcherClient类
from unitree_sdk2py.comm.motion_switcher.motion_switcher_client import MotionSwitcherClient
# 从sport客户端模块导入SportClient类
from unitree_sdk2py.go2.sport.sport_client import SportClient

# 默认网络接口名称
default_network = 'lo'

# 定义Custom类
class Go2_SIM2SIM:
    def __init__(self):
        # 初始化PID控制器参数
        self.Kp = 15
        self.Kd = 0.5
        

        
        # 初始化时间消耗计数器
        self.time_consume = 0
        # 初始化速率计数器
        self.rate_count = 0
        # 初始化正弦波计数器
        self.sin_count = 0
        # 初始化运动时间计数器
        self.motiontime = 0
        # 设置控制循环周期
        self.dt = 0.005  # 0.001~0.01

        #['standby','stand','lay','walk',]
        self.control_mode = 'standby'


        # 初始化低级命令对象
        self.low_cmd = unitree_go_msg_dds__LowCmd_()  
        # 初始化低级状态对象
        self.low_state = None  

        # 定义目标位置1 LAY
        self._targetPos_1 = [0.0, 1.36, -2.65, 0.0, 1.36, -2.65,
                             -0.2, 1.36, -2.65, 0.2, 1.36, -2.65]
        # 定义目标位置2 STAND
        self._targetPos_2 = [0.0, 0.67, -1.3, 0.0, 0.67, -1.3,
                             0.0, 0.67, -1.3, 0.0, 0.67, -1.3]
        # 定义目标位置3
        self._targetPos_3 = [-0.35, 1.36, -2.65, 0.35, 1.36, -2.65,
                             -0.5, 1.36, -2.65, 0.5, 1.36, -2.65]

        self.extent_targetPos = [0.0, 1.36, -2.65, 0.0, 1.36, -2.65,
                                -0.2, 1.36, -2.65, 0.2, 1.36, -2.65]



        # 初始化起始位置
        self.startPos = [0.0] * 12
        # 定义各阶段持续时间
        self.duration_1 = 500
        self.duration_2 = 500
        self.duration_3 = 1000
        self.duration_4 = 900
        self.net_duration = 2
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

        # 初始化低级命令
        self.InitLowCmd()
        print("Low command initialized.")
        # 创建低级命令发布者
        self.lowcmd_publisher = ChannelPublisher("rt/lowcmd", LowCmd_)
        self.lowcmd_publisher.Init()
        print("Low command publisher initialized.")
        # 创建低级状态订阅者
        self.lowstate_subscriber = ChannelSubscriber("rt/lowstate", LowState_)
        self.lowstate_subscriber.Init(self.LowStateMessageHandler, 1)

        print("Low state subscriber initialized.")

        self.wait_for_low_state()

        # 初始化CRC校验工具
        self.crc = CRC()
        print("CRC tool initialized.") 
        
    def wait_for_low_state(self):
        timeout = 10  # 设置超时时间为10秒
        start_time = time.time()
        while self.low_state is None:
            if time.time() - start_time > timeout:
                raise TimeoutError("Timeout waiting for low_state data")
            time.sleep(0.1)  # 每隔0.1秒检查一次
        print("Received initial low_state data.")
        
    def Start(self):
        # 启动低级命令写入线程
        self.lowCmdWriteThreadPtr = RecurrentThread(
            interval=0.005, target=self.LowCmdWrite, name="writebasiccmd"
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
        
    def LowCmdWrite(self):
        # 第一次运行时记录起始位置
        if self.firstRun:
            for i in range(12):  # 遍历12个电机
                self.startPos[i] = self.low_state.motor_state[i].q  # 记录每个电机的初始位置
            
            self.firstRun = False  # 标记首次运行已完成
        if self.control_mode == 'stand':
            # 计算第一阶段完成百分比
            self.percent_1 += 1.0 / self.duration_1  # 每次调用增加百分比
            self.percent_1 = min(self.percent_1, 1)  # 确保百分比不超过1
            if self.percent_1 < 1:  # 如果第一阶段未完成
                for i in range(12):  # 遍历12个电机
                    self.low_cmd.motor_cmd[i].q = (1 - self.percent_1) * self.startPos[i] + self.percent_1 * self._targetPos_1[i]  # 线性插值计算目标位置
                    self.low_cmd.motor_cmd[i].dq = 0  # 设置速度为0
                    self.low_cmd.motor_cmd[i].kp = self.Kp  # 设置位置控制增益
                    self.low_cmd.motor_cmd[i].kd = self.Kd  # 设置速度控制增益
                    self.low_cmd.motor_cmd[i].tau = 0  # 设置力矩为0

            # 计算第二阶段完成百分比
            if (self.percent_1 == 1) and (self.percent_2 <= 1):  # 如果第一阶段完成且第二阶段未完成
                self.percent_2 += 1.0 / self.duration_2  # 每次调用增加百分比
                self.percent_2 = min(self.percent_2, 1)  # 确保百分比不超过1
                for i in range(12):  # 遍历12个电机
                    self.low_cmd.motor_cmd[i].q = (1 - self.percent_2) * self._targetPos_1[i] + self.percent_2 * self._targetPos_2[i]  # 线性插值计算目标位置
                    self.low_cmd.motor_cmd[i].dq = 0  # 设置速度为0
                    self.low_cmd.motor_cmd[i].kp = self.Kp  # 设置位置控制增益
                    self.low_cmd.motor_cmd[i].kd = self.Kd  # 设置速度控制增益
                    self.low_cmd.motor_cmd[i].tau = 0  # 设置力矩为0   
        if self.control_mode == 'standby':
            self.percent_1 += 1.0 / self.duration_1  # 每次调用增加百分比
            self.percent_1 = min(self.percent_1, 1)  # 确保百分比不超过1
            for i in range(12):  # 遍历12个电机
                self.low_cmd.motor_cmd[i].q = 0  # 设置目标位置为初始位置
                self.low_cmd.motor_cmd[i].dq = 0  # 设置速度为0了
                self.low_cmd.motor_cmd[i].kp = 0  # 设置位置控制增益
                self.low_cmd.motor_cmd[i].kd = self.Kd  # 设置速度控制增益
                self.low_cmd.motor_cmd[i].tau = 0  # 设置力矩为0              
        if self.control_mode == 'walk':
            for i in range(12):  # 遍历12个电机
                self.low_cmd.motor_cmd[i].q = self.extent_targetPos[i]  # 线性插值计算目标位置
                self.low_cmd.motor_cmd[i].dq = 0  # 设置速度为0
                self.low_cmd.motor_cmd[i].kp = self.Kp  # 设置位置控制增益
                self.low_cmd.motor_cmd[i].kd = self.Kd  # 设置速度控制增益
                self.low_cmd.motor_cmd[i].tau = 0  # 设置力矩为0
            
            pass                 
        # 计算CRC校验值
        self.low_cmd.crc = self.crc.Crc(self.low_cmd)  # 计算命令的CRC校验值
        # 发布低级命令
        self.lowcmd_publisher.Write(self.low_cmd)  # 发布低级命令
        
    def reset(self):
        self.firstRun = True
        self.percent_1 = 0
        self.percent_2 = 0
        self.percent_3 = 0
        self.percent_4 = 0        

    def return_obs(self):
        return(
            self.low_state.imu_state,
            self.low_state.motor_state,
        )



import tty
import termios
import select

def get_key():
    """ 获取单个按键输入，不阻塞程序 """
    fd = sys.stdin.fileno()
    old_settings = termios.tcgetattr(fd)
    try:
        tty.setraw(fd)
        rlist, _, _ = select.select([fd], [], [], 0.5)  # 设置超时时间为0.1秒
        if rlist:
            ch = sys.stdin.read(1)
        else:
            ch = None  # 没有按键输入时返回None
    finally:
        termios.tcsetattr(fd, termios.TCSADRAIN, old_settings)
    return ch

def process_key(key):
    if key == 'w':
        print("Set control_mod to 'standby'")
        return 'walk',True
    if key == 's':
        print("Set control_mod to 'standby'")
        return 'standby',True
    if key == 'u':
        print("Set control_mod to 'stand'")
        return 'stand',True
    else:
        return None,False

if __name__ == '__main__':

    ChannelFactoryInitialize(1,"lo")

    # 创建Custom对象
    custom = Go2_SIM2SIM()
    
    custom.Start()
    # 主循环
    while True:
        key = get_key()
        if key is not None:
            custom.control_mode ,reset_mode= process_key(key)
            if reset_mode == True:
                custom.reset()
            if key == 'q':
                time.sleep(1)
                print("Done!")
                sys.exit(-1)

        time.sleep(1)