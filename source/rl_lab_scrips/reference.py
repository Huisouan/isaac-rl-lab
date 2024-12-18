from unitree_sdk2py.core.channel import  ChannelFactoryInitialize

from unitree_sdk2py.go2.low_level.go2_pd_sim2sim import Go2_SIM2SIM
# 默认网络接口名称
default_network = 'lo'

if __name__ == "__main__":
    ChannelFactoryInitialize(0,default_network)

    # 创建Custom对象
    go2 = Go2_SIM2SIM()
    # 初始化Custom对象
    go2.Init()
    # 启动Custom对象
    go2.Start()    
    