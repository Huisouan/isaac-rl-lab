import pygame

def init_pygame():
    """初始化pygame"""
    pygame.init()
    pygame.joystick.init()

def init_joystick():
    """初始化手柄"""
    if pygame.joystick.get_count() == 0:
        print("没有检测到手柄")
        return None
    else:
        joystick = pygame.joystick.Joystick(0)
        joystick.init()
        print(f"检测到手柄: {joystick.get_name()}")
        return joystick

def get_joystick_state(joystick):
    """
    获取手柄的状态，包括摇杆和按键。
    
    返回:
        - axis_values: 包含axis1, axis0, axis2的列表
        - button_pressed: 如果有按键被按下，则返回按键的值，否则返回None
    """
    axis_values = [joystick.get_axis(i) for i in range(3)]  # 获取axis0, axis1, axis2的值
    button_pressed = None
    
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            pygame.quit()
            exit()
        elif event.type == pygame.JOYBUTTONDOWN:
            button_pressed = event.button
        elif event.type == pygame.JOYBUTTONUP:
            pass  # 可以在这里处理按键释放事件
        elif event.type == pygame.JOYAXISMOTION:
            pass  # 轴移动已经在获取axis_values时处理过了
        elif event.type == pygame.JOYHATMOTION:
            pass  # 方向键移动可以在这里处理
    
    return axis_values, button_pressed

def get_command_from_key(button_value):
    """
    根据按键值返回对应的命令。
    
    参数:
        button_value (int): 按键的值
    
    返回:
        str: 'stand', 'standby', 或 'walk'
    """
    switcher = {
        3: 'stand',
        0: 'standby',
        4: 'walk'
    }
    
    # 使用字典的get方法获取对应值，默认返回None或其他默认值
    command = switcher.get(button_value, None)
    
    if command is None:
        print(f"未知的按键值: {button_value}")
    
    return command
