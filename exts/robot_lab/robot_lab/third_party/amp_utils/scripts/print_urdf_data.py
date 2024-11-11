import pybullet as p
import pybullet_data

# 连接到物理引擎
physicsClient = p.connect(p.GUI)

# 加载内置的数据路径
p.setAdditionalSearchPath(pybullet_data.getDataPath())

# 加载URDF文件
robot_id = p.loadURDF("exts/robot_lab/robot_lab/third_party/amp_utils/models/a1/urdf/a1.urdf")

# 获取并打印Body信息
num_bodies = p.getNumBodies()
for i in range(num_bodies):
    body_info = p.getBodyInfo(i)
    print(f"Body ID: {i}, Body Info: {body_info}")

# 获取并打印Joint信息
num_joints = p.getNumJoints(robot_id)
for i in range(num_joints):
    joint_info = p.getJointInfo(robot_id, i)
    print(f"Joint Index: {joint_info[0]}, Joint Name: {joint_info[1].decode('UTF-8')}, Joint Type: {joint_info[2]}")

# 断开与物理引擎的连接
p.disconnect()