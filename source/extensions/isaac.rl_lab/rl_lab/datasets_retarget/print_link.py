
import pybullet as p
import pybullet_data

def print_all_links(urdf_file):
    # 连接到pybullet物理引擎
    physics_client = p.connect(p.DIRECT)  # 使用DIRECT模式，不显示图形界面
    p.setAdditionalSearchPath(pybullet_data.getDataPath())  # 设置数据路径

    # 加载URDF文件
    robot_id = p.loadURDF(urdf_file, useFixedBase=True)

    # 获取所有link的信息
    num_joints = p.getNumJoints(robot_id)
    for joint_index in range(num_joints):
        joint_info = p.getJointInfo(robot_id, joint_index)
        link_name = joint_info[12].decode('utf-8')  # 获取link名称
        print(f"Link Name: {link_name}")

    # 断开与pybullet物理引擎的连接
    p.disconnect()

if __name__ == "__main__":
    urdf_file = "datasets/go2_description/urdf/go2_description.urdf"  # 替换为你的URDF文件路径
    print_all_links(urdf_file)