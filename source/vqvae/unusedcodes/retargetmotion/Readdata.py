import csv
import os
import math
import torch
from scipy.spatial.transform import Rotation as R
import numpy as np
import torch.nn.functional as F

class MotionData:
    def __init__(self, filename, scale = 1):
        """
        初始化一个包含文件名、表头和数据的对象。
        
        Args:
            filename (str): 文件名。
            header (list): 表头信息，由字符串列表组成。
            data (dict): 数据信息，由字典组成，键为节点名称，值为 [x, y, z] 形式的坐标列表。
        """
        self.filename = filename
        self.scale = scale
        self.header = []
        self.data = []
        self.totalframes = [0]
        self.load_data_from_csv()
        self.data_rescale()

    def load_data_from_csv(self):
        """
        从 CSV 文件中加载数据。
        
        Args:
            filename (str): CSV 文件名。
        
        Returns:
            None
        """
        with open(self.filename, 'r') as file:
            reader = csv.reader(file)
            # 假设第一行是表头
            headers = next(reader)
            self.header = [header[:-2] for header in headers if header.endswith('.X')]
            
            # 按照处理后的表头初始化数据字典
            data_dict = {header: [] for header in self.header}
            
            # 遍历每一行数据
            for row in reader:
                frame_data = {}
                for i, header in enumerate(headers):
                    if header.endswith('.X'):
                        name = header[:-2]
                        x = float(row[i])
                        y = float(row[i + 2])
                        z = float(row[i + 1])
                        frame_data[name] = [x, y, z]
                
                # 将当前帧的数据添加到数据字典中
                for name, xyz in frame_data.items():
                    data_dict[name].append(xyz)
            
            self.data = data_dict
            self.totalframes = len(self.data[self.header[0]])

    def data_rescale(self):
        """
        对数据重新缩放。
        
        Args:
            scale (float): 缩放比例。
        
        Returns:
            None
        """
        for name, data in self.data.items():
            for i, xyz in enumerate(data):
                data[i] = [x * self.scale for x in xyz]

    def get_data_by_frame(self, frame_number):
        """
        根据帧号获取数据。
        
        Args:
            frame_number (int): 帧号。
        
        Returns:
            dict: 包含指定帧号的节点名称和坐标的字典。
        """
        return {name: data[frame_number] for name, data in self.data.items()}
    
    def get_frame_without_name(self, frame_number):
        """
        获取数据，不包含节点名称。
        
        Args:
            frame_number (int): 帧号。
            
        Returns:
            list: 包含所有坐标的列表。
        """
        # 使用列表推导式收集所有节点在给定帧号的坐标数据
        return [data[frame_number] for name, data in self.data.items()]
    
    def get_data_by_name(self, node_name):
        """
        根据节点名称获取数据。
        
        Args:
            node_name (str): 节点名称。
        
        Returns:
            list: 包含指定节点名称的坐标列表。
        """
        return self.data[node_name]
# 示例用法


def read_all_csv_motions(folder_path,scale=0.01):
    """
    读取指定文件夹中所有 CSV 文件，并使用 MotionData 类处理每个文件。
    
    Args:
        folder_path (str): 包含 CSV 文件的文件夹路径。
    
    Returns:
        list: 包含所有 MotionData 对象的列表。
    """
    motion_data_list = []
    
    # 获取文件夹中所有的 CSV 文件
    csv_files = [f for f in os.listdir(folder_path) if f.endswith('.csv')]
    
    for csv_file in csv_files:
        full_path = os.path.join(folder_path, csv_file)
        
        # 创建 MotionData 对象并加载数据
        motion_data = MotionData(full_path,scale=scale)
        motion_data_list.append(motion_data)
    
    return motion_data_list


if __name__ == "__main__":
    motion_data = read_all_csv_motions("source/standalone/Mycode/retargetmotion/raw_mocap_data")
    print(motion_data.data)