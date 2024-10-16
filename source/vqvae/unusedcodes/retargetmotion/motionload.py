import os
import csv
import torch

class MotionData:
    def __init__(self, data_dir):
        """
        初始化方法，设置数据目录。
        
        :param data_dir: 包含CSV文件的目录路径
        """
        self.device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
        self.data_dir = data_dir
        self.data_tensors = []
        self.data_names = []
        self.data_length = []
        self.load_data()
    def load_data(self):
        """
        从指定目录加载所有CSV文件，并将每个文件转换为一个不包含表头的二维Tensor。
        """
        for filename in os.listdir(self.data_dir):
            if filename.endswith('.csv'):
                file_path = os.path.join(self.data_dir, filename)
                self.data_names.append(filename)
                with open(file_path, newline='') as csvfile:
                    csv_reader = csv.reader(csvfile)
                    next(csv_reader)  # 跳过表头
                    data = []
                    for row in csv_reader:
                        data.append([float(item) for item in row])
                        
                    tensor_data = torch.tensor(data, dtype=torch.float32).to(self.device)
                    self.data_length.append(tensor_data.shape[0])
                    self.data_tensors.append(tensor_data)
        self.data_length = torch.tensor(self.data_length,dtype=torch.int64).to(self.device)
        
    def get_tensors(self):
        """
        返回加载的所有数据的列表。
        
        :return: 包含所有CSV文件数据的二维Tensor列表
        """
        return self.data_tensors

    def get_frames(self,motion_id,frame_num):
        return self.data_tensors[motion_id][frame_num]
    
    def get_random_frame_batch(self,batch_size):
        random_frame_id = torch.randint(0,len(self.data_tensors),size=(batch_size,)).to(self.device)
        random_frame_index = torch.rand((1,len(random_frame_id)))[0].to(self.device)
        rand_frames = int(self.data_length[random_frame_id]*random_frame_index)
        
        frame = torch.stack([self.data_tensors[i][j] for i,j in zip(random_frame_id,rand_frames)])
        
        
        return frame,random_frame_id,rand_frames
    
    @staticmethod
    def root_state_w(frame):
        """
        提取根状态向量。
        
        :param frame: 一维或二维张量
        :return: 根状态向量
        """
        if frame.dim() == 1:
            return frame[:7]
        elif frame.dim() == 2:
            return frame[:, :7]

    @staticmethod
    def joint_position_w(frame):
        """
        提取关节位置向量。
        
        :param frame: 一维或二维张量
        :return: 关节位置向量
        """
        if frame.dim() == 1:
            return frame[7:19]
        elif frame.dim() == 2:
            return frame[:, 7:19]

    @staticmethod
    def joint_velocity_w(frame):
        """
        提取关节速度向量。
        
        :param frame: 一维或二维张量
        :return: 关节速度向量
        """
        if frame.dim() == 1:
            return frame[19:31]
        elif frame.dim() == 2:
            return frame[:, 19:31]

    @staticmethod
    def toe_position_w(frame):
        """
        提取脚趾位置向量。
        
        :param frame: 一维或二维张量
        :return: 脚趾位置向量
        """
        if frame.dim() == 1:
            return frame[31:43]
        elif frame.dim() == 2:
            return frame[:, 31:43]

    @staticmethod
    def toe_velocity_w(frame):
        """
        提取脚趾速度向量。
        
        :param frame: 一维或二维张量
        :return: 脚趾速度向量
        """
        if frame.dim() == 1:
            return frame[43:]
        elif frame.dim() == 2:
            return frame[:, 43:]
    
    
    
    
    
    
    