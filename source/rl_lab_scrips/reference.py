import torch
path = "weights/unitree_go2_him_rough/2024-12-16_17-33-52/model_46800.pt"
loaded_dict = torch.load(path)
print(loaded_dict.keys())