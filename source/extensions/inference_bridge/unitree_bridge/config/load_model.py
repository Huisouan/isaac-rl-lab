import torch

def load_model(path):
    # 加载模型
    model = torch.jit.load(path)
    
    # 设置目标设备为 CUDA 如果可用，否则使用 CPU
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # 将模型移动到目标设备
    model.to(device)
    
    # 设置模型为评估模式
    model.eval()
    
    print(f"Model loaded and moved to {device}")
    return model