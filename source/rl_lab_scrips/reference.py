import torch
path = "weights/unitree_go2_him_rough/2024-12-16_17-33-52/exported/policy.pt"
model = torch.jit.load(path)
model.eval()

# 假设输入数据是一个形状为 (batch_size, num_actor_obs) 的张量
batch_size = 1
num_actor_obs = 270  # 根据实际情况调整
input_data = torch.randn(batch_size, num_actor_obs)
print(model)
# 使用模型进行推理
with torch.no_grad():  # 关闭梯度计算以节省内存
    actions_mean = model(input_data)
    print("Predicted Actions Mean:", actions_mean)

