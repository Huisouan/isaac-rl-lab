import torch

def load_model(path):
    model = torch.jit.load(path)
    model.eval()
    print(model)
    return model

