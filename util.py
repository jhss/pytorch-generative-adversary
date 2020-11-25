import torch

def one_hot(y):
    return torch.eye(10)[y]
