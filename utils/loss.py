import torch

def CEloss(outputs, labels):
    return torch.nn.functional.cross_entropy(outputs, labels)