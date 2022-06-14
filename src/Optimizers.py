import numpy as np
import torch
from torch import nn
import torch_optimizer as optim
"""
First order:
Adam
SGD
Adagrad

Second order:
AdaHessian
"""

def get_optimizer(model, name = 'sgd', lr = 0.1, momentum = 0.0):
    name = name.lower()
    
    ### First order optimizers
    
    if name == 'adam':
        return torch.optim.Adam(model.parameters(), lr = lr)
    if name == 'sgd':
        return torch.optim.SGD(model.parameters(), lr = lr, momentum = momentum)
    if name == 'adagrad':
        return torch.optim.Adagrad(model.parameters(), lr = lr)
    
    ### Second order optimizers
    if name == 'adahessian':
        return optim.Adahessian(model.parameters(),lr = lr)