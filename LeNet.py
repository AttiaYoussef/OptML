import torch
from torch import nn
from torch.nn import functional as F



class LeNet(nn.Module): # LeNet 5 ; input is of size [nb_samples, nb_channels, 32, 32]
    
    def __init__(self, batch_size): # Define just the layers, not the activation functions
        super(LeNet, self).__init__()
        self.conv1 = nn.Conv2d(in_channels = 1, out_channels = 6, kernel_size = 5, stride = 1)
        self.pool1 = nn.AvgPool2d(kernel_size = 2, stride = 2)
        
        self.conv2 = nn.Conv2d(in_channels = 6, out_channels = 16, kernel_size = 5, stride = 1)
        self.pool2 = nn.AvgPool2d(kernel_size = 2, stride = 2)
        
        self.conv3 = nn.Conv2d(in_channels = 16, out_channels = 120, kernel_size = 5, stride = 1)
        
        self.fc1 = nn.Linear(in_features = 120, out_features = 84)
        self.fc2 = nn.Linear(in_features = 84,  out_features = 10)
        self.batch_size = batch_size
        
    def forward(self, x):
        x = self.pool1(torch.tanh(self.conv1(x)))
        x = self.pool2(torch.tanh(self.conv2(x)))
        x = torch.tanh(self.conv3(x))
        
        x = x.view(self.batch_size,-1)
        x = torch.tanh(self.fc1(x))
        logits = self.fc2(x)
        probs = F.softmax(logits, dim = 1)
        return logits
        