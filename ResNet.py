import torch
from torch import nn

class bottle_neck(nn.Module):
    def __init__(self, in_channels, out_channels, expansion, stride=1):
        super(bottle_neck, self).__init__()
        
        self.in_channels = in_channels
        self.out_channels = out_channels
        
        self.expansion = expansion
        self.stride = stride
        
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1)
        self.batch_norm1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1)
        self.batch_norm2 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU()
    
        
    def forward(self, X):
        identity = X.clone()
        if self.stride != 1:
            identity = nn.Conv2d(self.in_channels, self.in_channels*self.expansion, kernel_size=3, stride=self.stride, padding=1).cuda()(identity)
        
        y = self.relu(self.conv1(X))
        y = self.conv2(y)
        y += identity
        y = self.relu(y)
        return y
        
        
class ResNet(nn.Module):
    def __init__(self, in_channels, bottle_neck_in_each_block, expansion=2, downsampling=2):
        super(ResNet, self).__init__()
        self.expansion = expansion
        self.conv1 = nn.Conv2d(in_channels, 64, kernel_size=3, stride=1, padding=1)
        self.blocks = nn.ModuleList([])
        for i in range(len(bottle_neck_in_each_block)):
            self.blocks.append(self.make_layer(64*(expansion**i), n_blocks= bottle_neck_in_each_block[i], \
                                               expansion=self.expansion, stride=downsampling))
        self.flatten = nn.Flatten()
        #self.maxPool = nn.MaxPool2d(kernel_size=2)
        self.linear1 = nn.Linear(4*4*512, 10)
        
    def forward(self, X):
        X = self.conv1(X)
        #X = self.maxPool(X)
        for block in self.blocks:
            X = block(X)
        X = self.flatten(X)
        X = self.linear1(X)
        return X
        
    def make_layer(self, in_channels, n_blocks, expansion, stride=2):
        layers= []
        for i in range(n_blocks-1):
            layers.append(bottle_neck(in_channels, in_channels, self.expansion, stride=1))
        
        out_channels = in_channels * expansion
        layers.append(bottle_neck(in_channels, out_channels, expansion=self.expansion, stride=2))
        return nn.Sequential(*layers)