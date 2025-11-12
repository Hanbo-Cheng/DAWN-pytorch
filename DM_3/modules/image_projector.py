import torch
import torch.nn as nn
import torch.nn.functional as F

class ResBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1):
        super(ResBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        
        self.downsample = None
        if stride != 1 or in_channels != out_channels:
            self.downsample = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels)
            )

    def forward(self, x):
        identity = x
        
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        
        out = self.conv2(out)
        out = self.bn2(out)
        
        if self.downsample is not None:
            identity = self.downsample(x)
        
        out += identity
        out = self.relu(out)
        
        return out

class ResNet(nn.Module):
    def __init__(self, block, in_channels, layers):
        super(ResNet, self).__init__()
        self.in_channels = in_channels
        # self.conv1 = nn.Conv2d(self.in_channels, self.in_channels, kernel_size=7, stride=1, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(self.in_channels)
        self.relu = nn.ReLU(inplace=True)
        # self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        
        self.layer1 = self._make_layer(block, self.in_channels, self.in_channels * 2, layers[0])
        self.layer2 = self._make_layer(block, self.in_channels * 2, self.in_channels * 2, layers[1], stride=2)
        self.layer3 = self._make_layer(block, self.in_channels * 2, self.in_channels * 2, layers[2], stride=2)
        
    def _make_layer(self, block, in_channels, out_channels, blocks, stride=1):
        # downsample = None
        # if stride != 1 or self.in_channels != out_channels * block.expansion:
        #     downsample = nn.Sequential(
        #         nn.Conv2d(self.in_channels, out_channels * block.expansion, kernel_size=1, stride=stride, bias=False),
        #         nn.BatchNorm2d(out_channels * block.expansion),
        #     )
        
        layers = []
        layers.append(block(in_channels, out_channels, stride))
        for _ in range(1, blocks):
            layers.append(block(out_channels, out_channels))
        
        return nn.Sequential(*layers)
    
    def forward(self, x):
        # identity = x
        # x = self.conv1(x) # [1, 256, 32, 32]  ->  [1, 256, 32, 32]
        # x = self.bn1(x)
        # x = self.relu(x)
        # x = identity + x
        # x = self.maxpool(x)
        
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        
        return x

class ResNet_2(nn.Module):
    def __init__(self, block, in_channels, layers):
        super(ResNet_2, self).__init__()
        self.in_channels = in_channels
        # self.conv1 = nn.Conv2d(self.in_channels, self.in_channels, kernel_size=7, stride=1, padding=3, bias=False)
        # self.bn1 = nn.BatchNorm2d(self.in_channels)
        # self.relu = nn.ReLU(inplace=True)
        # self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        
        self.layer1 = self._make_layer(block, self.in_channels, self.in_channels * 2, layers[0])
        self.layer2 = self._make_layer(block, self.in_channels * 2, self.in_channels * 2, layers[1], stride=2)
        self.layer3 = self._make_layer(block, self.in_channels * 2, self.in_channels * 2, layers[2], stride=2)
        self.layer4 = self._make_layer(block, self.in_channels * 2, self.in_channels * 2, layers[2], stride=2)
        
    def _make_layer(self, block, in_channels, out_channels, blocks, stride=1):
        # downsample = None
        # if stride != 1 or self.in_channels != out_channels * block.expansion:
        #     downsample = nn.Sequential(
        #         nn.Conv2d(self.in_channels, out_channels * block.expansion, kernel_size=1, stride=stride, bias=False),
        #         nn.BatchNorm2d(out_channels * block.expansion),
        #     )
        
        layers = []
        layers.append(block(in_channels, out_channels, stride))
        for _ in range(1, blocks):
            layers.append(block(out_channels, out_channels))
        
        return nn.Sequential(*layers)
    
    def forward(self, x):
        # identity = x
        # x = self.conv1(x) # [1, 256, 32, 32]  ->  [1, 256, 32, 32]
        # x = self.bn1(x)
        # x = self.relu(x)
        # x = identity + x
        # x = self.maxpool(x)
        
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        
        return x


if __name__ == "__main__":
    # # Example usage:
    x = torch.ones((1, 256, 32, 32)).cuda()
    model = ResNet(ResBlock, in_channels = 256, layers = [2, 2, 2]).cuda()
    model(x)

    total_params = sum(p.numel() for p in model.parameters())
    print(f'Total number of parameters: {total_params/1024/1024}')
    # # print(model)

