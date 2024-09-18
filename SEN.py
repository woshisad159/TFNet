import torch
import torch.nn as nn
import torch.utils.model_zoo as model_zoo
import torch.nn.functional as F
__all__ = [
    'ResNet', 'resnet18', 'resnet34'
]
model_urls = {
    'resnet18': 'https://download.pytorch.org/models/resnet18-f37072fd.pth',
    'resnet34': 'https://download.pytorch.org/models/resnet34-333f7ec4.pth',
}

def conv3x3(in_planes, out_planes, stride=1):
    # 3x3x3 convolution with padding
    return nn.Conv3d(
        in_planes,
        out_planes,
        kernel_size=(1,3,3),
        stride=(1,stride,stride),
        padding=(0,1,1),
        bias=False)

class TSEM(nn.Module):
    def __init__(self, input_size ):
        super(TSEM, self).__init__()
        hidden_size = input_size//16
        self.conv_transform = nn.Conv1d(input_size, hidden_size, kernel_size=1, stride=1, padding=0)
        self.conv_back = nn.Conv1d(hidden_size, input_size, kernel_size=1, stride=1, padding=0)
        #self.conv_enhance = nn.Conv1d(hidden_size, hidden_size, kernel_size=9, stride=1, padding=4)
        self.num = 5
        self.conv_enhance = nn.ModuleList([
            nn.Conv1d(hidden_size, hidden_size, kernel_size=3, stride=1, padding=int(i+1), groups=hidden_size, dilation=int(i+1)) for i in range(self.num)
        ])
        self.weights = nn.Parameter(torch.ones(self.num) / self.num, requires_grad=True)
        self.alpha = nn.Parameter(torch.zeros(1), requires_grad=True)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        out = self.conv_transform(x.mean(-1).mean(-1))
        aggregated_out = 0
        for i in range(self.num):
            aggregated_out += self.conv_enhance[i](out) * self.weights[i]
        out = self.conv_back(aggregated_out)
        return x*(torch.sigmoid(out.unsqueeze(-1).unsqueeze(-1))-0.5) * self.alpha


class SSEM(nn.Module):
    def __init__(self, input_size ):
        super(SSEM, self).__init__()
        div_channel = input_size//16
        self.conv_transform = nn.Conv3d(input_size, div_channel, kernel_size=(1,1,1))
        self.num = 3
        self.conv_enhance = nn.ModuleList([
            nn.Conv3d(div_channel, div_channel, kernel_size=(9,3,3), padding=(4,i+1,i+1), dilation=(1,i+1,i+1), groups=div_channel) for i in range(self.num)
        ])
        
        self.weights = nn.Parameter(torch.ones(self.num) / self.num, requires_grad=True)
        self.conv_back = nn.Conv3d(div_channel, input_size, kernel_size=(1,1,1))
        self.alpha = nn.Parameter(torch.ones(1), requires_grad=True)

    def forward(self, x):
        out = self.conv_transform(x)
        aggregated_out = 0
        for i in range(self.num):
            aggregated_out += self.conv_enhance[i](out) * self.weights[i]
        out = self.conv_back(aggregated_out)
        return x*(torch.sigmoid(out)-0.5) * self.alpha

class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(BasicBlock, self).__init__()
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = nn.BatchNorm3d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = nn.BatchNorm3d(planes)
        self.downsample = downsample

        self.tsem = TSEM(planes)
        self.ssem = SSEM(planes)

    def forward(self, x):
        residual = x

        out = self.conv1(x)

        out = self.bn1(out)
        out = self.relu(out)

        # SEN
        out = out + self.ssem(out) + self.tsem(out) 

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out


class ResNet(nn.Module):

    def __init__(self, block, layers, num_classes=1000):
        self.inplanes = 64
        super(ResNet, self).__init__()
        self.conv1 = nn.Conv3d(3, 64, kernel_size=(1,7,7), stride=(1,2,2), padding=(0,3,3),
                               bias=False)
        self.bn1 = nn.BatchNorm3d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool3d(kernel_size=(1,3,3), stride=(1,2,2), padding=(0,1,1))
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2)
        self.avgpool = nn.AvgPool2d(7, stride=1)
        self.fc = nn.Linear(512 * block.expansion, num_classes)
        #self.ssem = SSEM(self.inplanes)
            
        for m in self.modules():
            if isinstance(m, nn.Conv3d) or isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm3d) or isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv3d(self.inplanes, planes * block.expansion,
                          kernel_size=1, stride=(1,stride,stride), bias=False),
                nn.BatchNorm3d(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes))

        return nn.Sequential(*layers)

    def forward(self, x):
        N, C, T, H, W = x.size()
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        #x = x + self.ssem(x)
        x = x.transpose(1,2).contiguous()
        x = x.view((-1,)+x.size()[2:])

        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)

        return x

def resnet18(**kwargs):
    """Constructs a ResNet-18 based model.
    """
    model = ResNet(BasicBlock, [2, 2, 2, 2], **kwargs)
    checkpoint = model_zoo.load_url(model_urls['resnet18'])
    layer_name = list(checkpoint.keys())
    for ln in layer_name :
        if 'conv' in ln or 'downsample.0.weight' in ln:
            checkpoint[ln] = checkpoint[ln].unsqueeze(2)  
    model.load_state_dict(checkpoint, strict=False)
    return model


def resnet34(**kwargs):
    """Constructs a ResNet-34 model.
    """
    model = ResNet(BasicBlock, [3, 4, 6, 3], **kwargs)
    checkpoint = model_zoo.load_url(model_urls['resnet34'])
    layer_name = list(checkpoint.keys())
    for ln in layer_name :
        if 'conv' in ln or 'downsample.0.weight' in ln:
            checkpoint[ln] = checkpoint[ln].unsqueeze(2)  
    model.load_state_dict(checkpoint, strict=False)
    return model




def test():
    net = resnet18()
    y = net(torch.randn(1,3,224,224))
    print(y.size())

#test()