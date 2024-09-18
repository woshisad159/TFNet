import torch.nn as nn
import torch
import torch.utils.model_zoo as model_zoo
import torch.nn.functional as F
import copy

model_urls = {
    'resnet18': 'https://download.pytorch.org/models/resnet18-f37072fd.pth',
    'resnet34': 'https://download.pytorch.org/models/resnet34-333f7ec4.pth',
    'resnet50': 'https://download.pytorch.org/models/resnet50-19c8e357.pth',
    'resnet101': 'https://download.pytorch.org/models/resnet101-5d3b4d8f.pth',
    'resnet152': 'https://download.pytorch.org/models/resnet152-b121ed2d.pth',
    'vgg19_bn': 'https://download.pytorch.org/models/vgg19_bn-c79401a0.pth',
    'convnext_tiny': "https://download.pytorch.org/models/convnext_tiny-983f1562.pth",
    'convnext_base': "https://download.pytorch.org/models/convnext_base-6075fbad.pth",
}

class Identity(nn.Module):
    def __init__(self):
        super(Identity, self).__init__()

    def forward(self, x):
        return x

class TemporalConv(nn.Module):
    def __init__(self, input_size, hidden_size, conv_type=2):
        super(TemporalConv, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.conv_type = conv_type

        if self.conv_type == 0:
            self.kernel_size = ['K3']
        elif self.conv_type == 1:
            self.kernel_size = ['K5', "P2"]
        elif self.conv_type == 2:
            self.kernel_size = ['K5', "P2", 'K5', "P2"]
        elif self.conv_type == 3:
            self.kernel_size = ['K3', 'K3']

        modules = []
        for layer_idx, ks in enumerate(self.kernel_size):
            input_sz = self.input_size if layer_idx == 0 else self.hidden_size
            if ks[0] == 'P':
                modules.append(nn.MaxPool1d(kernel_size=int(ks[1]), ceil_mode=False))
            elif ks[0] == 'K':
                modules.append(
                    nn.Conv1d(input_sz, self.hidden_size, kernel_size=int(ks[1]), stride=1, padding=0)
                )
                modules.append(nn.BatchNorm1d(self.hidden_size))
                modules.append(nn.ReLU(inplace=True))

        self.temporal_conv = nn.Sequential(*modules)

    def update_lgt(self, lgt):
        feat_len = copy.deepcopy(lgt)
        for ks in self.kernel_size:
            if ks[0] == 'P':
                feat_len = [(torch.floor(i / 2)).int() for i in feat_len]
            else:
                feat_len = [(i - int(ks[1]) + 1) for i in feat_len]
        return feat_len

    def forward(self, frame_feat, lgt):
        visual_feat = self.temporal_conv(frame_feat)
        lgt = self.update_lgt(lgt)
        return {
            "visual_feat": visual_feat,
            "feat_len": lgt,
        }

class NormLinear(nn.Module):
    def __init__(self, in_dim, out_dim):
        super(NormLinear, self).__init__()
        self.weight = nn.Parameter(torch.Tensor(in_dim, out_dim))
        nn.init.xavier_uniform_(self.weight, gain=nn.init.calculate_gain('relu'))

    def forward(self, x):
        outputs = torch.matmul(x, F.normalize(self.weight, dim=0))
        return outputs

def make_layer(block, inputSize, hiddenSize, outputSize, blocks, stride=1):
    downsample = None
    if stride != 1 or inputSize != outputSize:
        downsample = nn.Sequential(
            nn.Conv3d(inputSize, outputSize,
                      kernel_size=1, stride=(1, stride, stride), bias=False),
            nn.BatchNorm3d(outputSize),
        )

    layers = []
    layers.append(block(inputSize, hiddenSize, outputSize, stride, downsample))
    for i in range(1, blocks):
        layers.append(block(outputSize, hiddenSize, outputSize))

    return nn.Sequential(*layers)

def conv3x3(in_planes, out_planes, stride=1):
    # 3x3x3 convolution with padding
    return nn.Conv3d(
        in_planes,
        out_planes,
        kernel_size=(1, 3, 3),
        stride=(1, stride, stride),
        padding=(0, 1, 1),
        bias=False,
        dilation=1)

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
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)

        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            residual = self.downsample(residual)

        out = out + residual

        out = self.relu(out)

        return out

class Get_Correlation(nn.Module):
    def __init__(self, channels):
        super().__init__()
        reduction_channel = channels // 16
        self.down_conv = nn.Conv3d(channels, reduction_channel, kernel_size=1, bias=False)

        self.down_conv2 = nn.Conv3d(channels, channels, kernel_size=1, bias=False)
        self.spatial_aggregation1 = nn.Conv3d(reduction_channel, reduction_channel, kernel_size=(9, 3, 3),
                                              padding=(4, 1, 1), groups=reduction_channel)
        self.spatial_aggregation2 = nn.Conv3d(reduction_channel, reduction_channel, kernel_size=(9, 3, 3),
                                              padding=(4, 2, 2), dilation=(1, 2, 2), groups=reduction_channel)
        self.spatial_aggregation3 = nn.Conv3d(reduction_channel, reduction_channel, kernel_size=(9, 3, 3),
                                              padding=(4, 3, 3), dilation=(1, 3, 3), groups=reduction_channel)
        self.weights = nn.Parameter(torch.ones(3) / 3, requires_grad=True)
        self.weights2 = nn.Parameter(torch.ones(2) / 2, requires_grad=True)
        self.conv_back = nn.Conv3d(reduction_channel, channels, kernel_size=1, bias=False)

        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x2 = self.down_conv2(x)
        affinities = torch.einsum('bcthw,bctsd->bthwsd', x,
                                  torch.concat([x2[:, :, 1:], x2[:, :, -1:]], 2))  # repeat the last frame
        affinities2 = torch.einsum('bcthw,bctsd->bthwsd', x,
                                   torch.concat([x2[:, :, :1], x2[:, :, :-1]], 2))  # repeat the first frame
        features = torch.einsum('bctsd,bthwsd->bcthw', torch.concat([x2[:, :, 1:], x2[:, :, -1:]], 2),
                                self.sigmoid(affinities) - 0.5) * self.weights2[0] + \
                   torch.einsum('bctsd,bthwsd->bcthw', torch.concat([x2[:, :, :1], x2[:, :, :-1]], 2),
                                self.sigmoid(affinities2) - 0.5) * self.weights2[1]

        x = self.down_conv(x)
        aggregated_x = self.spatial_aggregation1(x) * self.weights[0] + self.spatial_aggregation2(x) * self.weights[1] \
                       + self.spatial_aggregation3(x) * self.weights[2]
        aggregated_x = self.conv_back(aggregated_x)

        return features * (self.sigmoid(aggregated_x) - 0.5)

class ResNet(nn.Module):
    def __init__(self, block, layers, num_classes=1000):
        self.inplanes = 64
        super(ResNet, self).__init__()
        self.conv1 = nn.Conv3d(3, 64, kernel_size=(1, 7, 7), stride=(1, 2, 2), padding=(0, 3, 3),
                               bias=False)
        self.bn1 = nn.BatchNorm3d(64)
        self.relu = nn.ReLU(inplace=True)

        self.maxpool = nn.MaxPool3d(kernel_size=(1, 3, 3), stride=(1, 2, 2), padding=(0, 1, 1))

        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2)
        self.avgpool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Linear(512 * block.expansion, num_classes)

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
                          kernel_size=1, stride=(1, stride, stride), bias=False),
                nn.BatchNorm3d(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes))

        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        ###################################################
        x = self.maxpool(x)
        ###################################################
        x = self.layer1(x)
        ###################################################
        x = self.layer2(x)
        ###################################################
        x = self.layer3(x)
        ###################################################
        x = self.layer4(x)
        ###################################################
        x = x.transpose(1, 2).contiguous()
        x = x.view((-1,) + x.size()[2:])  # bt,c,h,w

        x = self.avgpool(x)
        x = x.view(x.size(0), -1)  # bt,c
        x = self.fc(x)  # bt,c

        return x

class ResNetCorr(nn.Module):
    def __init__(self, block, layers, num_classes=1000):
        self.inplanes = 64
        super(ResNetCorr, self).__init__()
        self.conv1 = nn.Conv3d(3, 64, kernel_size=(1, 7, 7), stride=(1, 2, 2), padding=(0, 3, 3),
                               bias=False)
        self.bn1 = nn.BatchNorm3d(64)
        self.relu = nn.ReLU(inplace=True)

        self.maxpool = nn.MaxPool3d(kernel_size=(1, 3, 3), stride=(1, 2, 2), padding=(0, 1, 1))

        self.layer1 = self._make_layer(block, 64, layers[0])

        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.corr1 = Get_Correlation(self.inplanes)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        self.corr2 = Get_Correlation(self.inplanes)
        self.alpha = nn.Parameter(torch.zeros(3), requires_grad=True)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2)
        self.corr3 = Get_Correlation(self.inplanes)
        self.avgpool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Linear(512 * block.expansion, num_classes)

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
                          kernel_size=1, stride=(1, stride, stride), bias=False),
                nn.BatchNorm3d(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes))

        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        ###################################################
        x = self.maxpool(x)
        ###################################################
        x = self.layer1(x)
        ###################################################
        x = self.layer2(x)
        x = x + self.corr1(x) * self.alpha[0]
        ###################################################
        x = self.layer3(x)
        x = x + self.corr2(x) * self.alpha[1]
        ###################################################
        x = self.layer4(x)
        x = x + self.corr3(x) * self.alpha[2]
        ###################################################
        x = x.transpose(1, 2).contiguous()
        x = x.view((-1,) + x.size()[2:])  # bt,c,h,w

        x = self.avgpool(x)
        x = x.view(x.size(0), -1)  # bt,c
        x = self.fc(x)  # bt,c

        return x

class ResNet34MAM(nn.Module):
    def __init__(self, block, layers, num_classes=1000):
        self.inplanes = 64
        super(ResNet34MAM, self).__init__()
        self.motorAttention1 = MotorAttention(3, 16)

        self.conv1 = nn.Conv3d(3, 64, kernel_size=(1, 7, 7), stride=(1, 2, 2), padding=(0, 3, 3),
                               bias=False)
        self.bn1 = nn.BatchNorm3d(64)
        self.relu = nn.ReLU(inplace=True)

        self.maxpool = nn.MaxPool3d(kernel_size=(1, 2, 2), stride=(1, 2, 2))

        self.layer1 = self._make_layer(block, 64, layers[0])
        self.motorAttention2 = MotorAttention(64, 64)

        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.motorAttention3 = MotorAttention(128, 64)

        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        self.motorAttention4 = MotorAttention(256, 64)

        self.layer4 = self._make_layer(block, 512, layers[3], stride=2)
        self.avgpool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Linear(512 * block.expansion, num_classes)

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
                          kernel_size=1, stride=(1, stride, stride), bias=False),
                nn.BatchNorm3d(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes))

        return nn.Sequential(*layers)

    def forward(self, x):
        outData1 = []
        outData2 = []
        outData3 = []

        x = self.motorAttention1(x)
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.motorAttention2(x)
        outData1.append(x)

        x = self.layer2(x)
        x = self.motorAttention3(x)

        outData1.append(x)
        outData2.append(x)

        x = self.layer3(x)
        x = self.motorAttention4(x)
        outData2.append(x)

        outData3.append(x)

        x = self.layer4(x)
        outData3.append(x)

        x = x.transpose(1, 2).contiguous()
        x = x.view((-1,) + x.size()[2:])  # bt,c,h,w

        x = self.avgpool(x)
        x = x.view(x.size(0), -1)  # bt,c
        x = self.fc(x)  # bt,c

        return x, outData1, outData2, outData3

class MotorAttention(nn.Module):
    def __init__(self, inChannels, hiddens):
        super().__init__()
        k = 3
        p = 1
        self.conv3d1 = nn.Conv3d(in_channels=inChannels, out_channels=hiddens, kernel_size=(k, 1, 1), stride=1, padding=(p, 0, 0))
        self.conv3d2 = nn.Conv3d(in_channels=hiddens, out_channels=hiddens, kernel_size=(k, 1, 1), stride=1, padding=(p, 0, 0))
        self.conv3d3 = nn.Conv3d(in_channels=hiddens, out_channels=hiddens, kernel_size=(k, 1, 1), stride=1,
                                 padding=(p, 0, 0))
        self.conv3d4 = nn.Conv3d(in_channels=hiddens, out_channels=inChannels, kernel_size=(k, 1, 1), stride=1,
                                 padding=(p, 0, 0))

        self.reLU = nn.LeakyReLU(inplace=True)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        out = self.conv3d1(x)
        out = self.reLU(out)

        out = self.conv3d2(out)
        out = self.reLU(out)

        out = self.conv3d3(out)
        out = self.reLU(out)

        out = self.conv3d4(out)
        outData = self.sigmoid(out)

        return x * outData

def resnet18(**kwargs):
    """Constructs a ResNet-18 based model.
    """
    model = ResNet(BasicBlock, [2, 2, 2, 2], **kwargs)
    checkpoint = model_zoo.load_url(model_urls['resnet18'])
    layer_name = list(checkpoint.keys())
    for ln in layer_name:
        if 'conv' in ln or 'downsample.0.weight' in ln:
            checkpoint[ln] = checkpoint[ln].unsqueeze(2)
    model.load_state_dict(checkpoint, strict=False)
    return model

def resnet18Corr(**kwargs):
    """Constructs a ResNet-18 based model.
    """
    model = ResNetCorr(BasicBlock, [2, 2, 2, 2], **kwargs)
    checkpoint = model_zoo.load_url(model_urls['resnet18'])
    layer_name = list(checkpoint.keys())
    for ln in layer_name:
        if 'conv' in ln or 'downsample.0.weight' in ln:
            checkpoint[ln] = checkpoint[ln].unsqueeze(2)
    model.load_state_dict(checkpoint, strict=False)
    return model

def resnet34MAM(**kwargs):
    """Constructs a ResNet-34 model.
    """
    model = ResNet34MAM(BasicBlock, [3, 4, 6, 3], **kwargs)

    checkpoint = model_zoo.load_url(model_urls['resnet34'])
    layer_name = list(checkpoint.keys())
    for ln in layer_name:
        if 'conv' in ln or 'downsample.0.weight' in ln:
            checkpoint[ln] = checkpoint[ln].unsqueeze(2)
    model_dict = model.state_dict()
    load_pretrained_dict = checkpoint

    pretrained_dict = {k: v for k, v in load_pretrained_dict.items() if k in model_dict.keys()}
    model_dict.update(pretrained_dict)
    model.load_state_dict(model_dict)
    return model

