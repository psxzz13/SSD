import math
import torch
import numpy
import numpy as np
from torch import nn,flatten
import torch.nn.functional as F
import torch.nn as nn
from torch.nn import init
from torch.nn.modules.activation import ReLU
from torch.nn.modules.batchnorm import BatchNorm2d
from CoTNetBlock import CoTNetLayer
from torch.nn.parameter import Parameter

# ssd_model = ssd_SE, ssd_cbam, ssd_resnet, ssd_ECA, ssd_Botnet, ssd_Cotnet
ssd_model = "ssd_cbam"


if ssd_model == "ssd_resnet":
    class Bottleneck(nn.Module):
        expansion = 4

        def __init__(self, in_channel, out_channel, stride=1, downsample=None):
            super(Bottleneck, self).__init__()
            self.conv1 = nn.Conv2d(in_channels=in_channel, out_channels=out_channel,
                                kernel_size=1, stride=1, bias=False)  # squeeze channels
            self.bn1 = nn.BatchNorm2d(out_channel)
            # -----------------------------------------
            self.conv2 = nn.Conv2d(in_channels=out_channel, out_channels=out_channel,
                                kernel_size=3, stride=stride, bias=False, padding=1)
            self.bn2 = nn.BatchNorm2d(out_channel)
            # -----------------------------------------
            self.conv3 = nn.Conv2d(in_channels=out_channel, out_channels=out_channel*self.expansion,
                                kernel_size=1, stride=1, bias=False)  # unsqueeze channels
            self.bn3 = nn.BatchNorm2d(out_channel*self.expansion)
            self.relu = nn.ReLU(inplace=True)
            self.downsample = downsample

        def forward(self, x):
            identity = x
            if self.downsample is not None:
                identity = self.downsample(x)

            out = self.conv1(x)
            out = self.bn1(out)
            out = self.relu(out)

            out = self.conv2(out)
            out = self.bn2(out)
            out = self.relu(out)

            out = self.conv3(out)
            out = self.bn3(out)

            out += identity
            out = self.relu(out)

            return out


    class ResNet(nn.Module):

        def __init__(self, block, blocks_num, num_classes=1000, include_top=True):
            super(ResNet, self).__init__()
            self.include_top = include_top
            self.in_channel = 64

            self.conv1 = nn.Conv2d(3, self.in_channel, kernel_size=7, stride=2,
                                padding=3, bias=False)
            self.bn1 = nn.BatchNorm2d(self.in_channel)
            self.relu = nn.ReLU(inplace=True)
            self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
            self.layer1 = self._make_layer(block, 64, blocks_num[0])
            self.layer2 = self._make_layer(block, 128, blocks_num[1], stride=2)
            self.layer3 = self._make_layer(block, 256, blocks_num[2], stride=2)
            self.layer4 = self._make_layer(block, 512, blocks_num[3], stride=2)
            if self.include_top:
                self.avgpool = nn.AdaptiveAvgPool2d((1, 1))  # output size = (1, 1)
                self.fc = nn.Linear(512 * block.expansion, num_classes)

            for m in self.modules():
                if isinstance(m, nn.Conv2d):
                    nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')

        def _make_layer(self, block, channel, block_num, stride=1):
            downsample = None
            if stride != 1 or self.in_channel != channel * block.expansion:
                downsample = nn.Sequential(
                    nn.Conv2d(self.in_channel, channel * block.expansion, kernel_size=1, stride=stride, bias=False),
                    nn.BatchNorm2d(channel * block.expansion))

            layers = []
            layers.append(block(self.in_channel, channel, downsample=downsample, stride=stride))
            self.in_channel = channel * block.expansion

            for _ in range(1, block_num):
                layers.append(block(self.in_channel, channel))

            return nn.Sequential(*layers)

        def forward(self, x):
            x = self.conv1(x)
            x = self.bn1(x)
            x = self.relu(x)
            x = self.maxpool(x)

            x = self.layer1(x)
            x = self.layer2(x)
            x = self.layer3(x)
            x = self.layer4(x)

            if self.include_top:
                x = self.avgpool(x)
                x = torch.flatten(x, 1)
                x = self.fc(x)

            return x


    def resnet50(num_classes=1000, include_top=True):
        return ResNet(Bottleneck, [3, 4, 6, 3], num_classes=num_classes, include_top=include_top)


elif ssd_model == "ssd_ECA":
    class eca_layer(nn.Module):

        def __init__(self, channel, k_size=3):
            super(eca_layer, self).__init__()
            self.avg_pool = nn.AdaptiveAvgPool2d(1)
            self.conv = nn.Conv1d(1, 1, kernel_size=k_size, padding=(k_size - 1) // 2, bias=False)
            self.sigmoid = nn.Sigmoid()

        def forward(self, x):
            # feature descriptor on the global spatial information
            y = self.avg_pool(x)

            # Two different branches of ECA module
            y = self.conv(y.squeeze(-1).transpose(-1, -2)).transpose(-1, -2).unsqueeze(-1)

            # Multi-scale information fusion
            y = self.sigmoid(y)

            return x * y.expand_as(x)

    def conv3x3(in_planes, out_planes, stride=1):

        return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                        padding=1, bias=False)




    class ECABottleneck(nn.Module):
        expansion = 4

        def __init__(self, inplanes, planes, stride=1, downsample=None, k_size=3):
            super(ECABottleneck, self).__init__()
            self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=1, bias=False)
            self.bn1 = nn.BatchNorm2d(planes)
            self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride,
                                padding=1, bias=False)
            self.bn2 = nn.BatchNorm2d(planes)
            self.conv3 = nn.Conv2d(planes, planes * 4, kernel_size=1, bias=False)
            self.bn3 = nn.BatchNorm2d(planes * 4)
            self.relu = nn.ReLU(inplace=True)
            self.eca = eca_layer(planes * 4, k_size)
            self.downsample = downsample
            self.stride = stride

        def forward(self, x):
            residual = x

            out = self.conv1(x)
            out = self.bn1(out)
            out = self.relu(out)
            out = self.eca(out)

            out = self.conv2(out)
            out = self.bn2(out)
            out = self.relu(out)
            out = self.eca(out)

            out = self.conv3(out)
            out = self.bn3(out)
            out = self.eca(out)
            # out = self.eca(out)

            if self.downsample is not None:
                residual = self.downsample(x)

            out += residual
            out = self.relu(out)

            return out


    class ResNet(nn.Module):

        def __init__(self, block, layers, num_classes=1000, k_size=[3, 3, 3, 3]):
            self.inplanes = 64
            super(ResNet, self).__init__()
            self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3,
                                bias=False)
            self.bn1 = nn.BatchNorm2d(64)
            self.relu = nn.ReLU(inplace=True)
            self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
            self.layer1 = self._make_layer(block, 64, layers[0], int(k_size[0]))
            self.layer2 = self._make_layer(block, 128, layers[1], int(k_size[1]), stride=2)
            self.layer3 = self._make_layer(block, 256, layers[2], int(k_size[2]), stride=2)
            self.layer4 = self._make_layer(block, 512, layers[3], int(k_size[3]), stride=2)
            self.avgpool = nn.AvgPool2d(7, stride=1)
            self.fc = nn.Linear(512 * block.expansion, num_classes)

            for m in self.modules():
                if isinstance(m, nn.Conv2d):
                    n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                    m.weight.data.normal_(0, math.sqrt(2. / n))
                elif isinstance(m, nn.BatchNorm2d):
                    m.weight.data.fill_(1)
                    m.bias.data.zero_()

        def _make_layer(self, block, planes, blocks, k_size, stride=1):
            downsample = None
            if stride != 1 or self.inplanes != planes * block.expansion:
                downsample = nn.Sequential(
                    nn.Conv2d(self.inplanes, planes * block.expansion,
                            kernel_size=1, stride=stride, bias=False),
                    nn.BatchNorm2d(planes * block.expansion),
                )

            layers = []
            layers.append(block(self.inplanes, planes, stride, downsample, k_size))
            self.inplanes = planes * block.expansion
            for i in range(1, blocks):
                layers.append(block(self.inplanes, planes, k_size=k_size))

            return nn.Sequential(*layers)

        def forward(self, x):
            x = self.conv1(x)
            x = self.bn1(x)
            x = self.relu(x)
            x = self.maxpool(x)

            x = self.layer1(x)
            x = self.layer2(x)
            x = self.layer3(x)
            x = self.layer4(x)

            x = self.avgpool(x)
            x = x.view(x.size(0), -1)
            x = self.fc(x)

            return x



    def  resnet50(k_size=[3, 3, 3, 3], num_classes=1000, pretrained=False):
        model = ResNet(ECABottleneck, [3, 4, 6, 3], num_classes=num_classes, k_size=k_size)
        model.avgpool = nn.AdaptiveAvgPool2d(1)
        return model


elif ssd_model == "ssd_Botnet":
    def get_n_params(model):
        pp = 0
        for p in list(model.parameters()):
            nn = 1
            for s in list(p.size()):
                nn = nn * s
            pp += nn
        return pp


    class MHSA(nn.Module):
        def __init__(self, n_dims, width=14, height=14, heads=4):
            super(MHSA, self).__init__()
            self.heads = heads
            self.n_dims = n_dims

            self.query = nn.Conv2d(n_dims, n_dims, kernel_size=1)
            self.key = nn.Conv2d(n_dims, n_dims, kernel_size=1)
            self.value = nn.Conv2d(n_dims, n_dims, kernel_size=1)

            self.rel_h = nn.Parameter(torch.randn([1, heads, n_dims // heads, 1, height]), requires_grad=True)
            self.rel_w = nn.Parameter(torch.randn([1, heads, n_dims // heads, width, 1]), requires_grad=True)

            self.softmax = nn.Softmax(dim=-1)

        def forward(self, x):
            n_batch, C, width, height = x.size()
            self.rel_h = nn.Parameter(torch.randn([1, self.heads, self.n_dims // self.heads, 1, height]), requires_grad=True)
            self.rel_w = nn.Parameter(torch.randn([1, self.heads, self.n_dims // self.heads, width, 1]), requires_grad=True)
            q = self.query(x).view(n_batch, self.heads, C // self.heads, -1)
            k = self.key(x).view(n_batch, self.heads, C // self.heads, -1)
            v = self.value(x).view(n_batch, self.heads, C // self.heads, -1)

            content_content = torch.matmul(q.permute(0, 1, 3, 2), k)

            content_position = (self.rel_h + self.rel_w).view(1, self.heads, C // self.heads, -1).permute(0, 1, 3, 2).to('cuda')
            content_position = torch.matmul(content_position, q)

            energy = content_content + content_position
            attention = self.softmax(energy)

            out = torch.matmul(v, attention.permute(0, 1, 3, 2))
            out = out.view(n_batch, C, width, height)

            return out


    class Bottleneck(nn.Module):
        expansion = 4

        def __init__(self, in_planes, planes, stride=1, heads=4, mhsa=False, resolution=None, downsample=None):
            super(Bottleneck, self).__init__()

            self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=1, bias=False)
            self.bn1 = nn.BatchNorm2d(planes)

            self.stride = stride

            if not mhsa:
                self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, padding=1, stride=stride, bias=False)
            else:
                self.conv2 = nn.ModuleList()
                self.conv2.append(MHSA(planes, width=int(resolution[0]), height=int(resolution[1]), heads=heads))
                if stride == 2:
                    self.conv2.append(nn.AvgPool2d(2, 2, ceil_mode=True))
                self.conv2 = nn.Sequential(*self.conv2)
            self.bn2 = nn.BatchNorm2d(planes)
            self.conv3 = nn.Conv2d(planes, self.expansion * planes, kernel_size=1, bias=False)
            self.bn3 = nn.BatchNorm2d(self.expansion * planes)

            self.downsample = nn.Sequential()
            if stride != 1 or in_planes != self.expansion * planes:
                self.downsample = nn.Sequential(
                    nn.Conv2d(in_planes, self.expansion * planes, kernel_size=1, stride=stride),
                    nn.BatchNorm2d(self.expansion * planes)
                )

        def forward(self, x):
            out = F.relu(self.bn1(self.conv1(x)))
            out = F.relu(self.bn2(self.conv2(out)))
            out = self.bn3(self.conv3(out))

            out += self.downsample(x)
            out = F.relu(out)
            return out


    
    class ResNet(nn.Module):
        def __init__(self, block, num_blocks, num_classes=1000, resolution=(224, 224), heads=4):
            super(ResNet, self).__init__()
            self.in_planes = 64
            self.resolution = list(resolution)

            self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False)
            # self.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
            if self.conv1.stride[0] == 2:
                self.resolution[0] /= 2
            if self.conv1.stride[1] == 2:
                self.resolution[1] /= 2
            self.bn1 = nn.BatchNorm2d(64)
            self.relu = nn.ReLU(inplace=True)
            self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)  # for ImageNet
            if self.maxpool.stride == 2:
                self.resolution[0] /= 2
                self.resolution[1] /= 2

            self.layer1 = self._make_layer(block, 64, num_blocks[0], stride=1)
            self.layer2 = self._make_layer(block, 128, num_blocks[1], stride=2, heads=1, mhsa=True)
            self.layer3 = self._make_layer(block, 256, num_blocks[2], stride=2)
            self.layer4 = self._make_layer(block, 512, num_blocks[3], stride=2, heads=heads, mhsa=True)

            self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
            self.fc = nn.Sequential(
                nn.Dropout(0.3),  # All architecture deeper than ResNet-200 dropout_rate: 0.2
                nn.Linear(512 * block.expansion, num_classes)
            )

        def _make_layer(self, block, planes, num_blocks, stride=1, heads=4, mhsa=False):
            strides = [stride] + [1] * (num_blocks - 1)
            layers = []
            for idx, stride in enumerate(strides):
                layers.append(block(self.in_planes, planes, stride, heads, mhsa, self.resolution))
                if stride == 2:
                    # self.resolution[0] /= 2
                    # self.resolution[1] /= 2
                    self.resolution[0] = math.ceil(self.resolution[0] / 2)
                    self.resolution[1] = math.ceil(self.resolution[1] / 2)

                self.in_planes = planes * block.expansion
            return nn.Sequential(*layers)

        def forward(self, x):
            out = self.relu(self.bn1(self.conv1(x)))
            out = self.maxpool(out)  # for ImageNet

            out = self.layer1(out)
            out = self.layer2(out)

            out = self.layer3(out)
            out = self.layer4(out)

            out = self.avgpool(out)
            out = torch.flatten(out, 1)
            out = self.fc(out)
            return out


    def resnet50(num_classes=1000, resolution=(300, 300), heads=4):
        return ResNet(Bottleneck, [3, 4, 6, 3], num_classes=num_classes, resolution=resolution, heads=heads)


    def main():
        x = torch.randn([2, 3, 224, 224])
        model = resnet50(resolution=tuple(x.shape[2:]), heads=8)
        print(model(x).size())
        print(get_n_params(model))


    if __name__ == '__main__':
        main()


elif ssd_model == "ssd_Cotnet":
    class Bottleneck(nn.Module):
        expansion = 4

        def __init__(self, inplanes, planes, stride=1, downsample=None):
            super(Bottleneck, self).__init__()

            self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=1, bias=False)
            self.bn1 = nn.BatchNorm2d(planes)
            #self.cot_layer = CoTNetLayer(dim=planes, kernel_size=3)
            self.conv2 = CoTNetLayer(dim=planes, kernel_size=3)
            self.bn2 = nn.BatchNorm2d(planes)
            self.conv3 = nn.Conv2d(planes, planes * self.expansion, kernel_size=1, bias=False)
            self.bn3 = nn.BatchNorm2d(planes * self.expansion)
            self.relu = nn.ReLU(inplace=True)
            self.downsample = downsample
            self.planes = planes
            self.stride = stride
            if stride > 1 and self.planes != 256:
                self.avd = nn.AvgPool2d(3, 2, padding=1)
            else:
                self.avd = None

        def forward(self, x):
            residual = x

            out = self.conv1(x)  # 1*1 Conv
            out = self.bn1(out)
            out = self.relu(out)

            if self.avd is not None:  
                out = self.avd(out)

            #out = self.cot_layer(out)  # CoTNetLayer
            out = self.conv2(out)
            out = self.bn2(out)
            out = self.relu(out)

            out = self.conv3(out)  # 1*1 Conv
            out = self.bn3(out)

            if self.downsample is not None:
                residual = self.downsample(x)

            out += residual
            out = self.relu(out)

            return out


    class ResNet(nn.Module):

        def __init__(self, block, layers, num_classes=1000):
            self.inplanes = 64
            super(ResNet, self).__init__()
            self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3,
                                bias=False)
            self.bn1 = nn.BatchNorm2d(64)
            self.relu = nn.ReLU(inplace=True)
            self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
            self.layer1 = self._make_layer(block, 64, layers[0], stride=1)
            self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
            self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
            self.layer4 = self._make_layer(block, 512, layers[3], stride=2)
            self.avgpool = nn.AvgPool2d(7, stride=1)
            self.fc = nn.Linear(512 * block.expansion, num_classes)

            for m in self.modules():
                if isinstance(m, nn.Conv2d):
                    nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                elif isinstance(m, nn.BatchNorm2d):
                    nn.init.constant_(m.weight, 1)
                    nn.init.constant_(m.bias, 0)

        def _make_layer(self, block, planes, blocks, stride=1):
            downsample = None
            if stride != 1 or self.inplanes != planes * block.expansion:
                downsample = nn.Sequential(
                    nn.Conv2d(self.inplanes, planes * block.expansion,
                            kernel_size=1, stride=stride, bias=False),
                    nn.BatchNorm2d(planes * block.expansion),
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
            x = self.maxpool(x)

            x = self.layer1(x)
            x = self.layer2(x)
            x = self.layer3(x)
            x = self.layer4(x)

            x = self.avgpool(x)
            x = x.view(x.size(0), -1)
            x = self.fc(x)

            return x

    def resnet50(num_classes=1000):
        return ResNet(Bottleneck, [3, 4, 6, 3], num_classes=num_classes)

    if __name__ == '__main__':
        x = torch.rand(1, 3, 224, 224)
        model = resnet50(Bottleneck, [3,4,6,3])
        y = model(x)
        print(y.shape)


elif ssd_model == "ssd_SE":
    class SELayer(nn.Module):
        def __init__(self, channel, reduction = 16):
            super(SELayer, self).__init__()
            self.avg_pool = nn.AdaptiveAvgPool2d(1)
            self.fc = nn.Sequential(
                nn.Linear(channel, channel // reduction, bias = False),
                nn.ReLU(inplace = True),
                nn.Linear(channel // reduction, channel, bias = False),
                nn.Sigmoid()
            )

        def forward(self, x):
            b, c, _, _ = x.size()
            y = self.avg_pool(x).view(b, c)
            y = self.fc(y).view(b, c, 1, 1)

            return x * y.expand_as(x)


    def conv3x3(in_planes, out_planes, stride = 1, groups = 1, dilation = 1):
        """3x3 convolution with padding"""
        return nn.Conv2d(in_planes, out_planes, kernel_size = 3, stride = stride,
                        padding = dilation, groups = groups, bias = False, dilation = dilation)


    def conv1x1(in_planes, out_planes, stride = 1):
        """1x1 convolution"""
        return nn.Conv2d(in_planes, out_planes, kernel_size = 1,stride = stride, bias = False)




    class Bottleneck(nn.Module):
        expansion = 4

        def __init__(self, inplanes, planes, stride = 1, downsample = None, groups = 1,
                    base_width = 64, dilation = 1, norm_layer = None, reduction = 16, se = False):
            super(Bottleneck, self).__init__()
            if norm_layer is None:
                norm_layer = nn.BatchNorm2d
            width = int(planes * (base_width / 64.)) * groups

            self.se = se
            self.conv1 = conv1x1(inplanes, width)
            self.bn1 = norm_layer(width)
            self.conv2 = conv3x3(width, width, stride, groups, dilation)
            self.bn2 = norm_layer(width)
            self.conv3 = conv1x1(width, planes * self.expansion)
            self.bn3 = norm_layer(planes * self.expansion)

            self.se_layer = SELayer(planes * self.expansion, reduction)
            self.relu = nn.ReLU(inplace = True)
            self.downsample = downsample
            self.stride = stride


        def forward(self, x):
            identity = x

            out = self.conv1(x)
            out = self.bn1(out)
            out = self.relu(out)

            out = self.conv2(out)
            out = self.bn2(out)
            out = self.relu(out)

            out = self.conv3(out)
            out = self.bn3(out)


            if self.se:
                out = self.se_layer(out)


            if self.downsample is not None:
                identity = self.downsample(x)

            out += identity
            out = self.relu(out)

            return out


    class ResNet(nn.Module):
        def __init__(self, block = None, blocks = None, zero_init_residual = False,
                    groups=1, width_per_group=64, replace_stride_with_dilation = None,
                    norm_layer=None, se=False):
            super().__init__()
            if norm_layer is None:
                norm_layer = nn.BatchNorm2d

            self._norm_layer = norm_layer

            self.inplanes = 64
            self.dilation = 1

            self.blocks = blocks
            if replace_stride_with_dilation is None:

                replace_stride_with_dilation = [False, False, False]
            if len(replace_stride_with_dilation) != 3:
                raise ValueError("replace_stride_with_dilation should be None "
                                "or a 3-element tuple, got {}".format(replace_stride_with_dilation))

            self.se = se  # Squeeze-and-Excitation Module
            self.groups = groups
            self.base_width = width_per_group
            self.conv1 = nn.Conv2d(3, self.inplanes, kernel_size = 7, stride = 2, padding = 3, bias = False)
            self.bn1 = self._norm_layer(self.inplanes)
            self.relu = nn.ReLU(inplace = True)
            self.maxpool = nn.MaxPool2d(kernel_size = 3, stride = 2, padding = 1)
            self.layer1 = self._make_layer(block, 64, self.blocks[0])
            self.layer2 = self._make_layer(block, 128, self.blocks[1], stride = 2,
                                        dilate = replace_stride_with_dilation[0])
            self.layer3 = self._make_layer(block, 256, self.blocks[2], stride = 2,
                                        dilate = replace_stride_with_dilation[1])
            self.bi1 = nn.UpsamplingBilinear2d(scale_factor=2)
            self.layer4 = self._make_layer(block, 512, self.blocks[3], stride = 2,
                                        dilate = replace_stride_with_dilation[2])

            for m in self.modules():
                if isinstance(m, nn.Conv2d):
                    nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                    nn.init.constant_(m.weight, 1)
                    nn.init.constant_(m.bias, 0)


            if zero_init_residual:
                for m in self.modules():
                    if isinstance(m, Bottleneck):
                        nn.init.constant_(m.bn3.weight, 0)

        def _make_layer(self, block, planes, blocks, stride = 1, dilate = False):
            norm_layer = self._norm_layer
            downsample = None
            previous_dilation = self.dilation
            if dilate:
                self.dilation *= stride
                stride = 1
            if stride != 1 or self.inplanes != planes * block.expansion:
                downsample = nn.Sequential(
                    conv1x1(self.inplanes, planes * block.expansion, stride),
                    norm_layer(planes * block.expansion),
                )

            layers = []
            layers.append(block(self.inplanes, planes, stride, downsample, self.groups,
                                self.base_width, previous_dilation, norm_layer, se=self.se))
            self.inplanes = planes * block.expansion
            for _ in range(1, blocks):
                layers.append(block(self.inplanes, planes, groups=self.groups,
                                    base_width=self.base_width, dilation=self.dilation,
                                    norm_layer=norm_layer, se=self.se))

            return nn.Sequential(*layers)



        def forward(self, x):

                x = self.conv1(x)
                x = self.bn1(x)
                x = self.relu(x)
                x = self.maxpool(x)
                x = self.layer1(x)
                x = self.layer2(x)
                x = self.layer3(x)
                x = self.layer4(x)
                return x


    def resnet50():
        return ResNet(Bottleneck,[3,4,6,3],se=True)

    if __name__ == '__main__':
        import torch
        from torchsummary import summary

        resnet50 = ResNet(block=Bottleneck, blocks=[3, 4, 6, 3],
                        se=True)

        summary(resnet50.to('cuda'), (3, 300, 300))


elif ssd_model == "ssd_cbam":
    class Channel_Attention(nn.Module):

        def __init__(self, channel, r = 16):
            super(Channel_Attention, self).__init__()
            self._avg_pool = nn.AdaptiveAvgPool2d(1)
            self._max_pool = nn.AdaptiveMaxPool2d(1)

            self._fc = nn.Sequential(
                nn.Conv2d(channel, channel // r, 1, bias = False),
                nn.ReLU(inplace = True),
                nn.Conv2d(channel // r, channel, 1, bias = False)
            )

            self._sigmoid = nn.Sigmoid()

        def forward(self, x):
            y1 = self._avg_pool(x)
            y1 = self._fc(y1)

            y2 = self._max_pool(x)
            y2 = self._fc(y2)

            y = self._sigmoid(y1 + y2)
            return x * y


    class Spatial_Attention(nn.Module):

        def __init__(self, kernel_size = 3):
            super(Spatial_Attention, self).__init__()

            assert kernel_size % 2 == 1, "kernel_size = {}".format(kernel_size)
            padding = (kernel_size - 1) // 2

            self._layer = nn.Sequential(
                nn.Conv2d(2, 1, kernel_size = kernel_size, padding = padding),
                nn.Sigmoid()
            )

        def forward(self, x):
            avg_mask = torch.mean(x, dim = 1, keepdim = True)
            max_mask, _= torch.max(x, dim = 1, keepdim = True)
            mask = torch.cat([avg_mask, max_mask], dim = 1)

            mask = self._layer(mask)
            return x * mask


    def conv3x3(in_planes, out_planes, stride = 1, groups = 1, dilation = 1):
        """3x3 convolution with padding"""
        return nn.Conv2d(in_planes, out_planes, kernel_size = 3, stride = stride,
                        padding = dilation, groups = groups, bias = False, dilation = dilation)


    def conv1x1(in_planes, out_planes, stride = 1):
        """1x1 convolution"""
        return nn.Conv2d(in_planes, out_planes, kernel_size = 1,stride = stride, bias = False)




    class Bottleneck(nn.Module):
        expansion = 4

        def __init__(self, inplanes, planes, stride = 1, downsample = None, groups = 1,
                    base_width = 64, dilation = 1, norm_layer = None, reduction = 16, cbam=False):
            super(Bottleneck, self).__init__()
            if norm_layer is None:
                norm_layer = nn.BatchNorm2d
            width = int(planes * (base_width / 64.)) * groups

            self.cbam = cbam
            self.conv1 = conv1x1(inplanes, width)
            # Both self.conv2 and self.downsample layers downsample the input when stride != 1
            self.bn1 = norm_layer(width)
            self.conv2 = conv3x3(width, width, stride, groups, dilation)
            self.bn2 = norm_layer(width)
            self.conv3 = conv1x1(width, planes * self.expansion)
            self.bn3 = norm_layer(planes * self.expansion)

            self.ca_layer = Channel_Attention(planes * self.expansion, reduction)
            self.sa_layer = Spatial_Attention()

            self.relu = nn.ReLU(inplace = True)
            self.downsample = downsample
            self.stride = stride


        def forward(self, x):
            identity = x

            out = self.conv1(x)
            out = self.bn1(out)
            out = self.relu(out)

            out = self.conv2(out)
            out = self.bn2(out)
            out = self.relu(out)

            out = self.conv3(out)
            out = self.bn3(out)


            if self.cbam:
                out = self.ca_layer(out)
                out = self.sa_layer(out)

            if self.downsample is not None:
                identity = self.downsample(x)

            out += identity
            out = self.relu(out)

            return out


    class ResNet(nn.Module):
        def __init__(self, block = None, blocks = None, zero_init_residual = False,
                    groups=1, width_per_group=64, replace_stride_with_dilation = None,
                    norm_layer=None, cbam=False):
            super().__init__()
            if norm_layer is None:
                norm_layer = nn.BatchNorm2d

            self._norm_layer = norm_layer

            self.inplanes = 64
            self.dilation = 1

            self.blocks = blocks
            if replace_stride_with_dilation is None:
                replace_stride_with_dilation = [False, False, False]
            if len(replace_stride_with_dilation) != 3:
                raise ValueError("replace_stride_with_dilation should be None "
                                "or a 3-element tuple, got {}".format(replace_stride_with_dilation))

            self.cbam = cbam  # Convolutional Block Attention Module
            self.groups = groups
            self.base_width = width_per_group
            self.conv1 = nn.Conv2d(3, self.inplanes, kernel_size = 7, stride = 2, padding = 3, bias = False)
            self.bn1 = self._norm_layer(self.inplanes)
            self.relu = nn.ReLU(inplace = True)
            self.maxpool = nn.MaxPool2d(kernel_size = 3, stride = 2, padding = 1)
            self.layer1 = self._make_layer(block, 64, self.blocks[0])
            self.layer2 = self._make_layer(block, 128, self.blocks[1], stride = 2,
                                        dilate = replace_stride_with_dilation[0])
            self.layer3 = self._make_layer(block, 256, self.blocks[2], stride = 2,
                                        dilate = replace_stride_with_dilation[1])
            self.bi1 = nn.UpsamplingBilinear2d(scale_factor=2)
            self.layer4 = self._make_layer(block, 512, self.blocks[3], stride = 2,
                                        dilate = replace_stride_with_dilation[2])

            for m in self.modules():
                if isinstance(m, nn.Conv2d):
                    nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                    nn.init.constant_(m.weight, 1)
                    nn.init.constant_(m.bias, 0)

            if zero_init_residual:
                for m in self.modules():
                    if isinstance(m, Bottleneck):
                        nn.init.constant_(m.bn3.weight, 0)

        def _make_layer(self, block, planes, blocks, stride = 1, dilate = False):
            norm_layer = self._norm_layer
            downsample = None
            previous_dilation = self.dilation
            if dilate:
                self.dilation *= stride
                stride = 1
            if stride != 1 or self.inplanes != planes * block.expansion:
                downsample = nn.Sequential(
                    conv1x1(self.inplanes, planes * block.expansion, stride),
                    norm_layer(planes * block.expansion),
                )

            layers = []
            layers.append(block(self.inplanes, planes, stride, downsample, self.groups,
                                self.base_width, previous_dilation, norm_layer, cbam=self.cbam))
            self.inplanes = planes * block.expansion
            for _ in range(1, blocks):
                layers.append(block(self.inplanes, planes, groups=self.groups,
                                    base_width=self.base_width, dilation=self.dilation,
                                    norm_layer=norm_layer, cbam=self.cbam))

            return nn.Sequential(*layers)



        def forward(self, x):

                x = self.conv1(x)
                x = self.bn1(x)
                x = self.relu(x)
                x = self.maxpool(x)
                x = self.layer1(x)
                x = self.layer2(x)
                x = self.layer3(x)
                x = self.layer4(x)
                return x


    def resnet50():
        return ResNet(Bottleneck,[3,4,6,3],cbam=True)

    if __name__ == '__main__':
        import torch
        from torchsummary import summary
        resnet50 = ResNet(block=Bottleneck, blocks=[3, 4, 6, 3],
                        cbam=True)

        summary(resnet50.to('cuda'), (3, 300, 300))