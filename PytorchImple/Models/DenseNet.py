# Writer：TuTTTTT
# 编写Time:2024/3/15 21:21
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Any, List, Optional, Tuple
from torch import Tensor
from collections import OrderedDict


class DenseLayer(nn.Module):
    # question 1:nn.Module
    """
    DenseLayer
    """
    def __init__(self, input_channels: int, growth_rate: int, bn_size: int, drop_rate: float) -> None:
        """

        :param input_channels: 输入特征图
        :param growth_rate: 增长率
        :param bn_size: 一批图片的数量
        :param drop_rate: 遗忘率
        """
        super(DenseLayer, self).__init__()
        self.norm1 = nn.BatchNorm2d(input_channels)
        self.relu1 = nn.ReLU(inplace=True)
        self.conv1 = nn.Conv2d(input_channels, bn_size * growth_rate, kernel_size=1, stride=1, bias=False)

        self.norm2 = nn.BatchNorm2d(bn_size * growth_rate)
        self.relu2 = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(bn_size * growth_rate, growth_rate, kernel_size=3, stride=1, padding=1,
                               bias=False)

        self.drop_rate = float(drop_rate)

    def bn_function(self, inputs: List[Tensor]) -> Tensor:  # bottleneck结构
        """

        :param inputs: 输入特征
        :return: 返回bottleneck处理过后的结果
        """
        concat = torch.cat(inputs, dim=1)
        bottleneck_out = self.conv1(self.relu1(self.norm1(concat)))
        return bottleneck_out

    def forward(self, x: Tensor) -> Tensor:
        """

        :param x:
        :return: 前向传播后的结果
        """
        bottleneck = self.bn_function(x)
        x_out = self.conv2(self.relu2(self.norm2(bottleneck)))
        if self.drop_rate > 0:
            x_out = F.dropout(x_out, p=self.drop_rate, training=self.training)
        return x_out


class DenseBlock(nn.ModuleDict):
    """
    DenseBlock
    """

    def __init__(self, num_layer: int, input_channel: int, bn_size: int, growth_rate: int, drop_rate: float) -> None:
        """

        :param num_layer: 密集块中的密集层的数量
        :param input_channel: 输入通道数
        :param bn_size: 批数量
        :param growth_rate: 增长率
        :param drop_rate: 遗忘率
        """
        super().__init__()
        for i in range(num_layer):  # 每个块有若干个DenseLayer,这里是添加DenseLayer数量
            layer = DenseLayer(input_channels=input_channel + i * growth_rate,
                               bn_size=bn_size,
                               growth_rate=growth_rate,
                               drop_rate=drop_rate
                               )
            self.add_module('denselayer%d' % (i + 1), layer)  # 添加DenseLayer

    def forward(self, x):
        features = [x]
        for name, layer in self.items():
            new_feature = layer(features)
            features.append(new_feature)
        return torch.cat(features, dim=1)


class Transition(nn.Sequential):
    """
    Transition
    """

    def __init__(self, input_channel: int, output_channel: int) -> None:
        """

        :param input_channel: 输入通道数
        :param output_channel: 输出通道数
        """
        super().__init__()
        self.norm = nn.BatchNorm2d(input_channel)
        self.relu = nn.ReLU(inplace=True)
        self.conv = nn.Conv2d(input_channel, output_channel, kernel_size=1, stride=1, bias=False)
        self.pool = nn.AvgPool2d(kernel_size=2, stride=2)


class DenseNet(nn.Module):
    """
    DenseNet
    """

    def __init__(self, growth_rate=32, block_config=(6, 12, 24, 16), num_init_channel=64, bn_size=4, drop_rate=0,
                 num_classes=1000):
        super().__init__()
        self.features = nn.Sequential(
            OrderedDict([
            ('conv0',
             nn.Conv2d(in_channels=3, out_channels=num_init_channel, kernel_size=7, stride=2, padding=3, bias=False)),
            ('norm0', nn.BatchNorm2d(num_init_channel)),
            ('relu0', nn.ReLU(inplace=True)),
            ('pool0', nn.MaxPool2d(kernel_size=3, stride=2, padding=1))
        ])
        )

        num_features = num_init_channel
        for i, num_layer in enumerate(block_config):
            denseblock = DenseBlock(
                num_layer=num_layer,
                input_channel=num_features,
                bn_size=bn_size,
                growth_rate=growth_rate,
                drop_rate=drop_rate
            )
            self.features.add_module('denseblock%d' % (i + 1), denseblock)
            num_features = num_features + num_layer * growth_rate
            if i != len(block_config) - 1:
                trans = Transition(input_channel=num_features, output_channel=num_features // 2)
                self.features.add_module('trans%d' % (i + 1), trans)
                num_features = num_features // 2

        self.features.add_module('norm5', nn.BatchNorm2d(num_features))

        self.classifier = nn.Linear(num_features, num_classes)

        # Official init from torch repo.
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.constant_(m.bias, 0)
    def forward(self, x):
        feature = self.features(x)
        out = F.relu(feature, inplace=True)
        out = F.adaptive_avg_pool2d(out, output_size=(1, 1))
        out = torch.flatten(out, 1)
        out = self.classifier(out)
        return out
