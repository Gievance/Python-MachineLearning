# Writer：TuTTTTT
# 编写Time:2024/3/20 16:40

import torch
import torch.nn as nn
from torch import Tensor, Type, Union, List
from typing import Optional, Callable


def conv3x3(in_channel: int, out_channel: int, stride: int = 1, dilation: int = 1, group: int = 1) -> nn.Conv2d:
    """
    定义3x3卷积
    :param in_channels: 输入通道数
    :param out_channels: 输出通道数
    :param stride: 步长
    :param dilation: 扩张率（也就是padding）
    :param group: 分组卷积参数，=1相当于没有分组
    :return: 一个3x3卷积
    """
    return nn.Conv2d(in_channels=in_channel,
                     out_channels=out_channel,
                     kernel_size=3,
                     stride=stride,
                     padding=dilation,
                     groups=group,
                     bias=False,
                     dilation=dilation,
                     )


def conv1x1(in_channel: int, out_channel: int, stride: int = 1) -> nn.Conv2d:
    """

    :param in_channel:输入通道数
    :param out_channel:输出通道数
    :param stride:步长
    :return:返回一个1x1卷积
    """
    return nn.Conv2d(
        in_channels=in_channel,
        out_channels=out_channel,
        kernel_size=1,
        stride=stride,
        bias=False,
    )


class BasicBlock(nn.Module):
    """
    这是第一种连接方式组成的块，用于18层和34层，大于等于50层的用第二种
    """
    expansion: int = 1
    # expansion是BasicBlock和Bottleneck的核心区别之一

    def __init__(
            self,
            in_channel: int,
            out_channel: int,
            stride: int = 1,
            downsample: Optional[nn.Module] = None,
            groups: int = 1,
            base_width: int = 64,
            dilation: int = 1,
            norm_layer: Optional[Callable[..., nn.Module]] = None
    ) -> None:
        super(BasicBlock, self).__init__()

        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        if groups != 1 or base_width != 64:
            raise ValueError("BasicBlock only supports groups=1 and base_width=64")
        if dilation > 1:
            raise NotImplementedError("Dilation > 1 not supported in BasicBlock")

        self.conv1 = conv3x3(in_channel, out_channel, stride=stride)
        self.bn1 = norm_layer(out_channel)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(out_channel, out_channel)
        self.bn2 = norm_layer(out_channel)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x: Tensor) -> Tensor:
        iden = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        # 默认downsample=None，表示不做downsample;若x和out的通道数不一致，则要做一个downsample
        # 且downsample专门用来改变x的通道数，使x和output的通道数一致；
        # 且downsample实际是一个1x1卷积来降采样；
        if self.downsample is not None:
            iden = self.downsample(x)

        out += iden
        out = self.relu(out)

        return out


class BottleNeck(nn.Module):
    """
    第二种连接方式，这种参数少，适合层数超多的
    """
    expansion: int = 4
    # expansion是块输出通道数与输入通道数的比值，方便后续计算。在BasicBlock中，expansion=1,相当于输入维度和输出维度一致。
    def __init__(
            self,
            in_channel: int,
            out_channel: int,
            stride: int = 1,
            downsample: Optional[nn.Module] = None,
            groups: int = 1,
            base_width: int = 64,
            dilation: int = 1,
            norm_layer: Optional[Callable[..., nn.Module]] = None,
    ) -> None:

        super(BottleNeck, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        width = int(out_channel * (base_width / 64.0)) * groups

        self.conv1 = conv1x1(in_channel, width)
        self.bn1 = norm_layer(width)
        self.conv2 = conv3x3(width, width, stride, groups, dilation)
        self.bn2 = norm_layer(width)
        self.conv3 = conv1x1(width, out_channel * self.expansion)
        self.bn3 = norm_layer(out_channel * self.expansion)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x: Tensor) -> Tensor:
        iden = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            iden = self.downsample(x)

        out += iden
        out = self.relu(out)

        return out


class ResNet(nn.Module):

    def __init__(
            self,
            block,
            layer: List[int],
            num_classes: int = 1000,
            zero_init_residual: bool = False,
            groups: int = 1,
            width_per_group: int = 64,
            norm_layer: Optional[Callable[..., nn.Module]] = None,
    ) -> None:

        super().__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        self.norm_layer = norm_layer

        self.in_channel = 64
        self.dilation = 1
        self.groups = groups
        self.base_width = width_per_group

        self.conv1 = nn.Conv2d(3, self.in_channel,kernel_size=7, stride=2, padding=3, bias=False) # size/2
        self.bn1 = norm_layer(self.in_channel)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1) # size/2
        self.layer1 = self.make_layer(block, 64, layer[0]) # size不变
        self.layer2 = self.make_layer(block, 128, layer[1], stride=2) # size/2
        self.layer3 = self.make_layer(block, 256, layer[2], stride=2) # size/2
        self.layer4 = self.make_layer(block, 512, layer[3], stride=2) # size/2
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1)) # size=size-7+1  / 兼容图像输入大小问题
        self.fc = nn.Linear(512 * block.expansion, num_classes)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

        if zero_init_residual:
            for m in self.modules():
                if isinstance(m, BottleNeck) and m.bn3.weight is not None:
                    nn.init.constant_(m.bn3.weight, 0)  # type: ignore[arg-type]
                elif isinstance(m, BasicBlock) and m.bn2.weight is not None:
                    nn.init.constant_(m.bn2.weight, 0)  # type: ignore[arg-type]

    def make_layer(
            self,
            block,
            planes: int,
            blocks: int,
            stride: int = 1,
            dilate: bool = False
    ) -> nn.Sequential:
        """

        :param block: 指定BasicBlock|BottleNeck
        :param planes: 基准通道数不是输出通道数
        :param blocks: block的layer数量
        :param stride: 步长
        :param dilate:
        :return:
        """
        norm_layer = self.norm_layer
        downsample = None
        pre_dilation = self.dilation
        if dilate:
            self.dilation *= stride
            stride = 1
        # 步长不为1 or 输入通道数！=基准通道数*扩展率 =特征通道不一致，需要downsample
        if stride != 1 or self.in_channel != planes * block.expansion:
            downsample = nn.Sequential(
                conv1x1(self.in_channel, planes * block.expansion, stride), # 针对x
                norm_layer(planes * block.expansion)
            )

        layer = []

        # 带有步长，尺寸缩小
        layer.append(
            block(self.in_channel, planes, stride, downsample, self.groups, self.base_width, pre_dilation,
                  norm_layer)
        )
        self.in_channel = planes * block.expansion

        for _ in range(1, blocks):
            # 无步长
            layer.append(
                block(
                    self.in_channel,
                    planes,
                    groups=self.groups,
                    base_width=self.base_width,
                    dilation=self.dilation,
                    norm_layer=norm_layer
                )
            )
        return nn.Sequential(*layer)

    def _forward_impl(self, x: Tensor) -> Tensor:
        # See note [TorchScript super()]
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)

        return x

    def forward(self, x: Tensor) -> Tensor:
        return self._forward_impl(x)
