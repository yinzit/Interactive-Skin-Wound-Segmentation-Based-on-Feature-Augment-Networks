import torch
import torch.nn as nn
from torchvision import models
import torch.nn.functional as F
from functools import partial
from torch.nn import init

nonlinearity = partial(F.elu, inplace=True)  # inplace=True：在pytorch中是指改变一个tensor的值的时候，不经过复制操作，而是直接在原来的内存上改变它的值，节省内（显）存


def init_weight(model):
    for m in model.modules():
        if isinstance(m, nn.Conv2d):
            torch.nn.init.kaiming_normal_(m.weight)
        elif isinstance(m, nn.BatchNorm2d):
            m.weight.data.fill_(1)
            m.bias.data.zero_()


def conv3x3(in_planes, out_planes, stride=1):
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride, padding=1, bias=False)


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(BasicBlock, self).__init__()
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = nn.BatchNorm2d(planes)
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
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out


# #######################################[ MultiConv ]########################################

# MultiConv(1, [256, 256, 256, 512, 512, 512], [3, 3, 3, 3, 3, 3], [2, 1, 1, 2, 1, 2])
class MultiConv(nn.Module):
    def __init__(self, in_ch, channels, kernel_sizes=None, strides=None, dilations=None, paddings=None, BatchNorm=nn.BatchNorm2d):
        super(MultiConv, self).__init__()
        self.num = len(channels)
        if kernel_sizes is None:
            kernel_sizes = [3 for c in channels]
        if strides is None:
            strides = [1 for c in channels]
        if dilations is None:
            dilations = [1 for c in channels]
        if paddings is None:
            paddings = [((kernel_sizes[i]//2) if dilations[i] == 1 else (kernel_sizes[i]//2 * dilations[i])) for i in range(self.num)]
        convs_tmp = []
        for i in range(self.num):
            if channels[i] == 1:
                convs_tmp.append(nn.Conv2d(in_ch if i == 0 else channels[i-1], channels[i], kernel_size=kernel_sizes[i], stride=strides[i], padding=paddings[i], dilation=dilations[i]))
            else:
                convs_tmp.append(nn.Sequential(nn.Conv2d(in_ch if i == 0 else channels[i-1], channels[i], kernel_size=kernel_sizes[i], stride=strides[i], padding=paddings[i], dilation=dilations[i],bias=False), BatchNorm(channels[i]), nn.ReLU()))
        self.convs = nn.Sequential(*convs_tmp)
        init_weight(self)

    def forward(self, x):

        return self.convs(x)


class DecoderBlock(nn.Module):
    def __init__(self, in_channels, n_filters):
        super(DecoderBlock, self).__init__()

        self.conv1 = nn.Conv2d(in_channels, in_channels // 4, 1)
        self.norm1 = nn.BatchNorm2d(in_channels // 4)
        self.relu1 = nonlinearity

        # nn.ConvTranspose2d : output = (input - 1) * s + out_padding - 2 * padding + kernel
        self.deconv2 = nn.ConvTranspose2d(in_channels // 4, in_channels // 4,
                                          3, stride=2, padding=1, output_padding=1)  # 尺寸翻倍
        # 转置卷积可以学习自适应映射以恢复具有更详细信息的特征。因此，作者采用转置卷积来恢复解码器中的更高分辨率特征
        self.norm2 = nn.BatchNorm2d(in_channels // 4)
        self.relu2 = nonlinearity

        self.conv3 = nn.Conv2d(in_channels // 4, n_filters, 1)
        self.norm3 = nn.BatchNorm2d(n_filters)
        self.relu3 = nonlinearity

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_uniform_(m.weight.data)
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm2d):
                init.normal_(m.weight.data, 1.0, 0.02)
                init.constant_(m.bias.data, 0.0)

    def forward(self, x):
        x = self.conv1(x)
        x = self.norm1(x)
        x = self.relu1(x)
        x = self.deconv2(x)
        x = self.norm2(x)
        x = self.relu2(x)
        x = self.conv3(x)
        x = self.norm3(x)
        x = self.relu3(x)

        return x


class CE_Net(nn.Module):
    def __init__(self, num_classes=1):  # 输出1通道
        super(CE_Net, self).__init__()

        filters = [64, 128, 256, 512]
        resnet = models.resnet34(pretrained=True)  # 预训练模型

        # self.conv = nn.Conv2d(3, 3, 1)  # in_channel,out_channel,kernel  1*1卷积
        self.conv1 = nn.Conv2d(6, 8, 1)
        self.conv2 = nn.Conv2d(8, 3, 1)

        self.firstconv = resnet.conv1  # self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.firstbn = resnet.bn1
        self.firstrelu = resnet.relu
        self.firstmaxpool = resnet.maxpool  # self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        self.encoder1 = resnet.layer1  # self.layer1 = self._make_layer(block, 64, layers[0])
        self.encoder2 = resnet.layer2  # self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.encoder3 = resnet.layer3  # self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        self.encoder4 = resnet.layer4  # self.layer4 = self._make_layer(block, 512, layers[3], stride=2)

        self.decoder4 = DecoderBlock(768, filters[2])  # 512
        self.decoder3 = DecoderBlock(filters[2], filters[1])
        self.decoder2 = DecoderBlock(filters[1], filters[0])
        self.decoder1 = DecoderBlock(filters[0], filters[0])

        self.finaldeconv1 = nn.ConvTranspose2d(filters[0], 32, 4, 2, 1)
        self.finalrelu1 = nonlinearity
        self.finalconv2 = nn.Conv2d(33, 33, 3, padding=1)
        self.finalrelu2 = nonlinearity
        self.finalconv3 = nn.Conv2d(33, num_classes, 3, padding=1)

        self.avgpool = nn.AvgPool2d(kernel_size=3, stride=2, padding=1)
        self.edgedeconv = nn.ConvTranspose2d(1, 1, 4, 2, 1)
        self.edgebn = nn.BatchNorm2d(1)
        self.edgerelu = nonlinearity
        self.edgeconv = nn.Conv2d(2, 1, kernel_size=3, stride=1, padding=1, bias=True)

        # module
        self.spatialrelation = SpatialRelation()
        self.res1 = BasicBlock(64, 64, stride=1, downsample=None)  #
        self.res2 = BasicBlock(64, 64, stride=1, downsample=None)  #
        self.res3 = BasicBlock(32, 32, stride=1, downsample=None)  #
        self.res4 = BasicBlock(16, 16, stride=1, downsample=None)

        self.conv1x1_1 = nn.Conv2d(64, 64, 1)
        self.conv1x1_2 = nn.Conv2d(64, 1, 1)  # 64, 1, 1
        self.conv1x1_3 = nn.Conv2d(128, 1, 1)  # 128, 1, 1
        self.conv1x1_4 = nn.Conv2d(256, 1, 1)  # 256, 1, 1
        self.conv1x1_5 = nn.Conv2d(512, 1, 1)  # 512, 1, 1

        self.d1 = nn.Conv2d(64, 64, 1)
        self.d2 = nn.Conv2d(64, 32, 1)
        self.d3 = nn.Conv2d(32, 16, 1)
        self.d4 = nn.Conv2d(16, 8, 1)

        self.sa = SpatialAttention()
        self.ca1 = ChannelAttention(32, 1)  # 32, 1
        self.ca2 = ChannelAttention(16, 1)  # 16, 1
        self.ca3 = ChannelAttention(8, 1)  # 8, 1

        self.fuse = nn.Conv2d(8, 1, kernel_size=1, padding=0, bias=False)  #

        # interactive
        self.first_conv = MultiConv(1, [256, 256, 256, 512, 512, 512], [3, 3, 3, 3, 3, 3], [2, 1, 1, 2, 1, 2])
        self.my_aspp = MyASPP(in_ch=1280, out_ch=512, dilations=[1, 6, 12, 18], BatchNorm=nn.BatchNorm2d, if_global=True)
        # self.interactive = nn.Conv2d(1024, 512, 1)

    def forward(self, x):
        # Encoder：resnet34
        # x = self.conv(x)  # 3*512*512
        x = self.conv1(x)
        x = self.conv2(x)

        e0 = self.firstconv(x)  # 64*256*256
        e0 = self.firstbn(e0)
        e0 = self.firstrelu(e0)

        e1 = self.firstmaxpool(e0)  # 64*128*128
        e1 = self.encoder1(e1)  # 64*128*128

        e2 = self.encoder2(e1)  # 128*64*64
        e3 = self.encoder3(e2)  # 256*32*32
        e4 = self.encoder4(e3)  # 512*16*16


        ################################################
        # Shape
        u1 = F.interpolate(self.conv1x1_2(e1), size=(512, 512), mode='bilinear', align_corners=True)
        u2 = F.interpolate(self.conv1x1_3(e2), size=(512, 512), mode='bilinear', align_corners=True)
        u3 = F.interpolate(self.conv1x1_4(e3), size=(512, 512), mode='bilinear', align_corners=True)
        u4 = F.interpolate(self.conv1x1_5(e4), size=(512, 512), mode='bilinear', align_corners=True)

        cs = self.res1(self.conv1x1_1(e0))  # 64*256*256
        cs = F.interpolate(cs, size=(512, 512), mode='bilinear', align_corners=True)  # 64*512*512
        cs = self.d1(cs)  # 64*256*256
        cs = self.sa(cs, u1)  # 64*256*256  # 更好地聚焦于有效的低层特征，并获得清晰的显著边界 对于低层特征，我们没有使用通道注意，因为低层特征的不同通道之间几乎没有语义差异

        cs = self.res2(cs)  # 64*256*256
        cs = self.d2(cs)  # 32*256*256
        cs = self.sa(cs, u2)
        # cs = self.ca1(cs, u2)  # 32*256*256  # 对显著性检测起重要作用的通道赋予较大的权重 不将空间注意力用于高级特征，因为高级特征包含较高的抽象语义

        cs = self.res3(cs)  # 32*256*256
        cs = self.d3(cs)  # 16*256*256
        cs = self.ca2(cs, u3)  # 16*256*256

        cs = self.res4(cs)  # 16*256*256
        cs = self.d4(cs)  # 8*256*256
        cs = self.ca3(cs, u4)  # 8*256*256

        # *********
        edge_out = F.interpolate(cs, size=(512, 512), mode='bilinear', align_corners=True)
        edge_out = self.fuse(edge_out)  # 1*512*512
        #####################################################

        # interactive
        # d = F.interpolate(d, size=(128, 128), mode='bilinear', align_corners=True)
        # d = torch.cat((d, e1), dim=1)
        # l1_first = F.interpolate(d, size=(16, 16), mode='bilinear', align_corners=True)
        # l1_first = self.first_conv(d)  # 512*16*16

        # middle
        e4 = self.spatialrelation(e4)
        # e4 = torch.cat((l1_first, e4), dim=1)

        # Decoder
        d4 = self.decoder4(e4) + e3  # 256*32*32  ‘+’点对点相加而非拼接
        d3 = self.decoder3(d4) + e2  # 128*64*64
        d2 = self.decoder2(d3) + e1  # 64*128*128
        d1 = self.decoder1(d2)  # 64*256*256

        out = self.finaldeconv1(d1)  # 32*512*512
        out = self.finalrelu1(out)

        out = torch.cat((out, edge_out), 1)  # 33*512*512

        out = self.finalconv2(out)  # 32*512*512
        out = self.finalrelu2(out)
        img_out = self.finalconv3(out)  # 1*512*512

        return img_out, edge_out
        # return img_out

