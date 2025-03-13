import torch
import torch.nn as nn
import torch.nn.functional as F

def add_conv(in_ch, out_ch, ksize, stride, leaky=True):
    """
    Add a conv2d / batchnorm / leaky ReLU block.
    Args:
        in_ch (int): number of input channels of the convolution layer.
        out_ch (int): number of output channels of the convolution layer.
        ksize (int): kernel size of the convolution layer.
        stride (int): stride of the convolution layer.
    Returns:
        stage (Sequential) : Sequential layers composing a convolution block.
    """
    stage = nn.Sequential()
    pad = (ksize - 1) // 2
    stage.add_module('conv', nn.Conv2d(in_channels=in_ch,
                                       out_channels=out_ch, kernel_size=ksize, stride=stride,
                                       padding=pad, bias=False))
    stage.add_module('batch_norm', nn.BatchNorm2d(out_ch))
    if leaky:
        stage.add_module('leaky', nn.LeakyReLU(0.1))
    else:
        stage.add_module('relu6', nn.ReLU6(inplace=True))
    return stage

# Conv: Conv2d+BN+ReLU
class BasicConv2d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0):
        super(BasicConv2d, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding)
        self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.relu(x)
        return x


# Stem:BasicConv2d+MaxPool2d
class Stem(nn.Module):
    def __init__(self, in_channels):
        super(Stem, self).__init__()

        # conv3x3(32 stride2 valid)
        self.conv1 = BasicConv2d(in_channels, 32, kernel_size=3, stride=2)
        # conv3*3(32 valid)
        self.conv2 = BasicConv2d(32, 32, kernel_size=3)
        # conv3*3(64)
        self.conv3 = BasicConv2d(32, 64, kernel_size=3, padding=1)

        # maxpool3*3(stride2 valid)
        self.maxpool4 = nn.MaxPool2d(kernel_size=3, stride=2)

        # conv1*1(80)
        self.conv5 = BasicConv2d(64, 80, kernel_size=1)
        # conv3*3(192 valid)
        self.conv6 = BasicConv2d(80, 192, kernel_size=3)

        # conv3*3(256 stride2 valid)
        self.conv7 = BasicConv2d(192, 256, kernel_size=3, stride=2)

    def forward(self, x):
        x = self.maxpool4(self.conv3(self.conv2(self.conv1(x))))
        x = self.conv7(self.conv6(self.conv5(x)))
        return x


# Inception_ResNet_A:BasicConv2d+MaxPool2d
class Inception_ResNet_A(nn.Module):
    def __init__(self, in_channels, ch1x1, ch3x3red, ch3x3, ch3x3redX2, ch3x3X2_1, ch3x3X2_2, ch1x1ext, scale=1.0):
        super(Inception_ResNet_A, self).__init__()

        self.scale = scale
        # conv1*1(32)
        self.branch_0 = BasicConv2d(in_channels, ch1x1, 1)
        # conv1*1(32)+conv3*3(32)
        self.branch_1 = nn.Sequential(
            BasicConv2d(in_channels, ch3x3red, 1),
            BasicConv2d(ch3x3red, ch3x3, 3, stride=1, padding=1)
        )
        # conv1*1(32)+conv3*3(32)+conv3*3(32)
        self.branch_2 = nn.Sequential(
            BasicConv2d(in_channels, ch3x3redX2, 1),
            BasicConv2d(ch3x3redX2, ch3x3X2_1, 3, stride=1, padding=1),
            BasicConv2d(ch3x3X2_1, ch3x3X2_2, 3, stride=1, padding=1)
        )
        # conv1*1(256)
        self.conv = BasicConv2d(ch1x1 + ch3x3 + ch3x3X2_2, ch1x1ext, 1)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        x0 = self.branch_0(x)
        x1 = self.branch_1(x)
        x2 = self.branch_2(x)

        x_res = torch.cat((x0, x1, x2), dim=1)
        x_res = self.conv(x_res)
        return self.relu(x + self.scale * x_res)


# Inception_ResNet_B:BasicConv2d+MaxPool2d
class Inception_ResNet_B(nn.Module):
    def __init__(self, in_channels, ch1x1, ch_red, ch_1, ch_2, ch1x1ext, scale=1.0):
        super(Inception_ResNet_B, self).__init__()

        self.scale = scale
        # conv1*1(128)
        self.branch_0 = BasicConv2d(in_channels, ch1x1, 1)
        # conv1*1(128)+conv1*7(128)+conv1*7(128)
        self.branch_1 = nn.Sequential(
            BasicConv2d(in_channels, ch_red, 1),
            BasicConv2d(ch_red, ch_1, (1, 7), stride=1, padding=(0, 3)),
            BasicConv2d(ch_1, ch_2, (7, 1), stride=1, padding=(3, 0))
        )
        # conv1*1(896)
        self.conv = BasicConv2d(ch1x1 + ch_2, ch1x1ext, 1)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        x0 = self.branch_0(x)
        x1 = self.branch_1(x)

        x_res = torch.cat((x0, x1), dim=1)
        x_res = self.conv(x_res)
        return self.relu(x + self.scale * x_res)


# Inception_ResNet_C:BasicConv2d+MaxPool2d
class Inception_ResNet_C(nn.Module):
    def __init__(self, in_channels, ch1x1, ch3x3redX2, ch3x3X2_1, ch3x3X2_2, ch1x1ext, scale=1.0, activation=True):
        super(Inception_ResNet_C, self).__init__()

        self.scale = scale

        self.activation = activation
        # conv1*1(192)
        self.branch_0 = BasicConv2d(in_channels, ch1x1, 1)
        # conv1*1(192)+conv1*3(192)+conv3*1(192)
        self.branch_1 = nn.Sequential(
            BasicConv2d(in_channels, ch3x3redX2, 1),
            BasicConv2d(ch3x3redX2, ch3x3X2_1, (1, 3), stride=1, padding=(0, 1)),
            BasicConv2d(ch3x3X2_1, ch3x3X2_2, (3, 1), stride=1, padding=(1, 0))
        )
        # conv1*1(1792)
        self.conv = BasicConv2d(ch1x1 + ch3x3X2_2, ch1x1ext, 1)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        x0 = self.branch_0(x)
        x1 = self.branch_1(x)

        x_res = torch.cat((x0, x1), dim=1)
        x_res = self.conv(x_res)
        if self.activation:
            return self.relu(x + self.scale * x_res)
        return x + self.scale * x_res


# redutionA:BasicConv2d+MaxPool2d
class redutionA(nn.Module):
    def __init__(self, in_channels, k, l, m, n):
        super(redutionA, self).__init__()
        # conv3*3(n stride2 valid)
        self.branch1 = nn.Sequential(
            BasicConv2d(in_channels, n, kernel_size=3, stride=2),
        )
        # conv1*1(k)+conv3*3(l)+conv3*3(m stride2 valid)
        self.branch2 = nn.Sequential(
            BasicConv2d(in_channels, k, kernel_size=1),
            BasicConv2d(k, l, kernel_size=3, padding=1),
            BasicConv2d(l, m, kernel_size=3, stride=2)
        )
        # maxpool3*3(stride2 valid)
        self.branch3 = nn.Sequential(nn.MaxPool2d(kernel_size=3, stride=2))


    def forward(self, x):
        branch1 = self.branch1(x)
        branch2 = self.branch2(x)
        branch3 = self.branch3(x)

        outputs = [branch1, branch2, branch3]
        return torch.cat(outputs, 1)

    
class GAM_Attention(nn.Module):
    def __init__(self, in_channels, out_channels, rate=16):
        super(GAM_Attention, self).__init__()

        self.channel_attention = nn.Sequential(
            nn.Linear(in_channels, int(in_channels / rate)),
            nn.ReLU(inplace=True),
            nn.Linear(int(in_channels / rate), in_channels)
        )

        self.spatial_attention = nn.Sequential(
            nn.Conv2d(in_channels, int(in_channels / rate), kernel_size=7, padding=3),
            nn.BatchNorm2d(int(in_channels / rate)),
            nn.ReLU(inplace=True),
            nn.Conv2d(int(in_channels / rate), out_channels, kernel_size=7, padding=3),
            nn.BatchNorm2d(out_channels)
        )

    def forward(self, x):
        b, c, h, w = x.shape
        x_permute = x.permute(0, 2, 3, 1).view(b, -1, c)
        x_att_permute = self.channel_attention(x_permute).view(b, h, w, c)
        x_channel_att = x_att_permute.permute(0, 3, 1, 2)

        x = x * x_channel_att

        x_spatial_att = self.spatial_attention(x).sigmoid()
        out = x * x_spatial_att

        return out
    
# redutionB:BasicConv2d+MaxPool2d
class redutionB(nn.Module):
    def __init__(self, in_channels, ch1x1, ch3x3_1, ch3x3_2, ch3x3_3, ch3x3_4):
        super(redutionB, self).__init__()
        # conv1*1(256)+conv3x3(384 stride2 valid)
        self.branch_0 = nn.Sequential(
            BasicConv2d(in_channels, ch1x1, 1),
            BasicConv2d(ch1x1, ch3x3_1, 3, stride=2, padding=0)
        )
        # conv1*1(256)+conv3x3(256 stride2 valid)
        self.branch_1 = nn.Sequential(
            BasicConv2d(in_channels, ch1x1, 1),
            BasicConv2d(ch1x1, ch3x3_2, 3, stride=2, padding=0),
        )
        # conv1*1(256)+conv3x3(256)+conv3x3(256 stride2 valid)
        self.branch_2 = nn.Sequential(
            BasicConv2d(in_channels, ch1x1, 1),
            BasicConv2d(ch1x1, ch3x3_3, 3, stride=1, padding=1),
            BasicConv2d(ch3x3_3, ch3x3_4, 3, stride=2, padding=0)
        )
        # maxpool3*3(stride2 valid)
        self.branch_3 = nn.MaxPool2d(3, stride=2, padding=0)

    def forward(self, x):
        x0 = self.branch_0(x)
        x1 = self.branch_1(x)
        x2 = self.branch_2(x)
        x3 = self.branch_3(x)
        return torch.cat((x0, x1, x2, x3), dim=1)
    

class Inception_ResNetv1(nn.Module):
    def __init__(self, num_classes=6, k=192, l=192, m=256, n=384):
        super(Inception_ResNetv1, self).__init__()
        blocks = []
        blocks.append(Stem(3))

        for i in range(5):
            blocks.append(Inception_ResNet_A(256, 32, 32, 32, 32, 32, 32, 256, 0.17))
        blocks.append(redutionA(256, k, l, m, n))

        for i in range(10):
            blocks.append(Inception_ResNet_B(896, 128, 128, 128, 128, 896, 0.10))
        blocks.append(redutionB(896, 256, 384, 256, 256, 256))

        # for i in range(4):
            # blocks.append(Inception_ResNet_C(1792, 192, 192, 192, 192, 1792, 0.20))
        # blocks.append(Inception_ResNet_C(1792, 192, 192, 192, 192, 1792, activation=False))
        blocks.append(GAM_Attention(1792, 1792))
        
        self.features = nn.Sequential(*blocks)
        self.conv = BasicConv2d(1792, 1536, 1)
        self.global_average_pooling = nn.AdaptiveAvgPool2d((1, 1))
        self.dropout = nn.Dropout(0.8)
        self.linear = nn.Linear(1536, num_classes)

    def forward(self, x):
        x = self.features(x)
        x = self.conv(x)
        x = self.global_average_pooling(x)
        x = x.view(x.size(0), -1)
        x = self.dropout(x)
        x = self.linear(x)
        return x

    def freeze_backbone(self):
        for param in self.features.parameters():
            param.requires_grad = False

    def Unfreeze_backbone(self):
        for param in self.features.parameters():
            param.requires_grad = True

def Maize-IRNet(pretrained=False, progress=True, num_classes=6):
    model = Maize-IRNet()
    if num_classes != 6:
        model.linear = nn.Linear(1536, num_classes)
    return model

