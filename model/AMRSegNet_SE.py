import torch
import torch.nn as nn
import pdb

def concat(x, y, **kwargs):
    return torch.cat((x, y), 1)


def RELUcons(relu, nchan):
    if relu:
        return nn.ReLU(inplace=True)
    else:
        return nn.PReLU(nchan)


class LUconv(nn.Module):
    def __init__(self, nchan, relu):
        super(LUconv, self).__init__()
        self.relu1 = RELUcons(relu, nchan)
        self.bn1 = nn.BatchNorm2d(nchan)
        self.conv1 = nn.Conv2d(nchan, nchan, kernel_size=3, padding=1)

    def forward(self, x):
        out = self.relu1(self.bn1(self.conv1(x)))

        return out


def _make_nConv(nchan, depth, relu):
    layers = []
    for _ in range(depth):
        layers.append(LUconv(nchan, relu))

    return nn.Sequential(*layers)

# Squeeze-Excitation Net  SENet
class SE_block(nn.Module):
    expansion = 1

    def __init__(self, planes, stride=1, downsample=None):
        super(SE_block, self).__init__()
        self.downsample = downsample
        self.stride = stride
        self.relu = nn.ReLU(inplace=True)

        if planes == 50:
            self.globalAvgPool = nn.AvgPool2d(256, stride=1)
        elif planes == 100:
            self.globalAvgPool = nn.AvgPool2d(128, stride=1)
        elif planes == 200:
            self.globalAvgPool = nn.AvgPool2d(64, stride=1)
        elif planes == 400:
            self.globalAvgPool = nn.AvgPool2d(32, stride=1)
        elif planes == 800:
            self.globalAvgPool = nn.AvgPool2d(16, stride=1)

        self.fc1 = nn.Linear(in_features=planes, out_features=round(planes / 8))
        self.fc2 = nn.Linear(in_features=round(planes / 8), out_features=planes)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        if self.downsample is not None:
            residual = self.downsample(x)
        original_x = x
        #pdb.set_trace()
        out = self.globalAvgPool(x)
        out = out.view(out.size(0), -1)
        out = self.fc1(out)
        out = self.relu(out)
        out = self.fc2(out)
        out = self.sigmoid(out)
        out = out.view(out.size(0), out.size(1), 1, 1)
        out = out * original_x

        # out += residual
        # out = self.relu(out)

        return out


# Block 1 ------ Input transition
class first_conv_block(nn.Module):
    def __init__(self, ch_in, ch_out, alpha, depth, relu=False):
        super(first_conv_block, self).__init__()
        self.alpha = alpha
        self.out_channels = ch_out
        self.relu1 = RELUcons(relu, ch_out)
        self.relu2 = RELUcons(relu, ch_out)
        self.conv = nn.Sequential(
            nn.Conv2d(ch_in, ch_out, kernel_size=3, stride=1, padding=1, bias=True),
            nn.BatchNorm2d(ch_out),
            nn.ReLU(inplace=True)
        )

        self.sequentialconv = _make_nConv(ch_out, depth, relu)
        self.SE_block1 = SE_block(planes=2*ch_out)
        self.SE_block2 = SE_block(planes=2*ch_out)
    # def __init__(self, ch_in, ch_out, relu=False):
    #     super(first_conv_block, self).__init__()
    #     self.out_channels = ch_out
    #     self.relu1 = RELUcons(relu, ch_out)
    #     self.relu2 = RELUcons(relu, ch_out)
    #     self.conv = nn.Sequential(
    #         nn.Conv2d(ch_in, ch_out, kernel_size=3, stride=1, padding=1, bias=True),
    #         nn.BatchNorm2d(ch_out),
    #         nn.ReLU(inplace=True)
    #     )

    def forward(self, x, y):
        out1 = self.conv(x)
        out2 = self.conv(y)
        x25 = x
        y25 = y

        for _ in range(self.out_channels - 1):
            x25 = torch.cat((x25, x), dim=1)
            y25 = torch.cat((y25, y), dim=1)

        out1 = self.sequentialconv(out1)
        out2 = self.sequentialconv(out2)

        # out1 = self.alpha * (self.relu1(torch.add(x25, out1)))
        # out2 = (1 - self.alpha) * (self.relu2(torch.add(y25, out2)))
        out1 = self.relu1(torch.add(x25, out1))
        out2 = self.relu2(torch.add(y25, out2))

        top = concat(out1, out2)
        bottom = concat(out2, out1)

        topout = self.SE_block1(top)
        bottomout = self.SE_block2(bottom)

        topout = torch.add(top, topout)
        bottomout = torch.add(bottom, bottomout)

        return topout, bottomout

# Block 2 ------Downsampling
class down_conv_block(nn.Module):
    def __init__(self, ch_in, ch_out, alpha, depth, relu=False):
        super(down_conv_block, self).__init__()
        self.alpha = alpha
        self.relu = RELUcons(relu, ch_out)
        self.conv = nn.Sequential(
            nn.Conv2d(ch_in, ch_out, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(ch_out),
            nn.ReLU(inplace=True),
            nn.Dropout2d()
        )
        self.sequentialconv = _make_nConv(ch_out, depth, relu)
        self.SE_block1 = SE_block(planes=4*ch_out)
        self.SE_block2 = SE_block(planes=4*ch_out)
    # def __init__(self, ch_in, ch_out, depth, relu=False):
    #     super(down_conv_block, self).__init__()
    #     self.relu = RELUcons(relu, ch_out)
    #     self.conv = nn.Sequential(
    #         nn.Conv2d(ch_in, ch_out, kernel_size=3, stride=1, padding=1),
    #         nn.BatchNorm2d(ch_out),
    #         nn.ReLU(inplace=True),
    #         nn.Dropout2d()
    #     )
    #     self.sequentialconv = _make_nConv(ch_out, depth, relu)

    def forward(self, x, y):
        out_x = self.conv(x)
        out_y = self.conv(y)
        out1 = self.sequentialconv(out_x)
        out2 = self.sequentialconv(out_y)

        # out1 = self.alpha * self.relu(torch.add(out1, out_x))
        # out2 = (1 - self.alpha) * self.relu(torch.add(out2, out_y))
        out1 = self.relu(torch.add(out1, out_x))
        out2 = self.relu(torch.add(out2, out_y))

        top = concat(out1, x)
        top = concat(top, out2)
        bottom = concat(out2, y)
        bottom = concat(bottom, out1)
        # pdb.set_trace()
        topout = self.SE_block1(top)
        bottomout = self.SE_block2(bottom)

        topout = torch.add(top, topout)
        bottomout = torch.add(bottom, bottomout)

        return topout, bottomout

# Block 3 ----- two modalities fusion as one-path feature map
class bottom_block(nn.Module):
    def __init__(self, ch_in, ch_out, alpha, depth, relu=False):
        super(bottom_block, self).__init__()
        self.alpha = alpha
        self.relu = RELUcons(relu, ch_out)
        self.conv = nn.Sequential(
            nn.Conv2d(ch_in, ch_out, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(ch_out),
            nn.ReLU(inplace=True),
            nn.Dropout2d()
        )
        self.sequentialconv = _make_nConv(ch_out, depth, relu)
        self.SE_block1 = SE_block(planes=4*ch_out)
    # def __init__(self, ch_in, ch_out, depth, relu=False):
    #     super(bottom_block, self).__init__()
    #     self.relu = RELUcons(relu, ch_out)
    #     self.conv = nn.Sequential(
    #         nn.Conv2d(ch_in, ch_out, kernel_size=3, stride=1, padding=1),
    #         nn.BatchNorm2d(ch_out),
    #         nn.ReLU(inplace=True),
    #         nn.Dropout2d()
    #     )
    #     self.sequentialconv = _make_nConv(ch_out, depth, relu)


    def forward(self, x, y):
        out_x = self.conv(x)
        out_y = self.conv(y)
        out1 = self.sequentialconv(out_x)
        out2 = self.sequentialconv(out_y)

        # out1 = self.alpha * self.relu(torch.add(out1, out_x))
        # out2 = (1 - self.alpha) * self.relu(torch.add(out2, out_y))
        out1 = self.relu(torch.add(out1, out_x))
        out2 = self.relu(torch.add(out2, out_y))

        out = concat(out1, x)
        out = concat(out, out2)
        # pdb.set_trace()
        out_tem = self.SE_block1(out)
        out = torch.add(out_tem, out)

        return out

# Block 4 ------ Upsampling
class up_conv_block(nn.Module):
    def __init__(self, ch_in, ch_out, depth, relu=False):
        super(up_conv_block, self).__init__()
        self.relu = RELUcons(relu, ch_out)
        self.up = nn.Sequential(
            nn.ConvTranspose2d(ch_in, ch_out//2, kernel_size=2, stride=2),
            nn.BatchNorm2d(ch_out//2),
            nn.ReLU(inplace=True),
            nn.Dropout2d()
        )
        self.sequentialconv = _make_nConv(ch_out, depth, relu)
        self.SE_block1 = SE_block(planes=4 * ch_out)

    def forward(self, x, skipx):
        out1 = self.up(x)
        x_cat = concat(out1, skipx)
        out2 = self.sequentialconv(x_cat)
        # out2 = self.SE_block1(out2)
        out = self.relu(torch.add(out2, x_cat))

        return out


# Block 5 -------- Output transition
class OutputTransition(nn.Module):
    def __init__(self, ch_in, n_classes, relu=False):
        super(OutputTransition, self).__init__()
        self.conv1 = nn.Conv2d(ch_in, n_classes, kernel_size=5, padding=2)
        #self.conv1 = nn.Conv2d(in_channels, n_classes, kernel_size=1, padding=0)
        #self.bn1 = ContBatchNorm2d(n_classes)
        self.bn1 = nn.BatchNorm2d(n_classes)
        self.relu1 = RELUcons(relu, n_classes)

    def forward(self, x):
        out = self.relu1(self.bn1(self.conv1(x)))
        #print('out.shape', out.shape)
        # make channels the last axis
        out = out.permute(0, 2, 3, 1).contiguous()
        # flatten
        #out = out.view(out.numel() // 2, 2)
        out = out.view((out.shape[0], out.numel()//out.shape[0]))

        return out



class mod2_HDUnet(nn.Module):
    def __init__(self, in_ch=1):
        super(mod2_HDUnet, self).__init__()
        self.Maxpool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.Maxpool2 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.Maxpool3 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.Maxpool4 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.input_tr = first_conv_block(ch_in=in_ch, ch_out=25, alpha=0.5, depth=5)
        self.down_tr100 = down_conv_block(ch_in=50, ch_out=25, alpha=0.5, depth=5)
        self.down_tr200 = down_conv_block(ch_in=100, ch_out=50, alpha=0.5, depth=5)
        self.down_tr400 = down_conv_block(ch_in=200, ch_out=100, alpha=0.5, depth=5)
        self.down_tr800 = bottom_block(ch_in=400, ch_out=200, alpha=0.5, depth=5)
        # self.input_tr = first_conv_block(ch_in=in_ch, ch_out=25)
        # self.down_tr100 = down_conv_block(ch_in=50, ch_out=25, depth=5)
        # self.down_tr200 = down_conv_block(ch_in=100, ch_out=50, depth=5)
        # self.down_tr400 = down_conv_block(ch_in=200, ch_out=100, depth=5)
        # self.down_tr800 = bottom_block(ch_in=400, ch_out=200, depth=5)

        self.up_tr800 = up_conv_block(ch_in=800, ch_out=800, depth=5)
        self.up_tr400 = up_conv_block(ch_in=800, ch_out=400, depth=5)
        self.up_tr200 = up_conv_block(ch_in=400, ch_out=200, depth=5)
        self.up_tr100 = up_conv_block(ch_in=200, ch_out=100, depth=5)

        self.out_tr = OutputTransition(ch_in=100, n_classes=1)

    def forward(self, x, y):
        # encoding path
        x1, y1 = self.input_tr(x, y)
        x2, y2 = self.down_tr100(self.Maxpool1(x1), self.Maxpool1(y1))
        x3, y3 = self.down_tr200(self.Maxpool2(x2), self.Maxpool2(y2))
        x4, y4 = self.down_tr400(self.Maxpool3(x3), self.Maxpool3(y3))
        out_bottom = self.down_tr800(self.Maxpool4(x4), self.Maxpool4(y4))

        # decoding path
        out800 = self.up_tr800(out_bottom, x4)
        out400 = self.up_tr400(out800, x3)
        out200 = self.up_tr200(out400, x2)
        out100 = self.up_tr100(out200, x1)

        out = self.out_tr(out100)
        out = torch.sigmoid(out)

        return out
