"""
Implementation of the Cellpose model architecture, as described in the paper
"Cellpose: a generalist algorithm for cellular segmentation",
which can be found via the Cellpose github repository: https://github.com/MouseLand/cellpose
"""

import torch
from torch import nn

# Standard Cellpose class loss
class ClassLoss:
    def __init__(self, class_loss):
        self.loss = class_loss

    def __call__(self, g, y):
        class_pred = g[:, 0]
        class_y = y[:, 0]
        class_loss = self.loss(class_pred, class_y)
        return class_loss


# Standard Cellpose flow loss
class FlowLoss:
    def __init__(self, flow_loss):
        self.loss = flow_loss

    def __call__(self, g, y):
        flow_pred = g[:, 1:]
        flow_y = 5. * y[:, 1:]
        flow_loss = self.loss(flow_pred, flow_y)
        return flow_loss

def conv_block(in_feat, out_feat):
    return nn.Sequential(
        nn.BatchNorm2d(num_features=in_feat),
        nn.ReLU(),
        nn.Conv2d(in_channels=in_feat, out_channels=out_feat, kernel_size=3, padding=(1, 1))
    )


class DownBlock(nn.Module):

    def __init__(self, in_features, out_features, pool=True):
        super().__init__()
        self.pool = pool
        if self.pool:
            self.max_pool = nn.MaxPool2d(kernel_size=2)
        self.c_layer1 = conv_block(in_features, out_features)
        self.c_layer2 = conv_block(out_features, out_features)
        self.res_conv = nn.Conv2d(in_channels=in_features, out_channels=out_features, kernel_size=1)
        self.c_layer3 = conv_block(out_features, out_features)
        self.c_layer4 = conv_block(out_features, out_features)

    def forward(self, x):
        if self.pool:
            x = self.max_pool(x)
        x_1 = self.c_layer1(x)
        x_1 = self.c_layer2(x_1)
        residual = self.res_conv(x)
        x_1 += residual
        x_2 = self.c_layer3(x_1)
        x_2 = self.c_layer4(x_2)
        x_2 += x_1
        return x_2


class UpBlock(nn.Module):

    def __init__(self, in_features, upsample=True):
        super().__init__()
        self.upsample = upsample
        if self.upsample:
            self.upsample_image = nn.Upsample(scale_factor=2)
            self.out_features = in_features // 2
        else:
            self.out_features = in_features
        self.proj_style1 = nn.Linear(256, self.out_features)
        self.proj_style2 = nn.Linear(256, self.out_features)
        self.proj_style3 = nn.Linear(256, self.out_features)
        self.c_layer1 = conv_block(in_features, self.out_features)
        self.c_layer2 = conv_block(self.out_features, self.out_features)
        self.res_conv = nn.Conv2d(in_channels=in_features, out_channels=self.out_features, kernel_size=1)
        self.c_layer3 = conv_block(self.out_features, self.out_features)
        self.c_layer4 = conv_block(self.out_features, self.out_features)

    def forward(self, z, fm, style):
        if self.upsample:
            z = self.upsample_image(z)
        z_1 = self.c_layer1(z)
        z_1 += fm
        z_1 = self.c_layer2(z_1)
        style1 = self.proj_style1(style).view(style.shape[0], self.out_features, 1, 1)
        z_1 += style1
        z_1 += self.res_conv(z)
        z_2 = self.c_layer3(z_1)
        style2 = self.proj_style2(style).view(style.shape[0], self.out_features, 1, 1)
        z_2 += style2
        z_2 = self.c_layer4(z_2)
        style3 = self.proj_style3(style).view(style.shape[0], self.out_features, 1, 1)
        z_2 += style3
        z_2 += z_1
        return z_2


class CellTranspose(nn.Module):
    def __init__(self, channels, device='cuda'):
        super().__init__()
        self.device = device  # TODO: Removing this should be fine
        self.d_block1 = DownBlock(channels, 32, pool=False)
        self.d_block2 = DownBlock(32, 64)
        self.d_block3 = DownBlock(64, 128)
        self.d_block4 = DownBlock(128, 256)

        self.u_block4 = UpBlock(256, upsample=False)
        self.u_block3 = UpBlock(256)
        self.u_block2 = UpBlock(128)
        self.u_block1 = UpBlock(64)

        self.out_block = nn.Sequential(
            nn.BatchNorm2d(num_features=32),
            nn.ReLU(),
            nn.Conv2d(in_channels=32, out_channels=3, kernel_size=1)
        )

    def forward(self, x, style_only=False):
        fm1 = self.d_block1(x)
        fm2 = self.d_block2(fm1)
        fm3 = self.d_block3(fm2)
        fm4 = self.d_block4(fm3)

        im_style = torch.sum(fm4, dim=(2, 3)).data
        im_style = torch.div(im_style, torch.norm(im_style)).data

        if style_only:
            return im_style
        else:
            z = self.u_block4(fm4, fm4, im_style)
            z = self.u_block3(z, fm3, im_style)
            z = self.u_block2(z, fm2, im_style)
            z = self.u_block1(z, fm1, im_style)
            y = self.out_block(z)
            return y


class SizeModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear1 = nn.Linear(256, 512)
        self.linear2 = nn.Linear(512, 1)

    def forward(self, x):
        x = self.linear1(x)
        return self.linear2(x)
