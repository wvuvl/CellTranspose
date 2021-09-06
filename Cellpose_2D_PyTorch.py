"""
Implementation of the Cellpose model architecture, as described in the paper
"Cellpose: a generalist algorithm for cellular segmentation",
which can be found via the Cellpose github repository: https://github.com/MouseLand/cellpose

Created by Matthew Keaton on 3/3/21
"""

import torch
from torch import nn


# Standard Cellpose class loss
def class_loss(lbl, y):
    class_pred = lbl[:, 0]
    class_y = y[:, 0]
    class_loss = nn.BCEWithLogitsLoss(reduction='mean')(class_y, class_pred)  #TODO: INITIALIZE LOSS TO MAKE MORE EFFICIENT
    return class_loss


# Standard Cellpose flow loss
def flow_loss(lbl, y):
    flow_pred = 5. * lbl[:, 1:]
    flow_y = 5. * y[:, 1:]
    flow_loss = nn.MSELoss(reduction='mean')(flow_y, flow_pred)  # TODO: INITIALIZE LOSS TO MAKE MORE EFFICIENT
    return flow_loss


# Semantic Alignment/Separation Contrastive Loss for classification
def sas_class_loss(g_source, lbl_source, g_target, lbl_target, margin=1, gamma=0.1):

    match_mask = torch.eq(lbl_source, lbl_target)  # Mask where each pixel is 1 (source GT = target GT) or 0 (source GT != target GT)
    st_dist = torch.linalg.norm(g_source - g_target) / g_source.data.nelement()

    sa_loss = (1 - gamma) * 0.5 * torch.square(st_dist)
    s_loss = (1 - gamma) * 0.5 * torch.square(torch.max(torch.tensor(0).to(torch.device('cuda' if torch.cuda.is_available() else 'cpu')), margin - st_dist))
    source_class_loss = nn.BCEWithLogitsLoss(reduction='mean')(g_source, lbl_source)
    # target_class_loss = nn.BCEWithLogitsLoss(reduction='mean')(g_target, lbl_target)

    loss = torch.mean(match_mask * sa_loss + (~match_mask * s_loss) + source_class_loss)
    # loss = torch.mean(match_mask * sa_loss + (~match_mask * s_loss) + target_class_loss)
    return loss


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
            self.upsample_image = nn.Upsample(scale_factor=2)  # mode='nearest'
            out_features = in_features // 2
        else:
            out_features = in_features
        self.proj_style = nn.Linear(256, out_features)
        self.c_layer1 = conv_block(in_features, out_features)
        self.c_layer2 = conv_block(out_features, out_features)
        self.res_conv = nn.Conv2d(in_channels=in_features, out_channels=out_features, kernel_size=1)
        self.c_layer3 = conv_block(out_features, out_features)
        self.c_layer4 = conv_block(out_features, out_features)

    def forward(self, z, fm, style):
        style = self.proj_style(style)
        style = style.view(style.shape[0], style.shape[1], 1, 1)
        if self.upsample:
            z = self.upsample_image(z)
        z_1 = self.c_layer1(z)
        z_1 += fm
        z_1 = self.c_layer2(z_1)
        z_1 += style
        z_1 += self.res_conv(z)
        z_2 = self.c_layer3(z_1)
        z_2 += style
        z_2 = self.c_layer4(z_2)
        z_2 += style
        z_2 += z_1
        return z_2


class UpdatedCellpose(nn.Module):
    def __init__(self, channels, device='cuda'):
        super().__init__()
        self.device = device
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
        # self.linear = nn.Linear(256, 1)

    def forward(self, x):
        x = self.linear1(x)
        return self.linear2(x)
        # return self.linear(x)


# if __name__ == '__main__()':
if __name__ == '__main__':
    from torchsummary import summary

    # db = DownBlock(3, 32, down_sample=1)
    # data = torch.zeros((8, 3, 96, 96))
    # zeros = torch.zeros((8, 32, 96, 96))
    # out = db(data)
    mc = UpdatedCellpose(3)
    summary(mc, (3, 168, 168))
    # data = torch.rand((8, 3, 8, 8))
    # out = mc(data)
    print('test')
    # ub = UpBlock(128, 64)
    # data = torch.rand((8, 128, 8, 8))
    # fm = torch.rand((8, 64, 16, 16))
    # style = torch.rand((8, 256))
    # out = ub(data, fm, style)

    # ub = UpBlock(256, 256, upsample=False)
    # data = torch.rand((8,256,4,4))
    # fm = torch.rand((8,256,4,4))
    # style = torch.rand((8,256))
    # out = ub(data, fm, style)
