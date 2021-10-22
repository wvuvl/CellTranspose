"""
Implementation of the Cellpose model architecture, as described in the paper
"Cellpose: a generalist algorithm for cellular segmentation",
which can be found via the Cellpose github repository: https://github.com/MouseLand/cellpose
"""

import torch
from torch import nn

import matplotlib.pyplot as plt


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
        # flow_pred = 5. * g[:, 1:]
        flow_pred = g[:, 1:]
        flow_y = 5. * y[:, 1:]
        flow_loss = self.loss(flow_pred, flow_y)
        return flow_loss


# Semantic Alignment/Separation Contrastive Loss for classification
class SASClassLoss:
    def __init__(self, sas_class_loss):
        self.class_loss = sas_class_loss

    def __call__(self, g_source, lbl_source, g_target, lbl_target, margin=1, gamma_1=0.2, gamma_2=0.5):
        # foreground_mask = lbl_source

        match_mask = torch.eq(lbl_source, lbl_target)  # Mask where each pixel is 1 (source GT = target GT) or 0 (source GT != target GT)
        # st_dist = torch.linalg.norm(g_source - g_target) / g_source.data.nelement()
        # dist = 0.5 * torch.square(g_source - g_target)

        frgd_mask = torch.logical_and(lbl_source, lbl_target)
        frgd_count = torch.count_nonzero(frgd_mask)
        bkgd_mask = torch.logical_and(torch.logical_not(lbl_source), torch.logical_not(lbl_target))
        bkgd_count = torch.count_nonzero(bkgd_mask)
        s_not_t_mask = torch.logical_and(lbl_source, torch.logical_not(lbl_target))
        snt_count = torch.count_nonzero(s_not_t_mask)
        t_not_s_mask = torch.logical_and(torch.logical_not(lbl_source), lbl_target)
        tns_count = torch.count_nonzero(t_not_s_mask)

        sa_loss = gamma_1 * 0.5 * torch.square(g_source - g_target)
        sa_loss = (frgd_mask * sa_loss)*((frgd_count + bkgd_count)/frgd_count) + (bkgd_mask * sa_loss)*((frgd_count + bkgd_count)/bkgd_count)
        s_loss = gamma_1 * 0.5 * torch.square(torch.max(torch.tensor(0).to(torch.device('cuda' if torch.cuda.is_available() else 'cpu')), margin - torch.abs(g_source - g_target)))
        # s_loss = (~match_mask * s_loss)/(snt_count + tns_count)
        s_loss = (s_not_t_mask * s_loss)*((snt_count + tns_count)/snt_count) + (t_not_s_mask * s_loss)*((snt_count + tns_count)/tns_count)

        # sa_loss = (1 - gamma_1) * 0.5 * torch.square(st_dist)
        # s_loss = (1 - gamma_1) * 0.5 * torch.square(torch.max(torch.tensor(0).to(torch.device('cuda' if torch.cuda.is_available() else 'cpu')), margin - st_dist))
        source_class_loss = (1 - gamma_1) * self.class_loss(g_source, lbl_source)
        # target_class_loss = gamma_1 * self.class_loss(g_target, lbl_target)

        # loss = torch.mean(match_mask * sa_loss + (~match_mask * s_loss) + source_class_loss)
        loss = torch.mean(sa_loss + s_loss) + source_class_loss
        # loss = torch.mean(match_mask * sa_loss + (~match_mask * s_loss)) + target_class_loss
        # print(loss)
        return loss


class ContrastiveFlowLoss:
    def __init__(self):
        # self.loss = c_flow_loss
        return

    def __call__(self, z_source, lbl_source, z_target, lbl_target, temperature=0.1):
        # z = torch.matmul(torch.transpose(torch.flatten(lbl_source, 2, -1), 1, 2),
        #                  torch.flatten(lbl_target, 2, -1)).view(-1, 112, 112, 112 * 112)
        # z_v, z_i = torch.max(z, dim=-1)

        return


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
        # self.linear = nn.Linear(256, 1)

    def forward(self, x):
        x = self.linear1(x)
        return self.linear2(x)


if __name__ == '__main__':
    from torchsummary import summary

    # db = DownBlock(3, 32, down_sample=1)
    # data = torch.zeros((8, 3, 96, 96))
    # zeros = torch.zeros((8, 32, 96, 96))
    # out = db(data)
    mc = CellTranspose(3)
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
