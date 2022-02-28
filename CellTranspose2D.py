"""
Implementation of the Cellpose model architecture, as described in the paper
"Cellpose: a generalist algorithm for cellular segmentation",
which can be found via the Cellpose github repository: https://github.com/MouseLand/cellpose
"""

import torch
from torch import nn
from numpy import pi

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
class SASMaskLoss:
    def __init__(self, sas_class_loss):
        self.class_loss = sas_class_loss

    def __call__(self, g_source, lbl_source, g_target, lbl_target, margin=1, gamma_1=0.001, gamma_2=0.5):
        frgd_mask = torch.logical_and(lbl_source, lbl_target)
        frgd_count = torch.count_nonzero(frgd_mask)
        bkgd_mask = torch.logical_and(torch.logical_not(lbl_source), torch.logical_not(lbl_target))
        bkgd_count = torch.count_nonzero(bkgd_mask)
        s_not_t_mask = torch.logical_and(lbl_source, torch.logical_not(lbl_target))
        snt_count = torch.count_nonzero(s_not_t_mask)
        t_not_s_mask = torch.logical_and(torch.logical_not(lbl_source), lbl_target)
        tns_count = torch.count_nonzero(t_not_s_mask)
        px_count = lbl_target.shape[-1] * lbl_target.shape[-2]

        sa_loss = gamma_1 * 0.5 * torch.square(g_source - g_target)
        sa_loss = gamma_2 * (frgd_mask * sa_loss)*((px_count)/(frgd_count+1)) + (bkgd_mask * sa_loss)*((px_count)/(bkgd_count+1))
        s_loss = gamma_1 * 0.5 * torch.square(torch.max(torch.tensor(0).to(torch.device('cuda' if torch.cuda.is_available() else 'cpu')), margin - torch.abs(g_source - g_target)))
        s_loss = (1-gamma_2) * (s_not_t_mask * s_loss)*((px_count)/(snt_count+1)) + (t_not_s_mask * s_loss)*((px_count)/(tns_count+1))

        # source_class_loss = (1-gamma_1) * self.class_loss(g_source, lbl_source)
        # target_class_loss = (1-gamma_1) * self.class_loss(g_target, lbl_target)

        adaptive_class_loss = torch.mean(sa_loss + s_loss)
        return adaptive_class_loss


class ContrastiveFlowLoss:
    def __init__(self, c_flow_loss):
        self.flow_loss = c_flow_loss
        return

    def __call__(self, z_source, lbl_source, z_target, lbl_target, k=10, lmbda=1e-1, n_thresh=0.05, temperature=0.1):

        # Normalize labels and outputs
        z_source = torch.div(z_source, torch.linalg.norm(z_source, dim=1)[:, None, :])
        z_target = torch.div(z_target, torch.linalg.norm(z_target, dim=1)[:, None, :])
        flow_source = lbl_source[:, 1:]
        mask_target = lbl_target[:, 0]
        flow_target = lbl_target[:, 1:]
        flow_target_norm = torch.div(flow_target, torch.linalg.norm(flow_target, dim=1)[:, None, :])
        flow_target_norm[flow_target_norm != flow_target_norm] = 0.0
        p_size = flow_target_norm.shape[-1]

        # similarity between target label and source output
        lbl_match = torch.matmul(torch.transpose(torch.flatten(flow_target_norm, 2, -1), 1, 2),
                                 torch.flatten(z_source.detach(), 2, -1)).reshape(-1, p_size, p_size, p_size, p_size)

        # position of highest similarity source label vector for each target label vector
        # p_v, p_i = torch.max(lbl_match.view(-1, 112, 112, 112 * 112), dim=-1)
        p_i = torch.argmax(lbl_match.view(-1, p_size, p_size, p_size * p_size), dim=-1)
        # p_i_new = torch.cat((torch.div(p_i[None, :], 112, rounding_mode='floor'), p_i[None, :] % 112))
        # Match target output with source output vectors
        pos = torch.zeros(z_source.shape).to('cuda')
        for b in range(pos.shape[0]):
            for c in range(pos.shape[1]):
                pos[b][c] = torch.take(z_source[b][c], p_i[b])
        # Numerator: temperature-modified exponent of the normalized dot product
        # of the target output and selected positive sample for each pixel
        p_sim = torch.sum(torch.mul(z_target, pos), dim=1)
        num = torch.exp(p_sim / temperature)
        # num = torch.where(mask_target == 0, num, torch.tensor(0.0).to('cuda'))
        # num = torch.mul(num, mask_target)

        # Denominator: similar to numerator, but summed across each target output
        # pixel to ALL sample pixels in source output
        icos = torch.acos(p_sim)
        # thresh = torch.cos(icos + (pi / 18)).unsqueeze(-1)
        thresh = torch.cos(icos + n_thresh).unsqueeze(-1)
        z_match = torch.matmul(torch.transpose(torch.flatten(pos, 2, -1), 1, 2),
                               torch.flatten(z_source, 2, -1)).view(-1, p_size, p_size, p_size * p_size)
        z_match = torch.where(z_match < thresh, z_match, torch.tensor(0.0).to('cuda'))
        top_n = torch.topk(z_match, k, dim=-1).values

        # z_match = torch.matmul(torch.transpose(torch.flatten(z_target, 2, -1), 1, 2),
        #                          torch.flatten(z_source, 2, -1)).view(-1, 112, 112, 112 * 112)
        # # z_match = torch.sigmoid((z_match - hardness_thresh) * 100)
        # z_match = torch.where(z_match < hardness_thresh, z_match, torch.tensor(0.0).to('cuda'))
        # top_n = torch.topk(z_match, k, dim=-1).values
        # z_match = torch.where(z_match < 0.9, z_match, torch.tensor(0.0).to('cuda'))
        # z_match = torch.where(z_match > hardness_thresh and z_match < 0.9, z_match, torch.tensor(0.0).to('cuda'))
        # den = torch.sum(torch.exp(top_n / temperature), dim=-1)

        top_n = torch.exp(top_n / temperature)
        den = torch.cat((torch.unsqueeze(num, -1), top_n), dim=-1)
        den = torch.sum(den, dim=-1)
        # NOTE: CONCAT NUM TO DEN IF NOT INCLUDED
        # den = torch.mul(den, mask_target)
        adaptive_flow_loss = torch.div(num, den)
        # adaptive_flow_loss = -torch.log(adaptive_flow_loss)
        adaptive_flow_loss = torch.where(mask_target == 1, -torch.log(adaptive_flow_loss), torch.tensor(0.0).detach().to('cuda'))

        # source_flow_loss = self.flow_loss(z_source, flow_source)
        # target_flow_loss = self.flow_loss(z_target, flow_target)

        # adaptive_flow_loss = lmbda * torch.mean(adaptive_flow_loss) + (1 - lmbda) * (source_flow_loss + target_flow_loss)
        adaptive_flow_loss = lmbda * torch.mean(adaptive_flow_loss)
        return adaptive_flow_loss


def conv_block(in_feat, out_feat):
    return nn.Sequential(
        nn.BatchNorm2d(num_features=in_feat),
        nn.ReLU(),
        nn.Conv2d(in_channels=in_feat, out_channels=out_feat, kernel_size=3, padding=(1, 1))
    )


class DownBlock(nn.Module):

    def __init__(self, in_features: int, out_features: int, pool: bool = True):
        super().__init__()
        self.pool = pool
        # if self.pool:
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

    def __init__(self, in_features: int, upsample: bool = True):
        super().__init__()
        self.upsample = upsample
        self.upsample_image = nn.Upsample(scale_factor=2)  # mode='nearest'
        if self.upsample:
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
    def __init__(self, channels: int, device='cuda'):
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

    def forward(self, x, style_only: bool = False):
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
