"""
Implementation of the Cellpose model architecture, as described in the paper
"Cellpose: a generalist algorithm for cellular segmentation",
which can be found via the Cellpose github repository: https://github.com/MouseLand/cellpose
"""

import torch
from torch import nn
import torch.nn.functional as F

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


# Semantic Alignment/Separation Contrastive Loss for classification
class SASMaskLoss:
    def __init__(self, sas_class_loss):
        self.class_loss = sas_class_loss

    def __call__(self, g_source, lbl_source, g_target, lbl_target, margin=10, gamma_1=0.1, lam=0.5):
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
        sa_loss = lam * (frgd_mask*sa_loss) * (px_count / (frgd_count + 1)) + \
                        (bkgd_mask*sa_loss) * (px_count/(bkgd_count+1))
        s_loss = gamma_1 * 0.5 * torch.square(torch.max(torch.tensor(0).to
                                                        (torch.device('cuda' if torch.cuda.is_available() else 'cpu')),
                                                        margin - torch.abs(g_source - g_target)))
        s_loss = (1 - lam) * (s_not_t_mask*s_loss) * (px_count / (snt_count + 1)) + \
                 (t_not_s_mask*s_loss) * (px_count/(tns_count+1))

        adaptive_class_loss = torch.mean(sa_loss + s_loss)
        return adaptive_class_loss


# Contrastive Loss for flow calculation
class ContrastiveFlowLoss:
    def __init__(self, c_flow_loss):
        self.flow_loss = c_flow_loss
        return

    def __call__(self, z_source, lbl_source, z_target, lbl_target, k=20, gamma_2=2, n_thresh=0.05, temperature=0.1):

        # Normalize labels and outputs
        z_source = torch.div(z_source, torch.linalg.norm(z_source, dim=1)[:, None, :])
        z_target = torch.div(z_target, torch.linalg.norm(z_target, dim=1)[:, None, :])
        mask_target = lbl_target[:, 0]
        flow_target = lbl_target[:, 1:]
        flow_target_norm = torch.div(flow_target, torch.linalg.norm(flow_target, dim=1)[:, None, :])
        flow_target_norm[flow_target_norm != flow_target_norm] = 0.0
        p_size = flow_target_norm.shape[-1]

        # similarity between target label and source output
        lbl_match = torch.matmul(torch.transpose(torch.flatten(flow_target_norm, 2, -1), 1, 2),
                                 torch.flatten(z_source.detach(), 2, -1)).reshape(-1, p_size, p_size, p_size, p_size)
        # position of the highest similarity source label vector for each target label vector
        p_i = torch.argmax(lbl_match.view(-1, p_size, p_size, p_size * p_size), dim=-1)
        # Match target output with source output vectors
        pos = torch.zeros(z_source.shape).to('cuda')
        for b in range(pos.shape[0]):
            for c in range(pos.shape[1]):
                pos[b][c] = torch.take(z_source[b][c], p_i[b])
        # Numerator: temperature-modified exponent of the normalized dot product
        # of the target output and selected positive sample for each pixel
        p_sim = torch.sum(torch.mul(z_target, pos), dim=1)
        num = torch.exp(p_sim / temperature)

        # Denominator: similar to numerator, but summed across each target output
        # pixel to top k sample pixels in source output, thresholded at some maximum similarity
        icos = torch.acos(p_sim)
        thresh = torch.cos(icos + n_thresh).unsqueeze(-1)
        z_match = torch.matmul(torch.transpose(torch.flatten(pos, 2, -1), 1, 2),
                               torch.flatten(z_source, 2, -1)).view(-1, p_size, p_size, p_size * p_size)
        z_match = torch.where(z_match < thresh, z_match, torch.tensor(0.0).to('cuda'))
        top_n = torch.topk(z_match, k, dim=-1).values
        top_n = torch.exp(top_n / temperature)
        den = torch.cat((torch.unsqueeze(num, -1), top_n), dim=-1)
        den = torch.sum(den, dim=-1)

        adaptive_flow_loss = torch.div(num, den)
        adaptive_flow_loss = torch.where(mask_target == 1, -torch.log(adaptive_flow_loss),
                                         torch.tensor(0.0).detach().to('cuda'))
        adaptive_flow_loss = gamma_2 * torch.mean(adaptive_flow_loss)
        return adaptive_flow_loss


# flow contrast self-supervised loss
class Flow_Contrast_Loss:
    def __init__(self, c_flow_loss):
        self.flow_loss = c_flow_loss
        return

    def __call__(self, z_source, lbl_source, z_target, lbl_target, k=20, gamma_2=2, n_thresh=0.05, temperature=0.1):

        # Normalize labels and outputs
        z_source = torch.div(z_source, torch.linalg.norm(z_source, dim=1)[:, None, :])
        flow_source = lbl_source[:, 1:]
        flow_source_norm = torch.div(flow_source, torch.linalg.norm(flow_source, dim=1)[:, None, :])
        flow_source_norm[flow_source_norm != flow_source_norm] = 0.0
        
        z_target = torch.div(z_target, torch.linalg.norm(z_target, dim=1)[:, None, :])
        mask_target = lbl_target[:, 0]
        flow_target = lbl_target[:, 1:]
        flow_target_norm = torch.div(flow_target, torch.linalg.norm(flow_target, dim=1)[:, None, :])
        flow_target_norm[flow_target_norm != flow_target_norm] = 0.0
        p_size = flow_target_norm.shape[-1]

        # similarity between target label and source label
        lbl_match = torch.matmul(torch.transpose(torch.flatten(flow_target_norm, 2, -1), 1, 2),
                                 torch.flatten(flow_source_norm, 2, -1)).reshape(-1, p_size, p_size, p_size, p_size)
        
        # position of the highest similarity source label vector for each target label vector
        p_i = torch.argmax(lbl_match.view(-1, p_size, p_size, p_size * p_size), dim=-1)
        
        # Match target output with source output vectors
        pos = torch.zeros(z_source.shape).to('cuda')
        for b in range(pos.shape[0]):
            for c in range(pos.shape[1]):
                pos[b][c] = torch.take(z_source[b][c], p_i[b])
                
        # Numerator: temperature-modified exponent of the normalized dot product
        # of the target output and selected positive sample for each pixel
        p_sim = torch.sum(torch.mul(z_target, pos), dim=1)
        num = torch.exp(p_sim / temperature)

        # Denominator: similar to numerator, but summed across each target output
        # pixel to top k sample pixels in source output, thresholded at some maximum similarity
        icos = torch.acos(p_sim)
        thresh = torch.cos(icos + n_thresh).unsqueeze(-1)
        z_match = torch.matmul(torch.transpose(torch.flatten(pos, 2, -1), 1, 2),
                               torch.flatten(z_source, 2, -1)).view(-1, p_size, p_size, p_size * p_size)
        z_match = torch.where(z_match < thresh, z_match, torch.tensor(0.0).to('cuda'))
        top_n = torch.topk(z_match, k, dim=-1).values
        top_n = torch.exp(top_n / temperature)
        den = torch.cat((torch.unsqueeze(num, -1), top_n), dim=-1)
        den = torch.sum(den, dim=-1)

        adaptive_flow_loss = torch.div(num, den)
        adaptive_flow_loss = torch.where(mask_target == 1, -torch.log(adaptive_flow_loss),
                                         torch.tensor(0.0).detach().to('cuda'))
        adaptive_flow_loss = gamma_2 * torch.mean(adaptive_flow_loss)
        return adaptive_flow_loss



class PixelContrastMorphologyLoss(nn.Module):
    def __init__(self, easy_contrast=True):
        super(PixelContrastMorphologyLoss, self).__init__()

        configer_contrast =  {
                            "proj_dim": 256,
                            "temperature": 0.1,
                            "base_temperature": 0.07,
                            "max_samples": 1024,
                            "max_views": 100,
                            "stride": 8,
                            "warmup_iters": 5000,
                            "loss_weight": 0.1,
                            "use_rmi": False
                            }
        self.easy_contrast = easy_contrast
        self.temperature = configer_contrast['temperature']
        self.base_temperature = configer_contrast['base_temperature']
        self.max_samples = configer_contrast['max_samples']
        self.max_views = configer_contrast['max_views']
        self.ignore_label = -1

    def _hard_anchor_sampling(self, X, y_within, y_boundary, y):
        batch_size, feat_dim = X.shape[0], X.shape[-1]

        classes = []
        total_classes = 0
        
        
        for ii in range(batch_size):
            this_classes_within =  torch.unique(y_within[ii])
            this_classes_boundary =  torch.unique(y_boundary[ii])
            
            this_classes = this_classes_within if len(this_classes_within) < len(this_classes_boundary) else this_classes_boundary
            this_classes = [x for x in this_classes if x != self.ignore_label]
            this_classes = [x for x in this_classes if (x == y_within[ii]).nonzero().shape[0] > self.max_views]
            this_classes = [x for x in this_classes if x in this_classes_within and x in this_classes_boundary]
            
            classes.append(this_classes)
            total_classes += len(this_classes)

        if total_classes == 0:
            return None, None

        n_view = self.max_samples // total_classes
        n_view = min(n_view, self.max_views)

        X_ = torch.zeros((total_classes, n_view, feat_dim), dtype=torch.float).cuda()
        y_ = torch.zeros(total_classes, dtype=torch.float).cuda()

        X_ptr = 0
        for ii in range(batch_size):
            this_y_within = y_within[ii]
            this_y_boundary = y_boundary[ii]
            this_y = y[ii]
            this_classes = classes[ii]

            for cls_id in this_classes:
                
                if self.easy_contrast:
                    # just easy
                               
                    # belong to whithin the cell
                    easy_indices = (this_y_within == cls_id).nonzero()
            
                    num_easy = easy_indices.shape[0]
                    
                    perm = torch.randperm(num_easy)
                    easy_indices = easy_indices[perm[:n_view]]
                    indices = easy_indices
                else:     
                    # mask belong to the boundary of the cell
                    hard_indices = (this_y_boundary == cls_id).nonzero()
                    
                    # belong to whithin the cell
                    easy_indices = (this_y_within == cls_id).nonzero()
                    
                    # # mask belong to the boundary of the cell + not predicted
                    # hard_indices = ((this_y_boundary == cls_id) & (this_y != 1)).nonzero()
                    
                    # # belong to whithin the cell + not 0
                    # easy_indices = ((this_y_within == cls_id) & (this_y != 0)).nonzero()

                    num_hard = hard_indices.shape[0]
                    num_easy = easy_indices.shape[0]

                    if num_hard >= n_view / 2 and num_easy >= n_view / 2:
                        num_hard_keep = n_view // 2
                        num_easy_keep = n_view - num_hard_keep
                    elif num_hard >= n_view / 2:
                        num_easy_keep = num_easy
                        num_hard_keep = n_view - num_easy_keep
                    elif num_easy >= n_view / 2:
                        num_hard_keep = num_hard
                        num_easy_keep = n_view - num_hard_keep
                    else:
                        print('this shoud be never touched! {} {} {}'.format(num_hard, num_easy, n_view))
                        raise Exception

                    perm = torch.randperm(num_hard)
                    hard_indices = hard_indices[perm[:num_hard_keep]]
                    perm = torch.randperm(num_easy)
                    easy_indices = easy_indices[perm[:num_easy_keep]]
                    indices = torch.cat((hard_indices, easy_indices), dim=0)

                
                X_[X_ptr, :, :] = X[ii, indices, :].squeeze(1)
                y_[X_ptr] = cls_id
                X_ptr += 1

        return X_, y_

    def _contrastive(self, feats_, labels_):
        anchor_num, n_view = feats_.shape[0], feats_.shape[1]

        labels_ = labels_.view(-1, 1)
        
        #identity matrix
        mask = torch.eq(labels_, torch.transpose(labels_, 0, 1)).float().cuda()

        contrast_count = n_view
        contrast_feature = torch.cat(torch.unbind(feats_, dim=1), dim=0)

        anchor_feature = contrast_feature
        anchor_count = contrast_count

        anchor_dot_contrast = torch.div(torch.matmul(anchor_feature, torch.transpose(contrast_feature, 0, 1)),
                                        self.temperature)
        logits_max, _ = torch.max(anchor_dot_contrast, dim=1, keepdim=True)
        logits = anchor_dot_contrast - logits_max.detach()

        mask = mask.repeat(anchor_count, contrast_count)
        neg_mask = 1 - mask

        logits_mask = torch.ones_like(mask).scatter_(1,
                                                     torch.arange(anchor_num * anchor_count).view(-1, 1).cuda(),
                                                     0)
        mask = mask * logits_mask

        neg_logits = torch.exp(logits) * neg_mask
        neg_logits = neg_logits.sum(1, keepdim=True)

        exp_logits = torch.exp(logits)

        log_prob = logits - torch.log(exp_logits + neg_logits)

        mean_log_prob_pos = (mask * log_prob).sum(1) / mask.sum(1)

        loss = - (self.temperature / self.base_temperature) * mean_log_prob_pos
        loss = loss.mean()

        return loss

    def forward(self, feats, mask_within=None, mask_boundary=None, predict=None):
    
        batch_size = feats.shape[0]

        mask_within = mask_within.view(batch_size, -1)
        mask_boundary = mask_boundary.view(batch_size, -1)
        predict = predict.view(batch_size, -1)
        feats = feats.permute(0, 2, 3, 1)
        feats = feats.view(feats.shape[0], -1, feats.shape[-1])

        losses = 0
        ptr = 0
        for feats_i, mask_within_j, mask_boundary_k, predict_l,  in zip(feats, mask_within, mask_boundary, predict):
            
            feats_, labels_ = self._hard_anchor_sampling(feats_i.unsqueeze(0), mask_within_j.unsqueeze(0), mask_boundary_k.unsqueeze(0), predict_l.unsqueeze(0))
            losses += self._contrastive(feats_, labels_) if feats_ is not None else 0
            ptr += 1
            
        return losses/ptr


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
        self.upsample_image = nn.Upsample(scale_factor=2)
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


class CellTransposeModel(nn.Module):
    def __init__(self, channels: int, device='cuda'):
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

        # self.dropout = nn.Dropout(0.1)
        
        self.out_block = nn.Sequential(
            nn.BatchNorm2d(num_features=32),
            nn.ReLU(),
            nn.Conv2d(in_channels=32, out_channels=3, kernel_size=1)
        )
        
        self.representation = nn.Sequential(
            # nn.Conv2d(in_channels=32, out_channels=32, kernel_size=1),
            nn.BatchNorm2d(num_features=32),
            nn.ReLU(),
            nn.Conv2d(in_channels=32, out_channels=256, kernel_size=1)
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
            
            # y = self.dropout(z)
            y = self.out_block(z)
            rep = self.representation(z)
            
            return y,  F.normalize(rep, p=2, dim=1)