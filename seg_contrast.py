import torch
import torch.nn as nn
import torch.nn.functional as F

from loguru import logger   

class PixelContrastLoss(nn.Module):
    def __init__(self):
        super(PixelContrastLoss, self).__init__()

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
    
        self.temperature = configer_contrast['temperature']
        self.base_temperature = configer_contrast['base_temperature']
        self.max_samples = configer_contrast['max_samples']
        self.max_views = configer_contrast['max_views']
        self.ignore_label = -1

    def _hard_anchor_sampling(self, X, y_hat, y):
        batch_size, feat_dim = X.shape[0], X.shape[-1]

        classes = []
        total_classes = 0
        for ii in range(batch_size):
            this_y = y_hat[ii]
            this_classes = torch.unique(this_y)
            this_classes = [x for x in this_classes if x != self.ignore_label]
            this_classes = [x for x in this_classes if (this_y == x).nonzero().shape[0] > self.max_views]

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
            this_y_hat = y_hat[ii]
            this_y = y[ii]
            this_classes = classes[ii]

            for cls_id in this_classes:
                hard_indices = ((this_y_hat == cls_id) & (this_y != cls_id)).nonzero()
                easy_indices = ((this_y_hat == cls_id) & (this_y == cls_id)).nonzero()

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
                    logger.info('this shoud be never touched! {} {} {}'.format(num_hard, num_easy, n_view))
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

    def forward(self, feats, labels=None, predict=None):
    
        assert labels.shape[-1] == feats.shape[-1], '{} {}'.format(labels.shape, feats.shape)

        batch_size = feats.shape[0]

        labels = labels.view(batch_size, -1)
        predict = predict.view(batch_size, -1)
        feats = feats.permute(0, 2, 3, 1)
        feats = feats.view(feats.shape[0], -1, feats.shape[-1])

        feats_, labels_ = self._hard_anchor_sampling(feats, labels, predict)

        loss = self._contrastive(feats_, labels_)
        return loss


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
        self.ignore_label = 0

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
                        logger.info('this shoud be never touched! {} {} {}'.format(num_hard, num_easy, n_view))
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

class PixelContrastMorphologyLossWithGaussianBlur(nn.Module):
    def __init__(self):
        super(PixelContrastMorphologyLossWithGaussianBlur, self).__init__()

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
    
        self.temperature = configer_contrast['temperature']
        self.base_temperature = configer_contrast['base_temperature']
        self.max_samples = configer_contrast['max_samples']
        self.max_views = configer_contrast['max_views']
        self.ignore_label = 0

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
                    logger.info('this shoud be never touched! {} {} {}'.format(num_hard, num_easy, n_view))
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

    def forward(self, feats_1, feats_2, mask_within=None, mask_boundary=None, predict_1=None, predict_2=None):
    
        batch_size = feats_1.shape[0]

        mask_within = mask_within.view(batch_size, -1)
        mask_boundary = mask_boundary.view(batch_size, -1)
        
        predict_1 = predict_1.view(batch_size, -1)
        feats_1 = feats_1.permute(0, 2, 3, 1)
        feats_1 = feats_1.view(feats_1.shape[0], -1, feats_1.shape[-1])
        
        predict_2 = predict_2.view(batch_size, -1)
        feats_2 = feats_2.permute(0, 2, 3, 1)
        feats_2 = feats_2.view(feats_2.shape[0], -1, feats_2.shape[-1])

        losses = 0
        ptr = 0
        for feats_1_i, feats_2_i, mask_within_j, mask_boundary_k, predict_1_l, predict_2_l  in zip(feats_1, feats_2, mask_within, mask_boundary, predict_1, predict_2):
            
            feats_pair = torch.cat((feats_1_i.unsqueeze(0), feats_2_i.unsqueeze(0)))
            mask_within_pair = torch.cat((mask_within_j.unsqueeze(0), mask_within_j.unsqueeze(0)))
            mask_boundary_pair = torch.cat((mask_boundary_k.unsqueeze(0), mask_boundary_k.unsqueeze(0)))
            predict_pair = torch.cat((predict_1_l.unsqueeze(0), predict_2_l.unsqueeze(0)))
            
            feats_, labels_ = self._hard_anchor_sampling(feats_pair, mask_within_pair, mask_boundary_pair, predict_pair)
            losses += self._contrastive(feats_, labels_) if feats_ is not None else 0
            
            ptr += 1
            
        return losses/ptr
  