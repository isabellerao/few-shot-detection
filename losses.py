import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from dataloader import dataset_train

import sys

def euclidean_dist(x, y):
    # x: N x D
    # y: M x D
    n = x.size(0)
    m = y.size(0)
    d = x.size(1)
    assert d == y.size(1)

    x = x.unsqueeze(1).expand(n, m, d).cuda()
    y = y.unsqueeze(0).expand(n, m, d).cuda()

    return torch.pow(x - y, 2).sum(2)

def calc_iou(a, b):
    # calculates IoU (intersection over union)
    area = (b[:, 2] - b[:, 0]) * (b[:, 3] - b[:, 1])

    iw = torch.min(torch.unsqueeze(a[:, 2], dim=1), b[:, 2]) - torch.max(torch.unsqueeze(a[:, 0], 1), b[:, 0])
    ih = torch.min(torch.unsqueeze(a[:, 3], dim=1), b[:, 3]) - torch.max(torch.unsqueeze(a[:, 1], 1), b[:, 1])

    iw = torch.clamp(iw, min=0)
    ih = torch.clamp(ih, min=0)

    ua = torch.unsqueeze((a[:, 2] - a[:, 0]) * (a[:, 3] - a[:, 1]), dim=1) + area - iw * ih

    ua = torch.clamp(ua, min=1e-8)

    intersection = iw * ih

    IoU = intersection / ua

    return IoU

class FocalLoss(nn.Module):
    #def __init__(self):

    def forward(self, classifications, regressions, anchors, annotations):
        dummy = 0
        alpha = 0.25
        gamma = 2.0
        batch_size = classifications.shape[0]
        classification_losses = []
        regression_losses = []
        classification_batch = []

        anchor = anchors[0, :, :]
        anchor_widths  = anchor[:, 2] - anchor[:, 0]
        anchor_heights = anchor[:, 3] - anchor[:, 1]
        anchor_ctr_x   = anchor[:, 0] + 0.5 * anchor_widths
        anchor_ctr_y   = anchor[:, 1] + 0.5 * anchor_heights

        for j in range(batch_size):

            classification = classifications[j, :, :]
            regression = regressions[j, :, :]
            bbox_annotation = annotations[j, :, :]
            bbox_annotation = bbox_annotation[bbox_annotation[:, 4] != -1]

            if bbox_annotation.shape[0] == 0:
                regression_losses.append(torch.tensor(0).float().cuda())
                classification_losses.append(torch.tensor(0).float().cuda())

                continue

            IoU = calc_iou(anchors[0, :, :], bbox_annotation[:, :4]) # num_anchors x num_annotations

            IoU_max, IoU_argmax = torch.max(IoU, dim=1) # num_anchors x 1

            num_classes = dataset_train.num_classes()
            num_anchors = classification.shape[0]
            z_dim = classification.shape[1]

            targets = torch.ones(num_anchors, 1) * -1 #classification.shape[0] = # anchors
            targets = targets.cuda()
            targets[torch.lt(IoU_max, 0.4), 0] = num_classes
            positive_indices = torch.ge(IoU_max, 0.5)
            num_positive_anchors = positive_indices.sum()
            assigned_annotations = bbox_annotation[IoU_argmax, :]
            targets[positive_indices, 0] = assigned_annotations[positive_indices, 4]

            classification_wclass = torch.cat((classification, targets), dim = 1)

            if dummy == 0: 
                classification_batch = classification_wclass
                dummy = 1
            else: 
                classification_batch = torch.cat((classification_batch, classification_wclass), dim = 0) 

            # compute the loss for regression

            if positive_indices.sum() > 0:
                assigned_annotations = assigned_annotations[positive_indices, :]

                anchor_widths_pi = anchor_widths[positive_indices]
                anchor_heights_pi = anchor_heights[positive_indices]
                anchor_ctr_x_pi = anchor_ctr_x[positive_indices]
                anchor_ctr_y_pi = anchor_ctr_y[positive_indices]

                gt_widths  = assigned_annotations[:, 2] - assigned_annotations[:, 0]
                gt_heights = assigned_annotations[:, 3] - assigned_annotations[:, 1]
                gt_ctr_x   = assigned_annotations[:, 0] + 0.5 * gt_widths
                gt_ctr_y   = assigned_annotations[:, 1] + 0.5 * gt_heights

                # clip widths to 1
                gt_widths  = torch.clamp(gt_widths, min=1)
                gt_heights = torch.clamp(gt_heights, min=1)

                targets_dx = (gt_ctr_x - anchor_ctr_x_pi) / anchor_widths_pi
                targets_dy = (gt_ctr_y - anchor_ctr_y_pi) / anchor_heights_pi
                targets_dw = torch.log(gt_widths / anchor_widths_pi)
                targets_dh = torch.log(gt_heights / anchor_heights_pi)

                targets = torch.stack((targets_dx, targets_dy, targets_dw, targets_dh))
                targets = targets.t()

                targets = targets/torch.Tensor([[0.1, 0.1, 0.2, 0.2]]).cuda()


                negative_indices = ~positive_indices # negative_indices = 1 - positive_indices

                regression_diff = torch.abs(targets - regression[positive_indices, :])

                regression_loss = torch.where(
                    torch.le(regression_diff, 1.0 / 9.0),
                    0.5 * 9.0 * torch.pow(regression_diff, 2),
                    regression_diff - 0.5 / 9.0
                )
                regression_losses.append(regression_loss.mean())
            else:
                regression_losses.append(torch.tensor(0).float().cuda())


        # Classification loss

        classification_support = classification_batch[:5*num_anchors, :] 
        classification_query = classification_batch[5*num_anchors:, :]
        classification_query = classification_query[torch.ne(classification_query[:, z_dim], -1), :]
        # remove ignored anchors

        #print('query', classification_query.size())

        embedding_support = classification_support[:, :z_dim]
        labels_support = classification_support[:, z_dim].detach()

        embedding_query = classification_query[:, :z_dim]
        labels_query = classification_query[:, z_dim].detach()

        num_positive_query = labels_query[torch.ne(labels_query, 80)].size(0)
        #print('query positive anchors', num_positive_query)

        classes = labels_support.unique(dim = 0).detach()
        assert len(classes) == 4
        #print(classes)

        prototypes = torch.zeros(classes.shape[0], z_dim)
        i = 0
        for cl in classes: 
            prototypes[i,:] = (embedding_support[torch.eq(labels_support, cl), :].mean(dim = 0))
            i += 1

        dists = euclidean_dist(embedding_query, prototypes)
        
        p_y = F.softmax(-dists, dim=1)
        print('p_y', p_y[:10, :])
        
        target_inds = (torch.nonzero(labels_query[..., None] == classes)[:, 1])[:, None].detach()

        prob = p_y.gather(1, target_inds).squeeze().view(-1)
        print('prob', prob.float().sum())
        cls_loss = - alpha * torch.pow(1-prob, gamma) * torch.log(prob)
        #print(cls_loss.size())
        #print(cls_loss.sum())
        classification_losses = cls_loss.sum()/max(num_positive_query, 1)

        _, y_hat = p_y.max(1)
        print(y_hat.float().mean(), target_inds.squeeze().float().mean())

        accuracy = torch.eq(y_hat, target_inds.squeeze()).float().mean()
        print('accuracy', accuracy)

        return classification_losses, torch.stack(regression_losses).mean(dim=0, keepdim=True)

    
