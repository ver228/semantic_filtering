#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jul  5 22:47:32 2019

@author: avelinojaver
"""

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Aug 30 14:15:42 2018

@author: avelinojaver
"""
import math
import torch
from torch import nn
import torch.nn.functional as F
from torchvision.models.detection.rpn import AnchorGenerator, concat_box_prediction_layers, det_utils
from torchvision.models.detection.transform import GeneralizedRCNNTransform
from torchvision.ops import boxes as box_ops
from collections import OrderedDict

def _norm_init_weights(m): 
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        nn.init.normal_(m.weight.data, mean=0.0, std=0.01)
        nn.init.constant_(m.bias.data, 0.0)
        

class BasicHead(nn.Sequential):
    def __init__(self, in_planes, num_anchors):
        layers = []
        for i in range(4):
            _n_in = in_planes if i == 0 else 256
            conv = nn.Conv2d(_n_in, 256, kernel_size=3, stride=1, padding=1)
            layers.append(conv)
            layers.append(nn.ReLU())
        super().__init__(*layers)
        
        for m in self.modules():
            _norm_init_weights(m)


class ClassificationHead(nn.Module):
    def __init__(self, in_planes, num_classes, num_anchors, is_classification = False):
        super().__init__()
        self.num_classes = num_classes
        self.num_anchors = num_anchors
        
        self._head = BasicHead(in_planes, num_anchors)
        
        conv = nn.Conv2d(256, num_classes*num_anchors, kernel_size=3, stride=1, padding=1)
        #add bias to make easier to train the classification layer 
        #"every anchor should be labeled as foreground with confidence of ∼π"
        probability = 0.04
        bias = -math.log((1-probability)/probability)
        nn.init.constant_(conv.bias.data, bias)
        nn.init.constant_(conv.weight.data, 0.0)
        
        self.clf = nn.Sequential(conv, 
                                 nn.Sigmoid()
                                 )
        
    def forward(self, x):
        x = self._head(x)
        pred = self.clf(x)
        
        
        return pred
    
class RegressionHead(nn.Module):
    def __init__(self, in_planes, num_anchors, is_classification = False):
        super().__init__()
        self.num_anchors = num_anchors
        
        self._head = BasicHead(in_planes, num_anchors)
        self.loc = nn.Conv2d(256, 4*num_anchors, kernel_size=3, stride=1, padding=1)
        _norm_init_weights(self.loc)
        
    def forward(self, x):
        x = self._head(x)
        pred = self.loc(x)
        return pred

class RetinaHead(nn.Module):
    def __init__(self, in_planes, num_classes, num_anchors):
        super().__init__()
        
        self.in_planes = in_planes
        self.num_classes = num_classes
        self.num_anchors = num_anchors
        
        self.bbox_pred = RegressionHead(self.in_planes, self.num_anchors)
        self.cls_pred = ClassificationHead(self.in_planes, self.num_classes, self.num_anchors)
        
    def forward(self, x):
        clf = []
        bbox_reg = []
        for feature in x:
            clf.append(self.cls_pred(feature))
            bbox_reg.append(self.bbox_pred(feature))
        return clf, bbox_reg

#%%
class FocalLoss(nn.Module):
    def __init__(self, num_classes, alpha = 0.25, gamma = 2.):
        super().__init__()
        self.num_classes = num_classes
        self.alpha = alpha
        self.gamma = gamma
    
    def focal_loss(self, preds, targets):
        target_onehot = torch.eye(self.num_classes+1)[targets]
        target_onehot = target_onehot[:,1:].contiguous() #the zero is the background class
        target_onehot = target_onehot.to(targets.device) #send to gpu if necessary
        
        focal_weights = self._get_weight(preds,target_onehot)
        
        #I already applied the sigmoid to the classification layer. I do not need binary_cross_entropy_with_logits
        return (focal_weights*F.binary_cross_entropy(preds, target_onehot, reduce=False)).sum()
    
    def _get_weight(self, x, t):
        pt = x*t + (1-x)*(1-t)
        w = self.alpha*t + (1-self.alpha)*(1-t)
        return w * (1-pt).pow(self.gamma)
    
    def forward(self, pred, target):
        #%%
        clf_target, loc_target = target
        clf_preds, loc_preds = pred
        
        ### regression loss
        pos = clf_target > 0
        num_pos = pos.sum().item()
        
        #since_average true is equal to divide by the number of possitives
        loc_loss = F.smooth_l1_loss(loc_preds[pos], loc_target[pos], size_average=False)
        loc_loss = loc_loss/max(1, num_pos)
        
        #### focal lost
        valid = clf_target >= 0  # exclude ambigous anchors (jaccard >0.4 & <0.5) labelled as -1
        clf_loss = self.focal_loss(clf_preds[valid], clf_target[valid])
        clf_loss = clf_loss/max(1, num_pos)  #inplace operations are not permitted for gradients
        
        
        #I am returning both losses because I want to plot them separately
        return clf_loss, loc_loss

#%%
        
class RetinaNet(nn.Module):
    def __init__(self, 
                 backbone, 
                 num_classes = 1, 
                 anchor_generator=None, 
                 head=None,
                 image_mean = [0., 0., 0.], 
                 image_std= [1., 1., 1.], 
                 min_size = 512,
                 max_size = 512,
                 fg_iou_thresh=0.5, 
                 bg_iou_thresh=0.4,
                 nms_thresh=0.5,
                 score_thresh = 0.05,
                 detections_per_img = 100
                 ):
        
        super().__init__()
        
        
        
        if anchor_generator is None:
            anchor_sizes = ((32,), (64,), (128,), (256,), (512,))
            aspect_ratios = ((0.5, 1.0, 2.0),) * len(anchor_sizes)
            anchor_generator = AnchorGenerator(
                anchor_sizes, aspect_ratios
            )    
            
        out_channels = backbone.out_channels
        if head is None:
            head = RetinaHead(
                out_channels, num_classes, anchor_generator.num_anchors_per_location()[0]
            )
        
        self.transform = GeneralizedRCNNTransform(min_size, max_size, image_mean, image_std)
        self.box_coder = det_utils.BoxCoder(weights=(1.0, 1.0, 1.0, 1.0))
        self.anchor_generator = anchor_generator
        self.head = head
        self.backbone = backbone
        self.focal_loss = FocalLoss(num_classes)
        self.num_classes = num_classes
        self.nms_thresh = nms_thresh
        self.score_thresh = score_thresh
        self.detections_per_img = detections_per_img
                 
        # used during training
        self.box_similarity = box_ops.box_iou
        self.proposal_matcher = det_utils.Matcher(
            fg_iou_thresh,
            bg_iou_thresh,
            allow_low_quality_matches=False,
        )
    
    def assign_targets_to_anchors(self, anchors, targets):
        matched_gt_labels = []
        matched_gt_boxes = []
        for anchors_per_image, targets_per_image in zip(anchors, targets):
            gt_boxes = targets_per_image["boxes"]
            gt_labels = targets_per_image["labels"]
            
            match_quality_matrix = self.box_similarity(gt_boxes, anchors_per_image)
            matched_idxs = self.proposal_matcher(match_quality_matrix)
            
            
            bb_matches = matched_idxs.clamp(min=0) 
            # get the targets corresponding GT for each proposal
            # NB: need to clamp the indices because we can have a single
            # GT in the image, and matched_idxs can be -2, which goes
            # out of bounds
            matched_gt_boxes_per_image = gt_boxes[bb_matches]
            matched_gt_labels_per_image = gt_labels[bb_matches]
            
            #I want to set the ranges 0 for bg, -1 for matches in between and the rest as the box label 
            bad = matched_idxs<0
            matched_gt_labels_per_image[bad] =  matched_idxs[bad] + 1
            
            
            matched_gt_labels.append(matched_gt_labels_per_image)
            matched_gt_boxes.append(matched_gt_boxes_per_image)
        return matched_gt_labels, matched_gt_boxes
    
    def postprocess_detections(self, pred_boxes, pred_scores, image_shapes):
        
        all_boxes = []
        all_scores = []
        all_labels = []
        for boxes, clf_scores, image_shape in zip(pred_boxes, pred_scores, image_shapes):
            
            # do not backprop throught objectness
            scores, labels = clf_scores.max(dim = -1)
            labels += 1 # the classes a shifted (class 1 is the index 0 and so on)
            
            
            boxes = box_ops.clip_boxes_to_image(boxes, image_shape)
            # remove low scoring boxes
            keep = scores > self.score_thresh
            boxes, scores, labels = boxes[keep], scores[keep], labels[keep]
            # remove empty boxes
            keep = box_ops.remove_small_boxes(boxes, min_size=1e-2)
            boxes, scores, labels = boxes[keep], scores[keep], labels[keep]

            # non-maximum suppression, independently done per class
            keep = box_ops.batched_nms(boxes, scores, labels, self.nms_thresh)
            # keep only topk scoring predictions
            keep = keep[:self.detections_per_img]
            boxes, scores, labels = boxes[keep], scores[keep], labels[keep]
            
            all_boxes.append(boxes)
            all_scores.append(scores)
            all_labels.append(labels)
            
        return all_boxes, all_scores, all_labels
    
    def forward(self, images, targets=None):
#        """
#        Arguments:
#            images (list[Tensor]): images to be processed
#            targets (list[Dict[Tensor]]): ground-truth boxes present in the image (optional)
#
#        Returns:
#            result (list[BoxList] or dict[Tensor]): the output from the model.
#                During training, it returns a dict[Tensor] which contains the losses.
#                During testing, it returns list[BoxList] contains additional fields
#                like `scores`, `labels` and `mask` (for Mask R-CNN models).
#
#        """
        if self.training and targets is None:
            raise ValueError("In training mode, targets should be passed")
        images, targets = self.transform(images, targets)
        
        features = self.backbone(images.tensors)
        
        if isinstance(features, torch.Tensor):
            features = OrderedDict([(0, features)])
        
        features = list(features.values())
        clf_scores, pred_bbox_deltas = self.head(features)
        anchors = self.anchor_generator(images, features)
        
        
        num_images = len(anchors)
        clf_scores, pred_bbox_deltas = \
            concat_box_prediction_layers(clf_scores, pred_bbox_deltas)
        
        if self.training:
            gt_labels, matched_gt_boxes = self.assign_targets_to_anchors(anchors, targets)
            regression_targets = self.box_coder.encode(matched_gt_boxes, anchors)
            
            gt_labels = torch.cat(gt_labels, dim=0)
            regression_targets = torch.cat(regression_targets, dim=0)
            
            _target = (gt_labels, regression_targets)
            _pred = (clf_scores, pred_bbox_deltas)
            classification_loss, box_loss = self.focal_loss(_pred, _target)
            
            losses = {}
            losses['classification_loss'] = classification_loss
            losses['box_loss'] = box_loss
            
            return losses
        else:
            proposals = self.box_coder.decode(pred_bbox_deltas.detach(), anchors)
            proposals = proposals.view(num_images, -1, 4)
            clf_scores = clf_scores.view(num_images, -1, self.num_classes)
            
            boxes, labels, scores = self.postprocess_detections(proposals, clf_scores, images.image_sizes)
            num_images = len(boxes)
            
            result = []
            for i in range(num_images):
                result.append(
                    dict(
                        boxes=boxes[i],
                        labels=labels[i],
                        scores=scores[i],
                    )
                )
            return result

if __name__ == '__main__':
    
    
    #%%
    from flow import BBBC042Dataset, collate_simple
    from pathlib import Path
    from torch.utils.data import DataLoader
    from torchvision.models.detection.backbone_utils import resnet_fpn_backbone
    
    data_dir = Path('/Users/avelinojaver/Downloads/BBBC042/')
    roi_size = 512
    gen = BBBC042Dataset(data_dir, max_samples = 10, roi_size = roi_size)
    loader = DataLoader(gen, batch_size = 2, collate_fn = collate_simple)
    
    
    for X, target in loader:
        break
    
    #%%
    backbone = resnet_fpn_backbone(backbone_name = 'resnet50', pretrained = False)
    model = RetinaNet(backbone = backbone, num_classes = 2, min_size = roi_size, max_size = roi_size)
    losses = model(X, target)
    
    model.eval()
    result = model(X, target)
