#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jul  4 16:09:35 2019

@author: avelinojaver
"""
from flow import BBBC042Dataset, collate_simple, get_transforms
from models.retinanet import RetinaNet
from models.backbone_utils import retinanet_fpn_backbone, fasterrcnn_fpn_backbone

from pathlib import Path
import tqdm
import numpy as np
import math
import sys
from collections import defaultdict

import os
import shutil
import time
import datetime
from scipy.optimize import linear_sum_assignment
from tensorboardX import SummaryWriter

import torch
from torch.utils.data import DataLoader


pretrained_path = Path.home() / 'workspace/pytorch/pretrained_models/'
if pretrained_path.exists():
    os.environ['TORCH_HOME'] = str(pretrained_path)
from torchvision.models.detection import FasterRCNN, fasterrcnn_resnet50_fpn

def get_fasterrcnn(backbone_name = 'resnet50', progress=True, pretrained_backbone=True, **kwargs):
    _backbone = retinanet_fpn_backbone(backbone_name, pretrained_backbone)
    model = FasterRCNN(_backbone,  **kwargs)
    return model

def get_retinanet(backbone_name = 'resnet50', progress=True, pretrained_backbone=True, **kwargs):
    _backbone = retinanet_fpn_backbone(backbone_name, pretrained_backbone)
    model = RetinaNet(_backbone,  **kwargs)
    return model


def get_scores(prediction, target, IoU_cutoff = 0.25):
    if prediction.size == 0:
        return 0, 0, len(target), None, None
    
    if target.size == 0:
        return 0, len(prediction), 0, None, None
    
    
    #bbox areas
    xt1, yt1, xt2, yt2 = target.T
    true_areas = (xt2 - xt1 + 1) * (yt2 - yt1 + 1)
    
    xp1, yp1, xp2, yp2 = prediction.T
    pred_areas = (xp2 - xp1 + 1) * (yp2 - yp1 + 1)
    
    #intersections
    xx1 = np.maximum(xp1[..., None], xt1)
    yy1 = np.maximum(yp1[..., None], yt1)
    xx2 = np.minimum(xp2[..., None], xt2)
    yy2 = np.minimum(yp2[..., None], yt2)
    
    w = np.maximum(0.0, xx2 - xx1 + 1)
    h = np.maximum(0.0, yy2 - yy1 + 1)
    inter = w * h
    
    union = pred_areas[..., None] + true_areas - inter
    IoU = inter/union
    
    cost_matrix = inter.copy()
    cost_matrix[cost_matrix==0] = 1e-3
    cost_matrix = 1/cost_matrix
    pred_ind, true_ind = linear_sum_assignment(cost_matrix)
    good = IoU[pred_ind, true_ind] > IoU_cutoff
    
    pred_ind, true_ind = pred_ind[good], true_ind[good]
    
    TP = pred_ind.size
    FP = inter.shape[0] - pred_ind.size
    FN = inter.shape[1] - pred_ind.size
    
    
    return TP, FP, FN, pred_ind, true_ind


def warmup_lr_scheduler(optimizer, warmup_iters, warmup_factor):
    # I got this from https://github.com/pytorch/vision/blob/master/references/detection/utils.py
    def f(x):
        if x >= warmup_iters:
            return 1
        alpha = float(x) / warmup_iters
        return warmup_factor * (1 - alpha) + alpha

    return torch.optim.lr_scheduler.LambdaLR(optimizer, f)

def train_one_epoch(basename, model, optimizer, data_loader, device, epoch, logger):
     # Modified from https://github.com/pytorch/vision/blob/master/references/detection/engine.py
    
    model.train()
    header = f'{basename} Train Epoch: [{epoch}]'
    
    lr_scheduler = None
    if epoch == 0:
        warmup_factor = 1. / 1000
        warmup_iters = min(1000, len(data_loader) - 1)

        lr_scheduler = warmup_lr_scheduler(optimizer, warmup_iters, warmup_factor)
    
    train_avg_loss = 0
    individual_losses = defaultdict(int)
    
    pbar = tqdm.tqdm(data_loader, desc = header)
    for images, targets in pbar:
        images = list(image.to(device) for image in images)
        targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

        loss_dict = model(images, targets)

        losses = sum([loss for loss in loss_dict.values()])
        loss_value = losses.item()

        if not math.isfinite(loss_value):
            print(f"Loss is {loss_value}, stopping training")
            print(loss_dict)
            
            sys.exit(1)
        
        optimizer.zero_grad()
        losses.backward()
        optimizer.step()
        
        if lr_scheduler is not None:
            lr_scheduler.step()
        
        train_avg_loss += loss_value
        for k,v in loss_dict.items():
            individual_losses[k] += v.item()
    
    train_avg_loss /= len(data_loader)
    logger.add_scalar('train_loss', train_avg_loss, epoch)
    for k,v in individual_losses.items():
        logger.add_scalar('train_' + k, v/len(data_loader), epoch)
    
    return train_avg_loss    



@torch.no_grad()
def evaluate_one_epoch(basename, model, data_loader, device, epoch, logger):
    model.eval()
    
    cpu_device = torch.device("cpu")
    header = f'{basename} Test Epoch: [{epoch}]'
    
    metrics = np.zeros(3)
    model_time_avg = 0
    
    pbar = tqdm.tqdm(data_loader, desc = header)
    for image, targets in pbar:
        image = list(img.to(device) for img in image)
        
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        model_time = time.time()
        outputs = model(image)

        outputs = [{k: v.to(cpu_device) for k, v in t.items()} for t in outputs]
        model_time_avg += time.time() - model_time
        
    
        for pred, true in zip(outputs, targets) :
            pred_bb = pred['boxes'].detach().cpu().numpy()
            true_bb = true['boxes'].detach().cpu().numpy()
            TP, FP, FN, pred_ind, true_ind = get_scores(pred_bb, true_bb)
            metrics += TP, FP, FN
        
    TP, FP, FN = metrics
    P = TP/(TP+FP)
    R = TP/(TP+FN)
    F1 = 2*P*R/(P+R)
    
    logger.add_scalar('model_time', model_time_avg/len(data_loader), epoch)
    logger.add_scalar('val_P', P, epoch)
    logger.add_scalar('val_R', R, epoch)
    logger.add_scalar('val_F1', F1, epoch)
    
    return F1

def get_device(cuda_id):
    if torch.cuda.is_available():
        print("THIS IS CUDA!!!!")
        dev_str = "cuda:" + str(cuda_id)
    else:
        dev_str = 'cpu'
    device = torch.device(dev_str)
    
    return device

def save_checkpoint(state, is_best, save_dir, filename='checkpoint.pth.tar'):
    checkpoint_path = Path(save_dir) / filename
    torch.save(state, checkpoint_path)
    if is_best:
        best_path = Path(save_dir) / 'model_best.pth.tar'
        shutil.copyfile(checkpoint_path, best_path)



#%%
def get_model(model_name, backbone_name, roi_size, pretrained_backbone = True):
    if isinstance(roi_size, (list, tuple)):
        min_size = min(roi_size)
        max_size = max(roi_size)
    else:
        min_size = max_size = roi_size
    
    model_argkws = dict(
                min_size = min_size,
                max_size = max_size,
                image_mean = [0, 0, 0],
                image_std = [1., 1., 1.],
                pretrained_backbone = pretrained_backbone
                )
    
    if model_name == 'fasterrcnn':
        #faster rcnn requires an extra class for background...
        if not backbone_name:
            model = fasterrcnn_resnet50_fpn(num_classes = 2, **model_argkws)
        else:
            model = get_fasterrcnn(backbone_name = backbone_name, num_classes = 2, **model_argkws)
    elif model_name == 'retinanet':
        model = get_retinanet(backbone_name = backbone_name, num_classes = 1, **model_argkws)
    else:
        raise ValueError('Not implemented {model_name}')
    
    
    return model
#%%
def main(
    cuda_id = 0,
    test_freq = 1,
    roi_size = 512,
    num_epochs = 1000,
    num_workers = 1,
    batch_size = 16,
    lr = 1e-4,
    weight_decay = 0.0,
    max_samples = None,
    optimizer_name = 'adam',
    lr_scheduler_name = '',
    model_name = 'fasterrcnn',
    backbone = None,
    transform_type = 'all' 
    ):
    
    #root_dir = Path('/Users/avelinojaver/Downloads/BBBC042/')
    data_dir = Path.home() / 'workspace/datasets/BBBC/BBBC042'
    log_dir_root = Path.home() / 'workspace/localization/results/bbox_detection'
    device = get_device(cuda_id)
    
    transforms = get_transforms(transform_type, roi_size)
    model_roi_size = roi_size if transform_type in ['all', 'crops'] else None
    
    
    flow_train = BBBC042Dataset(data_dir, set_type = 'train', max_samples = max_samples, transforms = transforms)
    flow_val = BBBC042Dataset(data_dir, set_type = 'val', transforms = transforms)
    
    train_loader = DataLoader(flow_train, 
                              batch_size = batch_size,
                              num_workers = num_workers,
                              shuffle = True,
                              collate_fn = collate_simple
                              )
    
    val_loader = DataLoader(flow_val, 
                              batch_size = batch_size,
                              num_workers = num_workers,
                              collate_fn = collate_simple
                              )
    
    if model_roi_size is None:
        model_roi_size = flow_train.img_shape[:2]
    
    model = get_model(model_name, backbone, model_roi_size)
    model = model.to(device)
    model_params = filter(lambda p: p.requires_grad, model.parameters())
    
    
    if optimizer_name == 'adam':
        optimizer = torch.optim.Adam(model_params, lr = lr, weight_decay=weight_decay)
    elif optimizer_name == 'sgd':
        optimizer = torch.optim.SGD(model_params, 
                                    lr=lr,
                                    momentum=0.9, 
                                    weight_decay=weight_decay
                                    )
    else:
        raise ValueError(f'Not implemented {optimizer_name}')
        
        
    if not lr_scheduler_name:
        lr_scheduler = None
    elif lr_scheduler_name.startswith('stepLR'):
        #'stepLR-3-0.1'
        _, step_size, gamma = lr_scheduler_name.split('-')
        # and a learning rate scheduler
        lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer,
                                                       step_size = int(step_size),
                                                       gamma = float(gamma)
                                                       )
    else:
        raise ValueError(f'Not implemented {lr_scheduler_name}')
    
    now = datetime.datetime.now()
    date_str = now.strftime('%Y%m%d_%H%M%S')
    
    transform_type = transform_type if transform_type else 'none'
    
    sb = 'coco-resnet50' if backbone is None else backbone
    
    
    roi_size_s = str(model_roi_size).replace(' ', '')
    max_s = '' if not max_samples else f'S{max_samples}'
    
    basename = f'V_BBBC042{max_s}-T{transform_type}-roi{roi_size_s}_{model_name}-{sb}_{date_str}_{optimizer_name}-{lr_scheduler_name}_lr{lr}_wd{weight_decay}_batch{batch_size}'
    log_dir = log_dir_root / model_name / basename
    logger = SummaryWriter(log_dir = str(log_dir))
    
    
    
    
    best_score = 0
    for epoch in range(num_epochs):
        
        train_one_epoch(basename, model, optimizer, train_loader, device, epoch, logger)
        if lr_scheduler is not None:
            lr_scheduler.step()
        
        F1 = evaluate_one_epoch(basename, model, val_loader, device, epoch, logger)
        is_best = F1 > best_score
        best_score = F1 if is_best else best_score 
        
        state = {
                'epoch': epoch,
                'state_dict': model.state_dict(),
                'optimizer' : optimizer.state_dict(),
            }
        save_checkpoint(state, is_best, save_dir = str(log_dir))

if __name__ == '__main__':
    import fire
    fire.Fire(main)
    
#    model = get_fasterrcnn(backbone_name = 'resnet18', num_classes = 2, min_size = 512, pretrained_backbone = False)
#    model.eval()
#    X = torch.rand((2, 3, 512, 512))
#    out = model(X)