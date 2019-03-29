#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Sep 27 11:12:10 2018

@author: avelinojaver
"""
import sys
from pathlib import Path

dname = Path(__file__).resolve().parents[2]
sys.path.append(str(dname))

from scipy.optimize import linear_sum_assignment
from bgnd_removal.models import UNet

import torch
import numpy as np
import cv2
import pandas as pd

from pathlib import Path
#%%

def data_reader(data_root):
    data_root = Path(data_root)
    
    fnames = list(data_root.rglob('*.tif'))
    fnames = [x for x in fnames if not (x.name.startswith('.') or x.name.startswith('_'))]
    for fname in tqdm.tqdm(fnames):
        fname = Path(fname)
        img = cv2.imread(str(fname), -1)
        
        fname_r = fname.parent / ('R_' + fname.stem + '.png')
        target = cv2.imread(str(fname_r), -1)
        
        if img is None:
            print(fname)
            continue
        
    
        if target is None:
            print(target)
            continue
            
        yield fname, img, target

def _process_img(model, xin, device):
    scale_ = (0, 255)
    
    x = xin[None].astype(np.float32)
    #x = np.log(x+1)
    x = (x - scale_[0])/(scale_[1] - scale_[0])
    
    
    with torch.no_grad():
        X = torch.from_numpy(x[None])
        X = X.to(device)
        Xhat = model(X)
    
    xhat = Xhat.squeeze().detach().cpu().numpy()
    xhat = (scale_[1] - scale_[0])*xhat + xhat[0]
    
    return xhat

#%%
def _get_bounding_boxes(bw, min_area = 250):
    bw = bw.astype(np.uint8)
    _, cnts, _ = cv2.findContours(bw, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    cnts = [x for x in cnts if cv2.contourArea(x) >= min_area]
    
    
    bboxes = [cv2.boundingRect(x) for x in cnts]
    bboxes = np.array(bboxes)
    return bboxes
#%%
#bb_pred, bb_target = bboxes[0]

def xywh2xyxy(bb):
    x1 = bb[..., 0]
    x2 = x1 + bb[..., 2]
    
    y1 = bb[..., 1]
    y2 = y1 + bb[..., 3]
    
    return x1, y1, x2, y2

def score_bboxes(bbox_pred, bbox_target, min_IoU = 0.5):
    max_cost = 1e3
    
    p_x1, p_y1, p_x2, p_y2 = xywh2xyxy(bbox_pred)
    pred_areas = bbox_pred[..., 2]*bbox_pred[..., 3]
    
    t_x1, t_y1, t_x2, t_y2 = xywh2xyxy(bbox_target)
    true_areas = bbox_target[..., 2]*bbox_target[..., 3]
    
    xx1 = np.maximum(p_x1[..., None], t_x1)
    yy1 = np.maximum(p_y1[..., None], t_y1)
    xx2 = np.minimum(p_x2[..., None], t_x2)
    yy2 = np.minimum(p_y2[..., None], t_y2)
    w = np.maximum(0.0, xx2 - xx1 + 1)
    h = np.maximum(0.0, yy2 - yy1 + 1)
    
    inter = w * h
    union = pred_areas[..., None] + true_areas - inter
    IoU = inter/union
    
    cost_matrix = IoU.copy()
    
    cost_matrix[cost_matrix <= min_IoU] = 1/max_cost
    cost_matrix = 1/cost_matrix
    pred_ind, true_ind = linear_sum_assignment(cost_matrix)
    
    good = cost_matrix[pred_ind, true_ind] < max_cost
    pred_ind, true_ind = pred_ind[good], true_ind[good]
    
    
    TP = pred_ind.size
    FP = inter.shape[0] - pred_ind.size
    FN = inter.shape[1] - pred_ind.size

    return TP, FP, FN

#%%
def get_mIoU(model_path, data_root, device):
    
    model = UNet(n_channels = 1, n_classes = 1)
    state = torch.load(model_path, map_location = 'cpu')
    model.load_state_dict(state['state_dict'])
    model = model.to(device)
    
    model.eval()
    th = 10#20
    gen = data_reader(data_root)
    
    all_IoU = []
    fnames = []
    scores = []
    for fname, X, Y in gen:
        xhat = _process_img(model, X, device)        
        pred_bw = (xhat - X.astype(np.float32)) > th
        target_bw = (Y>0)
        I = (target_bw & pred_bw).sum()
        U = (target_bw | pred_bw).sum()
        
        #if I/U < 0.9: print(fname)
        all_IoU.append((I, U))
        
        bbox_pred = _get_bounding_boxes(pred_bw)
        bbox_target = _get_bounding_boxes(target_bw)
        
        TP, FP, FN = score_bboxes(bbox_pred, bbox_target)
        
        #if FP > 0:
        #    import pdb
        #    pdb.set_trace()
        
        scores.append((TP, FP, FN))
        
        fnames.append(fname)
        
        
        
    return all_IoU, fnames, scores


if __name__ == '__main__':
    import tqdm
    
    cuda_id = 0
    if torch.cuda.is_available():
        print("THIS IS CUDA!!!!")
        dev_str = "cuda:" + str(cuda_id)
    else:
        dev_str = 'cpu'
    device = torch.device(dev_str)
    
    data_root = Path.home() / 'workspace//denoising/data/c_elegans_divergent/test/'
    
    results_root = Path.home() / 'workspace/denoising/results/'
    
    save_name = Path.home() / 'workspace/RESULTS_worms-divergent-samples.csv'
    models2process = [x.parent.name + '/' + x.name for x in results_root.glob('worms-divergent-samples*/*/')]
    models2process += [
            'worms-divergent/worms-divergent_l1_20190201_011625_unet_adam_lr0.0001_wd0.0_batch36', 
            'worms-divergent/worms-divergent_l2_20190201_011853_unet_adam_lr0.0001_wd0.0_batch36',
            'worms-divergent/worms-divergent_l1smooth_20190201_011918_unet_adam_lr0.0001_wd0.0_batch36'
            ]
    
    #%%
    _results = []
    for bn in tqdm.tqdm(models2process):
        model_path = results_root / bn / 'checkpoint.pth.tar'
        
        all_IoU, fnames, scores  = get_mIoU(model_path, data_root, device)
        
        Is, Us = map(np.sum, zip(*all_IoU))
        mIoU = Is/Us
        
        _results.append((all_IoU, scores, bn, fnames))
        
    #%%
    
    outs = []
    for all_IoU, scores, bn, fnames in _results:
        Is, Us = map(np.sum, zip(*all_IoU))
        mIoU = Is/Us
        
        #valid_fnames, IoUs = zip(*[(f, i/u) for f, (i,u) in zip(fnames, all_IoU) if u > 0])
        #p_IoU = np.percentile(IoUs, [0, 25, 50, 75, 100])
        
        TP, FP, FN = map(sum, zip(*scores))
        
        P = TP/(TP+FP)
        R = TP/(TP+FN)
        F1 = 2*P*R/(P+R)
        
        print(f'{bn}, mIoU={mIoU}, P={P}, R={R}, F1={F1}')
        
        
        outs.append((bn, mIoU, P, R, F1))
        
        
    df = pd.DataFrame(outs, columns = ['basename', 'mIoU', 'P', 'R', 'F1'])
    df.to_csv(str(save_name))
    
    #%%

#134 vs 160
'''
0.9652567032953968 : [0.90742774 0.95767688 0.96812459 0.97319382 0.98633619] : worms-divergent_l1_20190201_011625_unet_adam_lr0.0001_wd0.0_batch36
0.9555678820710335 : [0.85965898 0.94488393 0.96143169 0.97009082 0.98204755] : worms-divergent_l2_20190201_011853_unet_adam_lr0.0001_wd0.0_batch36
0.956105201643458 : [0.88383278 0.94346642 0.9607318  0.96963697 0.98301795] : worms-divergent_l1smooth_20190201_011918_unet_adam_lr0.0001_wd0.0_batch36
'''