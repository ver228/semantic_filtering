#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jul  3 11:24:10 2019

@author: avelinojaver
"""
import os
import sys
from pathlib import Path

dname = Path(__file__).resolve().parents[2]
sys.path.append(str(dname))


from bgnd_removal.models import UNet
from bgnd_removal.trainer import get_device

from torch.utils.data import Dataset, DataLoader
import torch
import pandas as pd
import cv2
import numpy as np
import tqdm
import pickle
from scipy.optimize import linear_sum_assignment
#import multiprocessing as mp
from skimage.filters import threshold_otsu

def collate_simple(batch):
    X, y = zip(*batch)
    X = torch.stack(X)
    return X, y

def _cleaned_to_belive_map(x_in):
    
    if x_in.shape[0] == 3:
        x2th = np.rollaxis(x_in, 0, 3)
        x2th = 1 - cv2.cvtColor(x2th, cv2.COLOR_RGB2GRAY)
    else:
        x2th = x_in[0]
        
    bot, top = x2th.min(), x2th.max()
    x2th = (x2th - bot)/(top - bot)
    return x2th

def _get_candidates(x2th, th = -1):
    if th < 0:
        th = threshold_otsu(x2th)
    
    mask = (x2th>th).astype(np.uint8)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(5,5))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=1)
    _, cnts, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    return cnts, mask

def _filter_boxes(cnts, min_area = 100):

    pred_bboxes = [cv2.boundingRect(x) for x in cnts if cv2.contourArea(x) > min_area]
    pred_bboxes = [(x, y, x + w, y + h) for (x,y,w,h) in pred_bboxes]
    pred_bboxes = np.array(pred_bboxes)
    return pred_bboxes

#def _get_candidates(x2th, th):
#    if th < 0:
#        th = threshold_otsu(x2th)
#    
#    mask = (x2th>th).astype(np.uint8)
#        
#    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(3,3))
#    mask = cv2.dilate(mask, kernel, iterations=2)
#    
#    _, cnts, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
#    pred_bboxes = [cv2.boundingRect(x) for x in cnts]
#    return pred_bboxes, mask
#
#def _filter_boxes(pred_bboxes, min_bbox_size):
#    preds = []
#    for bb in pred_bboxes:
#        b_size = min(bb[2], bb[3])
#        if b_size > min_bbox_size:
#            dat = (bb[0], bb[1], bb[0] + bb[2], bb[1] + bb[3])
#            preds.append(dat)
#    preds = np.array(preds)   
#    return preds

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

class DataFlow(Dataset):
    int_scale = (0,255)
    def __init__(self, data_dir, files2check, is_colour = False):
        
        self.is_colour = is_colour
        data2process = []
        for fname in tqdm.tqdm(files2check, desc = 'Preloading data...'):
            img_id = int(fname.stem)
            annotations = data_dir / 'positions' / f'{img_id}.txt'
            X, target = self.read_data(fname, annotations)
            data2process.append((X,target))
        self.data2process = data2process
        
    def __len__(self):
        return len(self.data2process)
    
    def __getitem__(self, ind):
        X, target = self.data2process[ind]
        if not self.is_colour:
            #0.299R + 0.587G  + 0.114B #https://docs.opencv.org/2.4/modules/imgproc/doc/miscellaneous_transformations.html#cvtcolor
            X = 0.299*X[0] + 0.587*X[1] + 0.114*X[2]
            X = 1 - X.unsqueeze(0)
        
        return X, target
    
    def read_data(self, image_file, annotations_file):
        df = pd.read_csv(str(annotations_file), delim_whitespace=True, header = None)
        target = df.loc[:, 4:7].values.copy()
        img_ori = cv2.imread(str(image_file), -1)[..., ::-1]  #opencv reads the channels as BGR so i need to switch them
        img = np.rollaxis(img_ori, 2, 0)
        x = img.astype(np.float32)
        x = (x - self.int_scale[0])/(self.int_scale[1] - self.int_scale[0])
        X = torch.from_numpy(x)
        return X, target

def process_out(dat):
        xx, target, (iepoch, IoU_cutoffs, thresh2check, min_bbox_sizes) = dat
        res = []
        x2th = _cleaned_to_belive_map(xx[:3])
        
        for ith, th in enumerate(thresh2check):
            pred_bboxes, mask = _get_candidates(x2th, th)
            for isize, min_size in enumerate(min_bbox_sizes):
                preds = _filter_boxes(pred_bboxes, min_size)
                for icut, cutoff in enumerate(IoU_cutoffs):
                    TP, FP, FN, pred_ind, true_ind = get_scores(preds, target, cutoff)
                    res.append([(icut, iepoch, ith, isize), (TP, FP, FN)])
                    
        return res

def main(set_type = 'val', 
         batch_size = 6,
         cuda_id = 0
         ):

    
    device = get_device(cuda_id)
    
    epochs2check = list(range(4, 100, 5))

#    min_bbox_sizes = [10, 20, 30, 40, 50]
#    thresh2check = np.arange(0.05, 0.5, 0.025).tolist()
#    IoU_cutoffs =  [0.1, 0.25, 0.5]
    
    
    min_bbox_sizes = [100, 200, 300, 400, 500]
    thresh2check = [-1]
    IoU_cutoffs =  [0.01, 0.1, 0.25, 0.5]
    
    results = {}
    bn2check = [
#                'BBBC042-colour-v4-S5_unet-filter_l1smooth_20190709_202139_adam_lr0.00032_wd0.0_batch32',
#                'BBBC042-colour-v4-S10_unet-filter_l1smooth_20190709_202129_adam_lr0.00032_wd0.0_batch32',
#                'BBBC042-colour-v4-S25_unet-filter_l1smooth_20190710_000135_adam_lr0.00032_wd0.0_batch32',
#                'BBBC042-colour-v4-S100_unet-filter_l1smooth_20190710_001136_adam_lr0.00032_wd0.0_batch32',
#                'BBBC042-colour-v4_unet-filter_l1smooth_20190709_223456_adam_lr0.00032_wd0.0_batch32',
#                
#                'BBBC042-colour-bgnd-S5_unet-filter_l1smooth_20190710_145222_adam_lr0.00032_wd0.0_batch32',
#                'BBBC042-colour-bgnd-S100_unet-filter_l1smooth_20190710_151403_adam_lr0.00032_wd0.0_batch32',
#                'BBBC042-colour-bgnd-S25_unet-filter_l1smooth_20190710_162339_adam_lr0.00032_wd0.0_batch32',
#                'BBBC042-colour-bgnd-S10_unet-filter_l1smooth_20190710_164420_adam_lr0.00032_wd0.0_batch32',
#                'BBBC042-colour-bgnd_unet-filter_l1smooth_20190710_150856_adam_lr0.00032_wd0.0_batch32',
#                
#                'BBBC042-colour-bgnd_unet-filter_l1smooth_20190711_180845_adam_lr0.00032_wd0.0_batch32',
#                
#                'BBBC042-simple_unet-filter_l1smooth_20190711_195905_adam_lr0.00032_wd0.0_batch32',
#                'BBBC042-simple-S5_unet-filter_l1smooth_20190712_045727_adam_lr0.00032_wd0.0_batch32',
#                'BBBC042-simple-S10_unet-filter_l1smooth_20190711_204742_adam_lr0.00032_wd0.0_batch32',
#                'BBBC042-simple-S25_unet-filter_l1smooth_20190711_204744_adam_lr0.00032_wd0.0_batch32',
#                'BBBC042-simple-S100_unet-filter_l1smooth_20190712_043730_adam_lr0.00032_wd0.0_batch32',
#                
#                'BBBC042-simple-bgnd-S5_unet-filter_l1smooth_20190712_144153_adam_lr0.00032_wd0.0_batch32',
#                'BBBC042-simple-bgnd-S25_unet-filter_l1smooth_20190712_105659_adam_lr0.00032_wd0.0_batch32',
#                 'BBBC042-simple-bgnd-S10_unet-filter_l1smooth_20190712_160608_adam_lr0.00032_wd0.0_batch32',
#                'BBBC042-simple-bgnd-S100_unet-filter_l1smooth_20190712_144219_adam_lr0.00032_wd0.0_batch32',
#                'BBBC042-simple-bgnd_unet-filter_l1smooth_20190712_104448_adam_lr0.00032_wd0.0_batch32',
#                
#                'BBBC042-colour-bgnd_unet-filter_l1smooth_20190712_110713_adam_lr0.00032_wd0.0_batch32',
#                'BBBC042-colour-bgnd_unet-filter_l1smooth_20190712_210900_adam_lr0.00032_wd0.0_batch32'
#                'BBBC042-simple-bgnd_unet-filter_l1smooth_20190712_104448_adam_lr0.00032_wd0.0_batch32'

                'BBBC042-simple-bgnd-S5_unet-filter_l1smooth_20190713_123452_adam-stepLR-30-0.1_lr0.00032_wd0.0_batch32',
                'BBBC042-simple-bgnd-S10_unet-filter_l1smooth_20190713_144738_adam-stepLR-30-0.1_lr0.00032_wd0.0_batch32',
                'BBBC042-simple-bgnd-S25_unet-filter_l1smooth_20190713_161618_adam-stepLR-30-0.1_lr0.00032_wd0.0_batch32',
                'BBBC042-simple-bgnd-S100_unet-filter_l1smooth_20190713_124826_adam-stepLR-30-0.1_lr0.00032_wd0.0_batch32',
                'BBBC042-simple-bgnd_unet-filter_l1smooth_20190713_124210_adam-stepLR-30-0.1_lr0.00032_wd0.0_batch32',

                'BBBC042-simple-bgnd_unet-filter_l1smooth_20190713_162625_adam-_lr0.00032_wd0.0_batch32',
                ]
    
    
    data_dir = Path.home() / 'workspace/datasets/BBBC/BBBC042'
    files2check = data_dir.rglob('*.tif')
    
    if set_type == 'val':   
        files2check = [x for x in files2check if (int(x.stem) >= 1000) and (int(x.stem) <= 1049)]
    elif set_type == 'test':
        files2check = [x for x in files2check if int(x.stem) >= 1050]
    else:
        raise ValueError(set_type)
    
    #preload data
    flow = DataFlow(data_dir, files2check)
    loader = DataLoader(flow, batch_size=batch_size, num_workers=0, collate_fn = collate_simple)
    
    
    
    
    for bn in tqdm.tqdm(bn2check):
        subdir = bn.partition('_')[0]
        model_dir = model_path = Path.home() / 'workspace/denoising/results' / subdir / bn
        
        metrics = np.zeros((len(IoU_cutoffs), 3, len(epochs2check), len(thresh2check), len(min_bbox_sizes)))  
        
        for iepoch, n_epoch in enumerate(epochs2check):
            is_colour = 'colour' in bn
            loader.dataset.is_colour = is_colour
            
            if is_colour:
                n_ch_in, n_ch_out  = 3, 3
            else:
                n_ch_in, n_ch_out  = 1, 1
                
            model = UNet(n_channels = n_ch_in, n_classes = n_ch_out)
            
            
            model_path = model_dir /  f'checkpoint-{n_epoch}.pth.tar'
            if not model_path.exists():
                continue
            
            state = torch.load(model_path, map_location = 'cpu')
            epoch = state['epoch']
            
            model.load_state_dict(state['state_dict'])
            model.eval()
            model = model.to(device)
            
            for X, targets in tqdm.tqdm(loader, desc = f'{set_type} {model_path.name} {bn}'):
                with torch.no_grad():
                    X = X.to(device)
                    Xhat = model(X)
                xhat = Xhat.cpu().detach().numpy()
                
                rr = (iepoch, IoU_cutoffs, thresh2check, min_bbox_sizes) 
                dat2process =[(xx, target, rr) for xx, target in zip(xhat, targets)]
                #with mp.Pool(batch_size) as p:
                for out in map(process_out, dat2process):
                    for (icut, iepoch, ith, isize), (TP, FP, FN) in out:
                        metrics[icut, :, iepoch, ith, isize] += (TP, FP, FN)
                    
        results[bn] = metrics
        
        save_name = model_dir / f'scores_otsu_{set_type}.p'
        with open(save_name, 'wb') as fid:
            dat = [metrics,  IoU_cutoffs, epochs2check, thresh2check, min_bbox_sizes]
            pickle.dump(dat, fid)
    
    
    os.system('clear')
    
    for bn, metrics in results.items():
        print('*'*20)
        #fig, axs = plt.subplots(1,2, figsize = (15, 5))
        for metric in metrics:
            TP, FP, FN = metric
            P = TP/(TP+FP)
            R = TP/(TP+FN)
            F1 = 2*P*R/(P+R)
            
            
            iepoch,  ith, isize = np.unravel_index(np.nanargmax(F1), F1.shape)
            F1best = F1[iepoch,  ith, isize]
            Pbest = P[iepoch,  ith, isize]
            Rbest = R[iepoch, ith, isize]
            
            th = thresh2check[ith]
            min_box = min_bbox_sizes[isize]
            epoch = epochs2check[iepoch]
            
            print(f'P {Pbest:.3f} | R {Rbest:.3f} | F1 {F1best:.3f} | epoch{epoch} | th{th:.3f} | min_box{min_box}' )

if __name__ == '__main__':
    import fire
    fire.Fire(main)