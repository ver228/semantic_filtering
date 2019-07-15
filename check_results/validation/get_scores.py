#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jul 10 22:13:31 2019

@author: avelinojaver
"""

from pathlib import Path

from train_fasterrcnn import get_scores, get_model, get_device
from flow import BBBC042Dataset, collate_simple

import torch
import numpy as np
import tqdm
import pickle

from torch.utils.data import DataLoader
    
if __name__ == '__main__':

    bn2check_dict = {
            
    'retinanet_adam' :
    [
    'V_BBBC042-max5-noaug-roi(708,990)_retinanet-resnet50_20190710_173955_adam-stepLR-40-0.1_lr0.0001_wd0.0_batch4',
    'V_BBBC042-max10-noaug-roi(708,990)_retinanet-resnet50_20190710_180056_adam-stepLR-40-0.1_lr0.0001_wd0.0_batch4',
    'V_BBBC042-max25-noaug-roi(708,990)_retinanet-resnet50_20190710_181953_adam-stepLR-40-0.1_lr0.0001_wd0.0_batch4',
    'V_BBBC042-max100-noaug-roi(708,990)_retinanet-resnet50_20190710_184320_adam-stepLR-40-0.1_lr0.0001_wd0.0_batch4',
    'V_BBBC042-noaug-roi(708,990)_retinanet-resnet50_20190710_141758_adam-stepLR-40-0.1_lr0.0001_wd0.0_batch4',],
    
     'fasterrcnn_adam' :
    ['V_BBBC042-noaug-roi(708,990)_fasterrcnn-resnet50_20190710_174018_adam-stepLR-40-0.1_lr0.0001_wd0.0_batch6',
    'V_BBBC042-max5-noaug-roi(708,990)_fasterrcnn-resnet50_20190710_223915_adam-stepLR-40-0.1_lr0.0001_wd0.0_batch6',   
    'V_BBBC042-max10-noaug-roi(708,990)_fasterrcnn-resnet50_20190710_234838_adam-stepLR-40-0.1_lr0.0001_wd0.0_batch6',
    'V_BBBC042-max25-noaug-roi(708,990)_fasterrcnn-resnet50_20190711_010323_adam-stepLR-40-0.1_lr0.0001_wd0.0_batch6',
    'V_BBBC042-max100-noaug-roi(708,990)_fasterrcnn-resnet50_20190711_024227_adam-stepLR-40-0.1_lr0.0001_wd0.0_batch6'],

    'fasterrcnn_sgd_augmentation' :
    ['V_BBBC042-roi512_fasterrcnn-resnet50_20190709_164825_sgd-stepLR-250-0.1_lr0.005_wd0.0005_batch16',
    'V_BBBC042-max5-roi512_fasterrcnn-resnet50_20190709_164921_sgd-stepLR-250-0.1_lr0.005_wd0.0005_batch16',
    'V_BBBC042-max10-roi512_fasterrcnn-resnet50_20190709_173214_sgd-stepLR-250-0.1_lr0.005_wd0.0005_batch16',
    'V_BBBC042-max25-roi512_fasterrcnn-resnet50_20190709_181628_sgd-stepLR-250-0.1_lr0.005_wd0.0005_batch16',
    'V_BBBC042-max100-roi512_fasterrcnn-resnet50_20190709_191314_sgd-stepLR-250-0.1_lr0.005_wd0.0005_batch16'],
    
    'fasterrcnn_sgd' :
    ['V_BBBC042-noaug-roi(708,990)_fasterrcnn-resnet50_20190709_165309_sgd-stepLR-250-0.1_lr0.0005_wd0.0005_batch6',
    'V_BBBC042-max5-noaug-roi(708,990)_fasterrcnn-resnet50_20190709_164835_sgd-stepLR-250-0.1_lr0.0005_wd0.0005_batch6',
    'V_BBBC042-max10-noaug-roi(708,990)_fasterrcnn-resnet50_20190709_175036_sgd-stepLR-250-0.1_lr0.0005_wd0.0005_batch6',
    'V_BBBC042-max25-noaug-roi(708,990)_fasterrcnn-resnet50_20190709_185958_sgd-stepLR-250-0.1_lr0.0005_wd0.0005_batch6',
    'V_BBBC042-max100-noaug-roi(708,990)_fasterrcnn-resnet50_20190709_203151_sgd-stepLR-250-0.1_lr0.0005_wd0.0005_batch6'],
    
    'retinanet_sgd' :
    ['V_BBBC042-noaug-roi(708,990)_retinanet-resnet50_20190710_074839_sgd-stepLR-150-0.1_lr0.0005_wd0.0005_batch4',
    'V_BBBC042-max5-noaug-roi(708,990)_retinanet-resnet50_20190709_180307_sgd-stepLR-250-0.1_lr0.0005_wd0.0005_batch4',
    'V_BBBC042-max10-noaug-roi(708,990)_retinanet-resnet50_20190709_192208_sgd-stepLR-250-0.1_lr0.0005_wd0.0005_batch4',
    'V_BBBC042-max25-noaug-roi(708,990)_retinanet-resnet50_20190709_205253_sgd-stepLR-250-0.1_lr0.0005_wd0.0005_batch4',
    'V_BBBC042-max100-noaug-roi(708,990)_retinanet-resnet50_20190709_230249_sgd-stepLR-250-0.1_lr0.0005_wd0.0005_batch4'],
    
    }
    
    _debug = False
    set_type = 'test'#'val'#
    
    batch_size = 6
    cuda_id = 0
    
    device = get_device(cuda_id)
    cpu_device = torch.device("cpu")
    
    IoU_cutoffs =  [0.01, 0.1, 0.25, 0.5]
    
    data_dir = Path.home() / 'workspace/datasets/BBBC/BBBC042'
    
    #preload data
    flow = BBBC042Dataset(data_dir, set_type = set_type)
    loader = DataLoader(flow, batch_size=batch_size, num_workers=0, collate_fn = collate_simple)
    
    results = {}
    for bn_set_name, bn2check in tqdm.tqdm(bn2check_dict.items()):
        results[bn_set_name] = {}
        for bn in tqdm.tqdm(bn2check):
            metrics = np.zeros((len(IoU_cutoffs), 3))  
            
            dd = bn[2:] if bn.startswith('V_') else bn
            model_name, _, backbone_name = dd.split('_')[1].partition('-')
            backbone_name = backbone_name if backbone_name else 'resnet50'
        
            
            model_path = Path.home() / 'workspace/localization/results/bbox_detection' / model_name / bn / 'model_best.pth.tar'
            state = torch.load(model_path, map_location = 'cpu')
            
            img_size = (708, 990)
            model = get_model(model_name, backbone_name, img_size, pretrained_backbone = False)
            model.load_state_dict(state['state_dict'])
            model = model.to(device)
            model.eval()
            
            
            for images, targets in tqdm.tqdm(loader, desc = f'{model_path.name} {bn}'):
                
                images = list(img.to(device) for img in images)
            
                with torch.no_grad():
                    outputs = model(images)
                    outputs = [{k: v.to(cpu_device) for k, v in t.items()} for t in outputs]
                
                
                
                for pred, true in zip(outputs, targets) :
                    pred_bb = pred['boxes'].detach().cpu().numpy()
                    true_bb = true['boxes'].detach().cpu().numpy()
                    for icut, cutoff in enumerate(IoU_cutoffs):
                        
                        TP, FP, FN, pred_ind, true_ind = get_scores(pred_bb, true_bb, cutoff)
                        metrics[icut] += TP, FP, FN
                
                            
            results[bn_set_name][bn] = metrics
        
    save_name = Path.home() / f'workspace/box_detection_BBBC042_{set_type}.p'
    with open(save_name, 'wb') as fid:
        pickle.dump(results, fid)
    
    
#    for bn, metrics in results.items():
#        print('*'*20)
#        print(bn)
#        for icut, (TP, FP, FN) in enumerate(metrics):
#            P = TP/(TP+FP)
#            R = TP/(TP+FN)
#            F1 = 2*P*R/(P+R)
#        
#            cut = IoU_cutoffs[icut]
#            print(f'cutoffs {cut:.2} | R={R:.3} P={P:.3} F1={F1:.3}')