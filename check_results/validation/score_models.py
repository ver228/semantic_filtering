#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Jul  6 13:24:49 2019

@author: avelinojaver
"""
from flow import BBBC042Dataset, collate_simple
from train_fasterrcnn import get_model, get_scores

import tqdm
import numpy as np
import torch
from pathlib import Path
import time
from torch.utils.data import DataLoader

import pickle
#%%
if __name__ == '__main__':
    
    batch_size = 4
    cuda_id = 0
    
    device = torch.device(cuda_id)
    
    
    #model_path =  / model_name / bn / 'model_best.pth.tar'
    root_dir = Path.home() / 'workspace/localization/results/bbox_detection'
    
    
    
    results = []
    
    model_paths = list(root_dir.rglob('model_best.pth.tar'))
    model_paths = [x for x in model_paths if x.parent.name.startswith('V_')]
    model_paths = [x for x in model_paths if x.parent.name.startswith('V_BBBC042-max5-noaug-roi512_fasterrcnn-resnet50_20190708_201807_sgd-stepLR-250-0.1_lr0.005_wd0.0005_batch16')]
    
    
    #data_dir = Path('/Users/avelinojaver/Downloads/BBBC042/')
    data_dir = Path.home() / 'workspace/datasets/BBBC/BBBC042'
    
    flow_val = BBBC042Dataset(data_dir, set_type = 'val', transforms = None)
    data_loader = DataLoader(flow_val, 
                                  batch_size = batch_size,
                                  num_workers = 1,
                                  collate_fn = collate_simple
                                  )

    for model_path in tqdm.tqdm(model_paths):
        bn = model_path.parent.name
        bn = bn[2:] if bn.startswith('V_') else bn
        
        model_name, _, backbone_name = bn.split('_')[1].partition('-')
        backbone_name = backbone_name if backbone_name else 'resnet50'
        backbone_name = 'resnet50' if backbone_name == 'coco-resnet50' else backbone_name
        
        
        state = torch.load(model_path, map_location = 'cpu')
        
        
        img_size = (708, 990)
        model = get_model(model_name, backbone_name, img_size, pretrained_backbone = False)
        model.load_state_dict(state['state_dict'])
        model.to(device)
        model.eval()
        
        metrics = np.zeros(3)
        model_time_avg = 0
        
        pbar = tqdm.tqdm(data_loader)
        with torch.no_grad():
            for images, targets in pbar:
                images = list(img.to(device) for img in images)
                
                if torch.cuda.is_available():
                    torch.cuda.synchronize()
                model_time = time.time()
                outputs = model(images)
                
                
                outputs = [{k: v.to(device) for k, v in t.items()} for t in outputs]
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
        
        import pdb
        pdb.set_trace()
        
        model_time_avg /= len(data_loader) 
        
        results.append((bn, state['epoch'], P, R, F1))
    
    results = sorted(results, key = lambda x : x[-1])
    
    save_name = Path.home() / 'workspace/BBBC042_fasterrcnn.p'
    with open(save_name, "wb") as fid:
        pickle.dump(results, fid)
        
    
    print('********************')
    for bn, epoch, P, R, F1 in results:
        print(f'{bn} {epoch}')
        print(f'R={R:.3} P={P:.3} F1={F1:.3}')