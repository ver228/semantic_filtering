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


from bgnd_removal.models import UNet

import torch
import numpy as np
import cv2

from pathlib import Path
#%%

def data_reader(data_root):
    data_root = Path(data_root)
    
    fnames = list(data_root.rglob('*.tif'))
    fnames = [x for x in fnames if not x.name.startswith('.')]
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

def get_mIoU(model_path, data_root, device):
    
    model = UNet(n_channels = 1, n_classes = 1)
    state = torch.load(model_path, map_location = 'cpu')
    model.load_state_dict(state['state_dict'])
    model = model.to(device)
    
    model.eval()
    th = 20
    gen = data_reader(data_root)
    
    all_IoU = []
    fnames = []
    for fname, X, Y in gen:
        xhat = _process_img(model, X, device)        
        pred_bw = (xhat - X.astype(np.float32)) > th
        target_bw = (Y>0)
        I = (target_bw & pred_bw).sum()
        U = (target_bw | pred_bw).sum()
        
        #if I/U < 0.9: print(fname)
        
        
        all_IoU.append((I, U))
        fnames.append(fname)
        
    return all_IoU, fnames


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
    
    
    models2process = [x.parent.name + '/' + x.name for x in results_root.glob('worms-divergent-samples*/*/')]
    
    
#    models2process = [
#            'worms-divergent/worms-divergent_l1_20190201_011625_unet_adam_lr0.0001_wd0.0_batch36', 
#            'worms-divergent/worms-divergent_l2_20190201_011853_unet_adam_lr0.0001_wd0.0_batch36',
#            'worms-divergent/worms-divergent_l1smooth_20190201_011918_unet_adam_lr0.0001_wd0.0_batch36'
#            ]
    
    
    
    _results = []
    for bn in tqdm.tqdm(models2process):
        model_path = results_root / bn / 'checkpoint.pth.tar'
        
        all_IoU, fnames  = get_mIoU(model_path, data_root, device)
        
        Is, Us = map(np.sum, zip(*all_IoU))
        mIoU = Is/Us
        print(f'{mIoU} :  {bn}')
        
        _results.append((all_IoU, bn, fnames))
        
    
    for all_IoU, bn, fnames in _results:
        Is, Us = map(np.sum, zip(*all_IoU))
        mIoU = Is/Us
        
        
        valid_fnames, IoUs = zip(*[(f, i/u) for f, (i,u) in zip(fnames, all_IoU) if u > 0])
        
        p_IoU = np.percentile(IoUs, [0, 25, 50, 75, 100])
        
        print(f'{mIoU} : {p_IoU} : {bn}')
        
#        print('WORSE::')
#        for ii in np.argsort(IoUs)[:10]:
#            print(IoUs[ii], valid_fnames[ii])
        
    
    
    #%%

#134 vs 160
'''
0.9652567032953968 : [0.90742774 0.95767688 0.96812459 0.97319382 0.98633619] : worms-divergent_l1_20190201_011625_unet_adam_lr0.0001_wd0.0_batch36
0.9555678820710335 : [0.85965898 0.94488393 0.96143169 0.97009082 0.98204755] : worms-divergent_l2_20190201_011853_unet_adam_lr0.0001_wd0.0_batch36
0.956105201643458 : [0.88383278 0.94346642 0.9607318  0.96963697 0.98301795] : worms-divergent_l1smooth_20190201_011918_unet_adam_lr0.0001_wd0.0_batch36
'''