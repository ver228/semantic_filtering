#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Aug 17 17:05:49 2018

@author: avelinojaver
"""

import sys
from pathlib import Path 
root_dir = Path(__file__).resolve().parent.parent
sys.path.append(str(root_dir))

from bgnd_removal.trainer import train, get_device, log_dir_root_dflt
from bgnd_removal.flow import MNISTFashionFlow
from bgnd_removal.models import UNet, get_loss

import torch
from torch.utils.data import DataLoader
import tqdm


mnistfashion_params = dict(
             fg_n_range = (0, 5),
             bg_n_range = (1, 25),
             int_range = (0.5, 1.1),
             epoch_size = 10240,
             output_size = 256,
             is_h_flip = True,
             is_v_flip = True,
             max_rotation = 90,
             max_overlap = 0.25
            )
        
        
def test(model, device, criterion, loader_validation, seg_threshold):
    model.eval()
    pbar = tqdm.tqdm(loader_validation, desc = 'Test')
    
    test_avg_loss = 0
    I_all = 0
    U_all = 0
    with torch.no_grad():
        for X, target in pbar:
            X = X.to(device)
            target = target.to(device)
            pred = model(X)
            pred = pred[:, :1] #just keep one dimenssion (useful for the case of separated output)
            
            #calculate average reconstruction loss
            loss = criterion(pred, target)
            test_avg_loss += loss.item()
            
            #calculate mIoU
            pred_bw = pred > seg_threshold
            target_bw = target > seg_threshold
            
            I = (target_bw & pred_bw)
            U = (target_bw | pred_bw)
            
            I_all += I.sum().item()
            U_all += U.sum().item()
            
    mIoU = I_all/U_all
    test_avg_loss /= len(loader_validation)
    
    
    _out = {'test_avg_loss' : test_avg_loss, 'mIoU' : mIoU}
    return _out 
       
def train_fmnist(
                data_type = 'microglia-fluo',
                model_name = 'unet',
                loss_type = 'l1',
                cuda_id = 0,
                log_dir_root = log_dir_root_dflt,
                batch_size = 16,
                num_workers = 1,
                **argkws
                ):
    save_prefix = f'{data_type}_{model_name}_{loss_type}'
    log_dir = log_dir_root / 'fmnist'
    
    n_ch_in, n_ch_out = 1, 1
    
    if data_type == 'fmnist-fg-fix':
        gen = MNISTFashionFlow(is_fix_bg = False, **mnistfashion_params)
        gen.train()
        
    elif data_type == 'fmnist-bg-fix':
        gen = MNISTFashionFlow(is_fix_bg = True, **mnistfashion_params)  
        gen.train()
    
    elif data_type == 'fmnist-clean-out':
        gen = MNISTFashionFlow(is_clean_output = True, **mnistfashion_params)
        gen.train()
    
    elif data_type == 'fmnist-separated-out':
        gen = MNISTFashionFlow(is_separated_output = True, **mnistfashion_params)
        gen.train()
        
        n_ch_out = 2
    else:
        raise ValueError(data_type)
    
    
    model = UNet(n_channels = n_ch_in, n_classes = n_ch_out)
    device = get_device(cuda_id)
    
    criterion = get_loss(loss_type)
    
    def test_func():
        seg_threshold = 0.1
        
        test_params = mnistfashion_params.copy()
        test_params['epoch_size'] = 512
        gen_validation = MNISTFashionFlow(is_clean_output = True, **test_params)
        gen_validation.test()
        
        loader_validation = DataLoader(gen_validation, 
                                       batch_size = batch_size,
                                       num_workers = num_workers)
        
        return test(model, device, criterion, loader_validation, seg_threshold)
    
    
    train(save_prefix,
        model,
        device,
        gen,
        criterion,
        test_func = test_func,
        log_dir = log_dir,
        batch_size = batch_size,
        num_workers = num_workers,
        **argkws
        )
    

if __name__ == '__main__':
    import fire
    
    fire.Fire(train_fmnist)
    