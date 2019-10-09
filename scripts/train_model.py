#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Aug 17 17:05:49 2018

@author: avelinojaver
"""
import warnings
warnings.simplefilter("ignore", UserWarning)

import sys
from pathlib import Path 
root_dir = Path(__file__).resolve().parent.parent
sys.path.append(str(root_dir))

import datetime
import torch

from semantic_filtering.trainer import train, get_device, log_dir_root_dflt
from semantic_filtering.flow import FluoMergedFlow, BasicFlow, data_types_basic, data_types_synthetic
from semantic_filtering.models import UNet, get_loss

def train_model(
                data_type = 'BBBC042-colour',
                model_name = 'unet-filter',
                loss_type = 'l1smooth',
                optimizer_name = 'adam',
                lr = 1e-4, 
                weight_decay = 0.0,
                lr_scheduler_name = '',
                cuda_id = 0,
                log_dir_root = log_dir_root_dflt,
                batch_size = 16,
                num_workers = 1,
                data_root_dir = None,
                is_preloaded = True,
                **argkws
                ):
    
    
    
    if data_type in data_types_synthetic:
    
        dflts = data_types_synthetic[data_type]
        n_ch_in = dflts['n_ch_in']
        n_ch_out = dflts['n_ch_out']
        is_separated_output = False
        if model_name.endswith('-decomposition'):
            is_separated_output = True
            n_ch_out = 3*n_ch_out
        
        gen = FluoMergedFlow(epoch_size = 20480,
                                **dflts['flow_args'],
                                is_separated_output = is_separated_output,
                                is_preloaded = is_preloaded
                                 )  
        
    else:
        dflts = data_types_basic[data_type]
        n_ch_in = dflts['n_ch_in']
        n_ch_out = dflts['n_ch_out']
        
        gen = BasicFlow(**dflts['flow_args'] )  
    
    
    
    
    
    model = UNet(n_channels = n_ch_in, n_classes = n_ch_out)
    device = get_device(cuda_id)
    model = model.to(device)
    criterion = get_loss(loss_type)
    
    model_params = filter(lambda p: p.requires_grad, model.parameters())
    if optimizer_name == 'adam':
        optimizer = torch.optim.Adam(model_params, lr = lr, weight_decay=weight_decay)
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
        print(lr_scheduler)
    else:
        raise ValueError(f'Not implemented {lr_scheduler_name}')
    

    now = datetime.datetime.now()
    date_str = now.strftime('%Y%m%d_%H%M%S')
    save_prefix = f'{data_type}_{model_name}_{loss_type}_{date_str}_{optimizer_name}-{lr_scheduler_name}_lr{lr}_wd{weight_decay}_batch{batch_size}'
    log_dir = log_dir_root / data_type
    
    train(save_prefix,
        model,
        device,
        gen,
        criterion,
        optimizer = optimizer,
        lr_scheduler = lr_scheduler,
        log_dir = log_dir,
        batch_size = batch_size,
        num_workers = num_workers,
        **argkws
        )

if __name__ == '__main__':
    import fire
    fire.Fire(train_model)
    