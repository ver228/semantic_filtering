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
from bgnd_removal.flow import FluoMergedFlow
from bgnd_removal.models import UNet, get_loss

def train_fmnist(
                data_type = 'microglia-fluo',
                model_name = 'unet',
                loss_type = 'l1',
                cuda_id = 0,
                log_dir_root = log_dir_root_dflt,
                batch_size = 16,
                num_workers = 1,
                data_root_dir = None,
                init_model_path = None,
                **argkws
                ):
    
    save_prefix = f'{data_type}_{model_name}_{loss_type}'
    log_dir = log_dir_root / 'BBBC042'
    
    n_ch_in, n_ch_out = 1, 1
    is_separated_output = False
    
    
    assert data_type.startswith('BBBC042')
    cells1_prefix = 'foreground'
    cells2_prefix = 'background_crops'
    bgnd_prefix = 'background'
    
    if data_type.endswith('-separated'):
        is_separated_output = True
        n_ch_out = 3
    
    if data_root_dir is None:
        data_root_dir = Path.home() / 'workspace/denoising/data'
    else:
        data_root_dir = Path(data_root_dir)
        
    if '-small' in data_type:
        data_subdir = 'BBBC042_small/train'
    elif '-v3' in data_type:
        data_subdir = 'BBBC042_v3/train'
    else:
        data_subdir = 'BBBC042_v2/train'
    data_root_dir = data_root_dir / data_subdir
    
    gen = FluoMergedFlow(epoch_size = 20480,
                         root_dir = data_root_dir,
                            bgnd_prefix = bgnd_prefix,
                            cells1_prefix = cells1_prefix,
                            cells2_prefix = cells2_prefix,
                            crop_size = (256, 256),
                             is_log_transform = False,
                             int_scale = (0, 255),
                             img_ext = '*.npy',
                             is_timeseries_dir = False,
                             n_cells_per_crop = 5,
                             n_bgnd_per_crop = 10,
                             int_factor = (0.9, 1.1),
                             bgnd_sigma_range = (0., 1.2),
                             frac_crop_valid = 0.2,
                             zoom_range = (0.9, 1.1),
                             noise_range = (0., 10.),
                             rotate_range = (0, 90),
                             max_overlap = 0.1,
                             is_separated_output = is_separated_output
                             )  
    
    model = UNet(n_channels = n_ch_in, n_classes = n_ch_out)
    device = get_device(cuda_id)
    
    criterion = get_loss(loss_type)
    
    
    train(save_prefix,
        model,
        device,
        gen,
        criterion,
        log_dir = log_dir,
        batch_size = batch_size,
        num_workers = num_workers,
        init_model_path = init_model_path,
        **argkws
        )
    

if __name__ == '__main__':
    import fire
    
    fire.Fire(train_fmnist)
    