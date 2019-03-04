#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Aug 17 16:46:42 2018

@author: avelinojaver
"""
import multiprocessing as mp
mp.set_start_method('spawn', force=True)

from pathlib import Path 

from .flow import BasicFlow, FluoMergedFlow, FluoSyntheticFlow, BFFlow, FromTableFlow, MNISTFashionFlow, FromMoviesFlow
from .models import UNet, L0AnnelingLoss, BootstrapedPixL2

from tensorboardX import SummaryWriter
import torch
from torch import nn
from torch.utils.data import DataLoader
import os
import datetime
import shutil
import tqdm
import numpy as np

log_dir_root_dflt = Path.home() / 'workspace/denoising/results/'

def save_checkpoint(state, is_best, save_dir, filename='checkpoint.pth.tar'):
    checkpoint_path = os.path.join(save_dir, filename)
    torch.save(state, checkpoint_path)
    if is_best:
        best_path = os.path.join(save_dir, 'model_best.pth.tar')
        shutil.copyfile(checkpoint_path, best_path)
        
def get_loss(loss_type):
    if loss_type == 'l1':
        criterion = nn.L1Loss()
    elif loss_type == 'l1smooth':
        criterion = nn.SmoothL1Loss()
    elif loss_type == 'l2':
        criterion = nn.MSELoss()
    elif loss_type == 'l0anneling':
        criterion = L0AnnelingLoss(anneling_rate=1/50)
    elif loss_type == 'bootpixl2':
        criterion = BootstrapedPixL2(bootstrap_factor=4)
    
    else:
        raise ValueError(loss_type)
    return criterion

def get_model(model_name):
    if model_name == 'unet':
        model = UNet(n_channels = 1, n_classes = 1)
    elif model_name == 'unet-ch3':
        model = UNet(n_channels = 3, n_classes = 3)
    elif model_name == 'unet-ch4':
        model = UNet(n_channels = 4, n_classes = 4)
    else:
        raise ValueError(model_name)
    return model

def get_flow(data_type, src_root_dir = None):
    def _get_dir(_src_dir):
        if src_root_dir is None:
            return _src_dir
        else:
            return src_root_dir
        
    gen_validation = None
    seg_threshold = 0
    if data_type.startswith('worms-divergent'):
        max_samples_per_set = data_type.partition('worms-divergent-samples-')[-1]
        max_samples_per_set = int(max_samples_per_set) if max_samples_per_set else None
        print('max_samples_per_set', max_samples_per_set)
        
        src_dir = Path.home() / 'workspace/denoising/data/c_elegans_divergent/train/'
        gen = BasicFlow(_get_dir(src_dir), 
                            is_log_transform = False, 
                            scale_int = (0, 255),
                            samples_per_epoch = 2790,
                            max_samples_per_set = max_samples_per_set)
        
    elif data_type == 'bertie-worms':
        src_dir = Path.home() / 'workspace/denoising_data/bertie_c_elegans/'
        gen = FromTableFlow(_get_dir(src_dir), is_log_transform = False, scale_int = (0, 255))
        
        
    elif data_type == 'microglia-fluo':
        gen = FluoMergedFlow()
    
    elif data_type == 'microglia-fluo-v2':
        gen = FluoMergedFlow(fgnd_prefix = 'cell_images_dilated')
    
    elif data_type == 'microglia-fluo-clean':
        gen = FluoMergedFlow(is_return_clean = True)
    
    elif data_type == 'microglia-fluo-v2-clean':
        gen = FluoMergedFlow(is_return_clean = True,
                             fgnd_prefix = 'cell_images_dilated')
    
    elif data_type.startswith('BBBC026'):
        
        cell_type = data_type.split('-')[1]
        if cell_type == 'fibroblasts':
            root_dir = Path.home() / 'workspace/denoising/data/BBBC026/fibroblasts/train/'
        elif cell_type == 'hepatocytes':
            root_dir = Path.home() / 'workspace/denoising/data/BBBC026/hepatocytes/train/'
        else:
            raise ValueError(data_type)
        
        
        if data_type.endswith('-log'):
            is_log_transform = True
            int_scale = (0, np.log(2**8))
        else:
            is_log_transform = False
            int_scale = (0, 255)
        
        
        gen = FluoMergedFlow(root_dir = root_dir,
                             crop_size = (256, 256),
                             is_log_transform = is_log_transform,
                             int_scale = int_scale,
                             fgnd_prefix = 'foreground',
                             bgnd_prefix = 'background',
                             img_ext = '*.png',
                             is_timeseries_dir = False,
                             n_cells_per_crop = 10,
                             int_factor = (0.5, 2.),
                             bgnd_sigma_range = (0., 1.2),
                             frac_crop_valid = 0.8,
                             zoom_range = (0.75, 1.25),
                             noise_range = (0., 10.),
                             rotate_range = (0, 90),
                             is_return_clean = False
                             )
    
    elif data_type.startswith('sBBBC026'):
    
        root_dir = Path.home() / 'workspace/denoising/data/BBBC026/'
        
        cell_type = data_type.split('-')[1]
        if cell_type == 'fibroblasts':
            bgnd_prefix = 'hepatocytes/train/foreground'
            fgnd_prefix = 'fibroblasts/train/foreground'
            
            
        elif cell_type == 'hepatocytes':
            bgnd_prefix = 'fibroblasts/train/foreground'
            fgnd_prefix = 'hepatocytes/train/foreground'
            
        else:
            raise ValueError(data_type)
        
        
        if data_type.endswith('-log'):
            is_log_transform = True
            int_scale = (0, np.log(2**8))
        else:
            is_log_transform = False
            int_scale = (0, 255)
            
        
                     
        
        gen = FluoSyntheticFlow(epoch_size = 20480,
                                root_dir = root_dir,
                                bgnd_prefix = bgnd_prefix,
                                fgnd_prefix = fgnd_prefix,
                                 crop_size = (256, 256),
                                 is_log_transform = is_log_transform,
                                 int_scale = int_scale,
                                 img_ext = '*.png',
                                 is_timeseries_dir = False,
                                 n_cells_per_crop = 10,
                                 int_factor = (0.5, 2.),
                                 bgnd_sigma_range = (0., 1.2),
                                 frac_crop_valid = 0.8,
                                 zoom_range = (0.75, 1.25),
                                 noise_range = (0., 10.),
                                 rotate_range = (0, 90),
                                 bngd_base_range = (10, 40),
                                 is_return_clean = False
                                 )  
    
    elif data_type == 'BBBC042':
        root_dir = Path.home() / 'workspace/denoising/data/BBBC042/cell_bgnd_divided_v2/train/'
        
        gen = BFFlow(epoch_size = 20480,
                     root_dir = root_dir,
                     crop_size = (256, 256),
                         is_log_transform = False,
                         int_scale = (0, 255),
                         fgnd_prefix = 'foreground',
                         bgnd_prefix = 'background',
                         img_ext = '*.tif',
                         is_timeseries_dir = False,
                         n_cells_per_crop = 3,
                         int_factor = (1., 1.),
                         bgnd_sigma_range = (0., 1.),
                         merge_type = 'replace',
                         frac_crop_valid = 0.,
                         is_return_clean = False,
                         noise_range = (0., 10.)
                         ) 
        
    elif data_type == 'nanoscopy-vesicles':
        root_dir = Path.home() / 'workspace/denoising/data/vesicles_nanoscopy/50min_bio_10fps_Airyscan_ProcessingSetting3-3/'
        gen = FluoMergedFlow(root_dir = root_dir,
                             epoch_size = 20480,
                             bgnd_path_size = (128, 128),
                             is_log_transform = False,
                             fgnd_prefix = 'foreground',
                             bgnd_prefix = 'background',
                             img_ext = '*.tif',
                             is_timeseries_dir = False,
                             n_cells_per_crop = 16,
                             int_scale = (0, 2**16-1),
                             bgnd_sigma_range = (0., 1.)
                             )
    
    elif data_type == 'nanoscopy-vesicles-log':
        root_dir = Path.home() / 'workspace/denoising/data/vesicles_nanoscopy/50min_bio_10fps_Airyscan_ProcessingSetting3-3/'
        gen = FluoMergedFlow(root_dir = root_dir,
                             epoch_size = 20480,
                             bgnd_path_size = (128, 128),
                             fgnd_prefix = 'foreground',
                             bgnd_prefix = 'background',
                             img_ext = '*.tif',
                             is_timeseries_dir = False,
                             n_cells_per_crop = 16,
                             is_log_transform = True,
                             int_scale = (0, np.log(2**16)),
                             bgnd_sigma_range = (0., 1.)
                             )
    
    elif data_type.startswith('fmnist'):
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
        
        
        test_params = mnistfashion_params.copy()
        test_params['epoch_size'] = 512
        gen_validation = MNISTFashionFlow(is_clean_output = True, **test_params)
        gen_validation.test()
        seg_threshold = 0.1
        
        if data_type == 'fmnist-fg-fix':
            gen = MNISTFashionFlow(is_fix_bg = False, **mnistfashion_params)
            gen.train()
            
        elif data_type == 'fmnist-bg-fix':
            gen = MNISTFashionFlow(is_fix_bg = True, **mnistfashion_params)  
            gen.train()
        
        elif data_type == 'fmnist-clean-out':
            gen = MNISTFashionFlow(is_clean_output = True, **mnistfashion_params)
            gen.train()
        
        else:
            raise ValueError(data_type)
    
    elif data_type.startswith('from-movies'):
        frame_gap = data_type.partition('-gap-')[-1]
        frame_gap = int(frame_gap) if frame_gap else None
        gen = FromMoviesFlow(frame_gap = frame_gap)
    
    elif data_type.startswith('toulouse'):
        root_dir = Path.home() / 'workspace/denoising/data/ToulouseCampusSurveillanceDataset/train'
        frame_gap = data_type.partition('-gap-')[-1]
        frame_gap = int(frame_gap) if frame_gap else None
        gen = FromMoviesFlow(frame_gap = frame_gap)
    
    else:
        raise ValueError(data_type)
    
    return gen, gen_validation, seg_threshold

def train(
        data_type = 'microglia-fluo',
        loss_type = 'l1',
        log_dir_root = log_dir_root_dflt,
        cuda_id = 0,
        batch_size = 16,
        model_name = 'unet',
        lr = 1e-4, 
        weight_decay = 0.0,
        n_epochs = 2000,
        num_workers = 1,
        is_to_align = False,
        data_src_dir = None,
        init_model_path = None,
        save_frequency = 200
        ):
    
    log_dir_root = Path(log_dir_root) / data_type
    
    
    if torch.cuda.is_available():
        print("THIS IS CUDA!!!!")
        dev_str = "cuda:" + str(cuda_id)
    else:
        dev_str = 'cpu'
    device = torch.device(dev_str)
    
    gen, gen_validation, seg_threshold = get_flow(data_type, data_src_dir)
    
    loader = DataLoader(gen, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    
    loader_validation = None
    if gen_validation is not None:
        loader_validation = DataLoader(gen_validation, 
                                       batch_size=batch_size, 
                                       shuffle=True, 
                                       num_workers=num_workers)
    
    
    
    model = get_model(model_name)
    model = model.to(device)
    
    criterion = get_loss(loss_type)
    
    model_params = filter(lambda p: p.requires_grad, model.parameters())
    optimizer = torch.optim.Adam(model_params, lr = lr, weight_decay=weight_decay)
    
    now = datetime.datetime.now()
    bn = now.strftime('%Y%m%d_%H%M%S') + '_' + model_name
    bn = '{}_{}_{}_{}_lr{}_wd{}_batch{}'.format(data_type, loss_type, bn, 'adam', lr, weight_decay, batch_size)

    epoch_init = 0 #useful to keep track in restarted models
    if init_model_path:
        #load weights
        init_model_path = Path(init_model_path)
        if not init_model_path.exists():
            init_model_path = log_dir_root / init_model_path
        state = torch.load(str(init_model_path), map_location = dev_str)
        model.load_state_dict(state['state_dict'])
        optimizer.load_state_dict(state['optimizer'])
        epoch_init = state['epoch']
        
        bn = 'R_' + bn
        print('{} loaded...'.format(init_model_path))

    log_dir = log_dir_root / bn
    logger = SummaryWriter(log_dir = str(log_dir))
    
    
    best_loss = 1e10
    pbar_epoch = tqdm.trange(n_epochs)
    for epoch in pbar_epoch:
        
        #train
        model.train()
        pbar = tqdm.tqdm(loader, desc = 'Train')        
        train_avg_loss = 0
        for X, target in pbar:
            assert not np.isnan(X).any()
            assert not np.isnan(target).any()
            
            X = X.to(device)
            target = target.to(device)
            pred = model(X)
            
            loss = criterion(pred, target)
            
            optimizer.zero_grad()               # clear gradients for this training step
            loss.backward()                     # backpropagation, compute gradients
            optimizer.step() 
        
            train_avg_loss += loss.item()
            
            
        
        train_avg_loss /= len(loader)
        logger.add_scalar('train_epoch_loss', train_avg_loss, epoch)
        
        
        if loader_validation is None:
            avg_loss = train_avg_loss
        
        else:
            model.train()
            pbar = tqdm.tqdm(loader_validation, desc = 'Test')
            
            test_avg_loss = 0
            I_all = 0
            U_all = 0
            with torch.no_grad():
                for X, target in pbar:
                    X = X.to(device)
                    target = target.to(device)
                    pred = model(X)
                    
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
            test_avg_loss /= len(loader)
            
            logger.add_scalar('test_avg_loss', test_avg_loss, epoch)
            logger.add_scalar('mIoU', mIoU, epoch)
            
            #if there is a validation use this as the loss to be printed and to select the model to save as best
            avg_loss = test_avg_loss
        
        desc = 'epoch {} , loss={}'.format(epoch, avg_loss)
        pbar_epoch.set_description(desc = desc, refresh=False)
        
        state = {
                'epoch': epoch + epoch_init,
                'state_dict': model.state_dict(),
                'optimizer' : optimizer.state_dict(),
            }
        
        is_best = avg_loss < best_loss
        save_checkpoint(state, is_best, save_dir = str(log_dir))
        
        if (epoch+1) % save_frequency == 0:
            checkpoint_path = log_dir / f'checkpoint-{epoch}.pth.tar'
            torch.save(state, checkpoint_path)
        
if __name__ == '__main__':
    import fire
    fire.Fire(train)