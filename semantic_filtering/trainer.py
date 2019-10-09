#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Aug 17 16:46:42 2018

@author: avelinojaver
"""
import multiprocessing as mp
mp.set_start_method('spawn', force=True)

from pathlib import Path 

from tensorboardX import SummaryWriter
import torch
from torch.utils.data import DataLoader
import os
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




def get_device(cuda_id):
    if torch.cuda.is_available():
        print("THIS IS CUDA!!!!")
        dev_str = "cuda:" + str(cuda_id)
    else:
        dev_str = 'cpu'
    device = torch.device(dev_str)
    
    return device




def train(save_prefix,
        model,
        device,
        flow,
        criterion,
        optimizer = None,
        lr_scheduler = None,
        test_func = None,
        log_dir = log_dir_root_dflt,
        batch_size = 16,
        n_epochs = 2000,
        num_workers = 1,
        init_model_path = None,
        save_frequency = 200
        ):
    
    
    loader = DataLoader(flow, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    
    epoch_init = 0 #useful to keep track in restarted models
    if init_model_path:
        #load weights
        init_model_path = Path(init_model_path)
        if not init_model_path.exists():
            init_model_path = log_dir / init_model_path
        state = torch.load(str(init_model_path), map_location = str(device))
        model.load_state_dict(state['state_dict'])
        optimizer.load_state_dict(state['optimizer'])
        epoch_init = state['epoch']
        save_prefix = 'R_' + save_prefix
        print('{} loaded...'.format(init_model_path))
    
    
    log_dir = log_dir / save_prefix
    logger = SummaryWriter(log_dir = str(log_dir))
    
    
    best_loss = 1e10
    pbar_epoch = tqdm.trange(n_epochs)
    for epoch in pbar_epoch:
        if lr_scheduler is not None:
            lr_scheduler.step()
        #train
        model.train()
        pbar = tqdm.tqdm(loader, desc = f'{save_prefix} Train')        
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
        
        
        avg_loss = train_avg_loss
        if test_func:
            test_out = test_func()
            
            for name, val in test_out.items():
                logger.add_scalar(name, val, epoch)
                
            #if there is a validation use this as the loss to be printed and to select the model to save as best
            if 'test_avg_loss' in test_out:
                avg_loss = test_out['test_avg_loss']
        
        desc = 'epoch {} , loss={}'.format(epoch, avg_loss)
        pbar_epoch.set_description(desc = desc, refresh=False)
        
        state = {
                'epoch': epoch + epoch_init,
                'state_dict': model.state_dict(),
                'optimizer' : optimizer.state_dict(),
            }
        
        is_best = avg_loss < best_loss
        if is_best:
            best_loss = avg_loss
        save_checkpoint(state, is_best, save_dir = str(log_dir))
        
        if (epoch+1) % save_frequency == 0:
            checkpoint_path = log_dir / f'checkpoint-{epoch}.pth.tar'
            torch.save(state, checkpoint_path)
        
if __name__ == '__main__':
    import fire
    fire.Fire(train)