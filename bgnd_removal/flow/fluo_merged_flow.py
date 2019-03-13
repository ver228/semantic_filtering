#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Sep 27 11:12:10 2018

@author: avelinojaver
"""
from pathlib import Path
import random
import cv2
import time
import numpy as np
from torch.utils.data import Dataset 

#_root_dir = Path.home() / 'OneDrive - Nexus365/microglia/data/cell_bgnd_divided/train'
#_root_dir = Path.home() / 'workspace/denoising_data/microglia/syntetic_data'
_root_dir = Path.home() / 'workspace/denoising/data/microglia/cell_bgnd_divided/train/'



def rotate_bound(image, angle):
    #https://www.pyimagesearch.com/2017/01/02/rotate-images-correctly-with-opencv-and-python/
    # grab the dimensions of the image and then determine the
    # center
    (h, w) = image.shape[:2]
    (cX, cY) = (w // 2, h // 2)
 
    # grab the rotation matrix (applying the negative of the
    # angle to rotate clockwise), then grab the sine and cosine
    # (i.e., the rotation components of the matrix)
    M = cv2.getRotationMatrix2D((cX, cY), -angle, 1.0)
    cos = np.abs(M[0, 0])
    sin = np.abs(M[0, 1])
 
    # compute the new bounding dimensions of the image
    nW = int((h * sin) + (w * cos))
    nH = int((h * cos) + (w * sin))
 
    # adjust the rotation matrix to take into account translation
    M[0, 2] += (nW / 2) - cX
    M[1, 2] += (nH / 2) - cY
 
    # perform the actual rotation and return the image
    return cv2.warpAffine(image, M, (nW, nH))


class OverlapsTracker():
    def __init__(self, image_shape, max_overlap):
        
        self.max_overlap = max_overlap
        self.overlap_mask = np.zeros(image_shape, np.int32)
        self.bbox_placed = []
        self.crops_placed = []
        
        
    def add(self, xi, yi, crop):
        crop_size = crop.shape
        crop_bw = crop>0
        
        xf = xi + crop_size[0]
        yf = yi + crop_size[1]
        
        rr = self.overlap_mask[xi:xf, yi:yf]
        overlap_frac = np.mean(rr>0)
        
        #check the fraction of pixels on the new data that will be cover by the previous data
        if overlap_frac > self.max_overlap:
            return False
        
        #check the fraction of pixels in each of the previous crops that will be cover with the new data
        bbox = (xi, yi, xf, yf)
        if len(self.bbox_placed):
            overlaps_frac, intersect_coords = self.bbox_overlaps(bbox, self.bbox_placed)
            bad_ = overlaps_frac > self.max_overlap
            if np.any(bad_):
                #i will refine this estimate at the pixel level rather than to the bbox level
                inds2check, = np.where(bad_)
                for ind in inds2check:
                    crop_placed = self.crops_placed[ind]
                    original_area = crop_placed.sum()
                    if original_area == 0:
                        continue
                    
                    bbox_placed = self.bbox_placed[ind]
                    bbox_intersect= intersect_coords[ind]
                    
                    x_bi, x_bf = bbox_intersect[[0, 2]] - bbox_placed[0]
                    assert x_bi >= 0
                    y_bi, y_bf = bbox_intersect[[1, 3]]  - bbox_placed[1]
                    assert y_bi >= 0
                    
                    x_ci, x_cf = bbox_intersect[[0, 2]] - bbox[0]
                    assert x_ci >= 0
                    y_ci, y_cf = bbox_intersect[[1, 3]]  - bbox[1]
                    assert y_ci >= 0
                    
                    
                    prev_area = crop_placed[x_bi:x_bf, y_bi:y_bf]
                    next_area = crop_bw[x_ci:x_cf, y_ci:y_cf]
                    
                    
                    intesected_area = (prev_area & next_area).sum()
                    pix_overlap_frac = intesected_area/original_area
                    
                    assert pix_overlap_frac <= 1.
                        
                    #print(original_area, intesercted_area, pix_overlap_frac)
                    if pix_overlap_frac > self.max_overlap: 
                        return False
                    
        
        
        self.bbox_placed.append(bbox)
        self.crops_placed.append(crop_bw)
        
        curr_ind = len(self.bbox_placed)
        self.overlap_mask[xi:xf, yi:yf][crop_bw] = curr_ind
        
        
        return True
    
    def bbox_overlaps(self, bbox_new, bbox_placed):
        """
        get bonding max overlaps
        bbox_new: [x1, y1, x2, y2]
        bbox_placed: [[x1, y1, x2, y2]]
        """
        
        bbox_new = np.array(bbox_new)
        bbox_placed  = np.array(bbox_placed)
        
        areas = (bbox_placed[:, 2] - bbox_placed[:, 0] + 1) * (bbox_placed[:, 3] - bbox_placed[:, 1] + 1)
        
        
        xx1 = np.maximum(bbox_new[0], bbox_placed[:, 0])
        yy1 = np.maximum(bbox_new[1], bbox_placed[:, 1])
        xx2 = np.minimum(bbox_new[2], bbox_placed[:, 2])
        yy2 = np.minimum(bbox_new[3], bbox_placed[:, 3])
            
        w = np.maximum(0.0, xx2 - xx1 + 1)
        h = np.maximum(0.0, yy2 - yy1 + 1)
        inter = w * h
        
        overlaps = inter/areas
        
        intersect_coords = np.stack((xx1, yy1, xx2, yy2)).T
        
        return overlaps, intersect_coords
            
        
class FluoMergedFlow(Dataset):
    def __init__(self, 
                 root_dir = _root_dir,
                 crop_size = (512, 512),
                 n_cells_per_crop = 4,
                 n_bgnd_per_crop = None,
                 int_factor = (0.1, 3.0),
                 epoch_size = 2000,
                 is_log_transform = True,
                 int_scale = (0, np.log(2**16)),
                 is_clean_output = False,
                 is_separated_output = False,
                 
                 cells1_prefix = 'cell_images',
                 cells2_prefix = None,
                 bgnd_prefix = None,
                 
                 img_ext = '*.tif',
                 is_timeseries_dir = True,
                 bgnd_sigma_range = (0., 3.),
                 bgnd_mu_range = (-0.7, 0.7),
                 merge_type = 'add',
                 frac_crop_valid = 0.9,
                 zoom_range = None,
                 noise_range = None,
                 rotate_range = None,
                 int_base_q_range = (0, 10),
                 max_overlap = 1.
                 ):
        
        root_dir = Path(root_dir)
        self.root_dir = root_dir
        self.img_ext = img_ext
        self.crop_size = crop_size
        self.n_cells_per_crop = n_cells_per_crop
        self.n_bgnd_per_crop = n_bgnd_per_crop
        self.int_factor = int_factor
        self.frac_crop_valid = frac_crop_valid
        self.is_log_transform = is_log_transform
        self.epoch_size = epoch_size
        self.int_scale = int_scale
        self.is_clean_output = is_clean_output
        self.is_separated_output = is_separated_output
        
        
        self.bgnd_sigma_range = bgnd_sigma_range
        self.bgnd_mu_range = bgnd_mu_range
        self.zoom_range = zoom_range
        self.noise_range = noise_range
        self.rotate_range = rotate_range
        self.int_base_q_range = int_base_q_range
        self.max_overlap = max_overlap
    
    
        if self.n_bgnd_per_crop is None:
            self.n_bgnd_per_crop = n_cells_per_crop
    
        def _extract_files(prefix):
            if not prefix:
                return []
                
            dname = root_dir / prefix
            print(dname)
            assert dname.exists()
            fnames = [x for x in dname.rglob(img_ext) if not x.name.startswith('.')]
            return fnames
            
        
        cells1_files = _extract_files(cells1_prefix)
        cells2_files = _extract_files(cells2_prefix)
        bgnd_files = _extract_files(bgnd_prefix)
        
        if is_timeseries_dir:
            dd = str(root_dir / cells1_prefix) + '/'
            cells1_files_d = {}
            for x in cells1_files:
                path_r = str(x.parent).replace(dd, '')
                
                if not path_r in cells1_files_d:
                    cells1_files_d[path_r] = []
                
                cells1_files_d[path_r].append(x)
            
            for path_r in cells1_files_d:
                cells1_files_d[path_r] =  sorted(cells1_files_d[path_r])
            
            cells1_files_d = list(cells1_files_d.values())
            
        else:
            cells1_files_d = [[x] for x in cells1_files]
        
        
        self.cells1_files_d = cells1_files_d
        self.cells2_files_d = [[x] for x in cells2_files]
        self.bgnd_files_d = bgnd_files
        
    
    
    def read_file(self, fname, MAX_N_TRIES = 10):
        #try to read file MAX_N_TRIES. Let's consider an unstable connection.
        for _ in range(MAX_N_TRIES):
            
            if fname.suffix == '.npy':
                img = np.load(str(fname))
            else:    
                img = cv2.imread(str(fname), -1)
            if img is not None:
                break
            time.sleep(0.1)
        else:
            raise IOError(f'Cannot read background file: {fname}')
            
        return img
    
    
    def _get_random_bgnd(self):
        if not self.bgnd_files_d:
            return np.zeros(self.crop_size, np.float32)
        
        bgnd_file = random.choice(self.bgnd_files_d)
        img_bgnd = self.read_file(bgnd_file)
        
        
        xi = random.randint(0, img_bgnd.shape[0] - self.crop_size[0])
        yi = random.randint(0, img_bgnd.shape[1] - self.crop_size[1])
        
        crop_bgnd = img_bgnd[xi:xi + self.crop_size[0], yi:yi + self.crop_size[1]]
        crop_bgnd = crop_bgnd.astype(np.float32)
        
        mask_b = crop_bgnd == 0
        if np.any(mask_b) and not np.all(mask_b):
            mask_ = crop_bgnd > 0
            valid_l = crop_bgnd[mask_]
            
            if self.is_log_transform:
                valid_l = np.log(valid_l)
                
            med = np.median(valid_l)
            mad = np.median(np.abs(valid_l - med))
            
            assert (med == med) and (mad == mad)
            
            sigma_factor = random.uniform(*self.bgnd_sigma_range)
            mu_factor = random.uniform(*self.bgnd_mu_range)
            
            bb = crop_bgnd <= 1

            ss = max(0., sigma_factor*mad)
            noise = np.random.normal(mu_factor + med, ss, bb.sum())
            
            if self.is_log_transform:
                noise = np.exp(noise)
            
            crop_bgnd[bb] = noise
        
        if self.rotate_range:
            angle = random.uniform(*self.rotate_range)
            #crop_bgnd = rotate_bound(crop_bgnd, _angle)
            (cX, cY) = (crop_bgnd.shape[0] // 2, crop_bgnd.shape[1] // 2)
            M = cv2.getRotationMatrix2D((cX, cY), -angle, 1.0)
            crop_bgnd = cv2.warpAffine(crop_bgnd, M, crop_bgnd.shape, borderMode = cv2.BORDER_REFLECT_101)
        
        

        #random flips
        if random.random() >= 0.5:
            crop_bgnd = crop_bgnd[::-1]
        if random.random() >= 0.5:
            crop_bgnd = crop_bgnd[:, ::-1]
        
        _int_fac = random.uniform(0.8, 1.3)
        crop_bgnd *= _int_fac
        
        return crop_bgnd
    
    def _scale(self, x):
        x[x<0] = 0
        if self.is_log_transform:
            x = np.log(x+1)
        x = (x-self.int_scale[0])/(self.int_scale[1]-self.int_scale[0])
        return x.astype(np.float32)
    
    def _sample(self):
        fngd_p1, fgnd_p2, overlap_tracker = self.get_cell_pairs(self.cells1_files_d, self.n_cells_per_crop)
        bgnd1_p1 = self._get_random_bgnd()
        
        bgnd2_p1 = self.get_cell_pairs(self.cells2_files_d, self.n_bgnd_per_crop, overlap_tracker)[0]
        
        out1 = self._merge(fngd_p1, bgnd1_p1, bgnd2_p1)
        out1 = self._add_noise(out1)
        out1 = self._scale(out1)
       
        
        if self.is_clean_output:
            base_int = np.mean(bgnd1_p1) 
            out2 = self._scale(base_int + fgnd_p2)
        elif self.is_separated_output:
            out2 = [fngd_p1, bgnd1_p1, bgnd2_p1]
            out2 = [self._scale(x) for x in out2]
            out2 = np.stack(out2)
            
        else:
            bgnd1_p2 = self._get_random_bgnd()
            bgnd2_p2 = self.get_cell_pairs(self.cells2_files_d, self.n_bgnd_per_crop, overlap_tracker)[0]
        
            out2 = self._merge(fgnd_p2, bgnd1_p2, bgnd2_p2)
            out2 = self._add_noise(out2)
            out2 = self._scale(out2)
            
        
        return out1, out2
    
    def _add_noise(self, img):
        if self.noise_range:
            _sigma = random.uniform(*self.noise_range)
            
            noise = np.random.normal(0, _sigma, img.shape)
            
        return img + noise
    
    def _merge(self, fgnd, bgnd1, bgnd2):
        _out = bgnd1 + bgnd2 + fgnd
        #here i am assuming i haven't done any scaling, and data should be int
        _out[_out<0] = 0
        return _out
            
    
    def get_cell_pairs(self, src_dirs, max_n_images, overlap_tracker = None):
        N = random.randint(0, max_n_images)
        raw_cell_imgs = self._read_random_imgs(src_dirs, N)
        cells_p1, cells_p2, overlap_tracker = self.rawimgs2pairs(raw_cell_imgs, overlap_tracker)
        return cells_p1, cells_p2, overlap_tracker
        
    
    def rawimgs2pairs(self, raw_cell_imgs, overlap_tracker = None):
        p_cell_imgs = []
        
        if overlap_tracker is None:
            overlap_tracker = OverlapsTracker(self.crop_size, self.max_overlap)
            
        
        for _imgs in raw_cell_imgs:
            augmented_imgs = self._random_augment_imgs(_imgs)
            _out = self._random_locate_imgs(augmented_imgs, overlap_tracker)
            
            if not _out:
                continue
            
            coords, located_imgs, overlap_mask = _out
            
            #form a pair if it does not already exists.
            if len(located_imgs) == 1:
                _pair = (located_imgs[0], located_imgs[0])
            else:
                _pair = located_imgs[0], located_imgs[1]
            
            #shift order (either before/after or after/before)
            if random.random() >= 0.5:
                _pair = _pair[::-1]
            
            p_cell_imgs.append((coords, _pair))
        
        #rearrange data so the first/second elements of a given pair are together
        if len(p_cell_imgs) > 0:
            cells_p1, cells_p2 = zip(*[((c, p[0]), (c, p[1])) for c,p in p_cell_imgs])
        else:
            cells_p1, cells_p2 = [], []
        
        #synthetize images form the pairs
        cells_p1 = self._cellimgs2crop(cells_p1)         
        cells_p2 = self._cellimgs2crop(cells_p2)     
            
        return cells_p1, cells_p2, overlap_tracker
    
    def _read_random_imgs(self, cell_crops_d, n_crops):
        random_images = []
        for _ in range(n_crops):
            fnames = random.choice(cell_crops_d)
            
            ind = random.randint(0, max(0, len(fnames)-2))
            ind_next = min(len(fnames)-1, ind+1)
            
            x1 = self.read_file(fnames[ind])
            if ind == ind_next:
                #I am reading the same file
                _imgs = x1,
            else:
                x2 = self.read_file(fnames[ind_next])
                _imgs = (x1, x2)
                
            random_images.append(_imgs)
        return random_images
    
    
    def _random_augment_imgs(self, _raw_imgs):
        '''
        Randomly exectute the same transforms for each tuple the list.
        '''
        _flipv = random.random() >= 0.5
        _fliph = random.random() >= 0.5
        
        
        _zoom = random.uniform(*self.zoom_range) if self.zoom_range else None
        _angle = random.uniform(*self.rotate_range) if self.rotate_range else None
        
        if self.is_log_transform:
            _int_fac = np.exp(random.uniform(*np.log(self.int_factor)))
        else:
            _int_fac = random.uniform(*self.int_factor)
            
        _base_quantile = random.uniform(*self.int_base_q_range)
    
        def _augment(cc):
            cc = cc.astype(np.float32)
            
            good = cc>1
            val = cc[good]
            if val.size > 0:
                base_int = np.percentile(val, _base_quantile)
            else:
                base_int = 0
            cc[good] -= base_int
            
            #random rotation
            if _angle:
                cc = rotate_bound(cc, _angle)
            
            
            #random flips
            if _fliph:
                cc = cc[::-1]
            if _flipv:
                cc = cc[:, ::-1]
            
            #zoom
            if _zoom:
                cc = cv2.resize(cc, (0,0), fx=_zoom, fy=_zoom, interpolation = cv2.INTER_LINEAR)
            
            cc = cc[:self.crop_size[0], :self.crop_size[1]] #crop in case it is larger than the crop
            cc *= _int_fac
            return cc
        
        new_imgs = [_augment(x) for x in _raw_imgs]
        
        
        
        return new_imgs
    
    def _random_locate_imgs(self, _imgs, overlap_tracker):
        #randomly located a pair in the final image. If part of the pair is 
        #located outside the final image, the the pair is cropped.        
        img_shape = _imgs[0].shape
            
        #crop if the x,y coordinate is outside the expected image 
        frac_cc = [int(round(x*self.frac_crop_valid)) for x in img_shape]
        
        
        max_ind_x = self.crop_size[0] - img_shape[0]
        max_ind_y = self.crop_size[1] - img_shape[1]
        
        BREAK_NUM = 20 #number of trials to add an image before giving up
        
        for _ in range(BREAK_NUM):
            xi = random.randint(-frac_cc[0], max_ind_x + frac_cc[0])
            yi = random.randint(-frac_cc[1], max_ind_y + frac_cc[1])
            
            
            _imgs_n = _imgs
            if xi < 0:
                _imgs_n = [cc[abs(xi):] for cc in _imgs_n]
                xi = 0
            
            if yi < 0:
                _imgs_n = [cc[:, abs(yi):] for cc in _imgs_n]
                yi = 0
            
            
            if xi > max_ind_x:
                ii = max_ind_x-xi
                _imgs_n = [cc[:ii] for cc in _imgs_n]
            
            if yi > max_ind_y:
                ii = max_ind_y - yi
                _imgs_n = [cc[:, :ii] for cc in _imgs_n]
            
            
            if not overlap_tracker.add(xi, yi, _imgs_n[0]):
                continue
            
            return (xi, yi), _imgs_n, overlap_tracker
        else:
            return
    
    
    
    def _cellimgs2crop(self, cell_imgs):
        _out = np.zeros(self.crop_size, dtype = np.float32)
        for (xi,yi), cc in cell_imgs:
            _out[xi:xi+cc.shape[0], yi:yi+cc.shape[1]] += cc
        return _out
        
    
    def __len__(self):
        return self.epoch_size
    
    
    def __getitem__(self, ind):
        
        return [x[None] if x.ndim == 2 else x for x in self._sample()]
    
    def __iter__(self):
        self.n = 0
        return self

    def __next__(self):
        if self.n <= len(self):
             
            self.n += 1
            return self[self.n-1]
            
        else:
            raise StopIteration
        

    
#%%
if __name__ == '__main__':
    from torch.utils.data import DataLoader
    import matplotlib.pylab as plt
    import tqdm

    #%%
    #root_dir = '/Volumes/loco/workspace/denoising/data/BBBC026/'
    #root_dir = Path.home() / 'workspace/denoising/data/BBBC026_v2/'
#    root_dir = '/Users/avelinojaver/Desktop/BBBC026_divided/'
#    
#    cells1_prefix = 'hepatocytes'
#    cells2_prefix = 'fibroblasts'
#    bgnd_prefix = 'background'
#    
#    gen = FluoMergedFlow(root_dir = root_dir,
#                            bgnd_prefix = bgnd_prefix,
#                            cells1_prefix = cells1_prefix,
#                            cells2_prefix = cells2_prefix,
#                            crop_size = (256, 256),
#                             is_log_transform = False,
#                             int_scale = (0, 255),
#                             img_ext = '*.png',
#                             is_timeseries_dir = False,
#                             n_cells_per_crop = 10,
#                             int_factor = (0.5, 1.25),
#                             bgnd_sigma_range = (0., 1.2),
#                             frac_crop_valid = 0.5,
#                             zoom_range = (0.75, 1.3),
#                             noise_range = (0., 10.),
#                             rotate_range = (0, 90),
#                             max_overlap = 0.9,
#                             is_separated_output = True
#                             )  
    
    #%%
    #root_dir = Path.home() / 'workspace/denoising/data/BBBC042_small/train'
#    root_dir = Path.home() / 'Desktop/BBBC042_divided/train'
#    cells1_prefix = 'foreground'
#    cells2_prefix = 'background_crops'
#    bgnd_prefix = 'background'
#    
#    gen = FluoMergedFlow(root_dir = root_dir,
#                            bgnd_prefix = bgnd_prefix,
#                            cells1_prefix = cells1_prefix,
#                            cells2_prefix = cells2_prefix,
#                            crop_size = (256, 256),
#                             is_log_transform = False,
#                             int_scale = (0, 255),
#                             img_ext = '*.tif',
#                             is_timeseries_dir = False,
#                             n_cells_per_crop = 5,
#                             n_bgnd_per_crop = 10,
#                             int_factor = (0.9, 1.1),
#                             bgnd_sigma_range = (0., 1.2),
#                             frac_crop_valid = 0.2,
#                             zoom_range = (0.9, 1.1),
#                             noise_range = (0., 10.),
#                             rotate_range = (0, 90),
#                             max_overlap = 0.1,
#                             is_separated_output = True
#                             )  
    #for _ in tqdm.tqdm(range(10)):
    #    x,y = gen._sample()
#%%
#    #root_dir = Path.home() / 'workspace/denoising/data/microglia_v2'
#    #root_dir = Path.home() / 'workspace/denoising/data/microglia_v2_tight'
#    
#    root_dir = Path.home() / 'Desktop/microglia'
#    #root_dir = Path.home() / 'Desktop/microglia_tight'
#    
#    cells1_prefix = 'foreground'
#    cells2_prefix = 'background_crops'
#    bgnd_prefix = 'background'
#    
#    gen = FluoMergedFlow(root_dir = root_dir,
#                            bgnd_prefix = bgnd_prefix,
#                            cells1_prefix = cells1_prefix,
#                            cells2_prefix = cells2_prefix,
#                            crop_size = (512, 512),
#                             is_log_transform = True,
#                             int_scale = (0, np.log(2**16)),
#                             img_ext = '*.tif',
#                             is_timeseries_dir = False,
#                             n_cells_per_crop = 4,
#                             n_bgnd_per_crop = 10,
#                             int_factor = (0.1, 3.0),
#                             bgnd_sigma_range = (0., 3.),
#                             bgnd_mu_range = (-0.7, 0.7),
#                             frac_crop_valid = 0.9,
#                             zoom_range = (0.9, 1.1),
#                             noise_range = (0., 10.),
#                             rotate_range = (0, 90),
#                             max_overlap = 1.,
#                             is_separated_output = True,
#                             epoch_size = 500,
#                             int_base_q_range = (0, 10)
#                             )  
    
#%%
    root_dir = Path.home() / 'Desktop/MNIST_fashion/train'
    #root_dir = Path.home() / 'workspace/denoising/data/MNIST_fashion/train'
    cells1_prefix = 'foreground'
    cells2_prefix = 'background_crops'
    bgnd_prefix = 'background'
    
    gen = FluoMergedFlow(root_dir = root_dir,
                            bgnd_prefix = bgnd_prefix,
                            cells1_prefix = cells1_prefix,
                            cells2_prefix = cells2_prefix,
                            crop_size = (128, 128),
                             is_log_transform = False,
                             int_scale = (0, 255),
                             img_ext = '*.png',
                             is_timeseries_dir = False,
                             n_cells_per_crop = 5,
                             n_bgnd_per_crop = 10,
                             int_factor = (0.9, 1.1),
                             bgnd_sigma_range = (0., 1.2),
                             frac_crop_valid = 0.25,
                             zoom_range = (0.9, 1.1),
                             noise_range = (0., 5.),
                             rotate_range = (0, 90),
                             max_overlap = 0.5,
                             is_separated_output = True
                             )  
    #%%
#    batch_size = 16    
#    gen = FluoMergedFlow()
#    loader = DataLoader(gen, batch_size=batch_size, shuffl'e=True)
#    
#    for Xin, Xout in loader:
#        break
#    
#    for ii in range(batch_size):
#        xin = Xin[ii].squeeze().detach().numpy()
#        xout = Xout[ii].squeeze().detach().numpy()
#        
#        fig, axs = plt.subplots(1,2, figsize = (12, 8), sharex=True, sharey=True)
#        
#        vmax = max(xout.max(), xin.max())
#        vmin = min(xout.min(), xin.min())
#        axs[0].imshow(xin, cmap='gray', interpolation='none', vmin=vmin, vmax=vmax)
#        axs[1].imshow(xout, cmap='gray', interpolation='none', vmin=vmin, vmax=vmax)
#        axs[0].axis('off')
#        axs[1].axis('off')

    
    #%%
    

#%%
    
#    for ii, (X,Y) in enumerate(tqdm.tqdm(gen)):
#        xin = X.squeeze()
#%%
    batch_size = 8
    loader = DataLoader(gen, 
                        batch_size=batch_size, 
                        num_workers = 8,
                        shuffle=True)
    
    for Xin, Xout in tqdm.tqdm(loader):
        assert not (np.isnan(Xin).any() or np.isnan(Xout).any())
        break
        
    #%%
    save_dir = '/Users/avelinojaver/OneDrive - Nexus365/papers/miccai2019/data/synthetic/'
    save_dir = Path(save_dir)
    
    for ii in range(batch_size):
        xin = Xin[ii].squeeze().detach().numpy()
        
        
        #xout = Xout[ii][0].detach().numpy()
        xhat = Xout[ii].detach().numpy().squeeze()
        if xhat.ndim == 3:
            xout = np.rollaxis(xhat, 0, 3)
        else:
            xout = xhat.squeeze()
        
        fig, axs = plt.subplots(1,2, figsize = (12, 8), sharex=True, sharey=True)
        
        #vmin = min(xout.min(), xin.min())
        #vmax = max(xout.max(), xin.max())
        vmin, vmax = xin.min(), xin.max()
        print(vmin, vmax)
        
        vmax = 1.
        
        axs[0].imshow(xin, cmap='gray', interpolation='none', vmin=vmin, vmax=vmax)
        axs[1].imshow(xout[..., 0], cmap='gray', interpolation='none', vmin=vmin, vmax=vmax)
        axs[0].axis('off')
        axs[1].axis('off')
        
        
        x2save = ((xin - vmin)/(vmax - vmin)*255).astype(np.uint8)
        cv2.imwrite(str(save_dir / f'synthetic_{ii}.png'), x2save)
        

    