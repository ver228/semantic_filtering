#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Sep 27 11:12:10 2018

@author: avelinojaver
"""
from pathlib import Path
import random
import cv2
import numpy as np
from torch.utils.data import Dataset 
import tqdm



def rotate_bound(image, angle, border_value = 0):
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
 
    if image.ndim == 3:
        border_value = 3*[border_value]
    
    # perform the actual rotation and return the image
    return cv2.warpAffine(image, M, (nW, nH),
                          borderValue = border_value)


class OverlapsTracker():
    def __init__(self, image_shape, max_overlap, null_value):
        
        self.max_overlap = max_overlap
        self.null_value = null_value
        self.overlap_mask = np.zeros(image_shape, np.int32)
        self.bbox_placed = []
        self.crops_placed = []
        
        
    def add(self, xi, yi, crop):
        if crop.ndim == 3: # I do not need the three dimensions to track locations
            crop = crop[..., 0]
        
        crop_size = crop.shape
        crop_bw = crop != self.null_value
        
        
        xf = xi + crop_size[0]
        yf = yi + crop_size[1]
        
        rr = self.overlap_mask[xi:xf, yi:yf]
        overlap_frac = np.mean(rr>0)
        
        #check the fraction of pixels on the new data that will be cover by the previous data
        if overlap_frac > self.max_overlap:
            return False
        
        #check the fraction of pixels in each of the previous crops that will be cover with the new data.
        #I want to speed this up by checking only the bounding boxes of previously placed objects
        bbox = (xi, yi, xf, yf)
        if len(self.bbox_placed):
            overlaps_frac, intersect_coords = self.bbox_overlaps(bbox, self.bbox_placed)
            bad_ = overlaps_frac > self.max_overlap
            
            #if i find a box that has a large overlap then i want to refine the predictions
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
                 root_dir = None,
                 crop_size = (512, 512),
                 n_cells_per_crop = 4,
                 n_bgnd_per_crop = None,
                 intensity_factor = (0.1, 3.0),
                 
                 epoch_size = 500,
                 is_log_transform = True,
                 
                 int_scale = (0, 2**16-1),
                 fg_quantile_range = (0, 10),
                 bg_quantile_range = (25, 75),
                 is_separated_output = False,
                 
                 cells1_prefix = 'cell_images',
                 cells2_prefix = None,
                 bgnd_prefix = None,
                 
                 img_ext = '*.tif',
                 is_timeseries_dir = True,
                 
                 frac_crop_valid = 0.9,
                 zoom_range = None,
                 rotate_range = None,
                 max_overlap = 1.,
                 
                 
                 null_value = 0,
                 is_preloaded = False,
                 merge_by_prod = False,
                 _debug = False
                 ):
        
        print(root_dir)
        root_dir = Path(root_dir)
        self.root_dir = root_dir
        self.img_ext = img_ext
        self.crop_size = crop_size
        
        
        self.n_cells_per_crop = self.int2range(n_cells_per_crop)
        if n_bgnd_per_crop is None:
            self.n_bgnd_per_crop = self.n_cells_per_crop
        else:
            self.n_bgnd_per_crop = self.int2range(n_bgnd_per_crop)
        
        self.intensity_factor = intensity_factor
        self.frac_crop_valid = frac_crop_valid
        
        self.is_log_transform = is_log_transform
        self.epoch_size = epoch_size
        
        self.int_scale = int_scale #range how the images will be scaled
        
        self.bg_quantile_range = bg_quantile_range
        self.fg_quantile_range = fg_quantile_range
        
        self.is_separated_output = is_separated_output
        
        
        self.zoom_range = zoom_range
        self.rotate_range = rotate_range
        self.max_overlap = max_overlap
        
        self.null_value = null_value
        self.is_preloaded = is_preloaded
        self.merge_by_prod = merge_by_prod
        self._debug = _debug
        
            
            
        self.cells1_files, self.cells2_files, self.bgnd_files = \
                self._get_available_files(root_dir, 
                                         img_ext, 
                                         is_timeseries_dir, 
                                         cells1_prefix, 
                                         cells2_prefix, 
                                         bgnd_prefix)
         
    @staticmethod
    def int2range(x):
        if isinstance(x, (list, tuple)):
            assert len(x) == 2
            return x
        else:
            return (1, x)
            
    
    def _get_available_files(self, 
                             root_dir, 
                             img_ext, 
                             is_timeseries_dir, 
                             cells1_prefix, 
                             cells2_prefix, 
                             bgnd_prefix
                             ):
        
        root_dir = Path(root_dir)
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
        
        
        cells2_files = [[x] for x in cells2_files]
        if is_timeseries_dir:
            # I want to use pairs on consecutive images (using real noise2noise)
            dd = str(root_dir / cells1_prefix) + '/'
            cells1_files_d = {}
            for x in cells1_files:
                path_r = str(x.parent).replace(dd, '')
                
                if not path_r in cells1_files_d:
                    cells1_files_d[path_r] = []
                
                cells1_files_d[path_r].append(x)
            
            for path_r in cells1_files_d:
                cells1_files_d[path_r] =  sorted(cells1_files_d[path_r])
            
            cells1_files = list(cells1_files_d.values())
            
        else:
            cells1_files = [[x] for x in cells1_files]
        
        
        
        
        if self._debug:
            cells1_files = cells1_files[:100]
            cells2_files = cells2_files[:100]
            bgnd_files = bgnd_files[:10]
        
        self.is_preloaded = self.is_preloaded
        if self.is_preloaded:
            cells1_files = self.read_file_list(cells1_files, is_bgnd = False, desc = 'Preloading Foreground-1...')
            cells2_files = self.read_file_list(cells2_files, is_bgnd = False, desc = 'Preloading Foreground-2...')
            bgnd_files = self.read_file_list(bgnd_files, is_bgnd = True, desc = 'Preloading Background...')
            
        return cells1_files, cells2_files, bgnd_files
    
    def read_file_list(self, file_list, is_bgnd, desc = None):
        images = []
        
        pbar = tqdm.tqdm(file_list, desc) if desc is not None else file_list
        for x in pbar:
            if isinstance(x, list):
                images.append(self.read_file_list(x, is_bgnd)) #at somepoint we must reach a list of files so retry
            else:
                img = self.read_file(x, is_bgnd)
                
                if img is not None:
                    images.append(img)
        return images
        
    def read_file(self, fname, is_bgnd):
        if fname.suffix == '.npy':
            img = np.load(str(fname))
        else:    
            img = cv2.imread(str(fname), -1)
        
        if img is None:
            return
        
        
        img = self._scale(img)
        
        img = np.ma.masked_equal(img, self.null_value)
        q_range = self.bg_quantile_range if is_bgnd else self.fg_quantile_range
        
        if img.ndim == 2:
            base_range = np.percentile(img.compressed(), q_range)
        elif img.ndim == 3:
            base_range = [np.percentile(img[..., i].compressed(), q_range) for i in range(3)]
            
        else:
            raise ValueError(f'Invalid mumber of dimensions `{img.ndim}` in image `{fname}`')
        
        return img, base_range
    
    def _get_image(self, element):
        if self.is_preloaded:
            return element # IT IS SLOW TO COPY EVERYTHING, BE CAREFUL WITH IN PLACE OPERATIONS...
        else:
            return self.read_file(element) #the element is a string so i have to read the file
        
    def _get_random_bgnd(self):
        if not self.bgnd_files:
            return np.full(self.crop_size, self.null_value , np.float32)
        
        bgnd_file = random.choice(self.bgnd_files)
        img_bgnd, base_range = self._get_image(bgnd_file)
        
        xi = random.randint(0, img_bgnd.shape[0] - self.crop_size[0])
        yi = random.randint(0, img_bgnd.shape[1] - self.crop_size[1])
        crop_bgnd = img_bgnd[xi:xi + self.crop_size[0], yi:yi + self.crop_size[1]]
        
        base_val = self._get_random_base(base_range)
        
        if crop_bgnd.ndim == 2:
            crop_bgnd = crop_bgnd.filled(base_val)
        else:
            crop_bgnd = [crop_bgnd[..., ii].filled(v)[..., None] for ii, v in enumerate(base_val)]
            crop_bgnd = np.concatenate(crop_bgnd, axis=2)
           
        if self.rotate_range:
            angle = random.uniform(*self.rotate_range)
            #crop_bgnd = rotate_bound(crop_bgnd, _angle)
            (cX, cY) = (crop_bgnd.shape[0] // 2, crop_bgnd.shape[1] // 2)
            M = cv2.getRotationMatrix2D((cX, cY), -angle, 1.0)
            
            crop_bgnd = cv2.warpAffine(crop_bgnd, 
                                       M, 
                                       crop_bgnd.shape[:2], 
                                       borderMode = cv2.BORDER_REFLECT_101
                                       )

        #random flips
        if random.random() >= 0.5:
            crop_bgnd = crop_bgnd[::-1]
        if random.random() >= 0.5:
            crop_bgnd = crop_bgnd[:, ::-1]
        
        crop_bgnd = self._adjust_channels(crop_bgnd)
        return crop_bgnd
    
    def _scale(self, x):
        x[x<self.int_scale[0]] = self.int_scale[0]
        x = (x-self.int_scale[0])/(self.int_scale[1]-self.int_scale[0])
        
        return x.astype(np.float32)
    
    def _sample(self):
        
        fngd_p1, fgnd_p2, overlap_tracker = self.get_cell_pairs(self.cells1_files, self.n_cells_per_crop)
        bgnd1_p1 = self._get_random_bgnd()
        bgnd2_p1 = self.get_cell_pairs(self.cells2_files, self.n_bgnd_per_crop, overlap_tracker)[0]
        
        
        out1 = self._merge(fngd_p1, bgnd1_p1, bgnd2_p1)
#        if self.is_clean_output:
#            base_int = np.mean(bgnd1_p1) 
#            out2 = self._scale(base_int + fgnd_p2)
        if self.is_separated_output:
            out2 = [fngd_p1, bgnd1_p1, bgnd2_p1]
            out2 = np.concatenate(out2)
            
        else:
            bgnd1_p2 = self._get_random_bgnd()
            bgnd2_p2 = self.get_cell_pairs(self.cells2_files, self.n_bgnd_per_crop, overlap_tracker)[0]
        
            out2 = self._merge(fgnd_p2, bgnd1_p2, bgnd2_p2)
            
        
        if self.is_log_transform:
            def _log_normalize(x):
                _scale_log = [np.log(x+1) for x in self.int_scale]
                
                #denormalize
                xd = x * (self.int_scale[1]-self.int_scale[0]) + self.int_scale[0]
                xd = np.log(xd + 1)
                xd = (xd-_scale_log[0])/(_scale_log[1]-_scale_log[0])
                return xd
            
            out1 = _log_normalize(out1)
            
            
            out2 = _log_normalize(out2)
            
        return out1, out2
    
    
    def _merge(self, fgnd, bgnd1, bgnd2):
        
        if self.merge_by_prod:
            bgnd1 = np.clip(bgnd1, 0, 1)
            bgnd2 = np.clip(bgnd2, 0, 1)
            fgnd = np.clip(fgnd, 0, 1)
            _out = bgnd1*bgnd2*fgnd
            
        else:
            _out = bgnd1 + bgnd2 + fgnd
            _out = np.clip(_out, 0, 1)
        return _out
            
    
    def get_cell_pairs(self, src_dirs, n_images_range, overlap_tracker = None):
        n_rois = random.randint(*n_images_range)
        raw_cell_imgs = self._read_random_imgs(src_dirs, n_rois)
        cells_p1, cells_p2, overlap_tracker = self.raw_imgs_to_pairs(raw_cell_imgs, overlap_tracker)
        
        cells_p1 = self._adjust_channels(cells_p1)
        cells_p2 = self._adjust_channels(cells_p2)
        
        return cells_p1, cells_p2, overlap_tracker
        
    def _adjust_channels(self, x):
        if x.ndim == 2:
            return x[None]
        else:
            return np.rollaxis(x, 2, 0)
            
    
    
    def raw_imgs_to_pairs(self, raw_cell_imgs, overlap_tracker = None):
        p_cell_imgs = []
        
        if overlap_tracker is None:
            overlap_tracker = OverlapsTracker(self.crop_size[:2], self.max_overlap, self.null_value)
        
        
        
        for imgs_data in raw_cell_imgs:
            augmented_imgs = self._random_augment_imgs(imgs_data)
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
            
            x1 = self._get_image(fnames[ind])
            if ind == ind_next:
                #I am reading the same file
                _imgs = x1,
            else:
                x2 = self._get_image(fnames[ind_next])
                _imgs = (x1, x2)
                
            random_images.append(_imgs)
        return random_images
    
#    def _shift_base_intensity(self, x, base_quantile = 5):
#        #random shift of the base
#        x = x.copy()
#        if x.ndim == 2:
#            x = x[..., None]
#        
#        valid_pix = x[..., 0] != self.null_value
#        
#        if valid_pix.any():
#            for ii in range(x.shape[2]):
#                ch = x[..., ii]
#                base_int = np.percentile(ch[valid_pix], base_quantile)
#                if self.merge_by_prod:
#                    ch[valid_pix] /= base_int
#                else:
#                    ch[valid_pix] -= base_int
#                    
#            x = np.clip(x, 0, 1)            
#        
#        return x.squeeze()
    
    @staticmethod
    def _get_random_base(base_range):
        
        if len(base_range) == 2:
            return random.uniform(*base_range)
        else:
            #in three dimensions i want a random value along the ranges for each color
            val = [random.uniform(*x) for x in base_range]
            return val
        
    
    def _random_augment_imgs(self, imgs_data):
        '''
        Randomly exectute the same transforms for each tuple the list.
        '''
        
        img, base_range = imgs_data[0] #i am doing this because since i could accept timeseries imgs_data is a list of [(img1, base_range1), (img2, base_range2) ...]
        base_val = self._get_random_base(base_range)
        if img.ndim == 3: #deal with the case of three dimensions where each color channel will have its own value
            base_val = np.array(base_val)[None, None]
        
        
        _flipv = random.random() >= 0.5
        _fliph = random.random() >= 0.5
        
        
        _zoom = random.uniform(*self.zoom_range) if self.zoom_range else None
        _angle = random.uniform(*self.rotate_range) if self.rotate_range else None
        
        if self.is_log_transform:
            _int_log = [np.log(x) for x in self.intensity_factor]
            _intensity = np.exp(random.uniform(*_int_log))
            
        else:
            _intensity = random.uniform(*self.intensity_factor)
        #_base_quantile = random.uniform(*self.int_base_q_range)
            
        def _augment(cc):
            cc = cc.astype(np.float32).copy() # If it is already a float32 it will not return a copy...
            
            if self.merge_by_prod:
                cc /= base_val
                cc = np.power(cc, _intensity)
                
            else:
                cc -= base_val
                cc *= _intensity
            
            #random rotation
            if _angle:
                cc = rotate_bound(cc, _angle, border_value = self.null_value)
            
            
            #random flips
            if _fliph:
                cc = cc[::-1]
            if _flipv:
                cc = cc[:, ::-1]
            
            #zoom
            if _zoom:
                cc = cv2.resize(cc, (0,0), fx=_zoom, fy=_zoom, interpolation = cv2.INTER_LINEAR)
            
            cc = cc[:self.crop_size[0], :self.crop_size[1]] #crop in case it is larger than the final image
            
            cc = np.clip(cc, 0, 1) # here I am assuming the image was originally scaled from 0 to 1
            return cc
        
        new_imgs = [_augment(img) for img, _ in imgs_data]
        return new_imgs
    
    
    
    def _random_locate_imgs(self, _imgs, overlap_tracker):
        #randomly located a pair in the final image. If part of the pair is 
        #located outside the final image, the the pair is cropped.        
        img_shape = _imgs[0].shape
            
        #crop if the x,y coordinate is outside the expected image 
        frac_cc = [int(round(x*self.frac_crop_valid)) for x in img_shape]
        
        
        max_ind_x = self.crop_size[0] - img_shape[0]
        max_ind_y = self.crop_size[1] - img_shape[1]
        
        BREAK_NUM = 3 #number of trials to add an image before giving up
        
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
        _out = np.full(self.crop_size, self.null_value, dtype = np.float32)
        for (xi,yi), cc in cell_imgs:
            
            if self.merge_by_prod:
                _out[xi:xi+cc.shape[0], yi:yi+cc.shape[1]] *= cc
            else:
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



def _test_load_BBBC42(_debug = False):
    #root_dir = Path.home() / 'Desktop/BBBC042_divided_color_v2/train'
    #root_dir = Path.home() / 'workspace/denoising/data/BBBC042_colour_v2/train'
    root_dir = Path.home() / 'workspace/denoising/data/BBBC042_colour_more_bgnd/train'
    
    cells1_prefix = 'foreground'
    cells2_prefix = 'background_crops'
    bgnd_prefix = 'background'
              
    gen = FluoMergedFlow(root_dir = root_dir,
                            bgnd_prefix = bgnd_prefix,
                            cells1_prefix = cells1_prefix,
                            cells2_prefix = cells2_prefix,
                            crop_size = (256, 256, 3),#(256, 256, 3),
                             is_log_transform = False,
                             int_scale = (0, 255),
                             img_ext = '*.tif',
                             is_timeseries_dir = False,
                             n_cells_per_crop = (2, 5),
                             n_bgnd_per_crop = (1, 3),
                             intensity_factor = (0.9, 1.1),
                             fg_quantile_range = (90, 100),
                             bg_quantile_range = (25, 75),
                             
                             frac_crop_valid = 0.1,
                             zoom_range = (0.9, 1.1),
                             rotate_range = (0, 90),
                             max_overlap = 0.5,
                             is_separated_output = False,
                             
                             null_value = 1.,
                             merge_by_prod = True,
                             is_preloaded = True,
                             _debug = _debug
                             )  
    return gen

def _test_load_BBBC42_simple(_debug = False):
    #%%
    #root_dir = Path.home() / 'Desktop/BBBC042_divided/train'
    root_dir = Path.home() / 'workspace/denoising/data/BBBC042_bgnd/train'
    
    cells1_prefix = 'foreground'
    cells2_prefix = 'background_crops'
    bgnd_prefix = 'background'
              
    gen = FluoMergedFlow(root_dir = root_dir,
                            bgnd_prefix = bgnd_prefix,
                            cells1_prefix = cells1_prefix,
                            cells2_prefix = cells2_prefix,
                            crop_size = (256, 256),
                             is_log_transform = False,
                             int_scale = (0, 255),
                             img_ext = '*.tif',
                             is_timeseries_dir = False,
                             n_cells_per_crop = (2, 5),
                             n_bgnd_per_crop = (1, 3),
                             intensity_factor = (0.9, 1.1),
                             fg_quantile_range = (0, 10),
                             bg_quantile_range = (25, 75),
                             
                             frac_crop_valid = 0.1,
                             zoom_range = (0.9, 1.1),
                             rotate_range = (0, 90),
                             max_overlap = 0.5,
                             is_separated_output = False,
                             
                             is_preloaded = True,
                             _debug = _debug
                             ) 
    #%%
    #plt.imshow(gen.bgnd_files[5][0])
    #%%
    return gen

def _test_load_BBBC26(_debug = False):
    #root_dir = '/Volumes/loco/workspace/denoising/data/BBBC026/'
    #root_dir = Path.home() / 'workspace/denoising/data/BBBC026_v2/'
    root_dir = Path.home() / 'workspace/denoising/data/BBBC026_bgnd/'
    root_dir = '/Users/avelinojaver/Desktop/BBBC026_divided/'
    
    cells1_prefix = 'hepatocytes'
    cells2_prefix = 'fibroblasts'
    bgnd_prefix = 'background'
    
    gen = FluoMergedFlow(root_dir = root_dir,
                            bgnd_prefix = bgnd_prefix,
                            cells1_prefix = cells1_prefix,
                            cells2_prefix = cells2_prefix,
                            crop_size = (256, 256),
                             is_log_transform = False,
                             int_scale = (0, 255),
                             img_ext = '*.png',
                             is_timeseries_dir = False,
                             n_cells_per_crop = (3, 8),
                             intensity_factor = (0.5, 1.5),
                             frac_crop_valid = 0.5,
                             zoom_range = (0.75, 1.3),
                             rotate_range = (0, 90),
                             max_overlap = 0.9,
                             is_separated_output = True,
                             is_preloaded = True, 
                             _debug = _debug
                             )  
    return gen
     

def _test_load_microglia(_debug = False):
    #root_dir = Path.home() / 'workspace/denoising/data/microglia_v2'
    #root_dir = Path.home() / 'workspace/denoising/data/microglia_v2_tight'
    
    #root_dir = Path.home() / 'Desktop/microglia'
    root_dir = Path.home() / 'Desktop/microglia_tight'
    
    cells1_prefix = 'foreground'
    cells2_prefix = 'background_crops'
    bgnd_prefix = 'background'
    
    gen = FluoMergedFlow(root_dir = root_dir,
                            bgnd_prefix = bgnd_prefix,
                            cells1_prefix = cells1_prefix,
                            cells2_prefix = cells2_prefix,
                            crop_size = (512, 512),
                             is_log_transform = True,
                             int_scale = (5, 40000),#(0, 2**16-1),
                             img_ext = '*.tif',
                             is_timeseries_dir = False,
                             n_cells_per_crop = (1, 5),
                             n_bgnd_per_crop = (3, 6),
                             intensity_factor = (0.05, 10),
                             frac_crop_valid = 0.5,
                             zoom_range = (0.9, 1.1),
                             rotate_range = (0, 90),
                             max_overlap = 1.,
                             is_separated_output = False,
                             epoch_size = 500,
                             #base_quantile = 5,
                             is_preloaded = True, 
                             _debug = _debug
                             )
    return gen
#%%
if __name__ == '__main__':
    from pathlib import Path
    from torch.utils.data import DataLoader
    import matplotlib.pylab as plt
    import tqdm

    #%%
    gen = _test_load_BBBC42_simple(True)
    #gen = _test_load_BBBC42(True)
    #gen = _test_load_BBBC26(True)
    #gen = _test_load_microglia(True)
   #%%
    for _ in tqdm.tqdm(range(3)):
        
        if gen.is_separated_output:
            fig, axs = plt.subplots(1, 4, sharex=True, sharey=True, figsize=(15, 5))
        else:
            fig, axs = plt.subplots(1, 2, sharex=True, sharey=True)
        
        xin, xout = gen._sample()
        
        n_per_channel = xin.shape[0]
        
        if gen.is_separated_output:
            xout = [xout[ii:ii+n_per_channel] for ii in range(0, 3*n_per_channel, n_per_channel)]
        else:
            xout = [xout]
        
        
        for ax, x in zip(axs, [xin] + xout):
            if n_per_channel == 1:
                x = x[0]
            else:
                x = np.rollaxis(x, 0, 3)
            
            ax.imshow(x)
            
    #%%
    
    #%%
#    batch_size = 16        
#    loader = DataLoader(gen, batch_size=batch_size, shuffle=True)
#    for Xin, Xout in tqdm.tqdm(loader):
#        continue    
#%%
#    root_dir = Path.home() / 'Desktop/MNIST_fashion/train'
#    #root_dir = Path.home() / 'workspace/denoising/data/MNIST_fashion/train'
#    cells1_prefix = 'foreground'
#    cells2_prefix = 'background_crops'
#    bgnd_prefix = 'background'
#    
#    gen = FluoMergedFlow(root_dir = root_dir,
#                            bgnd_prefix = bgnd_prefix,
#                            cells1_prefix = cells1_prefix,
#                            cells2_prefix = cells2_prefix,
#                            crop_size = (128, 128),
#                             is_log_transform = False,
#                             int_scale = (0, 255),
#                             img_ext = '*.png',
#                             is_timeseries_dir = False,
#                             n_cells_per_crop = 5,
#                             n_bgnd_per_crop = 10,
#                             intensity_factor = (0.9, 1.1),
#                             bgnd_sigma_range = (0., 1.2),
#                             frac_crop_valid = 0.25,
#                             zoom_range = (0.9, 1.1),
#                             noise_range = (0., 5.),
#                             rotate_range = (0, 90),
#                             max_overlap = 0.5,
#                             is_separated_output = True
#                             )  
    #%%

    
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
#    batch_size = 8
#    loader = DataLoader(gen, 
#                        batch_size=batch_size, 
#                        num_workers = 1,
#                        shuffle=True)
#    
#    for Xin, Xout in tqdm.tqdm(loader):
#        assert not (np.isnan(Xin).any() or np.isnan(Xout).any())
#        #break
        
    #%%
#    save_dir = '/Users/avelinojaver/OneDrive - Nexus365/papers/miccai2019/data/synthetic/'
#    save_dir = Path(save_dir)
#    
#    for ii in range(batch_size):
#        xin = Xin[ii].squeeze().detach().numpy()
#        
#        
#        #xout = Xout[ii][0].detach().numpy()
#        xhat = Xout[ii].detach().numpy().squeeze()
#        if xhat.ndim == 3:
#            xout = np.rollaxis(xhat, 0, 3)
#        else:
#            xout = xhat.squeeze()
#        
#        fig, axs = plt.subplots(1,2, figsize = (12, 8), sharex=True, sharey=True)
#        
#        #vmin = min(xout.min(), xin.min())
#        #vmax = max(xout.max(), xin.max())
#        vmin, vmax = xin.min(), xin.max()
#        print(vmin, vmax)
#        
#        vmax = 1.
#        
#        axs[0].imshow(xin, cmap='gray', interpolation='none', vmin=vmin, vmax=vmax)
#        axs[1].imshow(xout[..., 0], cmap='gray', interpolation='none', vmin=vmin, vmax=vmax)
#        axs[0].axis('off')
#        axs[1].axis('off')
#        
#        
#        x2save = ((xin - vmin)/(vmax - vmin)*255).astype(np.uint8)
#        cv2.imwrite(str(save_dir / f'synthetic_{ii}.png'), x2save)
#        

    