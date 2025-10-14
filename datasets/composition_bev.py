"""
@author: Ziyue Wang and Wen Li
@file: composition_bev.py
@time: 2025/3/12 14:20
"""

import torch
import numpy as np
import sys
sys.path.insert(0, '../')

from .oxford_bev import Oxford_BEV
from .nclt_bev import NCLT_BEV
from .hercules_bev import Hercules_BEV
from .hercules_bev_radar import Hercules_BEV_Radar
from torch.utils import data
from utils.pose_util import calc_vos_safe_fc


class MF_bev(data.Dataset):
    def __init__(self, dataset, config, split='train', include_vos=False):

        self.steps = config.train.steps #!  clip 中帧的数量 eg 10 
        self.skip = config.train.skip #! clip 内相邻帧的索引间隔  [6 8 10 12 14]
        self.use_merge = config.train.use_merge 
        self.train = split

        if dataset == 'Oxford':
            self.dset = Oxford_BEV(config, split)
        elif dataset == 'NCLT':
            self.dset = NCLT_BEV(config, split)
        elif dataset == 'Hercules':
            self.dset = Hercules_BEV(config, split, config.train.sequence)
            print('Loading Hercules_lidar dataset.......')
        elif dataset == 'Hercules_radar':
            self.dset = Hercules_BEV_Radar(config, split, config.train.sequence)
            print('Loading Hercules_radar dataset.......')

        else:
            raise NotImplementedError('{:s} dataset is not implemented!')

        self.L = self.steps * self.skip
        # GCS
        self.include_vos = include_vos
        self.vo_func = calc_vos_safe_fc


    def get_indices(self, index):
        skips = self.skip * np.ones(self.steps-1) # [2,2,2,2]
        offsets = np.insert(skips, 0, 0).cumsum()  # (self.steps,)  [0,2,2,2,2] - > [0,2,4,6,8]
        offsets -= offsets[len(offsets) // 2] 
        offsets = offsets.astype(np.int_)
        idx = index + offsets  
        idx = np.minimum(np.maximum(idx, 0), len(self.dset)-1)
        assert np.all(idx >= 0), '{:d}'.format(index)
        assert np.all(idx < len(self.dset))
        return idx
    
    def get_merge_indices(self, index): #! 没用上 
        skips = self.merge_skip * np.ones(self.merge_steps-1)
        offsets = np.insert(skips, 0, 0).cumsum()  # (self.steps,)
        offsets -= offsets[len(offsets) // 2]
        offsets = offsets.astype(np.int_)
        idx = index + offsets
        idx = np.minimum(np.maximum(idx, 0), len(self.dset)-1)
        assert np.all(idx >= 0), '{:d}'.format(index)
        assert np.all(idx < len(self.dset))
        return idx

    def __getitem__(self, index):
        idx         = self.get_indices(index)

        clip        = [self.dset[i] for i in idx] 
        pcs         = torch.stack([c[0] for c in clip], dim=0)  # (self.steps, 1, 251, 251)  ??  (self.steps, 3, 251, 251)
        poses       = torch.stack([c[1] for c in clip], dim=0)  # (self.steps, 3)
        
        if self.train == 'train' and self.use_merge:
            merge_pcs   = torch.cat([c[2] for c in clip], dim=0)  # (self.steps, 1, 251, 251) 
            merge_poses = torch.cat([c[3] for c in clip], dim=0)  # (self.steps, 3)
            
            pcs = torch.cat([pcs, merge_pcs], dim=0)
            poses = torch.cat([poses, merge_poses], dim=0)

        if self.include_vos:
            vos = self.vo_func(poses.unsqueeze(0))[0]
            poses = torch.cat((poses, vos), dim=0)

        batch = {
            "image": pcs,   # (self.steps*2, 1, 251, 251) // (self.steps*2, 3, 251, 251)
            "pose": poses,  # (self.steps*2, 3)
        }
        return batch

    def __len__(self):
        L = len(self.dset)
        return L