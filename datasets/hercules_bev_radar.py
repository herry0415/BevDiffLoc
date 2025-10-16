"""
@author: Ziyue Wang and Wen Li
@file: oxford_bev.py
@time: 2025/3/12 14:20
"""

import os
import cv2
import h5py
import torch
import random
import numpy as np
import os.path as osp
from sklearn.neighbors import KDTree
from copy import deepcopy
from torch.utils import data
from datasets.projection import getBEV, new_getBEV
from datasets.augmentor import Augmentor, AugmentParams

from utils.pose_util import process_poses, poses_to_matrices, poses_foraugmentaion
from datasets.robotcar_sdk.python.interpolate_poses import interpolate_ins_poses
from datasets.robotcar_sdk.python.transform import build_se3_transform, euler_to_so3

BASE_DIR = osp.dirname(osp.abspath(__file__))

class Hercules_BEV_Radar(data.Dataset):
    def __init__(self, config, split='train', sequence_name='Library'):
        if split == 'train':
            self.is_train = True
        else:
            self.is_train = False

        lidar = 'hercules_radar' # todo

        data_path =  config.train.dataroot

        bev = config.bev_fold #! 改为从yaml文件中加载
        bev_poses = config.bev_poses_path #! 改为从yaml文件中加载

        self.sequence_name = sequence_name #['Library', 'Mountain', 'Sports']

        self.data_dir = osp.join(data_path,sequence_name)


        # decide which sequences to use
        seqs = self._get_sequences(sequence_name, self.is_train)

        ps = {}
        ts = {}
        vo_stats = {}
        self.pcs = []
        self.bev = []
        self.bev_poses = []
        self.merge_num = config.train.merge_num

        #! 通过lidar找到对应radar代码进行训练 
        self.lidar_files = []
        self.radar_files = []
        
        for seq in seqs:
            seq_dir = os.path.join(self.data_dir, seq)         
           
            lidar_pose_file_path = os.path.join(seq_dir, 'PR_GT/Aeva_gt.txt')
        
            ts_raw = np.loadtxt(lidar_pose_file_path, dtype=np.int64, usecols=0)
            ts[seq] = ts_raw
            lidar_ts = ts_raw
            
            lidar_pose_file = np.loadtxt(lidar_pose_file_path)
            p = poses_to_matrices(lidar_pose_file) # (n,4,4) #激光雷达坐标系
            ps[seq] = np.reshape(p[:, :3, :], (len(p), -1))     #  (n, 12)
            
            radar_file_path = os.path.join(seq_dir, 'PR_GT/newContinental_gt.txt')
            radar_ts= np.loadtxt(radar_file_path, dtype=np.int64, usecols=0)
            radar_pose_file = np.loadtxt(radar_file_path)

            lidar_xy = self.get_xy(lidar_pose_file)
            radar_xy = self.get_xy(radar_pose_file)
            
            #build kdtree to find match
            lidar_files = [os.path.join(seq_dir, 'LiDAR/np8Aeva', str(t) + '.bin') for t in lidar_ts]
            radar_files = [os.path.join(seq_dir, 'Radar/multi_frame_w7', str(t)+'_multi_w7' + '.bin') for t in radar_ts]
            
            tree_radar = KDTree(radar_xy)
            dists, idxs = tree_radar.query(lidar_xy, k=1)  # for each lidar -> nearest radar
            idxs = idxs.ravel()
            dists = dists.ravel()
                # filter by distance threshold
            for li, ridx in enumerate(idxs):
                dist = float(dists[li])
                # print(dist)

                lf = lidar_files[li]
                rf = radar_files[ridx]
                # check existence of files
                if os.path.exists(lf) and os.path.exists(rf):
                    self.lidar_files.append(lf)
                    self.radar_files.append(rf)
                else:
                    # 缺文件则略过并打印警告
                    if not os.path.exists(lf): print(f"Missing LIDAR file: {lf}")
                    if not os.path.exists(rf): print(f"Missing RADAR file: {rf}")
            
            vo_stats[seq] = {'R': np.eye(3), 't': np.zeros(3), 's': 1}

        poses = np.empty((0, 12))
        for p in ps.values():
            poses = np.vstack((poses, p))

        pose_stats_filename = os.path.join(self.data_dir, self.sequence_name + '_radar'+'_pose_stats.txt')
        # pose_max_min_filename = os.path.join(self.data_dir, self.sequence_name + '_lidar'+ '_pose_max_min.txt')

        # for seq in seqs:
        #     seq_dir = os.path.join(self.data_dir, seq)
        #     # read the image timestamps
        #     # h5_path = osp.join(seq_dir, lidar + '_' + 'False.h5')

        #     # bev_path = osp.join(seq_dir, bev)
        #     # bev_poses_path = osp.join(seq_dir, bev_poses)
        #     # if not os.path.isfile(h5_path):
        #     print('interpolate ' + seq)
        #     pose_file_path = os.path.join(seq_dir, 'PR_GT/newContinental_gt.txt') # PR_GT/newContinental_gt.txt
        #     ts_raw = np.loadtxt(pose_file_path, dtype=np.int64, usecols=0) # float读取数字丢精度
        #     ts[seq] = ts_raw
            
        #     pose_file = np.loadtxt(pose_file_path) #保证pose值不变
        #     p = poses_to_matrices(pose_file) # (n,4,4) #毫米波雷达坐标系
        #     # ps[seq] = np.reshape(p[:, :3, :], (len(p), -1))     #  (n, 12)
        #     # p = np.asarray([np.dot(pose, G_posesource_laser) for pose in p])  # (n, 4, 4)

        #     ps[seq] = np.reshape(p[:, :3, :], (len(p), -1))  # (n, 12)
        #     vo_stats[seq] = {'R': np.eye(3), 't': np.zeros(3), 's': 1}

        #     if self.is_train:
        #         # self.pcs.extend([osp.join(seq_dir, 'LiDAR/np8Aeva', '{:d}.bin'.format(t)) for t in ts[seq]])
        #         self.pcs.extend(os.path.join(seq_dir, 'Radar/multi_frame_w7', str(t)+'_multi_w7' + '.bin') for t in ts[seq])
        #         vo_stats[seq] = {'R': np.eye(3), 't': np.zeros(3), 's': 1}
        #     else:
        #         # self.pcs.extend(os.path.join(seq_dir, 'LiDAR/np8Aeva', str(t) + '.bin') for t in ts[seq])
        #         self.pcs.extend(os.path.join(seq_dir, 'Radar/multi_frame_w7', str(t)+'_multi_w7' + '.bin') for t in ts[seq])
        #         vo_stats[seq] = {'R': np.eye(3), 't': np.zeros(3), 's': 1}

        if self.is_train:
            #! 处理bev图像 因为之前把两个序列的bev图像放在一起了
            bev_path = osp.join(self.data_dir, bev)
            bev_poses_path = osp.join(self.data_dir, bev_poses)
            if not os.path.exists(bev_path):
                assert False, f"BEV path not exists: {bev_path}"
            if not os.path.exists(bev_poses_path):
                assert False, f"BEV poses path not exists: {bev_poses_path}"

            merge_sum = 0
            with open(bev_poses_path, 'r') as file:
                for line in file:
                    bev_pose = list(map(float, line.split()))
                    self.bev_poses.append(bev_pose)
                    merge_sum = merge_sum + 1
            
            for i in range(merge_sum):
                self.bev.append(osp.join(bev_path, f"{i+1}.png"))

        # read / save pose normalization information
        
        if self.is_train:
            self.mean_t = np.mean(poses[:, [3, 7, 11]], axis=0)  # (3,)
            self.std_t = np.std(poses[:, [3, 7, 11]], axis=0)  # (3,)
            np.savetxt(pose_stats_filename, np.vstack((self.mean_t, self.std_t)), fmt='%8.7f')
            print(f'saving pose stats in {pose_stats_filename}')
        else:
            self.mean_t, self.std_t = np.loadtxt(pose_stats_filename)
            

        self.poses = np.empty((0, 6))
        self.rots = np.empty((0, 3, 3))
        for seq in seqs:
            #! 更改
            pss, rotation, pss_max, pss_min = process_poses(poses_in=ps[seq], mean_t=self.mean_t, std_t=self.std_t,
                                                            align_R=vo_stats[seq]['R'], align_t=vo_stats[seq]['t'],
                                                            align_s=vo_stats[seq]['s'])
            self.poses = np.vstack((self.poses, pss)) # (37666, 6) 前三列: 归一化和对齐后的平移向量 (x, y, z)。 后三列: 对齐后的旋转分量（四元数的虚部），以三维向量形式表示
            self.rots = np.vstack((self.rots, rotation)) #  每一帧的 3x3 旋转矩阵
            

        # normalize translation
        for bev_pose in self.bev_poses:
            bev_pose[:2] -= self.mean_t[:2]
            bev_pose[:2] /= self.std_t[:2]

        if split == 'train':
            print("train bev data num:" + str(len(self.bev_poses)))
        else:
            print("valid bev data num:" + str(len(self.bev_poses)))

        augment_params = AugmentParams()
        augment_config = config.augmentation

        # Point cloud augmentations
        if self.is_train:
            augment_params.setTranslationParams(
                p_transx=augment_config['p_transx'], trans_xmin=augment_config[
                    'trans_xmin'], trans_xmax=augment_config['trans_xmax'],
                p_transy=augment_config['p_transy'], trans_ymin=augment_config[
                    'trans_ymin'], trans_ymax=augment_config['trans_ymax'],
                p_transz=augment_config['p_transz'], trans_zmin=augment_config[
                    'trans_zmin'], trans_zmax=augment_config['trans_zmax'])
            augment_params.setRotationParams(
                p_rot_roll=augment_config['p_rot_roll'], rot_rollmin=augment_config[
                    'rot_rollmin'], rot_rollmax=augment_config['rot_rollmax'],
                p_rot_pitch=augment_config['p_rot_pitch'], rot_pitchmin=augment_config[
                    'rot_pitchmin'], rot_pitchmax=augment_config['rot_pitchmax'],
                p_rot_yaw=augment_config['p_rot_yaw'], rot_yawmin=augment_config[
                    'rot_yawmin'], rot_yawmax=augment_config['rot_yawmax'])
            if 'p_scale' in augment_config:
                augment_params.sefScaleParams(
                    p_scale=augment_config['p_scale'],
                    scale_min=augment_config['scale_min'],
                    scale_max=augment_config['scale_max'])
                print(
                    f'Adding scaling augmentation with range [{augment_params.scale_min}, {augment_params.scale_max}] and probability {augment_params.p_scale}')
            self.augmentor = Augmentor(augment_params)
        else:
            self.augmentor = None

    def __len__(self):
        return len(self.poses)
    
    def _get_sequences(self, sequence_name, train):
        mapping = {
            'Library': (['Library_01_Day','Library_02_Night'], ['Library_03_Day']),
            'Mountain': (['Mountain_01_Day','Mountain_02_Night'], ['Mountain_03_Day']),
            'Sports': (['Complex_01_Day','Complex_02_Night'], ['Complex_03_Day'])
        }
        return mapping[sequence_name][0] if train else mapping[sequence_name][1]
    
    def get_xy(self, pose_file):
        pose = poses_to_matrices(pose_file) # (n,4,4)
        xy = pose[:,:2, 3]
        return xy

    def __getitem__(self, idx_N):
        # scan_path = self.pcs[idx_N]
        scan_path = self.radar_files[idx_N]
        pointcloud = np.fromfile(scan_path, dtype=np.float32).reshape(-1, 8)   
        pointcloud = np.concatenate((pointcloud[:, :3], pointcloud[:, 5:6]), axis=1) # [N, 4] xyz intensity
        pointcloud = np.concatenate((pointcloud, np.zeros((len(pointcloud), 1))), axis=1)
        # flip z
        # pointcloud[:, 2] = -1 * pointcloud[:, 2] #! Z轴不能反转
        
        merge_bev_img = np.empty((0, 3, 251, 251))
        merge_pose = np.empty((0, 3))
        
        if self.is_train:
            random_bev = self.merge_num
            for i in range(random_bev):
                bev_idx = random.randint(0, len(self.bev) - 1)
                bev_path = self.bev[bev_idx]
                bev_pose = np.array(self.bev_poses[bev_idx]) #! 之前在merge txt里面的存储的3维 pose
                bev_merge = cv2.imread(bev_path, 0) #! 读取的时候也是灰度图片
                bev_merge = np.tile(bev_merge, (3, 1, 1)) #! 复制为3通道
                
                bev_pose = bev_pose.reshape(1, 3)
                bev_merge = np.expand_dims(bev_merge, axis=0)
                
                merge_bev_img = np.concatenate((merge_bev_img, bev_merge), axis=0)
                merge_pose = np.concatenate((merge_pose, bev_pose), axis=0)
        
        if self.is_train:
            pointcloud, rotation = self.augmentor.doAugmentation_bev(pointcloud)  # n, 5
            original_rots = self.rots[idx_N]  # [3, 3]
            present_rots = rotation @ original_rots
            poses = poses_foraugmentaion(present_rots, self.poses[idx_N])
        else:
            poses = self.poses[idx_N]
        
        # Generate BEV_Image
        yaw = poses[5] * 2 #! 这里用5作为一个yaw？
        poses_bev = poses[[0, 1]]
        poses_bev = np.hstack((poses_bev, yaw))
        
        pointcloud = pointcloud[:, :3]

        # #todo 设置处理范围 如果换投影方式也需要改
        # pointcloud = pointcloud[np.where(np.abs(pointcloud[:,0])<50)[0],:]
        # pointcloud = pointcloud[np.where(np.abs(pointcloud[:,1])<50)[0],:]
        # pointcloud = pointcloud[np.where(np.abs(pointcloud[:,2])<50)[0],:]
        # pointcloud = pointcloud.astype(np.float32)

        # 按范围裁剪
        x_min, x_max = 0, 100
        y_min, y_max = -50, 50
        z_min, z_max = -10, 10 #! 这是z
        mask_x = (pointcloud[:, 0] >= x_min) & (pointcloud[:, 0] <= x_max)
        mask_y = (pointcloud[:, 1] >= y_min) & (pointcloud[:, 1] <= y_max)
        # mask_z = (pointcloud[:, 2] >= z_min) & (pointcloud[:, 2] <= z_max)

        mask = mask_x & mask_y
        pointcloud = pointcloud[mask]
        bev_img,_ ,_ = new_getBEV(pointcloud, 0, 0, 0) # [251, 251]
        bev_img = np.tile(bev_img, (3, 1, 1))

        # bev_img = getBEV(pointcloud, 0, 0, 0) # [251, 251]
        # bev_img = np.tile(bev_img, (3, 1, 1))
        
        bev_img_tensor = torch.from_numpy(bev_img.astype(np.float32))
        pose_tensor = torch.from_numpy(poses_bev.astype(np.float32))
        
        merge_bev_img_tensor = torch.from_numpy(merge_bev_img.astype(np.float32))
        merge_pose_tensor = torch.from_numpy(merge_pose.astype(np.float32))
        
        return bev_img_tensor, pose_tensor, merge_bev_img_tensor, merge_pose_tensor #todo 随机选的多帧，训练增强，提供上下文信息
        #! 当前帧 BEV 图像，模型输入     (3, H, W)
        #! 当前帧位姿标签 [x, y, yaw]
        #! 多帧随机 BEV 图像增强，模型输入 (self.merge_num, 3, H, W) -> (1, 3, H, W)
        #! 对应多帧增强的位姿标签，监督信息

    