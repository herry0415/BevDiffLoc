import os
import h5py
import torch
import cv2
import numpy as np
import os.path as osp
from omegaconf import OmegaConf, DictConfig
from tqdm import tqdm
from copy import deepcopy
import open3d as o3d
from torch.utils import data
from datasets.projection import new_getBEV, getBEV
from utils.pose_util import process_poses, poses_to_matrices
from utils.pose_util import process_poses, filter_overflow_ts, poses_foraugmentaion

from datasets.augmentor import Augmentor, AugmentParams
from datasets.robotcar_sdk.python.transform import build_se3_transform, euler_to_so3
import time
import math

BASE_DIR = osp.dirname(osp.abspath(__file__))

class Hercules_merge(data.Dataset):
    def __init__(self, config, split='train', sequence_name='Library'):
        if split == 'train':
            self.is_train = True
        else:
            self.is_train = False

        lidar = 'hercules_lidar' #todo 

        data_path =  config.train.dataroot

        sequence_name='Library'
        self.sequence_name = sequence_name #['Library', 'Mountain', 'Sports']

        self.data_dir = os.path.join(data_path, sequence_name)
        
         # 根据 sequence_name 和 train/val 设置序列
        seqs = self._get_sequences(sequence_name, self.is_train)

        ps = {}
        ts = {}
        vo_stats = {}
        self.pcs = []

        # # extrinsic reading
        # with open(os.path.join(extrinsics_dir, lidar + '.txt')) as extrinsics_file:
        #     extrinsics = next(extrinsics_file)
        # G_posesource_laser = build_se3_transform([float(x) for x in extrinsics.split(' ')])
        # with open(os.path.join(extrinsics_dir, 'ins.txt')) as extrinsics_file:
        #     extrinsics = next(extrinsics_file)

        # G_posesource_laser = np.linalg.solve(build_se3_transform([float(x) for x in extrinsics.split(' ')]), G_posesource_laser)  # (4, 4)

        for seq in seqs:
            #! 不用h5保存了
            seq_dir = os.path.join(self.data_dir, seq)

            # h5_path = os.path.join(seq_dir, 'lidar_poses.h5')
         
            # if not os.path.isfile(h5_path):
            print('interpolate ' + seq)
            pose_file_path = os.path.join(seq_dir, 'PR_GT/Aeva_gt.txt')
            ts_raw = np.loadtxt(pose_file_path, dtype=np.int64, usecols=0) # float读取数字丢精度
            ts[seq] = ts_raw
            
            pose_file = np.loadtxt(pose_file_path) #保证pose值不变
            p = poses_to_matrices(pose_file) # (n,4,4) #毫米波雷达坐标系
            ps[seq] = np.reshape(p[:, :3, :], (len(p), -1))     #  (n, 12)
    

            if self.is_train:
                self.pcs.extend([osp.join(seq_dir, 'LiDAR/np8Aeva', '{:d}.bin'.format(t)) for t in ts[seq]])
                # self.pcs.extend(os.path.join(seq_dir, 'Radar/multi_frame_w7', str(t)+'_multi_w7' + '.bin') for t in ts[seq])
                vo_stats[seq] = {'R': np.eye(3), 't': np.zeros(3), 's': 1}
            else:
                self.pcs.extend(os.path.join(seq_dir, 'LiDAR/np8Aeva', str(t) + '.bin') for t in ts[seq])
                vo_stats[seq] = {'R': np.eye(3), 't': np.zeros(3), 's': 1}

        # read / save pose normalization information
        poses = np.empty((0, 12))
        for p in ps.values():
            poses = np.vstack((poses, p))
        
        pose_stats_filename = os.path.join(self.data_dir, self.sequence_name + '_lidar'+'_pose_stats.txt')
        # pose_max_min_filename = os.path.join(self.data_dir, self.sequence_name + '_lidar'+ '_pose_max_min.txt')
        
        if self.is_train:
            self.mean_t = np.mean(poses[:, [3, 7, 11]], axis=0)  # (3,)
            self.std_t = np.std(poses[:, [3, 7, 11]], axis=0)  # (3,)
            np.savetxt(pose_stats_filename, np.vstack((self.mean_t, self.std_t)), fmt='%8.7f')
            print(f'saving pose stats in {pose_stats_filename}')
        else:
            self.mean_t, self.std_t = np.loadtxt(pose_stats_filename)

        self.poses_3_4 = poses  
        self.poses = np.empty((0, 6))
        self.rots = np.empty((0, 3, 3))
        
         # convert the pose to translation + log quaternion, align, normalize
         
        # self.poses = np.empty((0, 6))
        # self.rots = np.empty((0, 3, 3))
        # self.poses_max  =  np.empty((0, 2))
        # self.poses_min  =  np.empty((0, 2))
        for seq in seqs:
            #! 更改
            pss, rotation, pss_max, pss_min = process_poses(poses_in=ps[seq], mean_t=self.mean_t, std_t=self.std_t,
                                                            align_R=vo_stats[seq]['R'], align_t=vo_stats[seq]['t'],
                                                            align_s=vo_stats[seq]['s'])
            self.poses = np.vstack((self.poses, pss)) # (37666, 6) 前三列: 归一化和对齐后的平移向量 (x, y, z)。 后三列: 对齐后的旋转分量（四元数的虚部），以三维向量形式表示
            self.rots = np.vstack((self.rots, rotation)) #  每一帧的 3x3 旋转矩阵

        if split == 'train':
            print("train data num:" + str(len(self.poses)))
        else:
            print("valid data num:" + str(len(self.poses)))

    def __len__(self):
        return len(self.poses)
    
    def _get_sequences(self, sequence_name, train):
        mapping = {
            'Library': (['Library_01_Day','Library_02_Night'], ['Library_03_Day']),
            'Mountain': (['Mountain_01_Day','Mountain_02_Night'], ['Mountain_03_Day']),
            'Sports': (['Complex_01_Day','Complex_03_Day'], ['Complex_02_Night'])
        }
        return mapping[sequence_name][0] if train else mapping[sequence_name][1]

    def __getitem__(self, idx_N):
        scan_path = self.pcs[idx_N]
        
        pointcloud = np.fromfile(scan_path, dtype=np.float32).reshape(-1,8)
        # pointcloud[:, 2] = -1 * pointcloud[:, 2] #! Z轴不能反转
        poses_3_4 = self.poses_3_4[idx_N]

        #!设置处理范围
        x_min, x_max = 0, 100
        y_min, y_max = -50, 50
        z_min, z_max = -10, 10  #! 这是z

        
        #todo  Generate BEV_Image 筛选的 XYZ 范围都是 [-50, 50] 筛选的范围要改
        pointcloud = pointcloud[:, :3] 
        # 按范围裁剪
        mask_x = (pointcloud[:, 0] >= x_min) & (pointcloud[:, 0] <= x_max)
        mask_y = (pointcloud[:, 1] >= y_min) & (pointcloud[:, 1] <= y_max)
        # mask_z = (pointcloud[:, 2] >= z_min) & (pointcloud[:, 2] <= z_max)

        mask = mask_x & mask_y
        pointcloud = pointcloud[mask]
        
        return pointcloud, poses_3_4

if __name__ == '__main__':
    cfg = OmegaConf.load('cfgs/hercules_bev.yaml')
    # 假设你有一个配置对象 cfg
    dataset = Hercules_merge(config=cfg, split='train', sequence_name='Library')

    merged_pointcloud = o3d.geometry.PointCloud()
    merged_x = []
    merged_y = []
    # all_pointcloud = []
    all_poses = []
    voxel_size = 0.4
    image_path = '/home/data/ldq/HeRCULES/Library/merge_bev/'  # 更新保存路径 
    # image_path = '/home/data/ldq/HeRCULES/Library/new_merge_bev/'  # 更新保存路径 

    pose_path = '/home/data/ldq/HeRCULES/Library/merge_bev.txt'  # 更新pose保存路径
    # pose_path = '/home/data/ldq/HeRCULES/Library/new_merge_bev.txt'  # 更新pose保存路径

    with open(pose_path, 'w', encoding='utf-8'):
        pass 
    
    if not os.path.exists(image_path):
    # 如果目录不存在，创建该目录
        os.makedirs(image_path)
        print(f"目录 '{image_path}' 已创建")
    else:
        print(f"目录 '{image_path}' 已存在")
    
    T1 = time.time()

    #! 之前写的bev投影
    # for i in tqdm(range(len(dataset)),desc='[merge_hercules] Processing'):
    #     pointcloud, poses = dataset[i]
    #     bev_img, _, _ = new_getBEV(pointcloud)
    #     cv2.imwrite(f"{image_path}{i}.png", bev_img)
    
    #  #todo 生成merge_bev.txt

    #! 原始的单帧bev投影
   
    for i in tqdm(range(len(dataset)),desc='[merge_hercules] Processing'):

        pointcloud, poses = dataset[i]
        curx = poses[3]
        cury = poses[7]
        # # merged_x.append(poses[3])
        # # merged_y.append(poses[7])
        # # all_pointcloud.append(pointcloud)
        
        poses = poses.reshape(3, 4)
    
        # 添加最后一行 [0, 0, 0, 1]
        last_row = np.array([0, 0, 0, 1]).reshape(1, 4)
        poses = np.vstack((poses, last_row))
        # # all_poses.append(poses)
        
        # # if i > 100:
            
        #     # merged_pointcloud.clear()
        pcd = o3d.geometry.PointCloud()
            
        #     # for j in range(0, len(all_pointcloud), 20):
        #         # 将pointcloud从numpy数组转为Open3D点云对象
        pcd.points = o3d.utility.Vector3dVector(pointcloud)
        pcd.transform(poses) #todo 要不要用?
        #         # merged_pointcloud += pcd
            
        #     # x_mean = np.mean(merged_x)
        #     # y_mean = np.mean(merged_y)
        #     # x_std = np.std(merged_x)
        #     # y_std = np.std(merged_y)
        bev_pointcloud = pcd.voxel_down_sample(voxel_size)
            
        # # 创建绕Z轴的旋转矩阵
        yaw_random = np.random.uniform(-3.14, 3.14)
        # yaw_random = 0
        
        # # x_new = np.random.normal(loc=x_mean, scale=x_std)
        # # y_new = np.random.normal(loc=y_mean, scale=y_std)
        bev_img = getBEV(bev_pointcloud.points,curx,cury,yaw_random) 
        bev_img = np.tile(bev_img, (3, 1, 1))
        bev_img = bev_img.transpose(1, 2, 0)

        
        # print(pointcloud.shape)
        # bev_img = np.tile(bev_img, (3, 1, 1))
        # bev_img = bev_img.transpose(1, 2, 0)
        
        cv2.imwrite(f"{image_path}{i+1}.png", bev_img)
        print(f"Saved LiDAR BEV: {image_path}{i+1}.png")
        with open(pose_path, 'a') as file:
            file.write(f"{curx} {cury} {yaw_random}\n")
        
        # all_pointcloud.pop(0)
        # all_poses.pop(0)
        # merged_x.pop(0)
        # merged_y.pop(0)
    
    
    T2 = time.time()
    print("Time used:", T2-T1)
    
    print("Done")