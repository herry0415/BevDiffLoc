"""
@author: Ziyue Wang and Lun Lou
@file: projection.py
@time: 2025/3/12 14:20
"""

import numpy as np
import matplotlib.pyplot as plt
import open3d as o3d
import cv2
import os
import argparse
from tqdm import trange
import math

def rotate_bev_image(mat_global_image, yaw):
    """
    绕图像中心旋转 BEV 图像
    参数:
        mat_global_image: np.ndarray，输入的 BEV 图像（2D 数组）
        yaw: float，旋转角度（单位：弧度），正数表示逆时针，负数表示顺时针
    返回:
        rotated_img: np.ndarray，旋转后的 BEV 图像（尺寸不变）
    """
    # 获取图像中心
    h, w = mat_global_image.shape[:2]
    center = (w / 2.0, h / 2.0)

    # 弧度转角度（OpenCV 用角度制）
    angle_deg = yaw * 180.0 / math.pi

    # 生成旋转矩阵，保持缩放比例不变
    rotation_matrix = cv2.getRotationMatrix2D(center, angle_deg, 1.0)

    # 使用 warpAffine 进行旋转（尺寸不变）
    rotated_img = cv2.warpAffine(
        mat_global_image,
        rotation_matrix,
        (w, h),
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=0
    )

    return rotated_img

def new_getBEV(all_points, midx, midy, yaw): #N*3
    x_min, x_max = 0, 100
    y_min, y_max = -50, 50
    z_min, z_max = -10, 10 #! 这是z
    all_points_pc = o3d.geometry.PointCloud()# pcl.PointCloud()
    all_points_pc.points = o3d.utility.Vector3dVector(all_points)#all_points_pc.from_array(all_points)
    all_points_pc = all_points_pc.voxel_down_sample(voxel_size=0.4) #f = all_points_pc.make_voxel_grid_filter()
    
    
    all_points = np.asarray(all_points_pc.points)# np.array(all_points_pc.to_list())


    x_min_ind = np.floor(x_min/0.4).astype(int)
    x_max_ind = np.floor(x_max/0.4).astype(int)
    y_min_ind = np.floor(y_min/0.4).astype(int)
    y_max_ind = np.floor(y_max/0.4).astype(int)

    x_num = x_max_ind - x_min_ind + 1
    y_num = y_max_ind - y_min_ind + 1


    mat_global_image = np.zeros((y_num, x_num), dtype=np.float32) #! dtype发生改变int->float
          
    for i in range(all_points.shape[0]):
        # x_ind = x_max_ind-np.floor(all_points[i,1]/0.4).astype(int)
        # y_ind = y_max_ind-np.floor(all_points[i,0]/0.4).astype(int)
        x_ind = np.floor((all_points[i,1] - y_min)/0.4).astype(int)
        y_ind = np.floor((all_points[i,0] - x_min)/0.4).astype(int) 

        if x_ind >= x_num or y_ind >= y_num or x_ind < 0 or y_ind < 0:
            continue
        # if mat_global_image[ y_ind,x_ind]<10:
        #     mat_global_image[ y_ind,x_ind] += 1
        mat_global_image[y_ind, x_ind] += 1  # 统计点数

    mat_global_image = np.log1p(mat_global_image)  # log(count+1)，防止log(0)
    mat_global_image = np.flipud(np.fliplr(mat_global_image))

    # mat_global_image[mat_global_image<=1] = 0  
    # mat_global_image = mat_global_image*10
    if np.max(mat_global_image) > 0:
        mat_global_image = mat_global_image / np.max(mat_global_image) * 255
    
    mat_global_image = rotate_bev_image(mat_global_image, -yaw)

    # mat_global_image[np.where(mat_global_image>255)]=255
    # mat_global_image = mat_global_image/np.max(mat_global_image)*255

    return mat_global_image,x_max_ind,y_max_ind



def getBEV(all_points, midx, midy, yaw): #N*3
    
    all_points_pc = o3d.geometry.PointCloud()# pcl.PointCloud()
    all_points_pc.points = o3d.utility.Vector3dVector(all_points)#all_points_pc.from_array(all_points)
    all_points_pc = all_points_pc.voxel_down_sample(voxel_size=0.4) #f = all_points_pc.make_voxel_grid_filter()
    
    # 定义平移向量（例如平移 [1, 2, 3]）
    translation = np.array([-midx, -midy, 0])

    # 创建平移矩阵（4x4），对角线为1，最后一列是平移向量
    transformation_matrix = np.eye(4)  # 生成一个4x4单位矩阵
    transformation_matrix[:3, 3] = translation  # 只修改最后一列，保持旋转部分为单位矩阵
    all_points_pc.transform(transformation_matrix)
    
    rotation = np.array([
        [math.cos(-yaw), -math.sin(-yaw), 0],
        [math.sin(-yaw),  math.cos(-yaw), 0],
        [0, 0, 1]
    ])
    # 创建旋转矩阵（4x4）
    rotation_matrix = np.eye(4)  # 生成一个4x4单位矩阵
    rotation_matrix[:3, :3] = rotation  # 只修改最后一列，保持旋转部分为单位矩阵
    all_points_pc.transform(rotation_matrix)
    
    all_points = np.asarray(all_points_pc.points)# np.array(all_points_pc.to_list())

    x_min = -50
    y_min = -50
    x_max = +50 
    y_max = +50

    x_min_ind = np.floor(x_min/0.4).astype(int)
    x_max_ind = np.floor(x_max/0.4).astype(int)
    y_min_ind = np.floor(y_min/0.4).astype(int)
    y_max_ind = np.floor(y_max/0.4).astype(int)

    x_num = x_max_ind - x_min_ind + 1
    y_num = y_max_ind - y_min_ind + 1

    mat_global_image = np.zeros((y_num,x_num),dtype=np.uint8)
    
    for i in range(all_points.shape[0]):
        x_ind = x_max_ind-np.floor((all_points[i,1])/0.4).astype(int)
        y_ind = y_max_ind-np.floor((all_points[i,0])/0.4).astype(int)
        if(x_ind >= x_num or y_ind >= y_num or x_ind < 0 or y_ind < 0):
            continue
        if mat_global_image[ y_ind,x_ind]<10:
            mat_global_image[ y_ind,x_ind] += 1

    max_pixel = np.max(np.max(mat_global_image))

    mat_global_image[mat_global_image<=1] = 0  
    mat_global_image = mat_global_image*10
    
    mat_global_image[np.where(mat_global_image>255)]=255
    mat_global_image = mat_global_image/np.max(mat_global_image)*255

    return mat_global_image