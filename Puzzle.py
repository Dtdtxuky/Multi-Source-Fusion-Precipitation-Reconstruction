# 将同一时间戳的patch拼成大图
# overlap的地方取平均
# patch有超过边界的地方，用0来填补了，最后要把0去掉

# 从集群中读出该时间戳下的所有文件
# path格式：cluster3:s3://zwl2/PreRec/gt_2021072600_0_0.npy
# 读文件：path：

import numpy as np
from collections import defaultdict
from petrel_client.client import Client
import io
import cv2
import time as times
import torch 
import random
import numpy as np
import xarray as xr
from petrel_client.client import Client
from torch.utils import data as Data
import os
import json
import argparse
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
import torch
import torch.nn.functional as F
from scipy.ndimage import uniform_filter

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from mpl_toolkits.basemap import Basemap

conf_path = '~/petreloss.conf'
client = Client(conf_path)


def get_result_matrix(target, pred, thrs, stats, name, visualizer = None):

    bs = target.shape[0]

    # pool操作
    def pool_data(batch, pool_size):
        h, w = batch.shape[-2:]
        new_h = (h // pool_size) * pool_size
        new_w = (w // pool_size) * pool_size
        if batch.ndim == 2:  # If 2D, add batch and channel dimensions
            batch = batch[np.newaxis, np.newaxis, :new_h, :new_w]
        batch_tensor = torch.from_numpy(batch)
        return F.max_pool2d(batch_tensor, kernel_size=pool_size, stride=pool_size)
    
    target = np.array(target)
    pred = np.array(pred)
    
    target_pool1 = target
    pred_pool1 = pred
    
    target_pool4 = pool_data(target, 4)
    pred_pool4 = pool_data(pred, 4)
    
    target_pool8 = pool_data(target, 8)
    pred_pool8 = pool_data(pred, 8)
    
    target_pool4 = target_pool4.numpy()
    pred_pool4 = pred_pool4.numpy()

    target_pool8 = target_pool8.numpy()
    pred_pool8 = pred_pool8.numpy()
    
    # 计算不同thr下的csi
    for thr in thrs:
        has_event_target_pool1 = (target_pool1 >= thr)
        has_event_predict_pool1 = (pred_pool1 >= thr)

        hit_pool1 = np.sum(has_event_target_pool1 & has_event_predict_pool1).astype(int)
        miss_pool1 = np.sum(has_event_target_pool1 & ~has_event_predict_pool1).astype(int)
        false_alarm_pool1 = np.sum(~has_event_target_pool1 & has_event_predict_pool1).astype(int)

        if hit_pool1 + miss_pool1 + false_alarm_pool1 != 0:
            csi_pool1 = hit_pool1 / (hit_pool1 + miss_pool1 + false_alarm_pool1)
        else: csi_pool1 = np.nan

        has_event_target_pool4 = (target_pool4 >= thr)
        has_event_predict_pool4 = (pred_pool4 >= thr)

        hit_pool4 = np.sum(has_event_target_pool4 & has_event_predict_pool4).astype(int)
        miss_pool4 = np.sum(has_event_target_pool4 & ~has_event_predict_pool4).astype(int)
        false_alarm_pool4 = np.sum(~has_event_target_pool4 & has_event_predict_pool4).astype(int)

        if hit_pool4 + miss_pool4 + false_alarm_pool4 != 0:
            csi_pool4 = hit_pool4 / (hit_pool4 + miss_pool4 + false_alarm_pool4)
        else: csi_pool4 = np.nan
        
        has_event_target_pool8 = (target_pool8 >= thr)
        has_event_predict_pool8 = (pred_pool8 >= thr)

        hit_pool8 = np.sum(has_event_target_pool8 & has_event_predict_pool8).astype(int)
        miss_pool8 = np.sum(has_event_target_pool8 & ~has_event_predict_pool8).astype(int)
        false_alarm_pool8 = np.sum(~has_event_target_pool8 & has_event_predict_pool8).astype(int)

        if hit_pool8 + miss_pool8 + false_alarm_pool8 != 0:
            csi_pool8 = hit_pool8 / (hit_pool8 + miss_pool8 + false_alarm_pool8)
        else: csi_pool8 = np.nan

        stats['csi pool1'][0][thr].append(csi_pool1)
        stats['csi pool4'][0][thr].append(csi_pool4)
        stats['csi pool8'][0][thr].append(csi_pool8)
        
        print(f'Step:{name}, csi_pool1_thr{thr}:{csi_pool1}, csi_pool4_thr{thr}:{csi_pool4}, csi_pool8_thr{thr}:{csi_pool8}' )
    
    rmse = np.sqrt(np.mean((target - pred) ** 2))
    print(f'rmse:{rmse}')
    stats['rmses'].append(rmse)
    
    return stats

    
def load_and_merge_patches_from_txt(timestamp, txt_file, label, patch_size=256):
    """
    从给定的 txt 文件中读取同一时间戳的 patch 路径，并将其拼成大图
    Args:
        timestamp (str): 时间戳，例如 "2021072600"
        txt_file (str): 存储所有 patch 路径的 txt 文件路径
        patch_size (int): 每个 patch 的尺寸 (默认是 256x256)
    Returns:
        np.ndarray: 拼接后的大图，去掉 0 填补部分
    """
    # 读取 txt 文件中的所有路径
    with open(txt_file, 'r') as f:
        all_paths = f.readlines()
    
    # 筛选符合时间戳的路径
    file_paths = [line.strip() for line in all_paths if f"{timestamp}" in line]
    
    if not file_paths:
        raise ValueError(f"No files found for timestamp {timestamp} in txt file {txt_file}")
    
    # 存储所有 patch 及其坐标信息
    patch_dict = defaultdict(list)
    
    # 解析每个文件的起始坐标并读取数据
    for file_path in file_paths:
        _, x,y  = file_path.split(",")
        
        x = int(x)
        y = int(y)
        
        path = '/mnt/petrelfs/xukaiyi/CodeSpace/DiT/SaveNpy/SavePatch/'+ label + '_' + timestamp + '_' + str(x) + '_' + str(y) +'.npy'
        # 加载 patch 数据, 从集群上读取
        

        patch = np.load(path) 
        
        # 将 patch 和坐标存储
        patch_dict[(x, y)].append(patch)
    
    # 计算拼接大图的尺寸
    max_x = max(coord[0] for coord in patch_dict.keys()) + patch_size
    max_y = max(coord[1] for coord in patch_dict.keys()) + patch_size
    
    # 创建空白大图，超出边界用 0 填补
    large_image = np.zeros((max_x, max_y), dtype=np.float32)
    overlap_count = np.zeros((max_x, max_y), dtype=np.int32)
    
    # 将每个 patch 添加到大图中
    for (x, y), patches in patch_dict.items():
        # 计算 patch 坐标范围
        x_start, x_end = x, x + patch_size
        y_start, y_end = y, y + patch_size
        
        # 累加到大图，同时记录计数
        large_image[x_start:x_end, y_start:y_end] += patches[0]
        overlap_count[x_start:x_end, y_start:y_end] += 1
    
    # 处理重叠区域取平均
    overlap_mask = overlap_count > 0
    large_image[overlap_mask] /= overlap_count[overlap_mask]
    
    # 提取子区域
    sub_image = large_image[0:1501, 0:1751]
    np.save('/mnt/petrelfs/xukaiyi/CodeSpace/DiT/SaveNpy/SaveAll/'+ label + '_' + timestamp + '.npy', sub_image)
    
    save_path_nc = '/mnt/petrelfs/xukaiyi/CodeSpace/DiT/SaveNpy/Savenc/'
    savenc(save_path_nc +  label + '_' + timestamp + '.nc', sub_image)
    
    return sub_image

def savenc(path, data):
    data = cv2.resize(data, (7001, 6001), interpolation=cv2.INTER_LINEAR)
    # 定义纬度和经度坐标
    lat = np.linspace(0, 60, data.shape[0])
    lon = np.linspace(70, 140, data.shape[1])

    # 创建 DataArray
    data_array = xr.DataArray(
        data,
        dims=["lat", "lon"],
        coords={"lat": lat, "lon": lon},
        name="data"
    )

    # 创建 Dataset
    dataset = xr.Dataset({"data": data_array})
    dataset.to_netcdf(path)
    
if __name__=='__main__':
    # 指标初始化
    thrs = [1., 2., 5., 10., 15., 20.]
    stats = {}    
    stats['thr'] = thrs
    stats['csi pool1'] = [{thr: [] for thr in thrs}]
    stats['csi pool4'] = [{thr: [] for thr in thrs}]
    stats['csi pool8'] = [{thr: [] for thr in thrs}]
    stats['rmses'] = []

    # 给出记录集群上路径的txt的地址
    p_path = '/mnt/petrelfs/xukaiyi/CodeSpace/DiT/DataPath/Inf_patchpath.txt'
    
    # 读取时间戳txt
    timetamp = '/mnt/petrelfs/xukaiyi/CodeSpace/DiT/DataPath/Inf_allpath.txt'
    
    with open(timetamp, 'r') as file:
        for path in file:
            tamp = path.split("/")[-1].split("_")[0]
            gt = load_and_merge_patches_from_txt(tamp, p_path, 'gt')
            pre = load_and_merge_patches_from_txt(tamp, p_path, 'pred')
            stats = get_result_matrix(gt, pre, thrs, stats, timetamp)
            
            # 插值到（6001，7001）
            gt_resized = cv2.resize(gt, (7001, 6001), interpolation=cv2.INTER_LINEAR)
            pre_resized = cv2.resize(pre, (7001, 6001), interpolation=cv2.INTER_LINEAR)
    
            

    # 计算所有指标的平均值
    avg_stats = {
        'avg_csi_pool1': {},
        'avg_csi_pool4': {},
        'avg_csi_pool8': {},
        'avg_rmse': None
    }
                    
    for thr in thrs:
        csi_pool1_values = [x for x in stats['csi pool1'][0][thr] if not np.isnan(x)]
        csi_pool4_values = [x for x in stats['csi pool4'][0][thr] if not np.isnan(x)]
        csi_pool8_values = [x for x in stats['csi pool8'][0][thr] if not np.isnan(x)]

        avg_stats['avg_csi_pool1'][thr] = np.mean(csi_pool1_values) if csi_pool1_values else np.nan
        avg_stats['avg_csi_pool4'][thr] = np.mean(csi_pool4_values) if csi_pool4_values else np.nan
        avg_stats['avg_csi_pool8'][thr] = np.mean(csi_pool8_values) if csi_pool8_values else np.nan

    rmse_values = [x for x in stats['rmses'] if not np.isnan(x)]
    avg_stats['avg_rmse'] = np.mean(rmse_values) if rmse_values else np.nan
    print(f'Average stats: {avg_stats}')

    