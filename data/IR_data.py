import io
import cv2
import time as times
import torch 
import random
import numpy as np
import xarray as xr
from petrel_client.client import Client
from torch.utils import data as Data

conf_path = '~/petreloss.conf'

class CMA_Dataset(Data.Dataset):
    def __init__(self, file_path, phase='train', data_size=256, invalid_path=None) -> None:
        self.phase = phase
        self.data_size = data_size
        self.path_list, self.name_list, self.x_list, self.y_list = self.get_data_path(file_path, invalid_path, phase, 1)
        
        self.client = Client(conf_path)
        
        
    def get_data_path(self, file_path, invalid_names_file, phase, train_ratio):
        np.random.seed(3407)
        
        # invalid_names = set()
        # with open(invalid_names_file, 'r') as file:
        #     for line in file:
        #         invalid_name = line.strip()
        #         invalid_names.add(invalid_name)
                
        
        path_list, name_list, pre_xs, pre_ys = [], [], [], []
        with open(file_path, 'r') as file:
            for line in file:
                path, shape_x, shape_y, pre_x, pre_y = line.strip().split(',')
                
                start_x = int(pre_x)
                start_y = int(pre_y)
                
                name = path.split('/')[-1].split('.')[0].split('_')[0]
                name = name.strip()
                
                # if name in invalid_names:
                #     continue
                
                path_list.append(path)
                name_list.append(name)
                pre_xs.append(start_x)
                pre_ys.append(start_y)
                
        # 打乱数据集,生成随机索引
        lens = len(path_list)
        indices = np.random.permutation(lens) 
        
        path_list = [path_list[i] for i in indices]
        name_list = [name_list[i] for i in indices]
        pre_xs = [pre_xs[i] for i in indices]
        pre_ys = [pre_ys[i] for i in indices]

        train_num = int(lens * train_ratio)
        if phase == 'train':      
            return path_list[:train_num], name_list[:train_num], pre_xs[:train_num], pre_ys[:train_num]
        else:
            return path_list[train_num:], name_list[train_num:], pre_xs[train_num:], pre_ys[train_num:]
        
    def __len__(self):
        return len(self.path_list)
    
    #! 要么数据全为0返回None 要么返回全部的值
    def get_satellite_data(self, idx):
        '''
        return shape: [13, data_size, data_size]
        Drop channel = 6 becaz there is no data in this channel 
        '''
        path = self.path_list[idx]
        start_x, start_y = self.x_list[idx], self.y_list[idx]
        end_x = start_x + int(self.data_size)
        end_y = start_y + int(self.data_size)
        all_data = np.zeros((13, self.data_size, self.data_size), dtype=np.float32)
        
        dict_mean_std = {'channel0':[313.011, 6.806],
                 'channel1':[313.960, 7.591],
                 'channel2':[310.269, 7.945],
                 'channel3':[330.709, 1.072],
                 'channel4':[321.074, 3.894],
                 'channel5':[325.653, 2.785],
                 'channel6':[210.173, 11.941],
                 'channel7':[251.064, 7.070],
                 'channel8':[266.398, 9.306],
                 'channel9':[267.421, 14.694],
                 'channel10':[261.457, 11.976],
                 'channel11':[265.047, 11.143],
                 'channel12':[297.262, 8.293],
                 }

        index = 0
        for i in range(14):
            
            #! Drop channel=6 becaz There is no data in this channel
            if i == 6:
                continue
            
            satellite_path = path.replace('cma_land', 'cma_satellite').replace('nc2001x2334', 'fy4a').replace('_pre', f'0000_agri_4000M_{i}').replace('cluster2:s3', 's3')
            
            try:
                satellite_data = self.client.get(satellite_path)
                with io.BytesIO(satellite_data) as f:
                    src_data = xr.open_dataset(f, engine='h5netcdf')
                    data = src_data.variables['var'][start_x:end_x, start_y:end_y].values
                    data[data == 65535] = 0
                    data = np.clip(data, a_min=0, a_max=340)
                    # mean-std norm
                    mean, std = dict_mean_std[f'channel{index}']
                    data = (data - mean) / std
                    
                    all_data[index] = data
            except Exception as e:
                # print(f"Error sat {self.name_list[idx]} channel {i}: {e}")
                #! 直接将不好的数据输出None就好
                return None
            
            index += 1

        return all_data
    
    def get_land_data(self, idx):
        '''
        return shape: [5, data_size, data_size]
        '''
        path = self.path_list[idx]
        start_x, start_y = self.x_list[idx], self.y_list[idx]
        end_x = start_x + int(self.data_size)
        end_y = start_y + int(self.data_size)
        land_suffixes = ['_u10.nc', '_v10.nc', '_r2.nc', '_t2m.nc', '_pre.nc']
        all_data = np.zeros((5, self.data_size, self.data_size), dtype=np.float32)

        for i, suffix in enumerate(land_suffixes):
            land_path = path.replace('_pre.nc', suffix)
            
            try:
                land_data = self.client.get(land_path)
                var_name = suffix.split('.')[0][1:]

                with io.BytesIO(land_data) as f:
                    data_array = xr.open_dataset(f, engine='h5netcdf')
                    var_name = 'OI_merge' if var_name == 'pre' else var_name
                    original_data = data_array[var_name].values

                    if var_name == 't2m':
                        original_data[original_data == 9999.0] = 0
                        original_data = np.clip(original_data, a_min=213.15, a_max=333.15)
                        
                        original_data = self.min_max_norm(original_data, 260., 312.)
                        
                    elif var_name == 'r2':
                        original_data = np.clip(original_data, a_min=0.0, a_max=100.0)
                        
                        original_data = self.min_max_norm(original_data, 20., 100.)
                        
                    elif var_name in ['u10', 'v10']:
                        original_data[original_data < -1000.0] = 0.0
                        original_data[original_data > 1000.0] = 0.0
                        original_data = np.clip(original_data, a_min=-1000.0, a_max=1000.0)
                        
                        original_data = self.min_max_norm(original_data, -12., 9.)
                        
                    elif var_name == 'OI_merge':
                        original_data[original_data < 0] = 0
                        original_data = np.clip(original_data, a_min=0.0, a_max=300.0)
                        
                        original_data = self.sqrlog_minmax_norm(original_data, 0., 50.)

                    data_resized = cv2.resize(original_data, (1751, 1501), interpolation=cv2.INTER_LINEAR)
                    data_resized = data_resized[start_x:end_x, start_y:end_y]
                    all_data[i] = data_resized
            except Exception as e:
                print(f"Error land {self.name_list[idx]} channel {i}: {e}")
                
        return all_data
    
    def get_radar_data(self, idx):
        '''
        return shape: [1, data_size, data_size]
        '''

        path = self.path_list[idx]
        radar_path = path.replace('cluster2:s3', 's3').replace('cma_land', 'cma_radar').replace('nc2001x2334', 'radar_nmic_nc').replace('_pre', '0000_CR')
        
        start_x, start_y = self.x_list[idx], self.y_list[idx]
        end_x = start_x + int(self.data_size)
        end_y = start_y + int(self.data_size)
        
        # print(path, start_x, start_y)
        
        
        radar_datas = np.zeros((1, self.data_size, self.data_size), dtype=np.float32)
        
                        
        try:
            radar_data = self.client.get(radar_path)
            with io.BytesIO(radar_data) as f:
                src_data = xr.open_dataset(f, engine='h5netcdf')
                original_data = src_data.variables['var'].values

                if original_data.shape[0] == 4100:
                    original_data = np.pad(original_data, 
                                        pad_width=((1220, 681), (300, 501)), 
                                        mode='constant', 
                                        constant_values=0)  

                data_resized = cv2.resize(original_data, (1751, 1501), interpolation=cv2.INTER_LINEAR)
                data_resized = data_resized[start_x:end_x, start_y:end_y]
                radar_datas[0] = data_resized
                radar_datas = self.min_max_norm(radar_datas, 0., 65.)
                
        except Exception as e:
            print(f"Error radar {self.name_list[idx]}: {e}")
        return radar_datas
            
    
    def min_max_norm(self, data, min_val, max_val):
        norm_data = (data - min_val) / (max_val - min_val)
        return norm_data
    
    def sqrlog_minmax_norm(self, data, min_val, max_val):
        
        norm_data = (data - min_val)/(max_val-min_val)
        sqrt_data = np.sqrt(norm_data)
        log_data = np.log1p(sqrt_data)
        
        return log_data
    
    def mean_std_norm(self, data, mean, std):
        norm_data = (data - mean) / std 
        return norm_data
    
    def __getitem__(self, idx):
        max_attempts = len(self.path_list)
        attempts = 0

        for attempt in range(max_attempts):

            sat_data = self.get_satellite_data(idx)
            if sat_data is not None:
                land_data = torch.from_numpy(self.get_land_data(idx))
                oup_data = land_data[-1].unsqueeze(0)
                inp_data = torch.from_numpy(sat_data[6:, :, :])
                mask = torch.ones_like(oup_data)
                return {'HR': oup_data, 'SR':inp_data, 'label_mask': mask, 'loss_wt_mask':mask}
            else:
                print(f"Invalid satellite data at idx {idx}")
                idx = (idx + 1) % len(self.path_list)
                attempts += 1
        
        
if __name__ == "__main__":
    file_path = '/mnt/petrelfs/zhouzhiwang/codeespace/cma_data/cma_mp/coordinates_3years_pre_window256_th2_effectivesize500.txt'
    dataset = CMA_Dataset(file_path, phase='train')
    
    
    loader = torch.utils.data.DataLoader(dataset, batch_size=8, num_workers=8)

    for inp, oup in loader:
        print(inp.shape, oup.shape)
              
         
    
        
