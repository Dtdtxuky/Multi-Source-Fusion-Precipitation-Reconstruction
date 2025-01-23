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

conf_path = '~/petreloss.conf'

class CMA_Dataset(Data.Dataset):
    def __init__(self, file_path, phase='train', save_path='/mnt/petrelfs/xukaiyi/CodeSpace/DiT/DataPath/Inf_patchpath.txt', sat_list=None, data_size=256, shape=(1501,1751), invalid_path=None) -> None:
        self.phase = phase
        self.data_size = data_size
        self.client = Client(conf_path)
        self.shape = shape
        self.save_path = save_path

        self.overlap = data_size // 2
        
        self.get_data_path_patch(file_path)
        self.path_list, self.name_list, self.x_list, self.y_list, self.all_path = self.get_data_path(save_path, invalid_path, phase, 1)
        
        if sat_list==None:
            self.sat_list = [7,8,11,13]
        
    # file_path继续存放需要处理的图片路径（只有路径）
    def get_data_path_patch(self, file_path):
        np.random.seed(3407)
        with open(file_path, 'r') as file:
            image_paths = [line.strip() for line in file.readlines()]  

        patch_list = []  

        for img_path in image_paths:
            for y in range(0, self.shape[1] - self.overlap + 1, self.overlap):  
                for x in range(0, self.shape[0] - self.overlap + 1, self.overlap):
                    start_y = y
                    start_x = x
                    patch_list.append(f"{img_path},{start_x},{start_y}")

        with open(self.save_path, 'w') as output_file:
            output_file.write("\n".join(patch_list))
       
            
    def get_data_path(self, file_path, invalid_names_file, phase, train_ratio):
        np.random.seed(3407)
                
        path_list, name_list, pre_xs, pre_ys, all_path = [], [], [], [], []
        with open(file_path, 'r') as file:
            for line in file:
                all_path.append(line)
                path, pre_x, pre_y = line.strip().split(',')
                
                start_x = int(pre_x)
                start_y = int(pre_y)
                
                name = path.split('/')[-1].split('.')[0].split('_')[0]
                name = name.strip()
                
                path_list.append(path)
                name_list.append(name)
                pre_xs.append(start_x)
                pre_ys.append(start_y)      
    
            return path_list, name_list, pre_xs, pre_ys, all_path
           
    def __len__(self):
        return len(self.path_list)
    
    def get_latten_data(self, idx):
        # TODO 修改
        path = "/mnt/petrelfs/xukaiyi/CodeSpace/DiT/latent_data_187000"
        year = self.all_path[idx].split('/')[-3]
        month = self.all_path[idx].split('/')[-2][:2]
        day = self.all_path[idx].split('/')[-2]
        name = self.all_path[idx].split('/')[-1].replace('.','_').replace(',','_').replace('\n','')+".npy"
        fpath = path + "/" + year + "/" + month +"/" + day +"/" +name
        return np.random.rand(1, 32, 32)
        # try:
        #     if not os.path.exists(fpath):
        #         raise FileNotFoundError  # 主动抛出异常
        #     data = np.load(fpath)  # 读取文件
        #     data = torch.from_numpy(data)
        #     return data

        # except FileNotFoundError:
        #     print('NOT FIND')
        #     return np.random.rand(1, 32, 32)
        
    #! 要么数据全为0返回None 要么返回全部的值
    def get_satellite_data(self, idx):
        '''
        return shape: [13, data_size, data_size]
        Drop channel = 6 becaz there is no data in this channel 
        '''
        path = self.path_list[idx]
        start_x, start_y = self.x_list[idx], self.y_list[idx]
        end_x = min(start_x + int(self.data_size), self.shape[0])
        end_y = min(start_y + int(self.data_size), self.shape[1])
        
        all_data = np.zeros((14, self.data_size, self.data_size), dtype=np.float32)
        
        dict_mean_std = {'channel0':[313.011, 6.806],
                 'channel1':[313.960, 7.591],
                 'channel2':[310.269, 7.945],
                 'channel3':[330.709, 1.072],
                 'channel4':[321.074, 3.894],
                 'channel5':[325.653, 2.785],
                 'channel7':[210.173, 11.941],
                 'channel8':[251.064, 7.070],
                 'channel9':[266.398, 9.306],
                 'channel10':[267.421, 14.694],
                 'channel11':[261.457, 11.976],
                 'channel12':[265.047, 11.143],
                 'channel13':[297.262, 8.293],
                 }

        for i in range(14):
            if i not in self.sat_list:
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
                    mean, std = dict_mean_std[f'channel{i}']
                    data = (data - mean) / std
                    
                    if data.shape != (self.data_size, self.data_size):
                        padded_data = np.zeros((self.data_size, self.data_size), dtype=np.float32)
                        padded_data[:data.shape[0], :data.shape[1]] = data
                        data = padded_data
                        
                    all_data[i] = data
        
            except Exception as e:
                return None
            

        return all_data
    
    def get_land_data(self, idx):
        '''
        return shape: [5, data_size, data_size]
        '''
        path = self.path_list[idx]
        start_x, start_y = self.x_list[idx], self.y_list[idx]
        end_x = min(start_x + int(self.data_size), self.shape[0])
        end_y = min(start_y + int(self.data_size), self.shape[1])
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
                    
                    
                    if data_resized.shape != (self.data_size, self.data_size):
                        padded_data = np.zeros((self.data_size, self.data_size), dtype=np.float32)
                        padded_data[:data_resized.shape[0], :data_resized.shape[1]] = data_resized
                        data_resized = padded_data
                        
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
        end_x = min(start_x + int(self.data_size), self.shape[0])
        end_y = min(start_y + int(self.data_size), self.shape[1])
        
        
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
                
                if data_resized.shape != (self.data_size, self.data_size):
                    padded_data = np.zeros((self.data_size, self.data_size), dtype=np.float32)
                    padded_data[:data_resized.shape[0], :data_resized.shape[1]] = data_resized
                    data_resized = padded_data
                        
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
            
            name = self.name_list[idx]

            x = self.x_list[idx]
            y = self.y_list[idx]
            
            sat_data = self.get_satellite_data(idx)
            
            if sat_data is not None:
                land_data = torch.from_numpy(self.get_land_data(idx))
                oup_data = land_data[-1].unsqueeze(0)
                file = self.all_path[idx]
                
                inp_sat_data = torch.from_numpy(sat_data[self.sat_list, :, :])
                radar_data = torch.from_numpy(self.get_radar_data(idx))
                
                inp_data_land = land_data[:-1,:,:]
                inp_data = torch.cat((inp_sat_data, inp_data_land, radar_data), dim=0)   
                
                # print(x,y)
                lat_data = self.get_latten_data(idx)
                
                return {'latent':lat_data, 'inputs':inp_data, 'original':oup_data ,'file_name':file, 'Name': name, 'x': x, 'y': y}
            else:
                print(f"Invalid satellite data at idx {idx}")
                idx = (idx + 1) % len(self.path_list)
                attempts += 1
        
        
if __name__ == "__main__":
    dataset = CMA_Dataset(file_path='/mnt/petrelfs/xukaiyi/CodeSpace/SplitDataSet/Path/Month7.txt', phase='train')
    
    loader = torch.utils.data.DataLoader(dataset, batch_size=8,num_workers=8)

    for inp in loader:
        print(1)
              
         
    