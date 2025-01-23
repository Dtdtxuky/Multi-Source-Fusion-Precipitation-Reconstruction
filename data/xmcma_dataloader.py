import os
import os.path as osp
import numpy as np
import torch
import torch.utils.data as data
import io
from petrel_client.client import Client
from data.config_v2 import cfg
from torch.utils.data import DataLoader
import torch.nn.functional as F
import matplotlib.pyplot as plt

# def plot_jet_colormap(data, name, save_path):
#     plt.imshow(data, cmap='jet')
#     plt.colorbar()
#     plt.title(f'{name}')
#     plt.savefig(save_path)
#     plt.close()

class ListDataset(data.Dataset):
    def __init__(self, cfg, target_label, listfile):
        '''
        Args:
          root: (str) ditectory to the dataset
          list_file: (str/[str]) path to index file.
          transform: (function) normalizations before input
        '''
        self.cfg = cfg
        self.sample_records = []
        self.npy_root = self.cfg.DIR.NPY_ROOT
        
        assert target_label in self.cfg.DATA.LABEL_CHANNELS
        self.target_label = target_label
        self.input_channels = self.cfg.DATA.INPUT_CHANNELS
        self.region = self.cfg.DIR.INPUT_BIN_ROOT[-2:]
        self.psize_in = self.cfg.PATCH.IN_PATCH_SIZE
        self.psize_out = self.cfg.PATCH.OUT_PATCH_SIZE

        self.mean_sate6 = torch.tensor(self.cfg.DATA.MEAN_INPUT_SATE6, dtype=torch.float).view(6, 1, 1)
        self.std_sate6 = torch.tensor(self.cfg.DATA.STD_INPUT_SATE6, dtype=torch.float).view(6, 1, 1)

        self.ln_max_half = self.cfg.DATA.LN_MAX_HALF

        self.label_min = self.cfg.DATA.CR_MIN if self.target_label == 'RADAR_CR' else self.cfg.DATA.VIL_MIN
        self.label_max = self.cfg.DATA.CR_MAX if self.target_label == 'RADAR_CR' else self.cfg.DATA.VIL_MAX
        self.label_mean = self.cfg.DATA.CR_MEAN if self.target_label == 'RADAR_CR' else self.cfg.DATA.VIL_MEAN
        self.label_std = self.cfg.DATA.CR_STD if self.target_label == 'RADAR_CR' else self.cfg.DATA.VIL_STD
        self.client = Client(self.cfg.DATA.PETREL_CONF)
        
        self.inchannel_max = torch.tensor([300.0, 250.0, 300.0, 50.0]).view(-1, 1, 1)
        self.inchannel_min = torch.tensor([200.0, 200.0, 200.0, 0.1]).view(-1, 1, 1)

        with open(listfile) as f:
            lines = f.readlines()
            self.num_samples = len(lines)
            # self.num_samples = 100000

        for line in lines:
            record = line.strip()
            yearmon, dt = record.split(',')[0:2]
            patch_ind_x, patch_ind_y, in_start_x, in_start_y, out_start_x, out_start_y = record.split(',')[-6:]
            rec_dict = {
                'yearmon': yearmon,
                'dt': dt,
                'patch_ind_x': int(patch_ind_x),
                'patch_ind_y': int(patch_ind_y),
                'in_start_x': int(in_start_x),
                'in_start_y': int(in_start_y),
                'out_start_x': int(out_start_x),
                'out_start_y': int(out_start_y)
            }
            self.sample_records.append(rec_dict)
            
    def __getitem__(self, idx):
        rec = self.sample_records[idx]
        inputs_sate = []
        for channel in self.input_channels:
            fnpy = osp.join(self.npy_root, channel, rec['yearmon'], self.region+'_'+channel+'_'+rec['dt']+'.npy')
            with io.BytesIO(self.client.get(fnpy)) as f:
                if channel != 'LN10':
                    inputs_sate.append(torch.from_numpy(np.load(f)).type(torch.float).unsqueeze(0))
                else:
                    input_ln = torch.from_numpy(np.load(f)).type(torch.float).unsqueeze(0)
        fnpy = osp.join(self.npy_root, self.target_label, rec['yearmon'], self.region+'_'+self.target_label+'_'+rec['dt']+'.npy')
        with io.BytesIO(self.client.get(fnpy)) as f:
            label = torch.from_numpy(np.load(f)).type(torch.float).unsqueeze(0)
        ## get patch slice
        inputs_sate = torch.cat(inputs_sate, dim=0)
        inputs_sate = inputs_sate[:, rec['in_start_x']:rec['in_start_x']+self.psize_in, rec['in_start_y']:rec['in_start_y']+self.psize_in]
        input_ln = input_ln[:, rec['in_start_x']:rec['in_start_x']+self.psize_in, rec['in_start_y']:rec['in_start_y']+self.psize_in]
        label = label[:, rec['out_start_x']:rec['out_start_x']+self.psize_out, rec['out_start_y']:rec['out_start_y']+self.psize_out]
        # print('label shape:', label.shape)
        label = F.interpolate(label.unsqueeze(0), size=(256, 256), mode='bilinear', align_corners=False).squeeze(0)
        # print('label begin shape is:', label.shape)
        ## mask invalid values and normalize
        sate_mask_invalid = inputs_sate < 0
        label_mask_invalid = label < -1279.9 #-1280
        loss_wts_delta = torch.tensor(self.cfg.TRAIN.REV_LOSS_WTS) - torch.tensor([0] + self.cfg.TRAIN.REV_LOSS_WTS[:-1])
        loss_wts_delta = loss_wts_delta.type(torch.float) # * 1e-3
        loss_wt_mask_ =  loss_wts_delta[0] * torch.ones_like(label) + loss_wts_delta[1] * (label >= 100) + \
                         loss_wts_delta[2] * (label >= 200) + loss_wts_delta[3] * (label >= 350)
                         
        # inputs_sate = (inputs_sate - self.mean_sate6) / self.std_sate6
        # input_ln = input_ln / self.ln_max_half
        # label = (label - self.label_mean) / self.label_std

        inputs_sate[sate_mask_invalid] = 0
        label[label_mask_invalid] = 0
        
        # #! Clip label [0, 60]
        # final_label = torch.clamp(label, min=0.0, max=60.0) 
        channel1 = inputs_sate[1]
        channel2 = inputs_sate[2]
        channel4 = inputs_sate[4]
        channel6 = input_ln[0]
               
        final_input = torch.stack([channel1, channel4, channel2, channel6], dim=0)
        
        #! IR39, VAP69, IR104, LN10 
        #! norm input data and oup data
        # final_input = (final_input - self.inchannel_min) / (self.inchannel_max - self.inchannel_min)
        final_input = (self.inchannel_max - final_input) / (self.inchannel_max - self.inchannel_min)
        final_input = torch.clamp(final_input, 0.0, 1.0)
        
        # label = (label - self.label_mean) / self.label_std
        label = label / 450.0
        label = torch.clamp(label, 0.0, 1.0)

        # sate_invalidity = sate_mask_invalid.to(torch.float).sum(dim=0, keepdim=True) / 6
        # inputs = torch.cat([inputs_sate, input_ln, sate_invalidity], dim=0)
        label_mask = 1 - label_mask_invalid.to(torch.float)
        loss_wt_mask = label_mask * loss_wt_mask_ * 1e-3
        loss_wt_mask = loss_wt_mask.type(torch.float)
        
        

        # return inputs, label, label_mask, loss_wt_mask
        return {'HR': label, 'SR':final_input, 'label_mask':label_mask, 'loss_wt_mask': loss_wt_mask}

    def __len__(self):
        return self.num_samples
 
 
if __name__ == "__main__":
       
    train_dataset = ListDataset(cfg=cfg, target_label='RADAR_CR', listfile=cfg.DIR.TRAIN_LIST_CR)

    train_loader = DataLoader(dataset=train_dataset, batch_size=cfg.TRAIN.BATCH_SIZE, shuffle=True, 
            num_workers=2, pin_memory=False, drop_last=False)

    for a, b, c, d in train_loader:
        inp, label, label_mask, losswt_mask = a, b, c, d


    print("ss")