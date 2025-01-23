import numpy as np
import os
from torch.utils.data import Dataset
from petrel_client.client import Client
import io

class Sat2RadDataset(Dataset):
    def __init__(
        self, data_folder, 
        transform=None, 
        conf_path='~/petrel-oss-python-sdk/conf/petreloss.conf',
        start_id=1,
        end_id=5
    ):
        self.data_folder = data_folder
        self.transform = transform
        self.client = Client(conf_path)
        self.samples = []
        v = data_folder
        for id in range(start_id, end_id + 1):
            f = f'conus3_regA_sclA_{id:06d}.npz'
            url = f'{data_folder}/{f}'
            self.samples.append(url)
        self.samples.sort()

        self.xshape, self.tshape = self._get_shape()

    def _get_shape(self):
        url = self.samples[0]
        bytes_data = self.client.get(url)
        data_file = io.BytesIO(bytes_data)
        with np.load(data_file) as data:
            xshape = np.moveaxis(data['xdata'], -1, 0).shape
            # xshape = data['xdata'].shape
            tshape = data['ydata'][np.newaxis, ...].shape
            # tshape = data['ydata'][..., np.newaxis].shape
        return xshape, tshape

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        url = self.samples[idx]
        bytes_data = self.client.get(url)
        data_file = io.BytesIO(bytes_data)
        with np.load(data_file) as data:  # C x H x W
            x = np.moveaxis(data['xdata'], -1, 0)
            # x = data['xdata']
            t = data['ydata'][np.newaxis, ...]
            # t = data['ydata'][..., np.newaxis]
            # t = np.repeat(t, 4, axis=0)
        if self.transform is not None:
            x = self.transform(x)
            t = self.transform(t)

        return {'HR': t, 'SR': x, 'Index': idx}

if __name__ == "__main__":
    data_folder = "cluster3:s3://zwl2/sat2rad_patch_dataset/val"
    dataset = Sat2RadDataset(data_folder)
    print(len(dataset))
