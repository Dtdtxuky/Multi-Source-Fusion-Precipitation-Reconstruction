import numpy as np
import os
from torch.utils.data import Dataset



class MyDataset(Dataset):
    def __init__(self, data_path, transform=None):
        self.data_path = data_path
        self.transform = transform

        self.samples = []
        v = data_path
        for f in os.listdir(v):
            if f.endswith('.npz'):
                self.samples.append(os.path.join(v, f))
        self.samples.sort()

        self.xshape, self.tshape = self._get_shape()

    def _get_shape(self):
        with np.load(self.samples[0]) as data:
            xshape = np.moveaxis(data['xdata'], -1, 0).shape
            # xshape = data['xdata'].shape
            tshape = data['ydata'][np.newaxis, ...].shape
            # tshape = data['ydata'][..., np.newaxis].shape
        return xshape, tshape

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        f = self.samples[idx]
        with np.load(f) as data:  # C x H x W
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
    data_path = "/mnt/petrelfs/hexuming/sat2rad_patch_dataset/train"
    dataset = MyDataset(data_path)
    print(len(dataset))
