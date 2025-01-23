import os
import json
import argparse
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from multiprocessing.pool import ThreadPool
# from skimage.metrics import peak_signal_noise_ratio as psnr
# from skimage.metrics import structural_similarity as ssim
from scipy.ndimage import uniform_filter
# from utils import sevir_visualizer

import torch
import torch.nn.functional as F
parser = argparse.ArgumentParser()
parser.add_argument('--input', type=str,
                    default='/mnt/petrelfs/hexuming/test_2month',
                    help='data directory')
parser.add_argument('--results', type=str,
                    default='/mnt/petrelfs/hexuming/compare/SR3_1cond_iter=200_pred_2month',
                    help='results directory')

################################################################

refthrs_default = np.arange(5, 55, 5)
ymax = 60.0  # value from Hilburn et al. (2020)

################################################################

def restore_large_image(patches, coords, image_shape=(1501,1751), patch_size=256):
    """
    将小图根据坐标还原为完整大图。
    
    Args:
        patches (list of np.ndarray): 小图列表。
        coords (list of tuple): 每个小图的 (x, y) 坐标。
        image_shape (tuple): 大图的目标形状 (height, width)。
        patch_size (int): 小图的大小。

    Returns:
        np.ndarray: 还原后的完整大图。
    """
    large_image = np.zeros(image_shape, dtype=patches[0].dtype)

    for patch, (x, y) in zip(patches, coords):
        x = int(x)
        y = int(y)
        large_image[x:x + patch_size, y:y + patch_size] = patch
    
    return large_image

def load_ty(args):
    xtf, yf = args
    with np.load(xtf) as data:  # C x H x W
        # x = np.flip(np.moveaxis(data['xdata'], -1, 0), axis=1)
        t = np.flip(data['ydata'][np.newaxis, ...], axis=1) * ymax
    y = np.load(yf).squeeze()[np.newaxis, ...]
    y = np.flip(y, axis=1) * ymax
    # y = np.flip(np.load(yf), axis=1) * ymax
    print("t.shape", t.shape, "y.shape", y.shape)
    return t, y


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

        if(visualizer != None):
            hit1 = has_event_target_pool1 & has_event_predict_pool1
            miss1 = has_event_target_pool1 & ~has_event_predict_pool1
            false_alarm1 = ~has_event_target_pool1 & has_event_predict_pool1        
            no_event1 = ~has_event_target_pool1 & ~has_event_predict_pool1  # Both are 0
            
            # Create a visualization array
            visualization = np.zeros_like(target, dtype=int)
            visualization[hit1] = 1          # Mark "hit" regions
            visualization[miss1] = 2         # Mark "miss" regions
            visualization[false_alarm1] = 3  # Mark "false alarm" regions
            visualization[no_event1] = 4     # Mark "no event" regions

            # Define a colormap
            from matplotlib.colors import ListedColormap
            colors = ['black', 'green', 'red', 'blue', 'gray']  # Colors for each category
            cmap = ListedColormap(colors)

            # Visualize
            plt.figure(figsize=(15, 10))
            plt.imshow(visualization[::-1], cmap=cmap, interpolation='none')
            plt.colorbar(ticks=range(5), label='Category')
            plt.clim(0, 4)
            plt.title("Event Classification Visualization")
            plt.axis('off')
            
            path = f'{visualizer.save_dir}/{name}_{thr}.png'
            plt.savefig(path, dpi=300, bbox_inches='tight')
            print(path)
            
            
            
    rmse = np.sqrt(np.mean((target - pred) ** 2))
    print(f'rmse:{rmse}')
    stats['rmses'].append(rmse)
    
    return stats
        
    
def get_refc_stats(goes_filenames, mrms_filenames, batch_size=100, refthrs=refthrs_default):
    '''
    inputs:
        goes_filenames: list of str, filenames of GOES data (npy files)
        mrms_filenames: list of str, filenames of MRMS data (npy files)
        batch_size: int, batch size for processing
        refthrs: np array, thresholds for statistics computation
    outputs:
        stats: dictionary of stats
    '''

    goes_sample = np.load(goes_filenames[0])
    goes_shape = goes_sample.shape
    num_samples = len(goes_filenames)
    n_batches = int(np.ceil(num_samples / batch_size))
    # print("num samples: ", num_samples)
    print('=> starting...')
    stats = {}
    stats['ref'] = refthrs
    stats['pod'] = []
    stats['far'] = []
    stats['csi pool1'] = []
    stats['csi pool4'] = []
    stats['csi pool8'] = []
    stats['bias'] = []
    stats['rmses'] = []
    # diff_mean_total = 0
    diff_sumsq_total = 0
    # r2 vars
    tsum = 0
    tsumsqrd = 0

    for i, rthr in tqdm(enumerate(refthrs), total=len(refthrs)):
        nhasrads_total = 0
        nhit_total = 0
        nmis_total = 0
        nfal_total = 0
        rthr_diff_sumsq_total = 0

        nhasrads_total_pool4 = 0
        nhit_total_pool4 = 0
        nmis_total_pool4 = 0
        nfal_total_pool4 = 0

        nhasrads_total_pool8 = 0
        nhit_total_pool8 = 0
        nmis_total_pool8 = 0
        nfal_total_pool8 = 0

        with ThreadPool(32) as pool:
            for batch_start in tqdm(range(0, num_samples, batch_size), total=n_batches, leave=False):
                batch_end = min(batch_start + batch_size, num_samples)

                goes_batch, mrms_batch = zip(*pool.map(load_ty, [(mrms_filenames[j], goes_filenames[j])
                                                                 for j in range(batch_start, batch_end)]))

                def pool_data(batch, pool_size):
                    h, w = batch.shape[-2:]
                    new_h = (h // pool_size) * pool_size
                    new_w = (w // pool_size) * pool_size
                    new_batch = batch[..., :new_h, :new_w]
                    batch_tensor = torch.from_numpy(new_batch)
                    return F.max_pool2d(batch_tensor, kernel_size=pool_size, stride=pool_size)
                    return F.avg_pool2d(batch_tensor, kernel_size=pool_size, stride=pool_size)


                goes_batch = np.array(goes_batch)
                mrms_batch = np.array(mrms_batch)
                
                goes_batch_pool4 = pool_data(goes_batch, 4)
                mrms_batch_pool4 = pool_data(mrms_batch, 4)

                goes_batch_pool8 = pool_data(goes_batch, 8)
                mrms_batch_pool8 = pool_data(mrms_batch, 8)

                goes_batch_pool4 = goes_batch_pool4.numpy()
                mrms_batch_pool4 = mrms_batch_pool4.numpy()

                goes_batch_pool8 = goes_batch_pool8.numpy()
                mrms_batch_pool8 = mrms_batch_pool8.numpy()

                goes_batch[goes_batch < 0] = 0.
                mrms_batch[mrms_batch < 0] = 0.

                goes_batch_pool4[goes_batch_pool4 < 0] = 0.
                mrms_batch_pool4[mrms_batch_pool4 < 0] = 0.

                goes_batch_pool8[goes_batch_pool8 < 0] = 0.
                mrms_batch_pool8[mrms_batch_pool8 < 0] = 0.

                hasrad = mrms_batch > rthr
                hassat = goes_batch > rthr

                hasrad_pool4 = mrms_batch_pool4 > rthr
                hassat_pool4 = goes_batch_pool4 > rthr

                hasrad_pool8 = mrms_batch_pool8 > rthr
                hassat_pool8 = goes_batch_pool8 > rthr

                nhit = np.sum(hasrad & hassat)
                nmis = np.sum(hasrad & ~hassat)
                nfal = np.sum(~hasrad & hassat)

                nhit_pool4 = np.sum(hasrad_pool4 & hassat_pool4)
                nmis_pool4 = np.sum(hasrad_pool4 & ~hassat_pool4)
                nfal_pool4 = np.sum(~hasrad_pool4 & hassat_pool4)

                nhit_pool8 = np.sum(hasrad_pool8 & hassat_pool8)
                nmis_pool8 = np.sum(hasrad_pool8 & ~hassat_pool8)
                nfal_pool8 = np.sum(~hasrad_pool8 & hassat_pool8)
                

                nhasrads_total += np.sum(hasrad)
                nhit_total += nhit
                nmis_total += nmis
                nfal_total += nfal

                nhasrads_total_pool4 += np.sum(hasrad_pool4)
                nhit_total_pool4 += nhit_pool4
                nmis_total_pool4 += nmis_pool4
                nfal_total_pool4 += nfal_pool4

                nhasrads_total_pool8 += np.sum(hasrad_pool8)
                nhit_total_pool8 += nhit_pool8
                nmis_total_pool8 += nmis_pool8
                nfal_total_pool8 += nfal_pool8

                diff = goes_batch - mrms_batch
                rthr_diff_sumsq_total += np.sum(np.square(diff[hasrad]))
                if i == 0:
                    # diff_mean_total += np.mean(diff) * diff.size
                    diff_sumsq_total += np.sum(np.square(diff))
                    tsum += np.sum(mrms_batch)
                    tsumsqrd += np.sum(np.square(mrms_batch))

        if nhit_total == 0:
            stats['pod'].append(np.nan)
            stats['far'].append(np.nan)
            stats['csi pool1'].append(np.nan)
            stats['bias'].append(np.nan)
            stats['rmses'].append(np.nan)
        else:
            csi = float(nhit_total) / \
                float(nhit_total + nmis_total + nfal_total)
            pod = float(nhit_total) / float(nhit_total + nmis_total)
            far = float(nfal_total) / float(nhit_total + nfal_total)
            bias = float(nhit_total + nfal_total) / \
                float(nhit_total + nmis_total)
            rmse = np.sqrt(rthr_diff_sumsq_total / nhasrads_total)
            stats['pod'].append(pod)
            stats['far'].append(far)
            stats['csi pool1'].append(csi)
            stats['bias'].append(bias)
            stats['rmses'].append(rmse)
        
        if nhit_total_pool4 == 0:
            stats['csi pool4'].append(np.nan)
        else:
            csi_pool4 = float(nhit_total_pool4) / \
                float(nhit_total_pool4 + nmis_total_pool4 + nfal_total_pool4)
            stats['csi pool4'].append(csi_pool4)
        
        if nhit_total_pool8 == 0:
            stats['csi pool8'].append(np.nan)
        else:
            csi_pool8 = float(nhit_total_pool8) / \
                float(nhit_total_pool8 + nmis_total_pool8 + nfal_total_pool8)
            stats['csi pool8'].append(csi_pool8)

    n_samples_pixels = num_samples * goes_shape[1] * goes_shape[2]
    # diff_mean = diff_mean_total / n_samples_pixels
    rmse = np.sqrt(diff_sumsq_total / n_samples_pixels)
    ss_tot = tsumsqrd - (tsum**2 / n_samples_pixels)
    rsqrd = 1 - (diff_sumsq_total / ss_tot)

    # stats['diff_mean'] = diff_mean
    stats['rmse'] = rmse
    stats['rsqrd'] = rsqrd

    return stats


class NumpyEncoder(json.JSONEncoder):
    """ Special json encoder for numpy types """

    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        return json.JSONEncoder.default(self, obj)


def main(args):
    # 1) get target and predicted data files
    xt_samples = []
    v = args.input
    for f in os.listdir(v):
        # if 'regA' in f and f.endswith('.npz'):
        if f.endswith('.npz'):
            xt_samples.append(os.path.join(v, f))
    xt_samples.sort()

    y_samples = []
    v = args.results
    for f in os.listdir(v):
        if f.endswith('.npy'):
            y_samples.append(os.path.join(v, f))
    y_samples.sort()

    # xt_samples = xt_samples[:2000]
    # y_samples = y_samples[:2000]

    # 2) compute statistics
    stats = get_refc_stats(y_samples, xt_samples)

    dumped = json.dumps(stats, cls=NumpyEncoder)
    output_file = os.path.join(args.results, 'stats1.json')
    with open(output_file, 'w') as f:
        json.dump(dumped, f)
    print(f'=> results saved to {output_file}')


if __name__ == '__main__':
    # args = parser.parse_args()
    # main(args)
    get_result_matrix()
