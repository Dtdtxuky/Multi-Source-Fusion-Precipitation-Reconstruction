from data.custom_utils import OrderedEasyDict as edict
import numpy as np
import os
import os.path as osp
#import torch

cfg = edict()
#cfg_ex = cfg
cfg.DIR = edict()
cfg.DIR.INPUT_BIN_ROOT = 's3://cma_radar/recons-nmic/SW'
cfg.DIR.LABEL_BIN_ROOT = 's3://cma_radar/recons-nmic/SW'
cfg.DIR.LIST_ROOT = '/mnt/petrelfs/zhouzhiwang/codeespace/cma_region/radar-recons-sh/data_list/SW'
cfg.DIR.NPY_ROOT = 's3://cma_radar/recons-nmic/SW-decoded'
cfg.DIR.INTER_LIST = '/mnt/petrelfs/zhouzhiwang/codeespace/cma_region/radar-recons-sh/data_list/SW/SW_CR_VIL_inter_dtlist.txt'

cfg.DIR.PATCH_SAMPLE_LIST_CR = '/mnt/petrelfs/zhouzhiwang/codeespace/cma_region/radar-recons-sh/data_list/SW/SW_CR_patched_sample_list-v2.txt'

cfg.DIR.TRAIN_LIST_CR = '/mnt/petrelfs/zhouzhiwang/codeespace/cma_region/radar-recons-sh/data_list/SW/SW_CR_train_sample_list-v2.txt'
cfg.DIR.VAL_LIST_CR = '/mnt/petrelfs/zhouzhiwang/codeespace/cma_region/radar-recons-sh/data_list/SW/SW_CR_val_sample_list-v2.txt'
cfg.DIR.TEST_LIST_CR = '/mnt/petrelfs/zhouzhiwang/codeespace/cma_region/radar-recons-sh/data_list/SW/SW_CR_test_sample_list-v2.txt'

cfg.TRAIN = edict()
#cfg.GLOBAL.DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
cfg.TRAIN.BATCH_SIZE = 16
cfg.TRAIN.MODEL_SAVE_ROOT = '/mnt/petrelfs/zhouzhiwang/codeespace/cma_region/radar-recons-sh/ckpts'
cfg.TRAIN.OPTIMIZER = 'AdamW'
cfg.TRAIN.INIT_LR = 0.0002
cfg.TRAIN.WEIGHT_DECAY = 0.02
cfg.TRAIN.WARMUP_EPOCHS = 2
cfg.TRAIN.REV_LOSS_WTS = [2.485361, 3.280593, 4.098358, 20.483035]
#cfg.TRAIN.DP_GPUS = '0,1,2,3,4,5,6,7'
#cfg.TRAIN.DP_GPUS = '0,3,4,5'

cfg.DATA = edict()
cfg.DATA.INPUT_CHANNELS = ['SATE_IR16', 'SATE_IR39', 'SATE_IR104', 'SATE_IR133', 
                              'SATE_VAP69', 'SATE_VIS', 'LN10']
cfg.DATA.LABEL_CHANNELS = ['RADAR_CR', 'RADAR_VIL']
cfg.DATA.MEAN_INPUT_SATE6 = [0.08, 275.12, 265.25, 253.35, 239.82, 0.12]
cfg.DATA.STD_INPUT_SATE6 = [0.12, 20.56, 22.45, 17.42, 10.79, 0.19]
cfg.DATA.LN_MAX_HALF = 48.00
cfg.DATA.CR_MIN = -863.00
cfg.DATA.CR_MAX = 1275.00
cfg.DATA.CR_MEAN = 137.63
cfg.DATA.CR_STD = 117.67
cfg.DATA.VIL_MIN = -1280.00
cfg.DATA.VIL_MAX = 800.00
cfg.DATA.VIL_MEAN = -1116.21
cfg.DATA.VIL_STD = 440.65
cfg.DATA.PETREL_CONF = '~/petreloss.conf'

cfg.PATCH = edict()
cfg.PATCH.IN_PATCH_SIZE = 256
cfg.PATCH.OUT_PATCH_SIZE = 512
cfg.PATCH.PATCH_IND_TO_START_IN = {0:0, 1:198, 2:395}
cfg.PATCH.PATCH_IND_TO_START_OUT = {0:0, 1:395, 2:789}
cfg.PATCH.PVALID_THRES = 0.5 # for SATE
#! 这个文件没有
# cfg.PATCH.MERGE_WEIGHTS = '/mnt/petrelfs/xinyuhang.p/radar-recons-sh/merge_weights_9patches.npy'

cfg.EVAL = edict()
cfg.EVAL.EVAL_ROOT = '/mnt/petrelfs/zhouzhiwang/codeespace/cma_region/radar-recons-sh/eval_results'
cfg.EVAL.SAVE_FREQ = 350
cfg.EVAL.EVAL_THRES_L = [10]
cfg.EVAL.EVAL_THRES_GE = [10, 20, 35, 40]
cfg.EVAL.SCALE_EPS = 1e-5

cfg.MODEL = edict()