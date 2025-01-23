import torch
from models.model import basemodel
import torch.cuda.amp as amp
from torch.functional import F
from torch.distributions import Normal
import time
import copy
from megatron_utils import mpu
import numpy as np
import utils.misc as utils
from tqdm.auto import tqdm
import torch.distributed as dist
import math
import wandb
from utils.misc import get_rank, get_world_size, is_dist_avail_and_initialized
from einops import rearrange
from utils.misc import reduce_mean, synchronize 
from utils.results import get_result_matrix, restore_large_image
import os 
from datetime import datetime
from collections import defaultdict
import cv2

class latent_diffusion_model(basemodel):
    def __init__(self, logger, tblogger, wandb, rank,**params) -> None:
        super().__init__(logger, tblogger, wandb, rank,**params)
        self.logger_print_time = False
        self.data_begin_time = time.time()
        self.tblogger = tblogger
        self.wandb = wandb
        self.rank = rank
        
        self.diffusion_kwargs = params.get('diffusion_kwargs', {})

        ## init noise scheduler ##
        self.noise_scheduler_kwargs = self.diffusion_kwargs.get('noise_scheduler', {})
        self.noise_scheduler_type = list(self.noise_scheduler_kwargs.keys())[0]
        if self.noise_scheduler_type == 'DDPMScheduler':
            from src.diffusers import DDPMScheduler
            self.noise_scheduler = DDPMScheduler(**self.noise_scheduler_kwargs[self.noise_scheduler_type])
            num_train_timesteps = self.noise_scheduler_kwargs[self.noise_scheduler_type]['num_train_timesteps']
            self.noise_scheduler.set_timesteps(num_train_timesteps)
        elif self.noise_scheduler_type == 'DPMSolverMultistepScheduler':
            from src.diffusers import DPMSolverMultistepScheduler
            self.noise_scheduler = DPMSolverMultistepScheduler(**self.noise_scheduler_kwargs[self.noise_scheduler_type])
            num_train_timesteps = self.noise_scheduler_kwargs[self.noise_scheduler_type]['num_train_timesteps']
            self.noise_scheduler.set_timesteps(num_train_timesteps)
        else:
            raise NotImplementedError
        
        ## init noise scheduler for sampling ##
        self.sample_noise_scheduler_type = 'DDIMScheduler'
        if self.sample_noise_scheduler_type == 'DDIMScheduler':
            print("############# USING SAMPLER: DDIMScheduler #############")
            from src.diffusers import DDIMScheduler
            self.sample_noise_scheduler = DDIMScheduler(**self.noise_scheduler_kwargs[self.noise_scheduler_type])
            ## set num of inference
            self.sample_noise_scheduler.set_timesteps(200)
        elif self.sample_noise_scheduler_type == 'DDPMScheduler':
            print("############# USING SAMPLER: DDPMScheduler #############")
            from src.diffusers import DDPMScheduler
            self.sample_noise_scheduler = DDPMScheduler(**self.noise_scheduler_kwargs[self.noise_scheduler_type])
            self.sample_noise_scheduler.set_timesteps(2000)
        else:
            raise NotImplementedError

        ## important: scale the noise to get a reasonable noise process ##
        self.noise_scale = self.noise_scheduler_kwargs.get('noise_scale', 1.0)
        self.logger.info(f'####### noise scale: {self.noise_scale} ##########')

        ## load pretrained checkpoint ##
        self.predictor_ckpt_path = self.extra_params.get("predictor_checkpoint_path", None)
        print(f'load from predictor_ckpt_path: {self.predictor_ckpt_path}')
        self.load_checkpoint(self.predictor_ckpt_path, load_model=True, load_optimizer=False, load_scheduler=False, load_epoch=False, load_metric_best=False)
        
        self.autoencoder_ckpt_path = self.extra_params.get("autoencoder_checkpoint_path", None)
        print(f'load from autoencoder_ckpt_path: {self.autoencoder_ckpt_path}')
        self.load_checkpoint(self.autoencoder_ckpt_path, load_model=True, load_optimizer=False, load_scheduler=False, load_epoch=False, load_metric_best=False)

        ## scale factor ##
        self.scale_factor = 1.0 ## 1/std TODO: according to latent space
        self.logger.info(f'####### USE SCALE_FACTOR: {self.scale_factor} ##########')

        ## classifier free guidance ##
        self.classifier_free_guidance_kwargs = self.diffusion_kwargs.get('classifier_free_guidance', {})
        self.p_uncond = self.classifier_free_guidance_kwargs.get('p_uncond', 0.0)
        self.guidance_weight = self.classifier_free_guidance_kwargs.get('guidance_weight', 0.0)
        


    def data_preprocess(self, data):
        # print(list(data.keys()))
        data_dict = {}
        inp_data = data['inputs'].float().to(self.device, non_blocking=True, dtype=self.data_type)
        tar_data = data['latent'].float().to(self.device, non_blocking=True, dtype=self.data_type)
        original_tar = data['original'].float().to(self.device, non_blocking=True, dtype=self.data_type)
        data_dict.update({'inputs': inp_data, 'data_samples': tar_data, 'original': original_tar})
        return data_dict
    
    @torch.no_grad()
    def denoise(self, template_data, cond_data, bs=1, vis=False, cfg=1, ensemble_member=1):
        """
        denoise from gaussian.
        """
        _, c, h, w = template_data.shape

        generator = torch.Generator(device=template_data.device) #torch.manual_seed(0)
        generator.manual_seed(0)
        latents = torch.randn(
            (bs*ensemble_member, c, h, w),
            generator=generator,
            device=template_data.device,
        ) 
        latents = latents * self.sample_noise_scheduler.init_noise_sigma

        print("start sampling")
        if cfg == 1:
            assert ensemble_member == 1
            ## iteratively denoise ##
            for t in tqdm(self.sample_noise_scheduler.timesteps) if (self.debug or vis) else self.sample_noise_scheduler.timesteps:
                ## predict the noise residual ##
                timestep = torch.ones((bs,), device=template_data.device) * t
                noise_pred = self.model[list(self.model.keys())[0]](x=latents, time=timestep, cond=cond_data)
                ## compute the previous noisy sample x_t -> x_{t-1} ##
                latents = self.sample_noise_scheduler.step(noise_pred, t, latents).prev_sample
            print("end sampling")
            return latents
        else:
            print(f"guidance strength: {cfg}")
            ## for classifier free sampling ##
            cond_data = torch.cat([cond_data, torch.zeros_like(cond_data)])
            avg_latents = []
            for member in range(ensemble_member):
                member_latents = latents[member*bs:(member+1)*bs, ...]
                for t in tqdm(self.sample_noise_scheduler.timesteps) if (self.debug or vis) else self.sample_noise_scheduler.timesteps:
                    ## predict the noise residual ##
                    timestep = torch.ones((bs*2,), device=template_data.device) * t
                    latent_model_input = torch.cat([member_latents]*2)
                    latent_model_input = self.sample_noise_scheduler.scale_model_input(latent_model_input, t)
                    noise_pred = self.model[list(self.model.keys())[0]](x=latent_model_input, timesteps=timestep, cond=cond_data)
                    ########################
                    noise_pred_cond, noise_pred_uncond = noise_pred.chunk(2)
                    noise_pred = noise_pred_uncond + cfg*(noise_pred_cond - noise_pred_uncond)
                    ## compute the previous noisy sample x_t -> x_{t-1} ##
                    member_latents = self.sample_noise_scheduler.step(noise_pred, t, member_latents).prev_sample
                avg_latents.append(member_latents)
            print('end sampling')
            avg_latents = torch.stack(avg_latents, dim=1)
            return avg_latents

    @torch.no_grad()
    def encode_stage(self, x):
        if utils.get_world_size() == 1 :
            z = self.model[list(self.model.keys())[1]].net.encode(x)
        else:
            z = self.model[list(self.model.keys())[1]].module.net.encode(x)
        return z.sample() * self.scale_factor

    @torch.no_grad()
    def decode_stage(self, z):
        z = z/self.scale_factor
        if utils.get_world_size() == 1 :
            z = self.model[list(self.model.keys())[1]].net.decode(z)
        else:
            z = self.model[list(self.model.keys())[1]].module.net.decode(z)
        return z

    @torch.no_grad()
    def init_scale_factor(self, z_tar):
        del self.scale_factor
        self.logger.info("### USING STD-RESCALING ###")
        _std = z_tar.std()
        if utils.get_world_size() == 1 :
            pass
        else:
            dist.barrier()
            dist.all_reduce(_std)
            _std = _std / dist.get_world_size()
        scale_factor = 1/_std
        self.logger.info(f'####### scale factor: {scale_factor.item()} ##########')
        self.register_buffer('scale_factor', scale_factor)

    def train_one_step(self, batch_data, step):
        data_dict = self.data_preprocess(batch_data)
        inp, tar = data_dict['inputs'], data_dict['data_samples'] # data_sample是latten data
        original_tar = data_dict['original'] # 最开始的降水归一化的结果

        b, c, h, w = tar.shape
        ## inp is coarse prediction in latent space, tar is gt in latent space
        z_tar = tar 
        z_coarse_prediction = inp # inp可以改成卫星与实况
        ## init scale_factor ##
        if self.scale_factor == 1.0:
            self.init_scale_factor(tar)
        ## scale ##
        z_tar = z_tar * self.scale_factor
        z_coarse_prediction = z_coarse_prediction * self.scale_factor
        ## classifier free guidance ##
        p = torch.rand(1)
        if p < self.p_uncond: ## discard condition 我们需要吗？
            z_coarse_prediction_cond = torch.zeros_like(z_coarse_prediction)
        else:
            z_coarse_prediction_cond = z_coarse_prediction
        ## sample noise to add ##
        noise = torch.randn_like(z_tar)
        ## sample random timestep for each ##
        bs = inp.shape[0]
        timesteps = torch.randint(0, self.noise_scheduler.config.num_train_timesteps, (bs,), device=inp.device)
        noisy_tar = self.noise_scheduler.add_noise(z_tar, noise, timesteps)

        ## predict the noise residual ##
        noise_pred = self.model[list(self.model.keys())[0]](x=noisy_tar, time=timesteps, cond=z_coarse_prediction_cond)

        loss = self.loss(noise_pred, noise) ## important: rescale the loss
        loss.backward()

        ## update params of diffusion model ##
        self.optimizer[list(self.model.keys())[0]].step()
        self.optimizer[list(self.model.keys())[0]].zero_grad()


        return {self.loss_type: loss.item()}
    
    @torch.no_grad()
    def test_one_step(self, batch_data, step, batch_idx, epoch):
        data_dict = self.data_preprocess(batch_data)
        inp, tar, original_tar = data_dict['inputs'], data_dict['data_samples'], data_dict['original']
        b, c, h, w = tar.shape
        ## inp is coarse prediction in latent space, tar is gt in latent space
        z_tar = tar
        z_coarse_prediction = inp
        ## scale ##
        z_tar = z_tar * self.scale_factor
        z_coarse_prediction = z_coarse_prediction * self.scale_factor
        ## sample noise to add ##
        noise = torch.randn_like(z_tar)
        ## sample random timestep for each ##
        bs = inp.shape[0]
        timesteps = torch.randint(0, self.noise_scheduler.config.num_train_timesteps, (bs,), device=inp.device)
        noisy_tar = self.noise_scheduler.add_noise(z_tar, noise, timesteps)

        ## predict the noise residual ##
        noise_pred = self.model[list(self.model.keys())[0]](x=noisy_tar, time=timesteps, cond=z_coarse_prediction)

        loss_records = {}
        ## evaluate other metrics ##

        z_sample_prediction = self.denoise(template_data=z_tar, cond_data=z_coarse_prediction, bs=b)
        sample_prediction = self.decode_stage(z_sample_prediction) # 得到反归一化之后的结果
        
        figpath = self.visualizer.save_pixel_image_rain(pred_image=sample_prediction, target_img=original_tar, epoch = epoch, step = step)
        
        if get_rank() == 0:  # 只记录主进程         
            # 加载图片并记录到 W&B
            from PIL import Image
            # 从 figpath 读取图片
            image = Image.open(figpath)
            # 使用 wandb 记录读取的图片
            wandb.log({
                f"Val/Epoch_{epoch}_Iteration_{step}_Batch_{batch_idx}": wandb.Image(image)
            })
            
        data_dict = {}
        data_dict['gt'] = noise
        data_dict['pred'] = noise_pred
        
        MSE_loss = torch.mean((noise_pred - noise) ** 2).item()
        loss = self.loss(noise_pred, noise) ## important: rescale the loss
        
        # self.tblogger.add_scalar("Val/NoiseMSE_loss", MSE_loss, step)
        # if get_rank() == 0:  # 确保只在主进程记录
        #     self.wandb.log({"Val/NoiseMSE_loss": MSE_loss}, step=step)

        
        ## evaluation ##
        if self.metrics_type == 'SEVIRSkillScore':
            csi_total = 0
            ## to pixel ##
            data_dict['gt'] = data_dict['gt'].squeeze(2) * 255
            data_dict['pred'] = data_dict['pred'].squeeze(2) * 255
            self.eval_metrics.update(target=data_dict['gt'].cpu(), pred=data_dict['pred'].cpu())
            metrics = self.eval_metrics.compute()
            for i, thr in enumerate(self.eval_metrics.threshold_list):
                loss_records.update({f'CSI_{thr}': metrics[thr
                ]['csi']})
                csi_total += metrics[thr]['csi']
            loss_records.update({'CSI_m': csi_total / len(self.eval_metrics.threshold_list)})
            loss_records.update({'MSE': MSE_loss})
        else:
            loss_records.update({'MSE': MSE_loss})

        return loss_records, original_tar, sample_prediction
    
    @torch.no_grad()
    def test(self, test_data_loader, trainstep, epoch):
        Trainstep = trainstep
        metric_logger = utils.MetricLogger(delimiter="  ", sync=True)
        # set model to eval
        for key in self.model:
            self.model[key].eval()
        data_loader = test_data_loader
        
        from megatron_utils.tensor_parallel.data import get_data_loader_length
        total_step = get_data_loader_length(test_data_loader)
        print('total_step:', total_step)
        ALLRMSE=[]
        thresholds = [1., 2., 5., 10., 15., 20.]
        total_pod = {thr: 0.0 for thr in thresholds}
        total_far = {thr: 0.0 for thr in thresholds}
        total_csi = {thr: 0.0 for thr in thresholds}
        idx = 0
        
        for step, batch in enumerate(data_loader):
            if self.debug and step>= 2:
                break
            idx += 1
            print(f'rank_{get_rank()}')
            print("idx:", idx)
            
            loss, gt, pred = self.test_one_step(batch, trainstep, step)
            metric_logger.update(**loss)

        
            if isinstance(batch, int):
                batch = None
            
                
            b,_,_,_ = gt.shape
            
            hr = torch.squeeze(gt)
            sr = torch.squeeze(pred)
            print('shape_hr:',hr.shape)
            
            hr = hr.detach().cpu().numpy()
            sr = sr.detach().cpu().numpy()
            
            renorm_hr = self.reverse_sqrtlog_minmax_norm(hr)
            renorm_sr = self.reverse_sqrtlog_minmax_norm(sr)
                    
            mse,_ = self.calculate_mse_psnr(renorm_sr, renorm_hr)
            rmse = math.sqrt(mse) # 该批次的平均值
            ALLRMSE.append(rmse)
            
            for thr in thresholds:                       
                has_event_target = (renorm_hr >= thr) 
                has_event_predict = (renorm_sr >= thr) 
                
                hit = np.sum(has_event_target & has_event_predict).astype(int)
                miss = np.sum(has_event_target & ~has_event_predict).astype(int)
                false_alarm = np.sum(~has_event_target & has_event_predict).astype(int)
                no_event = np.sum(~has_event_target).astype(int)
                
                pod = hit / (hit + miss) if (hit + miss) > 0 else float(2)
                far = false_alarm / no_event if no_event > 0 else float(2)
                csi = hit / (hit + miss + false_alarm) if (hit + miss + false_alarm) > 0 else float(2)
                
                total_pod[thr] += pod
                total_far[thr] += far
                total_csi[thr] += csi
                
                self.logger.info(f"Step:{step}, Epoch:{epoch}, Threshold: {thr}, Rank:{get_rank()}, Idx: {idx}, AVG POD: {pod:.4f}, AVG FAR: {far:.4f}, AVG CSI: {csi:.4f}, AVG RMSE:{rmse:.4f}")
        
        self.logger.info('  '.join(
                [f'Step [{trainstep + 1}](val stats)',
                 f'Epoch[{epoch}]',
                 "{meters}"]).format(
                    meters=str(metric_logger)
                 ))
        
        AVG_RMSE = sum(ALLRMSE) / len(ALLRMSE)
        self.tblogger.add_scalar('Val/avg_rmse', AVG_RMSE, Trainstep)
        

        self.wandb.log({
            f"Val/Avg_RMSE_Rank_{get_rank()}": AVG_RMSE,
            "Epoch": epoch,
            "Step": trainstep
        })
            
        for thr in thresholds:
            avg_pod = total_pod[thr] / idx
            avg_far = total_far[thr] / idx
            avg_csi = total_csi[thr] / idx
            self.tblogger.add_scalar(f'Val/POD_{thr}', avg_pod, Trainstep)
            self.tblogger.add_scalar(f'Val/FAR_{thr}', avg_far, Trainstep)
            self.tblogger.add_scalar(f'Val/CSI_{thr}', avg_csi, Trainstep)

            self.logger.info(f"Fin: Step:{Trainstep}, Epoch:{epoch},Threshold: {thr}, AVG POD: {avg_pod:.4f}, AVG FAR: {avg_far:.4f}, AVG CSI: {avg_csi:.4f}, AVG RMSE:{AVG_RMSE:.4f}")
        return metric_logger

    def test_all_ranks(self, test_data_loader, trainstep, epoch):
        Trainstep = trainstep
        metric_logger = utils.MetricLogger(delimiter="  ", sync=True)
        for key in self.model:
            self.model[key].eval()
        data_loader = test_data_loader

        from megatron_utils.tensor_parallel.data import get_data_loader_length
        total_step = get_data_loader_length(test_data_loader)
        print('total_step:', total_step)
        
        ALLRMSE = []
        thresholds = [5., 10., 15., 30., 25., 40.]
        total_pod = {thr: 0.0 for thr in thresholds}
        total_far = {thr: 0.0 for thr in thresholds}
        total_csi = {thr: 0.0 for thr in thresholds}
        idx = 0

        for valstep, batch in enumerate(data_loader):
            if self.debug and valstep >= 2:
                break
            idx += 1
            print(f'rank_{get_rank()}')
            print("idx:", idx)

            loss, gt, pred = self.test_one_step(batch, trainstep, valstep, epoch)
            metric_logger.update(**loss)

            if isinstance(batch, int):
                batch = None

            b, _, _, _ = gt.shape

            hr = torch.squeeze(gt)
            sr = torch.squeeze(pred)
            print('shape_hr:', hr.shape)

            hr = hr.detach().cpu().numpy()
            sr = sr.detach().cpu().numpy()

            renorm_hr = self.reverse_sqrtlog_minmax_norm(hr)
            renorm_sr = self.reverse_sqrtlog_minmax_norm(sr)

            mse, _ = self.calculate_mse_psnr(renorm_sr, renorm_hr)
            rmse = math.sqrt(mse)  # 该批次的平均值
            ALLRMSE.append(rmse)

            for thr in thresholds:
                has_event_target = (renorm_hr >= thr)
                has_event_predict = (renorm_sr >= thr)

                hit = np.sum(has_event_target & has_event_predict).astype(int)
                miss = np.sum(has_event_target & ~has_event_predict).astype(int)
                false_alarm = np.sum(~has_event_target & has_event_predict).astype(int)
                no_event = np.sum(~has_event_target).astype(int)

                pod = hit / (hit + miss) if (hit + miss) > 0 else float(2)
                far = false_alarm / no_event if no_event > 0 else float(2)
                csi = hit / (hit + miss + false_alarm) if (hit + miss + false_alarm) > 0 else float(2)

                total_pod[thr] += pod
                total_far[thr] += far
                total_csi[thr] += csi

        # 使用 misc.reduce_mean 计算全局平均 RMSE
        ALLRMSE_tensor = torch.tensor(ALLRMSE, device='cuda')
        AVG_RMSE = reduce_mean(ALLRMSE_tensor.mean()).item()

        # 使用 misc.reduce_mean 对 total_pod、total_far、total_csi 进行全局平均
        for thr in thresholds:
            total_pod[thr] = reduce_mean(torch.tensor(total_pod[thr], device='cuda')).item()
            total_far[thr] = reduce_mean(torch.tensor(total_far[thr], device='cuda')).item()
            total_csi[thr] = reduce_mean(torch.tensor(total_csi[thr], device='cuda')).item()

        # 同步所有进程，确保统计和记录一致
        synchronize()

        # 打印和记录（仅主进程）
        if get_rank()==0:
            # WandB 日志记录全局 AVG_RMSE
            self.wandb.log({
                "Val/Global_Avg_RMSE": AVG_RMSE,
                "Epoch": epoch
            })

            # WandB 日志记录全局 POD, FAR, CSI
            for thr in thresholds:
                self.wandb.log({
                    f"Val/Global_CSI_{thr}": total_csi[thr]
                })

            # 控制台输出全局结果
            self.logger.info(f"Epoch: {epoch}, Step:{trainstep}")
            self.logger.info(f"Global AVG_RMSE: {AVG_RMSE}")
            for thr in thresholds:
                self.logger.info(f"Threshold: {thr}, AVG POD: {total_pod[thr]:.4f}, AVG FAR: {total_far[thr]:.4f}, AVG CSI: {total_csi[thr]:.4f}")

        return metric_logger

    @torch.no_grad()
    def eval_step(self, batch_data, step):
        data_dict = self.data_preprocess(batch_data)
        inp, tar = data_dict['inputs'], data_dict['data_samples']
        original_tar = data_dict['original']
        b, c, h, w = tar.shape
        ## inp is coarse prediction in latent space, tar is gt in latent space
        z_tar = tar 
        z_coarse_prediction = inp
        ## scale ##
        z_tar = z_tar * self.scale_factor
        z_coarse_prediction = z_coarse_prediction * self.scale_factor
        ## sample image ##
        losses = {}
        z_sample_prediction = self.denoise(template_data=z_tar, cond_data=z_coarse_prediction, bs=b)
        sample_prediction = self.decode_stage(z_sample_prediction) # 得到归一化之后的结果
        
        return original_tar, sample_prediction
    
    @torch.no_grad()
    def test_final(self, test_data_loader, predict_length):
        self.test_data_loader = test_data_loader
        metric_logger = utils.MetricLogger(delimiter="  ", sync=True)
        if self.metrics_type == 'SEVIRSkillScore':
            self.scale_factor = 0.49429234862327576
            
        else:
            raise NotImplementedError
        print(self.scale_factor)
        
        for key in self.model:
            self.model[key].eval()

        if test_data_loader is not None:
            data_loader = test_data_loader
        else:
            raise ValueError("test_data_loader is None")

        from megatron_utils.tensor_parallel.data import get_data_loader_length
        total_step = get_data_loader_length(test_data_loader)

        # from utils.metrics import cal_FVD
        # self.fvd_computer = cal_FVD(use_gpu=True)
        
        ALLRMSE=[]
        thresholds = [5., 10., 15., 30., 25., 40.]
        total_pod = {thr: 0.0 for thr in thresholds}
        total_far = {thr: 0.0 for thr in thresholds}
        total_csi = {thr: 0.0 for thr in thresholds}
        
        idx = 0
        Vis = False
        for step, batch in enumerate(data_loader):
            idx=idx+1
            if isinstance(batch, int):
                batch = None
            # gt, pred = self.eval_step(batch_data=batch, step=step)
            
            gt = torch.zeros((1, 1, 16, 16))
            pred = torch.zeros((1, 1, 16, 16))
            
            print(gt.shape)
            if(Vis):
                current_time = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
                path = os.path.join(self.save_dir, f"TestInf_{current_time}")
                figpath = self.visualizer.save_pixel_image_rain(pred_image=pred, target_img=gt)
                
            b,_,_,_ = gt.shape
            
            hr = torch.squeeze(gt)
            sr = torch.squeeze(pred)
            
            hr = hr.detach().cpu().numpy()
            sr = sr.detach().cpu().numpy()
            
            renorm_hr = self.reverse_sqrtlog_minmax_norm(hr)
            renorm_sr = self.reverse_sqrtlog_minmax_norm(sr)
                    
            mse,_ = self.calculate_mse_psnr(renorm_sr, renorm_hr)
            rmse = math.sqrt(mse) # 该批次的平均值
            ALLRMSE.append(rmse)
            
            for thr in thresholds:                       
                has_event_target = (renorm_hr >= thr) 
                has_event_predict = (renorm_sr >= thr) 
                
                hit = np.sum(has_event_target & has_event_predict).astype(int)
                miss = np.sum(has_event_target & ~has_event_predict).astype(int)
                false_alarm = np.sum(~has_event_target & has_event_predict).astype(int)
                no_event = np.sum(~has_event_target).astype(int)
                
                pod = hit / (hit + miss) if (hit + miss) > 0 else float(2)
                far = false_alarm / no_event if no_event > 0 else float(2)
                csi = hit / (hit + miss + false_alarm) if (hit + miss + false_alarm) > 0 else float(2)
                
                total_pod[thr] += pod
                total_far[thr] += far
                total_csi[thr] += csi
                
                if step % 10 == 0:
                    self.logger.info(f"Step:{step}, Threshold: {thr}, AVG POD: {pod:.4f}, AVG FAR: {far:.4f}, AVG CSI: {csi:.4f}, AVG RMSE:{rmse:.4f}")
                # 使用 misc.reduce_mean 计算全局平均 RMSE
        # ALLRMSE_tensor = torch.tensor(ALLRMSE, device='cuda')
        # AVG_RMSE = reduce_mean(ALLRMSE_tensor.mean()).item()

        # 使用 misc.reduce_mean 对 total_pod、total_far、total_csi 进行全局平均
        # for thr in thresholds:
        #     total_pod[thr] = reduce_mean(torch.tensor(total_pod[thr], device='cuda')).item()
        #     total_far[thr] = reduce_mean(torch.tensor(total_far[thr], device='cuda')).item()
        #     total_csi[thr] = reduce_mean(torch.tensor(total_csi[thr], device='cuda')).item()
        AVG_RMSE = sum(ALLRMSE) / len(ALLRMSE)
        for thr in thresholds:
            avg_pod = total_pod[thr] / idx
            avg_far = total_far[thr] / idx
            avg_csi = total_csi[thr] / idx
        # self.logger.info(f"Fin: Step:{step}, Threshold: {thr}, AVG POD: {avg_pod:.4f}, AVG FAR: {avg_far:.4f}, AVG CSI: {avg_csi:.4f}, AVG RMSE:{AVG_RMSE:.4f}")
        self.logger.info(f"Global AVG_RMSE: {AVG_RMSE}")
        for thr in thresholds:
            self.logger.info(f"Threshold: {thr}, AVG POD: {total_pod[thr]:.4f}, AVG FAR: {total_far[thr]:.4f}, AVG CSI: {total_csi[thr]:.4f}")
                
        return None
  
    def test_final_pool(self, test_data_loader, predict_length):
        self.test_data_loader = test_data_loader
        metric_logger = utils.MetricLogger(delimiter="  ", sync=True)
        if self.metrics_type == 'SEVIRSkillScore':
            self.scale_factor = 0.49429234862327576
            
        else:
            raise NotImplementedError
        print(self.scale_factor)
        
        for key in self.model:
            self.model[key].eval()

        if test_data_loader is not None:
            data_loader = test_data_loader
        else:
            raise ValueError("test_data_loader is None")

        from megatron_utils.tensor_parallel.data import get_data_loader_length
        total_step = get_data_loader_length(test_data_loader)

        # from utils.metrics import cal_FVD
        # self.fvd_computer = cal_FVD(use_gpu=True)
        
        ALLRMSE=[]
        thrs = [1., 2., 5., 10., 15., 20.]
        
        # 指标初始化
        stats = {}    
        stats['thr'] = thrs
        stats['csi pool1'] = [{thr: [] for thr in thrs}]
        stats['csi pool4'] = [{thr: [] for thr in thrs}]
        stats['csi pool8'] = [{thr: [] for thr in thrs}]
        stats['rmses'] = []
        
        idx = 0
        Vis = True
        
        for step, batch in enumerate(data_loader):
            idx=idx+1
            
            if isinstance(batch, int):
                batch = None
                
            gt, pred = self.eval_step(batch_data=batch, step=step) 
            
            
            if(Vis):
                figpath = self.visualizer.save_pixel_image_rain(pred_image=pred, target_img=gt, epoch = 1, step = step)
                print(figpath)
            b,_,_,_ = gt.shape
            
            hr = torch.squeeze(gt)
            sr = torch.squeeze(pred)
            
            hr = hr.detach().cpu().numpy()
            sr = sr.detach().cpu().numpy()
            
            renorm_hr = self.reverse_sqrtlog_minmax_norm(hr)
            renorm_sr = self.reverse_sqrtlog_minmax_norm(sr)
            
            stats = get_result_matrix(renorm_hr, renorm_sr, thrs, stats, step)
        
        # 计算所有指标的平均值
        avg_stats = {
            'avg_csi_pool1': {},
            'avg_csi_pool4': {},
            'avg_csi_pool8': {},
            'avg_rmse': None
        }

        def remove_nan_and_convert(values):
            return torch.tensor([x for x in values if not np.isnan(x)], device='cuda')

        # for thr in thrs:
        #     csi_pool1_values = [x for x in stats['csi pool1'][0][thr] if not np.isnan(x)]
        #     csi_pool4_values = [x for x in stats['csi pool4'][0][thr] if not np.isnan(x)]
        #     csi_pool8_values = [x for x in stats['csi pool8'][0][thr] if not np.isnan(x)]

        #     avg_stats['avg_csi_pool1'][thr] = np.mean(csi_pool1_values) if csi_pool1_values else np.nan
        #     avg_stats['avg_csi_pool4'][thr] = np.mean(csi_pool4_values) if csi_pool4_values else np.nan
        #     avg_stats['avg_csi_pool8'][thr] = np.mean(csi_pool8_values) if csi_pool8_values else np.nan

        # rmse_values = [x for x in stats['rmses'] if not np.isnan(x)]
        # avg_stats['avg_rmse'] = np.mean(rmse_values) if rmse_values else np.nan
        
        # 全局平均 RMSE
        ALLRMSE_tensor = torch.tensor([x for x in stats['rmses'] if not np.isnan(x)], device='cuda')
        AVG_RMSE = reduce_mean(ALLRMSE_tensor.mean()).item()
        avg_stats['avg_rmse'] = AVG_RMSE
        
        # 针对多个进程的全局平均
        for thr in thrs:
            # 处理 csi_pool1
            csi_pool1_values = remove_nan_and_convert(stats['csi pool1'][0][thr])
            avg_stats['avg_csi_pool1'][thr] = (reduce_mean(csi_pool1_values.mean()).item() if len(csi_pool1_values) > 0 else np.nan)
            
            # 处理 csi_pool4
            csi_pool4_values = remove_nan_and_convert(stats['csi pool4'][0][thr])
            avg_stats['avg_csi_pool4'][thr] = (reduce_mean(csi_pool4_values.mean()).item() if len(csi_pool4_values) > 0 else np.nan)
            
            # 处理 csi_pool8
            csi_pool8_values = remove_nan_and_convert(stats['csi pool8'][0][thr])
            avg_stats['avg_csi_pool8'][thr] = (reduce_mean(csi_pool8_values.mean()).item() if len(csi_pool8_values) > 0 else np.nan)

        
        if get_rank() == 0:
            self.logger.info(f'Average stats: {avg_stats}')
            
        
        stats['average'] = avg_stats


        return None
    
    def patch2all(self, test_data_loader, predict_length):
        self.test_data_loader = test_data_loader
        metric_logger = utils.MetricLogger(delimiter="  ", sync=True)
        if self.metrics_type == 'SEVIRSkillScore':
            self.scale_factor = 0.49429234862327576
            
        else:
            raise NotImplementedError
        print(self.scale_factor)
        
        for key in self.model:
            self.model[key].eval()

        if test_data_loader is not None:
            data_loader = test_data_loader
        else:
            raise ValueError("test_data_loader is None")

        from megatron_utils.tensor_parallel.data import get_data_loader_length
        total_step = get_data_loader_length(test_data_loader)
        print("total_step:", total_step)
    
        idx = 0
        
        # 指标初始化
        thrs = [1., 2., 5., 10., 15., 20.]
        stats = {}    
        stats['thr'] = thrs
        stats['csi pool1'] = [{thr: [] for thr in thrs}]
        stats['csi pool4'] = [{thr: [] for thr in thrs}]
        stats['csi pool8'] = [{thr: [] for thr in thrs}]
        stats['rmses'] = []
        
        for step, batch in enumerate(data_loader):
            idx=idx+1
            print('idx:', idx)
            if isinstance(batch, int):
                batch = None
            
            name = batch['Name'][0]
            x = int(batch['x'] )        
            y = int(batch['y'] )

            gt, pred = self.eval_step(batch_data=batch, step=step) 
            
            b,_,_,_ = gt.shape
            
            hr = torch.squeeze(gt)
            sr = torch.squeeze(pred)
            
            hr = hr.detach().cpu().numpy()
            sr = sr.detach().cpu().numpy()
            
            renorm_hr = self.reverse_sqrtlog_minmax_norm(hr)
            renorm_sr = self.reverse_sqrtlog_minmax_norm(sr)
            
            save_path = '/mnt/petrelfs/xukaiyi/CodeSpace/DiT/SaveNpy/SavePatch'
            
            np.save(save_path + '/gt_' + name + '_' + str(x) + '_' + str(y), renorm_hr)
            np.save(save_path + '/pred_' + name + '_' + str(x) + '_' + str(y), renorm_sr)
            
            stats = get_result_matrix(renorm_hr, renorm_sr, thrs, stats, name)
            
        # 计算所有指标的平均值
        avg_stats = {
            'avg_csi_pool1': {},
            'avg_csi_pool4': {},
            'avg_csi_pool8': {},
            'avg_rmse': None
        }

        def remove_nan_and_convert(values):
            return torch.tensor([x for x in values if not np.isnan(x)], device='cuda')

        # for thr in thrs:
        #     csi_pool1_values = [x for x in stats['csi pool1'][0][thr] if not np.isnan(x)]
        #     csi_pool4_values = [x for x in stats['csi pool4'][0][thr] if not np.isnan(x)]
        #     csi_pool8_values = [x for x in stats['csi pool8'][0][thr] if not np.isnan(x)]

        #     avg_stats['avg_csi_pool1'][thr] = np.mean(csi_pool1_values) if csi_pool1_values else np.nan
        #     avg_stats['avg_csi_pool4'][thr] = np.mean(csi_pool4_values) if csi_pool4_values else np.nan
        #     avg_stats['avg_csi_pool8'][thr] = np.mean(csi_pool8_values) if csi_pool8_values else np.nan

        # rmse_values = [x for x in stats['rmses'] if not np.isnan(x)]
        # avg_stats['avg_rmse'] = np.mean(rmse_values) if rmse_values else np.nan
        
        # 全局平均 RMSE
        ALLRMSE_tensor = torch.tensor([x for x in stats['rmses'] if not np.isnan(x)], device='cuda')
        AVG_RMSE = reduce_mean(ALLRMSE_tensor.mean()).item()
        avg_stats['avg_rmse'] = AVG_RMSE
        
        # 针对多个进程的全局平均
        for thr in thrs:
            # 处理 csi_pool1
            csi_pool1_values = remove_nan_and_convert(stats['csi pool1'][0][thr])
            avg_stats['avg_csi_pool1'][thr] = (reduce_mean(csi_pool1_values.mean()).item() if len(csi_pool1_values) > 0 else np.nan)
            
            # 处理 csi_pool4
            csi_pool4_values = remove_nan_and_convert(stats['csi pool4'][0][thr])
            avg_stats['avg_csi_pool4'][thr] = (reduce_mean(csi_pool4_values.mean()).item() if len(csi_pool4_values) > 0 else np.nan)
            
            # 处理 csi_pool8
            csi_pool8_values = remove_nan_and_convert(stats['csi pool8'][0][thr])
            avg_stats['avg_csi_pool8'][thr] = (reduce_mean(csi_pool8_values.mean()).item() if len(csi_pool8_values) > 0 else np.nan)

        
        if get_rank() == 0:
            self.logger.info(f'Average stats: {avg_stats}')
            
        
        stats['average'] = avg_stats

        return None
       
    @torch.no_grad()
    def reverse_sqrtlog_minmax_norm(self, transformed_data, mask=None):
        min_val = 0.0
        max_val = 50.0

        sqrt_data = np.expm1(transformed_data)
        norm_data = sqrt_data ** 2

        original_data = norm_data * (max_val - min_val) + min_val

        if mask is not None:
            if isinstance(original_data, np.ndarray):
                original_data = torch.tensor(original_data) 
            mask = mask.bool()
            original_data = torch.where(mask, original_data, torch.tensor(-1280.0))
            original_data = original_data.cpu().numpy()  

        return original_data
    
    def calculate_mse_psnr(self, img1, img2):
        img1 = img1.astype(np.float64)
        img2 = img2.astype(np.float64)
        mse = np.mean((img1 - img2)**2)
        if mse == 0:
            return mse, float('inf')
        psnr = 20 * math.log10(255.0 / math.sqrt(mse))
        return mse, psnr
