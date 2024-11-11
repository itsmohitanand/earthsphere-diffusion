import lightning as L 
import torch
import torch.nn.functional as F
from einops import rearrange, reduce, repeat
from torch.optim.optimizer import Optimizer
import wandb

from ema_pytorch import EMA


from es.diffusion.core.elucidated_diffusion import ElucidateDiffusion

class LElucidateDiffusion(L.LightningModule):
    def __init__(self, backbone, cfg):
        super().__init__()

        self.cfg = cfg

        self.model = ElucidateDiffusion(backbone = backbone, device="cuda", **self.cfg.model.ElucidateDiffusionParams.toDict())

        self.ema = EMA(self.model, **self.cfg.model.EMAParams.toDict())
        
        self.lat_weight_arr = None
        if self.cfg.latitude_weight:
            lat_weight_arr = torch.from_numpy(self.trainer.dl.get_cos_latitude()).to("cuda")
            self.lat_weight_arr = rearrange(lat_weight_arr, "h w -> 1 1 h w")

    def configure_optimizers(self):
        params = self.cfg.optimizer.params.toDict()
        lr = float(params["lr"])
        T_max = float(params["T_max"])
        eta_min = float(params["eta_min"])

        optimizer = torch.optim.Adam(self.model.parameters(), lr=lr)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=T_max, eta_min=eta_min)
        
        return {
            'optimizer': optimizer,
            'lr_scheduler': scheduler
        }
    
    def input_T(self, batch):
        
        x_cond, x_tar = batch

        tar_features = x_tar.shape[1]

        #TODO: Reverse the difference operation
        x_tar = x_tar - x_cond[:, :tar_features, :, :]

        return x_cond, x_tar
    
    def output_T(self, x_cond, x_tar):

        tar_features = x_tar.shape[1]

        x_tar = x_tar + x_cond[:, :tar_features, :, :]

        return x_cond, x_tar
    
    def training_step(self, batch, batch_idx):
        
        x_cond, x_tar = self.input_T(batch)
      
        denoised, sigmas = self.model(x_target=x_tar, x_conditional=x_cond)

        loss = self.compute_loss(denoised, x_tar, sigmas)
        
        if batch_idx % 200 == 0:
            x_pred = self.ema.ema_model.sample(x_conditional=x_cond)
            self.log_images(x_tar, x_pred, "train")

        self.log("loss/train", loss, prog_bar=True, sync_dist=True)

        return loss

    def log_images(self, x_tar, x_pred, split):

        wind_u_obj = []
        wind_v_obj = []
        pred_u_obj = []
        pred_v_obj = []

        for i in range(2):
            wind_speed_u = x_tar[i, 0, :, :].cpu().detach().numpy()
            pred_speed_u = x_pred[i, 0, :, :].cpu().detach().numpy()
            wind_u_obj.append(wandb.Image(wind_speed_u))
            pred_u_obj.append(wandb.Image(pred_speed_u))
            wind_speed_v = x_tar[i, 1, :, :].cpu().detach().numpy()
            pred_speed_v = x_pred[i, 1, :, :].cpu().detach().numpy()
            wind_v_obj.append(wandb.Image(wind_speed_v))
            pred_v_obj.append(wandb.Image(pred_speed_v))

        self.logger.experiment.log({
            f"{split}/true/wind_speed_u": wind_u_obj,
            f"{split}/true/wind_speed_v": wind_v_obj,
            f"{split}/pred/wind_speed_u": pred_u_obj,
            f"{split}/pred/wind_speed_v": pred_v_obj
        })
        
    def validation_step(self, batch, batch_idx):

        x_cond, x_tar = self.input_T(batch)

        denoised, sigmas = self.model(x_target=x_tar, x_conditional=x_cond)

        self.ema.ema_model.eval()
        loss = self.compute_loss(denoised, x_tar, sigmas)

        if batch_idx == 0:
            x_pred = self.ema.ema_model.sample(x_conditional=x_cond)
            self.log_images(x_tar, x_pred, "valid")

        self.log("loss/val", loss, prog_bar=True, sync_dist=True)

    def on_before_zero_grad(self, *args, **kwargs) -> None:        
        self.ema.update()
    

    def compute_loss(self, denoised, target, sigmas):
        loss = F.mse_loss(denoised, target, reduction='none')

        if self.cfg.loss_mask:
            loss = loss * self.loss_mask

        if self.lat_weight_arr:
            loss = loss * self.lat_weight_arr
        
        if self.cfg.variable_weight:
            loss = loss * self.variable_weight

        loss = reduce(loss, 'b c h w -> b', 'mean')

        loss = loss*self.model.loss_weight(sigmas)

        loss = loss.mean()

        return loss

    def sample(self, batch):

        x_cond, x_tar  = batch

        denoised = self.model.sample(x_conditional=x_cond)

        _, x_pred = self.output_T(x_cond, denoised)

        return x_tar, x_pred, x_cond