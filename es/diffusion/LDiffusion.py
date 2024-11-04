import lightning as L 
import torch
import torch.nn.functional as F
from einops import rearrange, reduce, repeat
import wandb

from es.diffusion.core.elucidated_diffusion import ElucidateDiffusion

class LElucidateDiffusion(L.LightningModule):
    def __init__(self, backbone, cfg):
        super().__init__()

        self.cfg = cfg

        self.model = ElucidateDiffusion(backbone = backbone, device="cuda", **self.cfg.model.params.toDict())

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
    
    def training_step(self, batch, batch_idx):
        x_cond, x_tar = batch

        denoised, sigmas = self.model(x_target=x_tar, x_conditional=x_cond) # 

        loss = self.compute_loss(denoised, x_tar, sigmas)
        
        self.log("loss/train", loss, prog_bar=True, sync_dist=True)

        if batch_idx % 1000 == 0:
            x_pred = self.model.sample(x_conditional=x_cond)
            self.log_images(x_tar, x_pred)
            
        return loss

    def log_images(self, x_tar, x_pred):

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
            "true/wind_speed_u": wind_u_obj,
            "true/wind_speed_v": wind_v_obj,
            "pred/wind_speed_u": pred_u_obj,
            "pred/wind_speed_v": pred_v_obj
        })
        
        # log with WandbImage

    def validation_step(self, batch, batch_idx):
        x_cond, x_tar = batch

        denoised, sigmas = self.model(x_target=x_tar, x_conditional=x_cond)

        loss = self.compute_loss(denoised, x_tar, sigmas)

        self.log("loss/val", loss, prog_bar=True, sync_dist=True)



    def compute_loss(self, denoised, target, sigmas):
        loss = F.mse_loss(denoised, target, reduction='none')

        if self.cfg.loss_mask:
            loss = loss * self.loss_mask

        if self.cfg.latitude_weight:
            loss = loss * self.latitude_weight
        
        if self.cfg.variable_weight:
            loss = loss * self.variable_weight

        loss = reduce(loss, 'b c h w -> b', 'mean')

        loss = loss*self.model.loss_weight(sigmas)

        loss = loss.mean()

        return loss