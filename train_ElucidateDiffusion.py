from es.utils.config_loaders import Config 
from es.dataloader.loader import HighResWindSpeed
from es.backbone import KarrasUnet
from es.diffusion import LElucidateDiffusion

from lightning import Trainer

from lightning.pytorch.loggers import WandbLogger
from lightning.pytorch.callbacks import LearningRateMonitor
from lightning.pytorch.strategies import DDPStrategy


## backbone config


config = Config(system="Mohit").get_config()

dl = HighResWindSpeed(config)
dl.setup(stage="train")

backbone_params = config.experiment.model.backbone.params.toDict()
diffusion_config = config.experiment.model.diffusion
diffusion_config

backbone = KarrasUnet(**backbone_params)
diffusion = LElucidateDiffusion(backbone, diffusion_config)

wandb_project =  "test_hrws"
wandb_entity = "earth-ai"
wandb_logger = WandbLogger(project=wandb_project, 
                           entity=wandb_entity, 
                           experiment=None, 
                           save_dir="/home/mila/m/mohit.anand/scratch/earthsphere/wandb")


lr_monitor = LearningRateMonitor(logging_interval='step')

strategy = DDPStrategy()

trainer = Trainer(accelerator="gpu", 
                logger=wandb_logger, 
                callbacks=[lr_monitor],
                devices=8, 
                strategy=strategy, 
                num_nodes=1, 
                default_root_dir="/home/mila/m/mohit.anand/scratch/earthsphere/lightning/",
                precision=16, 
                gradient_clip_val=0.5,)
if trainer.global_rank == 0:
    wandb_logger.experiment.config.update(config.toDict())
trainer.fit(model=diffusion, datamodule=dl)