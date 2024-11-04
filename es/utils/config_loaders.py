import yaml
from munch import munchify
from pathlib import Path
import wandb
from dotenv import dotenv_values
import os

class Config:
    def __init__(self, system):
        system_config = self.load("config/config.yaml")
        self.system_config = getattr(system_config, system)

        self.system_config.experiment.dataset = self.load("config/dataset/" + self.system_config.experiment.dataset)
        self.system_config.experiment.model.backbone = self.load("config/model/backbone/" + self.system_config.experiment.model.backbone)
        self.system_config.experiment.model.diffusion = self.load("config/model/diffusion/" + self.system_config.experiment.model.diffusion)

        self.system_config.path.root = Path(self.system_config.path.root)

        self.login_wandb()

    def load(self, path):
        if ".yaml" not in path:
            path = path + ".yaml"
        return munchify(yaml.load(open(path, 'r'), Loader=yaml.FullLoader))

    def get_config(self):
        return self.system_config   

    def login_wandb(self):
        config = dotenv_values(".env")
        wandb.login(key=config["WANDB_API_KEY"], force=True)