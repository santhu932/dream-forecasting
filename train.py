import numpy as np
import argparse
import wandb
import os
import einops
import lightning.pytorch as pl
import torch


from src.models.dream_cast.dreamcast import DreamCast
from src.data_processing import data
from src.utils import metric

class DreamLitModel(pl.LightningModule):
    
    def __init__(self, config, save_dir: str = None):
        super(DreamLitModel, self).__init__()
        
        info = {
            'stoch_size' : (1, config.img_height, config.img_width, config.latent_stoch_channels),
            'deter_size' : (1, config.img_height, config.img_width, config.latent_deter_channels),
            'min_std' : config.min_std
        }
        
        self.model = DreamCast(
            config=config, 
            info=info, 
            latent_deter_channels=config.latent_deter_channels, 
            lssm_type=config.lssm_type,
            )
        
        
        
        






