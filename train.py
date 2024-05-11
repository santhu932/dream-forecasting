import numpy as np
import argparse
import wandb
import argparse
import os
import einops
import lightning.pytorch as pl
from lightning.pytorch import Trainer, seed_everything, loggers as pl_loggers
from pytorch_lightning.loggers import WandbLogger
import torch
from omegaconf import OmegaConf
import torch.distributions as td


from src.models.dream_cast.dreamcast import DreamCast
from src.data_processing import data
from src.utils import metric
from src.data_processing.data_module import TCDataModule

class DreamLitModel(pl.LightningModule):
    
    def __init__(self, config, save_dir: str = None):
        super(DreamLitModel, self).__init__()
        
        info = {
            'stoch_size' : (1, config.img_height, config.img_width, config.latent_stoch_channels),
            'deter_size' : (1, config.img_height, config.img_width, config.latent_deter_channels),
            'min_std' : config.min_std
        }
        
        self.loss_cfg = OmegaConf.to_object(config.loss)
        self.optim_cfg = OmegaConf.to_object(config.optim)
        
        self.lr = config.optim.lr
        self.mode = config.mode
        
        self.model = DreamCast(
            config=config, 
            info=info, 
            latent_deter_channels=config.latent_deter_channels, 
            lssm_type=config.lssm_type,
            )
    
    
    def mae_recon(self, y_pred, y_true):
        assert y_pred.size() == y_true.size()
        return torch.sum(torch.abs(y_true - y_pred)) / y_true.size(0) / y_true.size(-1) # average over batch and channels
    
    def mse_recon(self, y_pred, y_true):
        assert y_pred.size() == y_true.size()
        return torch.sum((y_pred - y_true) ** 2) / y_true.size(0) / y_true.size(-1) # average over batch and channels
    
    def kl_loss(self, prior, posterior):
        prior_dist = self.model.get_dist(prior)
        posterior_dist = self.model.get_dist(posterior)
        if self.loss_cfg['kl_balance'] == True:
            alpha = self.loss_cfg['kl_balance_scale']
            kl_lhs = torch.mean(td.kl_divergence(self.model.get_dist(self.model.lssm_detach(posterior)), prior_dist))
            kl_rhs = torch.mean(td.kl_divergence(posterior_dist, self.model.get_dist(self.model.lssm_detach(prior))))
            if self.loss_cfg['use_free_nats']:
                free_nats = self.loss_cfg['free_nats']
                kl_lhs = torch.max(kl_lhs, kl_lhs.new_full(kl_lhs.size(), free_nats))
                kl_rhs = torch.max(kl_rhs, kl_rhs.new_full(kl_rhs.size(), free_nats))
            kl_loss = alpha * kl_lhs + (1 - alpha) * kl_rhs
        else:
            kl_loss = torch.mean(td.kl_divergence(posterior_dist, prior_dist))
            if self.loss_cfg['use_free_nats']:
                free_nats = self.loss_cfg['free_nats']
                kl_loss = torch.max(kl_loss, kl_loss.new_full(kl_loss.size(), free_nats))
        return kl_loss
                
    

    def total_loss(self, y_pred, y_true, priors, posteriors):
        if self.loss_cfg['recon_loss_type'] == 'mse':
            recon_loss = self.mse_recon(y_pred=y_pred, y_true=y_true)
        else:
            recon_loss = self.mae_recon(y_pred=y_pred, y_true=y_true)
            
        kl_loss = self.kl_loss(prior=priors, posterior=posteriors) / y_true.size(-1)
        loss = recon_loss + self.loss_cfg['kl_beta'] * kl_loss
        return loss
        
        
    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr = self.lr)
        return optimizer
    
    def training_step(self, batch, batch_idx):
        _, X, Y = batch
        X_hat, priors, posteriors = self.model(X, mode=self.mode)
        loss = self.total_loss(X_hat, X, priors=priors, posteriors=posteriors)
        return loss
    
def get_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('--save', default='saved_models', type=str)
    parser.add_argument('--gpus', default=1, type=int)
    parser.add_argument('--cfg', default=None, type=str)
    parser.add_argument('--project_name', type=str)
    return parser
        
def main():
    parser  = get_parser()  
    args = parser.parse_args()
    project_id = 'dream_cast_net' if not args.project_name else args.project_name
    
    try:
        config = OmegaConf.load(open(args.cfg, "r"))
    except Exception as e:
        print("Config file path is required!")
        raise
    
    seed_everything(config.optim.seed, workers=True)
    torch.set_float32_matmul_precision(config.optim.float32_matmul_precision)  
    
    data_module = TCDataModule(config=config)
    
    data_module.prepare_data()
    data_module.setup()
    wandb_logger = WandbLogger(project=project_id, name=config.name)
    
    model = DreamLitModel(config=config)
    
    trainer = Trainer(logger=wandb_logger, max_epochs=config.optim.max_epochs)
    trainer.fit(model=model, datamodule=data_module)
    
    wandb.finish()
        
if __name__ == '__main__':
    main()
        
        
        
        
        
        






