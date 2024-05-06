import torch
import torch.nn as nn
from omegaconf import OmegaConf
from ..taming.vae import Encoder, Decoder
from ..cuboid_transformer.cuboid_transformer import CuboidTransformerModel


class DreamCast(nn.module):
    
    def __init__(
        self,
        config,
        latent_height: int = 5,
        latent_width: int = 10,
        seq_length: int = 4,
        task: str = 'forecasting',
        ):
        super(DreamCast, self).__init__()
        
        #VAE Model configurations
        vae_cfg = OmegaConf.to_object(config.vae_model)
        #Earthfromer Model configurations
        earthformer_cfg = OmegaConf.to_object(config.earthformer_model)
        
        self.seq_length = seq_length
        self.latent_channels = vae_cfg['latent_channels']
        
        #initialize encoder
        self.encoder = Encoder(
            in_channels=vae_cfg['in_channels'],
            out_channels=vae_cfg['latent_channels'],
            block_out_channels=vae_cfg['block_out_channels'],
            down_block_types=vae_cfg['down_block_types'],
            layers_per_block=vae_cfg['layers_per_block'],
            act_fn=vae_cfg['act_fn'],
            norm_num_groups=vae_cfg['norm_num_groups'],
            double_z=vae_cfg['double_z'],
        )
        
        #initialize decoder
        self.decoder = Decoder(
            in_channels=vae_cfg['latent_channels'],
            out_channels=vae_cfg['out_channels'],
            up_block_types=vae_cfg['up_block_types'],
            block_out_channels=vae_cfg['block_out_channels'],
            layers_per_block=vae_cfg['layers_per_block'],
            norm_num_groups=vae_cfg['norm_num_groups'],
            double_z=vae_cfg['double_z'],
        )
        
        #initialize earthformer
        self.earthformer = CuboidTransformerModel(
            input_shape=(1, latent_height, latent_width, vae_cfg['latent_channels']),
            target_shape=(1, latent_height, latent_width, vae_cfg['latent_channels']),
            **config.earthformer_model
        )
        
        self.quant_conv = nn.Conv2d(2 * self.latent_channels , 2 * self.latent_channels , 1)
        self.post_quant_conv = nn.Conv2d(self.latent_channels , self.latent_channels , 1)
        
        
    def encode(self, x: torch.FloatTensor) -> torch.FloatTensor:
        x_encoded = self.encoder(x)
        print("Hidden shape:", x_encoded.shape)
        return x_encoded
    
    def decode(self, h_combined: torch.FloatTensor) -> torch.FloatTensor:
        x_recon = self.decoder(h_combined)
        return x_recon
    
    def compute_hidden_observe(self, ):
        return
    
    def rollout_observation(self, obs_encoded: torch.Tensor, prev_hidden_state):
        priors = []
        posteriors = []
        for t in range(self.seq_length):
            prior_hidden, posterior_hidden = self.compute_hidden_observe(obs_encoded[t], prev_hidden_state)
            priors.append(prior_hidden)
            posteriors.append(posterior_hidden)
            prev_hidden_state = posterior_hidden
        return priors, posteriors
            
            
        
        
        
        
        
    
    
        
        
        
        
        
        
        
        
        
    