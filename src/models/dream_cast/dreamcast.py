import torch
import torch.nn as nn
from omegaconf import OmegaConf
from ..taming.vae import Encoder, Decoder
from ..cuboid_transformer.cuboid_transformer import CuboidTransformerModel
from ...utils.lssm import LSSMUtils, LSSMContState
from ..cuboid_transformer.cuboid_utils import (get_activation, get_norm_layer)
from ..cuboid_transformer.cuboid_transformer import PositionwiseFFN


class DreamCast(nn.module, LSSMUtils):
    
    def __init__(
        self,
        config,
        info,
        latent_height: int = 5,
        latent_width: int = 10,
        seq_length: int = 4,
        task: str = 'forecasting',
        lssm_type = 'continuous',
        pre_norm: bool = True,
        normalization: str = 'layer_norm',
        layer_norm_eps: float = 1E-5,
        activation='relu',
        activation_dropout: float = 0.0,
        dropout: float = 0.1,
        ):
        nn.Module.__init__(self)
        LSSMUtils.__init__(self, rssm_type=lssm_type, info=info)
        
        #VAE Model configurations
        vae_cfg = OmegaConf.to_object(config.vae_model)
        #Earthfromer Model configurations
        earthformer_cfg = OmegaConf.to_object(config.earthformer_model)
        
        self.seq_length = seq_length
        self.latent_channels = vae_cfg['latent_channels']
        self._pre_norm = pre_norm
        self.normalization = normalization
        self.layer_norm_eps = layer_norm_eps
        self.activation = activation
        self.activation_dropout = activation_dropout
        self.dropout = dropout
        self.hidden_size = 4 * self.latent_channels
        
        self.fc_prior = self._compute_prior()
        self.fc_posterior = self._compute_posterior()
        self.quant_conv = nn.Conv2d(2 * self.latent_channels , 2 * self.latent_channels , 1)
        self.post_quant_conv = nn.Conv2d(self.latent_channels , self.latent_channels , 1)
        
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
            input_shape=(1, latent_height, latent_width, 2 * vae_cfg['latent_channels']),
            target_shape=(1, latent_height, latent_width, vae_cfg['latent_channels']),
            **config.earthformer_model
        )
        
        
    def _compute_prior(self):
        prior_module = [nn.Linear(in_features=self.latent_channels, out_features=self.hidden_size, bias=True)]
        if self._pre_norm:
            prior_module += [get_norm_layer(normalization=self.normalization, in_channels=self.latent_channels, epsilon=self.layer_norm_eps)]
        prior_module += [get_activation(self.activation)]
        prior_module += [nn.Dropout(self.activation_dropout)]
        prior_module += [nn.Linear(in_features=self.hidden_size, out_features=2*self.latent_channels, bias=True)]
        prior_module += [nn.Dropout(self.dropout)]
        return nn.Sequential(*prior_module)
    
    
    
    def _compute_posterior(self):
        posterior_module = [nn.Linear(in_features=self.latent_channels, out_features=self.hidden_size, bias=True)]
        if self._pre_norm:
            posterior_module += [get_norm_layer(normalization=self.normalization, in_channels=self.latent_channels, epsilon=self.layer_norm_eps)]
        posterior_module += [get_activation(self.activation)]
        posterior_module += [nn.Dropout(self.activation_dropout)]
        posterior_module += [nn.Linear(in_features=self.hidden_size, out_features=2*self.latent_channels, bias=True)]
        posterior_module += [nn.Dropout(self.dropout)]
        return nn.Sequential(*posterior_module)
        
        
    def encode(self, x: torch.FloatTensor) -> torch.FloatTensor:
        x_encoded = self.encoder(x)
        print("Hidden shape:", x_encoded.shape)
        return x_encoded
    
    def decode(self, h_combined: torch.FloatTensor) -> torch.FloatTensor:
        x_recon = self.decoder(h_combined)
        return x_recon
    
    def compute_hidden_imagine(self, prev_latent_state):
        stoch_state = prev_latent_state.stoch
        h_prev = prev_latent_state.deter
        h = self.earthformer(torch.cat([h_prev, stoch_state], dim = -1))
        prior_mean, prior_logvar = torch.chunk(self.fc_prior(h), 2, dim = -1)
        stats = {'mean':prior_mean, 'std': torch.exp(0.5 * prior_logvar)}
        prior_stoch_state, std = self.get_stoch_state(stats)
        prior_latent_state = LSSMContState(prior_mean, std, prior_stoch_state, h)
        return prior_latent_state
    
    def compute_hidden_observe(self, obs_encoded, prev_latent_state):
        prior_latent_state = self.compute_hidden_imagine(prev_latent_state)
        h = prior_latent_state.deter
        x = torch.cat([h, obs_encoded], dim=-1)
        posterior_mean, posterior_logvar = torch.chunk(self.fc_posterior(x), 2, dim = -1)
        stats = {'mean':posterior_mean, 'std': torch.exp(0.5 * posterior_logvar)}
        posterior_stoch_state, std = self.get_stoch_state(stats)
        posterior_latent_state = LSSMContState(posterior_mean, std, posterior_stoch_state, h)
        return prior_latent_state, posterior_latent_state
    
    def rollout_imagination(self, lead_time, prev_latent_state):
        latent_state = prev_latent_state
        next_latent_states = []
        for t in range(lead_time):
            latent_state = self.compute_hidden_imagine(latent_state)
            next_latent_states.append(latent_state)
        next_latent_states = self.lssm_stack_states(next_latent_states, dim=0)
        return next_latent_states
        
    
    def rollout_observation(self, obs_encoded: torch.Tensor, prev_latent_state):
        priors = []
        posteriors = []
        for t in range(self.seq_length):
            prior_latent, posterior_latent = self.compute_hidden_observe(obs_encoded[t], prev_latent_state)
            priors.append(prior_latent)
            posteriors.append(posterior_latent)
            prev_latent_state = posterior_latent
        prior = self.lssm_stack_states(priors, dim=0)
        posterior = self.lssm_stack_states(posteriors, dim=0)
        return prior, posterior
    
    
    def forward(self):
        
        return
            
            
        
        
        
        
        
    
    
        
        
        
        
        
        
        
        
        
    