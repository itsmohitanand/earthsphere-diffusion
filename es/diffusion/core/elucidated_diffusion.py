import torch
from einops import rearrange, reduce, repeat
from random import random
import torch.nn.functional as F
from tqdm import tqdm
from math import sqrt
import warnings

from torch import nn

def log(t, eps=1e-20):
    return torch.log(t.clamp(min=eps))

class ElucidateDiffusion(nn.Module):
    def __init__(self,
                    backbone,
                    device,
                    p_mu=-1.2,
                    p_sigma=1.2,
                    sigma_min=0.002,
                    sigma_max = 80,
                    s_churn=80,
                    s_tmin=0.05,
                    s_tmax=50,
                    s_noise=1.003,
                    sigma_data=0.5,
                    rho = 7,
                    mask=None,
                    latitude_weight=None,
                    variable_weight=None,
                    num_sample_steps=32,
                    self_conditioning=False,

                 ):
        super().__init__()

        self.net = backbone
        self.p_mu = p_mu
        self.p_sigma = p_sigma
        self.sigma_min = sigma_min
        self.sigma_max = sigma_max
        self.sigma_data = sigma_data
        self.s_churn = s_churn
        self.s_tmin = s_tmin
        self.s_tmax = s_tmax
        self.s_noise = s_noise
        self.mask = mask
        self.latitude_weight = latitude_weight
        self.variable_weight = variable_weight
        self.num_sample_steps = num_sample_steps
        self.self_conditioning = self_conditioning
        self.device = device
        self.rho = rho

        self.dtype = torch.float32


    def forward(self, x_target, x_conditional):
        
        bs = x_target.size(0)

        sigmas = self.noise_distribution(bs, self.p_mu, self.p_sigma)

        # padd sigma
        sigmas = rearrange(sigmas, 'b -> b 1 1 1')
        noise = torch.randn_like(x_target)
        x_noisy = x_target + sigmas * noise

        assert x_noisy.size(1) == 2, f"{x_noisy.size()}"
       
        self_conditional = None
        if self.self_conditioning and random() < 0.5:
            with torch.no_grad():
                x_noisy = self(x_target, x_noisy)
                self_conditional = self.conditional_network(x_noisy, x_conditional, sigmas)
                self_conditional.detach_()

        denoised = self.conditional_network(x_noisy, x_conditional, sigmas, self_conditional)

        return denoised, sigmas
        

    def loss_weight(self, sigma):
        return (sigma**2 + self.sigma_data**2) * (sigma * self.sigma_data) ** -2


    def conditional_network(self, x_noisy, x_conditional, sigma, self_conditional=None):
        
        network_input = torch.cat([x_noisy, x_conditional], dim=1)
        
        bs = x_noisy.size(0)

        if isinstance(sigma, float): # This happens during sampling
            sigma = torch.full((bs,), sigma,  device = self.device)
            sigma = rearrange(sigma, 'b -> b 1 1 1')

        net_out = self.net(
            self.c_in(sigma)*network_input, self.c_noise(sigma), self_conditional
        )

        out = (
            self.c_skip(sigma)*x_noisy 
            + self.c_out(sigma) * net_out
            )
        

        return out
    
    def noise_distribution(self, batch_size, mu, sigma):
        return mu + sigma * torch.randn((batch_size,), device='cuda')
    
    def c_skip(self, sigma):
        return (self.sigma_data**2) / (sigma**2 + self.sigma_data**2)

    def c_out(self, sigma):
        return sigma * self.sigma_data * (self.sigma_data**2 + sigma**2) ** -0.5

    def c_in(self, sigma):
        return 1 * (sigma**2 + self.sigma_data**2) ** -0.5

    def c_noise(self, sigma):
        return log(sigma) * 0.25

    
    @torch.no_grad()
    def sample(self, x_conditional, x_target_channels=2, num_sample_steps=None, clamp=True):
        
        if num_sample_steps is None:
            num_sample_steps = self.num_sample_steps

        sigmas = self.sample_schedule(num_sample_steps)

        gammas = torch.where(
            (sigmas>=self.s_tmin) & (sigmas<=self.s_tmax),
            min(self.s_churn / num_sample_steps, sqrt(2)-1), 0.0
        )

        sigmas_and_gammas = list(zip(sigmas[:-1], sigmas[1:], gammas[:-1] ))

        init_sigma = sigmas[0]

        shape = [s if i!=1 else x_target_channels for i, s in enumerate(x_conditional.size()) ] 

        shape = tuple(shape)

        x = init_sigma * torch.randn(shape).to(self.device)

        if self.mask is not None:
            x = x * self.mask
        
        x_start = None

        for sigma, next_sigma, gamma in tqdm(sigmas_and_gammas, desc="sampling_timestep"):
            
            sigma, next_sigma, gamma = map(lambda t: t.item(), (sigma, next_sigma, gamma)) # scalar sigma

            eps = self.s_noise * torch.randn(shape).to(self.device)
            
            sigma_hat = sigma+gamma*sigma

            x_hat = x + sqrt(sigma_hat**2 - sigma**2) * eps

            if self.mask is not None:
                x_hat = x_hat * self.mask

            self_cond = x_start if self.self_conditioning else None

            model_out = self.conditional_network(x_hat, x_conditional, sigma_hat, self_cond)

            denoised_over_sigma = (x_hat - model_out) / sigma_hat

            next_x = x_hat + (next_sigma - sigma_hat) * denoised_over_sigma
            
            if self.mask is not None:
                next_x = next_x * self.mask

            if next_sigma != 0:
                self_cond = model_out if self.self_conditioning else None

                next_model_out = self.conditional_network(next_x, x_conditional, next_sigma, self_cond)
                denoised_prime_over_sigma = (next_x - next_model_out) / next_sigma

                next_x = next_x + 0.5 * (next_sigma - sigma_hat) * (denoised_over_sigma + denoised_prime_over_sigma)

            x = next_x

            if self.mask is not None:
                x = x * self.mask
            
            x_start = next_model_out if next_sigma != 0 else model_out

        return x 

    def sample_schedule(self, num_sample_steps):
        
        if self.num_sample_steps != num_sample_steps:
            warnings.warn(f"num_sample_steps {num_sample_steps} is different from the num_sample_steps used during training {self.num_sample_steps}")
            
        inv_rho = 1/self.rho

        steps = torch.arange(num_sample_steps, device=self.device, dtype = self.dtype)

        sigmas = (
            self.sigma_max**inv_rho + steps / (num_sample_steps - 1) * (self.sigma_min**inv_rho - self.sigma_max**inv_rho)
        ) ** self.rho

        sigmas = F.pad(sigmas, (0, 1), value=0.0)

        return sigmas
