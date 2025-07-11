from dataclasses import dataclass, field
import os
import numpy as np
import json
import copy
import torch
import torch.nn.functional as F
from skimage import measure
from einops import repeat
from tqdm import tqdm
from PIL import Image
import random

from diffusers import (
    DDPMScheduler,
    DDIMScheduler,
    UniPCMultistepScheduler,
    KarrasVeScheduler,
    DPMSolverMultistepScheduler
)

import craftsman
from craftsman.systems.base import BaseSystem
from craftsman.utils.misc import get_rank
from craftsman.utils.typing import *
from diffusers import DDIMScheduler
from craftsman.systems.utils import compute_snr, ddim_sample


# DEBUG = True
@craftsman.register("shape-diffusion-system-nods")
class ShapeDiffusionSystem(BaseSystem):
    @dataclass
    class Config(BaseSystem.Config):
        val_samples_json: str = None
        extract_mesh_func: str = "mc"

        # diffusion config
        num_class: int = 2
        z_scale_factor: float = 1.0
        guidance_scale: float = 7.5
        num_inference_steps: int = 50
        eta: float = 0.0
        snr_gamma: float = 5.0

        # shape vae model
        shape_model_type: str = None
        shape_model: dict = field(default_factory=dict)

        # condition model
        condition_model_type: str = None
        condition_model: dict = field(default_factory=dict)

        # diffusion model
        denoiser_model_type: str = None
        denoiser_model: dict = field(default_factory=dict)

        # noise scheduler
        noise_scheduler_type: str = None
        noise_scheduler: dict = field(default_factory=dict)

        # denoise scheduler
        denoise_scheduler_type: str = None
        denoise_scheduler: dict = field(default_factory=dict)

    cfg: Config

    def configure(self):
        super().configure()
        self.shape_model = craftsman.find(self.cfg.shape_model_type)(self.cfg.shape_model)
        self.shape_model.eval()
        self.shape_model.requires_grad_(False)

        self.condition = None
        
        self.denoiser_model = craftsman.find(self.cfg.denoiser_model_type)(self.cfg.denoiser_model)

        self.noise_scheduler = craftsman.find(self.cfg.noise_scheduler_type)(**self.cfg.noise_scheduler)

        self.denoise_scheduler = craftsman.find(self.cfg.denoise_scheduler_type)(**self.cfg.denoise_scheduler)

    def forward(self, batch: Dict[str, Any], skip_noise=False) -> Dict[str, Any]:
        # 1. encode shape latents
        latents = batch['kl_embed'] * self.cfg.z_scale_factor

        # 3. sample noise that we"ll add to the latents
        noise = torch.randn_like(latents).to(latents) # [batch_size, n_token, latent_dim]
        bs = latents.shape[0]
            
        # 4. Sample a random timestep for each motion
        timesteps = torch.randint(
            0,
            self.cfg.noise_scheduler.num_train_timesteps,
            (bs,),
            device=latents.device,
        )
        timesteps = timesteps.long()
        # 5. add noise
        noisy_z = self.noise_scheduler.add_noise(latents, noise, timesteps)
        # 6. diffusion model forward
        uids = batch['uid']
        class_tokens = [
            0 if "table" in uid else 1
            for uid in uids
        ]
        #null class token for cfg
        if random.random() < 0.1: 
            for i in range(len(class_tokens)):
                class_tokens[i] = self.cfg.num_class

        class_tokens = torch.tensor(class_tokens)
        noise_pred = self.denoiser_model(noisy_z, timesteps, class_token=class_tokens)

        # 7. compute loss
        if self.noise_scheduler.config.prediction_type == "epsilon":
            target = noise 
        elif self.noise_scheduler.config.prediction_type == "v_prediction":
            target = self.noise_scheduler.get_velocity(latents, noise, timesteps)
        else:
            raise ValueError(f"Prediction Type: {self.noise_scheduler.prediction_type} not supported.")
        if self.cfg.snr_gamma == 0:
            if self.cfg.loss.loss_type == "l1":
                loss = F.l1_loss(noise_pred, target, reduction="mean")
            elif self.cfg.loss.loss_type in ["mse", "l2"]:
                loss = F.mse_loss(noise_pred, target, reduction="mean")
            else:
                raise ValueError(f"Loss Type: {self.cfg.loss.loss_type} not supported.")
        else:
            # Compute loss-weights as per Section 3.4 of https://arxiv.org/abs/2303.09556.
            # Since we predict the noise instead of x_0, the original formulation is slightly changed.
            # This is discussed in Section 4.2 of the same paper.
            snr = compute_snr(self.noise_scheduler, timesteps)
            mse_loss_weights = torch.stack([snr, self.cfg.snr_gamma * torch.ones_like(timesteps)], dim=1).min(
                dim=1
            )[0]
            if self.noise_scheduler.config.prediction_type == "epsilon":
                mse_loss_weights = mse_loss_weights / snr
            elif self.noise_scheduler.config.prediction_type == "v_prediction":
                mse_loss_weights = mse_loss_weights / (snr + 1)
            if self.cfg.loss.loss_type == "l1":
                loss = F.l1_loss(noise_pred, target, reduction="none")
            elif self.cfg.loss.loss_type in ["mse", "l2"]:
                loss = F.mse_loss(noise_pred, target, reduction="none")
            else:
                raise ValueError(f"Loss Type: {self.cfg.loss.loss_type} not supported.")
            loss = loss.mean(dim=list(range(1, len(loss.shape)))) * mse_loss_weights
            loss = loss.mean()


        return {
            "loss_diffusion": loss,
            "latents": latents,
            "x_t": noisy_z,
            "noise": noise,
            "noise_pred": noise_pred,
            "timesteps": timesteps,
            }

    def training_step(self, batch, batch_idx):
        out = self(batch)
        loss = 0.
        for name, value in out.items():
            if name.startswith("loss_"):
                self.log(f"train/{name}", value)
                loss += value * self.C(self.cfg.loss[name.replace("loss_", "lambda_")])

        for name, value in self.cfg.loss.items():
            if name.startswith("lambda_"):
                self.log(f"train_params/{name}", self.C(value))

        print(f"Batch {batch_idx}: Average loss per sample = {loss}")

        return {"loss": loss}

    @torch.no_grad()
    def validation_step(self, batch, batch_idx):
        self.eval()
        os.makedirs(f"shapenet_class_condtioned_nods_06B_1024", exist_ok=True)
        cfgs = [1.0, 2.0, 3.0, 5.0, 7.5]
        for cfg in cfgs:
            if get_rank() == 0:
                uids = batch['uid']
                class_tokens = [
                    0 if "table" in uid else 1
                    for uid in uids
                ]
                class_tokens = torch.tensor(class_tokens)
                sample_outputs = self.sample(class_token=class_tokens, guidance_scale=cfg)
                
                for i, sample_output in enumerate(sample_outputs):
                    torch.save(sample_output, f"shapenet_class_condtioned_nods_06B_1024/it{self.true_global_step}_{batch['uid'][i]}_cfg{cfg}.pt")

        out = self(batch)
        if self.global_step == 0:
            torch.save(out["latents"], f"shapenet_class_condtioned_nods_06B_1024/sanity_check.pt")

        return {"val/loss": out["loss_diffusion"]}
 
    @torch.no_grad()
    def sample(self,
               class_token: torch.Tensor = None,
               sample_times: int = 1,
               steps: Optional[int] = None,
               eta: float = 0.0,
               guidance_scale: Optional[float] = None,
               seed: Optional[int] = None,
               **kwargs):

        if steps is None:
            steps = self.cfg.num_inference_steps
        if guidance_scale is None:
            guidance_scale = self.cfg.guidance_scale
        do_classifier_free_guidance = guidance_scale != 1.0

        # conditional encode
        if do_classifier_free_guidance:
            unclass_token = torch.tensor([self.cfg.num_class]).to(class_token)
            class_token = torch.cat([unclass_token, class_token], dim=0)

        outputs = []
        latents = None
        
        if seed != None:
            generator = torch.Generator(device="cuda").manual_seed(seed)
        else:
            generator = None

        for _ in range(sample_times):
            sample_loop = ddim_sample(
                self.denoise_scheduler,
                self.denoiser_model.eval(),
                shape=self.shape_model.latent_shape,
                bsz=1,
                steps=steps,
                guidance_scale=guidance_scale,
                do_classifier_free_guidance=do_classifier_free_guidance,
                device=self.device,
                eta=eta,
                disable_prog=False,
                generator= generator,
                class_token=class_token
            )
            for sample, t in sample_loop:
                latents = sample
            # outputs.append(self.shape_model.decode(latents / self.cfg.z_scale_factor, **kwargs))
        
        # return outputs
        return latents / self.cfg.z_scale_factor

    def on_validation_epoch_end(self):
        pass
