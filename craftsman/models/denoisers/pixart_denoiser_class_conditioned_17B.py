from dataclasses import dataclass

import torch
import torch.nn as nn
import math
import importlib
import craftsman
import re
from transformers import AutoTokenizer, AutoModel
from typing import Optional
from craftsman.utils.base import BaseModule
from craftsman.models.denoisers.utils_class_conditioned import *

@craftsman.register("pixart-denoiser-class-conditioned-17B")
class PixArtDinoDenoiser(BaseModule):
    @dataclass
    class Config(BaseModule.Config):
        pretrained_model_name_or_path: Optional[str] = None
        input_channels: int = 32
        output_channels: int = 32
        class_dim: int = 2
        n_ctx: int = 512
        width: int = 768
        layers: int = 28
        heads: int = 16
        context_dim: int = 1024
        n_views: int = 1
        context_ln: bool = True
        init_scale: float = 0.25
        use_checkpoint: bool = False
        drop_path: float = 0.
        qkv_fuse: bool = True
        clip_weight: float = 1.0
        dino_weight: float = 1.0
        condition_type: str = "clip_dinov2"
        use_RMSNorm: bool = False

    cfg: Config

    def configure(self) -> None:
        super().configure()

        # timestep embedding
        self.class_embed = nn.Embedding(self.cfg.class_dim+1, self.cfg.width).requires_grad_(True)
        self.time_embed = TimestepEmbedder(self.cfg.width)

        # x embedding
        self.x_embed = nn.Linear(self.cfg.input_channels, self.cfg.width, bias=True)

        # context embedding
        if self.cfg.context_ln:
            if "clip" in self.cfg.condition_type:
                self.clip_embed = nn.Sequential(
                    nn.RMSNorm(self.cfg.context_dim) if self.cfg.use_RMSNorm else nn.LayerNorm(self.cfg.context_dim),
                    nn.Linear(self.cfg.context_dim, self.cfg.width),
                )

            if "dino" in self.cfg.condition_type:
                self.dino_embed = nn.Sequential(
                    nn.RMSNorm(self.cfg.context_dim) if self.cfg.use_RMSNorm else nn.LayerNorm(self.cfg.context_dim),
                    nn.Linear(self.cfg.context_dim, self.cfg.width),
                )
        else:
            if "clip" in self.cfg.condition_type:
                self.clip_embed = nn.Linear(self.cfg.context_dim, self.cfg.width)
            if "dino" in self.cfg.condition_type:
                self.dino_embed = nn.Linear(self.cfg.context_dim, self.cfg.width)

        init_scale = self.cfg.init_scale * math.sqrt(1.0 / self.cfg.width)
        drop_path = [x.item() for x in torch.linspace(0, self.cfg.drop_path, self.cfg.layers)]
        
        path = "Qwen/Qwen3-1.7B"
        llm = AutoModel.from_pretrained(
            path,
            torch_dtype=torch.bfloat16,
            local_files_only=False,
            trust_remote_code=True,
            cache_dir="./models_cache",
            resume_download=True
        )
        llm.gradient_checkpointing_enable()
        llm.config.use_cache = False
        self.denoiser = QwenVLDenoiser(llm=llm)

        self.t_block = nn.Sequential(
                        nn.SiLU(),
                        nn.Linear(self.cfg.width, self.cfg.width, bias=True)
                    )
        self.c_block = nn.Sequential(
                        nn.SiLU(),
                        nn.Linear(self.cfg.width, self.cfg.width, bias=True)
                    )
         # final layer
        self.final_layer = T2IFinalLayer(self.cfg.width, self.cfg.output_channels, self.cfg.use_RMSNorm)

        if self.cfg.pretrained_model_name_or_path:
            print(f"Loading pretrained model from {self.cfg.pretrained_model_name_or_path}")
            ckpt = torch.load(self.cfg.pretrained_model_name_or_path, map_location="cpu")
            if 'state_dict' in ckpt:
                ckpt = ckpt['state_dict']
            self.denoiser_ckpt = {}
            for k, v in ckpt.items():
                if k.startswith('denoiser_model.'):
                    self.denoiser_ckpt[k.replace('denoiser_model.', '')] = v
            self.load_state_dict(self.denoiser_ckpt, strict=False)


    def forward(self,
                model_input: torch.FloatTensor,
                timestep: torch.LongTensor,
                class_token: torch.Tensor = None):
        
        r"""
        Args:
            model_input (torch.FloatTensor): [bs, n_data, c]
            timestep (torch.LongTensor): [bs,]
            context (torch.FloatTensor): [bs, context_tokens, c]

        Returns:
            sample (torch.FloatTensor): [bs, n_data, c]

        """
        B, n_data, _ = model_input.shape
        # 1. time + class
        t_emb = self.time_embed(timestep)       
        class_emb = self.class_embed(class_token.to(self.class_embed.weight.device))
        # 4. denoiser
        latent = self.x_embed(model_input)
        # visual_cond = torch.zeros_like(latent).to(device=latent.device, dtype=latent.dtype)
        
        c0 = self.c_block(class_emb).unsqueeze(dim=1)
        t0 = self.t_block(t_emb).unsqueeze(dim=1)
        
        latent = self.denoiser(latent, t0, c0)
        condition_latent = t_emb + class_emb
        latent = self.final_layer(latent, condition_latent)

        return latent

