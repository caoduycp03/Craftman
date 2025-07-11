import math
import torch
import torch.nn as nn
import numpy as np
from einops import rearrange
from itertools import repeat
from collections.abc import Iterable
from torch.utils.checkpoint import checkpoint, checkpoint_sequential
from timm.models.layers import DropPath
from craftsman.models.transformers.utils import MLP
from craftsman.models.transformers.attention import MultiheadAttention, MultiheadCrossAttention

class DiTBlock(nn.Module):
    """
    A DiT block with adaptive layer norm (adaLN-single) conditioning.
    """

    def __init__(self, width, heads, init_scale=1.0, qkv_bias=True, qkv_fuse=False, use_RMSNorm=False, use_flash=True, drop_path=0.0):
        super().__init__()
        if use_RMSNorm:
            self.norm1 = nn.RMSNorm(width, elementwise_affine=True, eps=1e-6)
        else:
            self.norm1 = nn.LayerNorm(width, elementwise_affine=True, eps=1e-6)
        self.attn = MultiheadAttention(
            n_ctx=None,
            width=width,
            heads=heads,
            init_scale=init_scale,
            qkv_bias=qkv_bias,
            qkv_fuse=qkv_fuse,
            use_flash=use_flash
        )
        self.cross_attn = MultiheadCrossAttention(
            n_data=None,
            width=width,
            heads=heads,
            data_width=None,
            init_scale=init_scale,
            qkv_bias=qkv_bias,
            qkv_fuse=qkv_fuse,
            use_flash=use_flash,
        )
        if use_RMSNorm:
            self.norm2 = nn.RMSNorm(width, elementwise_affine=True, eps=1e-6)
        else:
            self.norm2 = nn.LayerNorm(width, elementwise_affine=True, eps=1e-6)

        self.mlp = MLP(width=width, init_scale=init_scale)
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.scale_shift_table = nn.Parameter(torch.randn(6, width) / width ** 0.5)

    def forward(self, x, y, t, **kwargs):
        B, N, C = x.shape

        shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = (self.scale_shift_table[None] + t.reshape(B, 6, -1)).chunk(6, dim=1)
        x = x + self.drop_path(gate_msa * self.attn(t2i_modulate(self.norm1(x), shift_msa, scale_msa)).reshape(B, N, C))
        x = x + self.cross_attn(x, y)
        x = x + self.drop_path(gate_mlp * self.mlp(t2i_modulate(self.norm2(x), shift_mlp, scale_mlp)))

        return x
    
def t2i_modulate(x, shift, scale):
    return x * (1 + scale) + shift

def auto_grad_checkpoint(module, *args, **kwargs):
    if getattr(module, 'grad_checkpointing', False):
        if not isinstance(module, Iterable):
            return checkpoint(module, *args, **kwargs)
        gc_step = module[0].grad_checkpointing_step
        return checkpoint_sequential(module, gc_step, *args, **kwargs)
    return module(*args, **kwargs)


class TimestepEmbedder(nn.Module):
    """
    Embeds scalar timesteps into vector representations.
    """

    def __init__(self, hidden_size, frequency_embedding_size=256):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(frequency_embedding_size, hidden_size, bias=True),
            nn.SiLU(),
            nn.Linear(hidden_size, hidden_size, bias=True),
        )
        self.frequency_embedding_size = frequency_embedding_size

    @staticmethod
    def timestep_embedding(t, dim, max_period=10000):
        """
        Create sinusoidal timestep embeddings.
        :param t: a 1-D Tensor of N indices, one per batch element.
                          These may be fractional.
        :param dim: the dimension of the output.
        :param max_period: controls the minimum frequency of the embeddings.
        :return: an (N, D) Tensor of positional embeddings.
        """
        # https://github.com/openai/glide-text2im/blob/main/glide_text2im/nn.py
        half = dim // 2
        freqs = torch.exp(
            -math.log(max_period) * torch.arange(start=0, end=half, dtype=torch.float32, device=t.device) / half)
        args = t[:, None].float() * freqs[None]
        embedding = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
        if dim % 2:
            embedding = torch.cat([embedding, torch.zeros_like(embedding[:, :1])], dim=-1)
        return embedding

    def forward(self, t):
        t_freq = self.timestep_embedding(t, self.frequency_embedding_size).to(self.dtype)
        t_emb = self.mlp(t_freq)
        return t_emb

    @property
    def dtype(self):
        return next(self.parameters()).dtype
    

class FinalLayer(nn.Module):
    """
    The final layer of DiT.
    """
    def __init__(self, hidden_size, out_channels):
        super().__init__()
        self.norm_final = nn.LayerNorm(hidden_size, elementwise_affine=True, eps=1e-6)
        self.linear = nn.Linear(hidden_size, out_channels, bias=True)
        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(),
            nn.Linear(hidden_size, 2 * hidden_size, bias=True)
        )

    def forward(self, x, c):
        shift, scale = self.adaLN_modulation(c).chunk(2, dim=-1)
        x = t2i_modulate(self.norm_final(x), shift, scale)
        x = self.linear(x)
        return x

class T2IFinalLayer(nn.Module):
    """
    The final layer of PixArt.
    """

    def __init__(self, hidden_size, out_channels, use_RMSNorm=False):
        super().__init__()
        if use_RMSNorm:
            self.norm_final = nn.RMSNorm(hidden_size, elementwise_affine=True, eps=1e-6)
        else:
            self.norm_final = nn.LayerNorm(hidden_size, elementwise_affine=True, eps=1e-6)
        self.linear = nn.Linear(hidden_size, out_channels, bias=True)
        self.scale_shift_table = nn.Parameter(torch.randn(2, hidden_size) / hidden_size ** 0.5)
        self.out_channels = out_channels

    def forward(self, x, t):
        shift, scale = (self.scale_shift_table[None] + t[:, None]).chunk(2, dim=1)
        x = t2i_modulate(self.norm_final(x), shift, scale)
        x = self.linear(x)
        return x
    
class QwenVLDenoiser(nn.Module):
    def __init__(self, llm=None) -> None:
        super().__init__()
        self.llm = llm
        self.embeds_3d_len = 1024
        self.initialize_weights()

    def initialize_weights(self):
        # Initialize transformer layers:
        def _basic_init(module):
            if isinstance(module, nn.Linear):
                torch.nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)
        self.apply(_basic_init)
        
    def forward(self, x, timestep, class_token, past_key_values=None):
        cache_position = None

        inputs_embeds = torch.cat([class_token, timestep, x], dim=1)
        attention_mask = torch.ones(inputs_embeds.shape[1]).unsqueeze(0).repeat(inputs_embeds.shape[0], 1)
        
        output_lm = self.llm(
            input_ids=None,
            attention_mask=attention_mask,
            inputs_embeds=inputs_embeds,
            output_hidden_states=True,
            return_dict=True,
            past_key_values=past_key_values,
        )
        output = output_lm.hidden_states[-1]

        recon_3d_embeds = output[:, -self.embeds_3d_len:, :] #bz x n_3d_tokens x dim
        return recon_3d_embeds
    
class QwenVLDenoiserStack(nn.Module):
    def __init__(self, llm=None) -> None:
        super().__init__()
        self.llm = llm
        self.vocab_dim = 2048
        self.embeds_3d_dim = 64
        self.embeds_3d_len = 256
        self.initialize_weights()

    def initialize_weights(self):
        # Initialize transformer layers:
        def _basic_init(module):
            if isinstance(module, nn.Linear):
                torch.nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)
        self.apply(_basic_init)
        

    def forward(self, x, timestep, input_ids, attention_mask, pixel_values, image_grid_thw, position_ids=None, past_key_values=None, return_past_key_values=False, offload_model:bool=False):
        cache_position = None
 
        block_size = int(self.vocab_dim/self.embeds_3d_dim)
        x = x.view(-1, int(self.embeds_3d_len/block_size), block_size, self.embeds_3d_dim).permute(0, 1, 3, 2).reshape(-1, int(self.embeds_3d_len/block_size), self.vocab_dim)
        inputs_embeds = torch.cat([timestep, x], dim=1)
        attention_mask = torch.ones(inputs_embeds.shape[1], device=attention_mask.device, dtype=attention_mask.dtype).unsqueeze(0).repeat(attention_mask.shape[0], 1)
        
        output_lm = self.llm(
            input_ids=None,
            attention_mask=attention_mask,
            inputs_embeds=inputs_embeds,
            output_hidden_states=True,
            return_dict=True,
            past_key_values=past_key_values,
        )
        output = output_lm.last_hidden_state

        recon_3d_embeds = output[:, -(int(self.embeds_3d_len/block_size)):, :] #bz x n_3d_tokens x dim
        recon_3d_embeds = recon_3d_embeds.view(-1, int(self.embeds_3d_len/block_size), self.embeds_3d_dim, block_size).permute(0, 1, 3, 2).reshape(-1, self.embeds_3d_len, self.embeds_3d_dim)

        return recon_3d_embeds