import math
import os
import json
import re
import cv2
from dataclasses import dataclass, field

import random
import imageio
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from PIL import Image

from craftsman.utils.typing import *

        
@dataclass
class BaseDataModuleConfig:
    local_dir: str = None

    ################################# Geometry part #################################
    load_geometry: bool = True           # whether to load geometry data
    geo_data_type: str = "occupancy"     # occupancy, sdf
    geo_data_path: str = ""              # path to the geometry data
    # for occupancy and sdf data
    n_samples: int = 4096                # number of points in input point cloud
    upsample_ratio: int = 1              # upsample ratio for input point cloud
    sampling_strategy: Optional[str] = None    # sampling strategy for input point cloud
    scale: float = 1.0                   # scale of the input point cloud and target supervision
    load_supervision: bool = True        # whether to load supervision
    supervision_type: str = "occupancy"  # occupancy, sdf, tsdf
    tsdf_threshold: float = 0.05         # threshold for truncating sdf values, used when input is sdf
    n_supervision: int = 10000           # number of points in supervision

    ################################# Image part #################################
    load_image: bool = False             # whether to load images 
    image_data_path: str = ""            # path to the image data
    image_type: str = "rgb"              # rgb, normal
    background_color: Tuple[float, float, float] = field(
            default_factory=lambda: (0.5, 0.5, 0.5)
        )
    idx: Optional[List[int]] = None      # index of the image to load
    n_views: int = 1                     # number of views
    marign_pix_dis: int = 30             # margin of the bounding box
    batch_size: int = 32
    num_workers: int = 8


class BaseDataset(Dataset):
    def __init__(self, cfg: Any, split: str) -> None:
        super().__init__()
        self.cfg: BaseDataModuleConfig = cfg
        self.split = split
        breakpoint()
        self.uids = os.listdir(self.cfg.geo_data_path)
        print(f"Loaded {len(self.uids)} {split} uids")
    
    def __len__(self):
        return len(self.uids)


    def _load_shape_from_occupancy_or_sdf(self, index: int) -> Dict[str, Any]:
        # for sdf data with our own format
        kl_embed = torch.load(f'{self.cfg.geo_data_path}/{self.uids[index]}.pt')
        
        ret = {
            "uid": self.uids[index].split('/')[-1],
            "kl_embed": kl_embed,
        }

        return ret


    def _get_data(self, index):
        ret = {"uid": self.uids[index]}
        # load geometry
        if self.cfg.load_geometry:
            if self.cfg.geo_data_type == "occupancy" or self.cfg.geo_data_type == "sdf":
                # load shape
                ret = self._load_shape_from_occupancy_or_sdf(index)
            else:
                raise NotImplementedError(f"Geo data type {self.cfg.geo_data_type} not implemented")

        return ret
        
    def __getitem__(self, index):
        try:
            return self._get_data(index)
        except Exception as e:
            print(f"Error in {self.uids[index]}: {e}")
            return self.__getitem__(np.random.randint(len(self)))

    def collate(self, batch):
        from torch.utils.data._utils.collate import default_collate_fn_map
        return torch.utils.data.default_collate(batch)
