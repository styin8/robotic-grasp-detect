from abc import ABC, abstractmethod
from torch.utils.data import Dataset
import torch
import numpy as np
import random


class BaseDataset(Dataset, ABC):
    def __init__(self, opt):
        self.opt = opt
        self.root = opt.dataroot
        self.random_rotate = opt.random_rotate
        self.random_zoom = opt.random_zoom
        self.output_size = opt.output_size

    @abstractmethod
    def __len__(self):
        pass


    @abstractmethod
    def get_gtbb(self, idx, rot=0, zoom=1.0):
        pass

    @abstractmethod
    def get_depth(self, idx, rot=0, zoom=1.0):
        pass

    @abstractmethod
    def get_rgb(self, idx, rot=0, zoom=1.0):
        pass

    @staticmethod
    def numpy_to_torch(s):
        if len(s.shape) == 2:
            return torch.from_numpy(np.expand_dims(s, 0).astype(np.float32))
        else:
            return torch.from_numpy(s.astype(np.float32))

    def __getitem__(self, idx):
        if self.random_rotate:
            rotations = [0, np.pi/2, 2*np.pi/2, 3*np.pi/2]
            rot = random.choice(rotations)
        else:
            rot = 0.0

        if self.random_zoom:
            zoom_factor = np.random.uniform(0.5, 1.0)
        else:
            zoom_factor = 1.0

        # Load the depth image
        if self.include_depth:
            depth_img = self.get_depth(idx, rot, zoom_factor)

        # Load the RGB image
        if self.include_rgb:
            rgb_img = self.get_rgb(idx, rot, zoom_factor)

        # Load the grasps
        bbs = self.get_gtbb(idx, rot, zoom_factor)

        pos_img, ang_img, width_img = bbs.draw(
            (self.output_size, self.output_size))
        width_img = np.clip(width_img, 0.0, 150.0)/150.0

        if self.include_depth and self.include_rgb:
            x = self.numpy_to_torch(
                np.concatenate(
                    (np.expand_dims(depth_img, 0),
                     rgb_img),
                    0
                )
            )
        elif self.include_depth:
            x = self.numpy_to_torch(depth_img)
        elif self.include_rgb:
            x = self.numpy_to_torch(rgb_img)

        pos = self.numpy_to_torch(pos_img)
        cos = self.numpy_to_torch(np.cos(2*ang_img))
        sin = self.numpy_to_torch(np.sin(2*ang_img))
        width = self.numpy_to_torch(width_img)

        return x, (pos, cos, sin, width), idx, rot, zoom_factor
