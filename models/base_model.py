import os
import torch
from abc import ABC, abstractmethod


class BaseModel(ABC):
    def __init__(self, opt) -> None:
        self.opt = opt
        self.gpu_ids = opt.gpu_ids
        self.isTrain = opt.isTrain


