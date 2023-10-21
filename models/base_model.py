import os
import torch
from abc import ABC, abstractmethod


class BaseModel(ABC):
    def __init__(self, opt) -> None:
        self.opt = opt
        self.gpu_ids = opt.gpu_ids
        self.isTrain = opt.isTrain
        self.device = torch.device(
            f"cuda:{self.gpu_ids}" if torch.cuda.is_available() else "cpu")
        self.net = None

    @abstractmethod
    def set_input(self, input):
        pass

    @abstractmethod
    def forward(self):
        pass

    @abstractmethod
    def optimize_parameters(self):
        pass

    def setup(self):
        self.net = self.load_network()
        self.print_network()

    @abstractmethod
    def load_network(self):
        pass

    def save_network(self, net, opt):
        pass

    def print_network(self):
        message = ""
        message += '--------------- Networks ----------------\n'
        for _, module in self.net.named_modules():
            if module.__class__.__name__ != self.net.__class__.__name__:
                message += '{:<25}: {:<30}\n'.format(str(module.__class__.__name__), str(
                    sum(p.numel() for p in module.parameters())))
        message += '-----------------------------------------\n'
        message += f'Total number of parameters : {sum(p.numel() for p in self.net.parameters())/1e6:.3f} M\n'
        print(message)

        # save in the disk
        save_dir = os.path.join(self.opt.checkpoints_dir, self.opt.name)
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
        file_name = os.path.join(save_dir, f"{self.opt.name}net.txt")
        with open(file_name, "wt") as f:
            f.write(message)
            f.write("\n")
