import os
import torch
from abc import ABC, abstractmethod
from skimage.filters import gaussian


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
        self.load_network()
        self.print_network()

    @abstractmethod
    def load_network(self):
        pass

    @abstractmethod
    def test(self):
        pass

    def post_process_output(self, q_img, sin_img, cos_img, width_img):
        """
        :param q_img: Q output of net (as torch Tensors)
        :param cos_img: cos output of net
        :param sin_img: sin output of net
        :param width_img: Width output of net
        :return: Filtered Q output, Filtered Angle output, Filtered Width output
        """
        q_img = q_img.cpu().detach().numpy().squeeze()
        ang_img = (torch.atan2(sin_img, cos_img) /
                   2.0).cpu().detach().numpy().squeeze()
        width_img = width_img.cpu().detach().numpy().squeeze() * 150.0

        q_img = gaussian(q_img, 2.0, preserve_range=True)
        ang_img = gaussian(ang_img, 2.0, preserve_range=True)
        width_img = gaussian(width_img, 1.0, preserve_range=True)

        return q_img, ang_img, width_img

    def save_network(self, epoch, accuracy):
        save_path = os.path.join(
            self.opt.checkpoints_dir, self.opt.name, f'{self.opt.model}_net_epoch_{epoch}_acc_{accuracy}.pth')
        torch.save(self.net.state_dict(), save_path)

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
        if self.opt.isTrain:
            save_dir = os.path.join(self.opt.checkpoints_dir, self.opt.name)
            if not os.path.exists(save_dir):
                os.makedirs(save_dir)
            file_name = os.path.join(save_dir, f"{self.opt.name}_net.txt")
            with open(file_name, "wt") as f:
                f.write(message)
                f.write("\n")
