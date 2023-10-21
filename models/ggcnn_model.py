from .base_model import BaseModel
import torch
import torch.nn.functional as F


class GGCNNModel(BaseModel):
    def __init__(self, opt):
        BaseModel.__init__(self, opt)
        if self.isTrain:
            self.optimizer = torch.optim.Adam(
                self.net.parameters(), lr=opt.lr)

    def load_network(self):
        from .networks import GGCNN
        input_channels = self.opt.enable_rgb * 3 + self.opt.enable_depth
        return GGCNN(input_channels)

    def set_input(self, x, gt):
        self.x = x.to(self.device)
        self.gt = gt.to(self.device)

    def forward(self):
        self.pred = self.net(self.x)
        self.compute_loss()

    def compute_loss(self):
        gt_pos, gt_sin, gt_cos, gt_width = self.gt
        pred_pos, pred_sin, pred_cos, pred_width = self.pred

        loss_pos = F.mse_loss(pred_pos, gt_pos)
        loss_sin = F.mse_loss(pred_sin, gt_sin)
        loss_cos = F.mse_loss(pred_cos, gt_cos)
        loss_width = F.mse_loss(pred_width, gt_width)

        self.loss = loss_pos + loss_sin + loss_cos + loss_width
        self.loss = {
            'loss_pos': loss_pos,
            'loss_sin': loss_sin,
            'loss_cos': loss_cos,
            'loss_width': loss_width
        }

        self.pred = {'pos': pred_pos,
                     'sin': pred_sin,
                     'cos': pred_cos,
                     'width': pred_width}

    def backward(self):
        self.loss.backward()

    def optimize_parameters(self):
        self.forward()
        self.optimizer.zero_grad()
        self.backward()
        self.optimizer.step()
