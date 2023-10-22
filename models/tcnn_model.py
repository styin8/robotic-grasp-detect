from .base_model import BaseModel
import torch
import torch.nn.functional as F
from utils import evaluation


class GGCNNModel(BaseModel):
    def __init__(self, opt):
        BaseModel.__init__(self, opt)

    def load_network(self):
        from .networks import TCNN
        input_channels = self.opt.enable_rgb * 3 + self.opt.enable_depth
        self.net = TCNN(input_channels).to(self.device)
        if self.isTrain:
            self.optimizer = torch.optim.Adam(
                self.net.parameters())
        else:
            self.net.load_state_dict(torch.load(
                self.opt.model_path, map_location=self.device))
        return self.net

    def set_input(self, x, gt):
        self.x = x.to(self.device)
        self.gt = [y.to(self.device) for y in gt]

    def forward(self):
        self.pred = self.net(self.x)
        self.compute_loss()

    def _bin_focal_loss(self, pred, target, gamma=2, alpha=0.6):
        n, c, h, w = pred.size()

        _loss = -1 * target * \
            torch.log(pred + 1e-7) - (1 - target) * \
            torch.log(1 - pred+1e-7)    # (N, C, H, W)
        _gamma = torch.abs(pred - target) ** gamma

        zeros_loc = torch.where(target == 0)
        _alpha = torch.ones_like(pred) * alpha
        _alpha[zeros_loc] = 1 - alpha

        loss = _loss * _gamma * _alpha
        loss = loss.sum() / (n*c*h*w)
        return loss

    def compute_loss(self):
        gt_pos, gt_sin, gt_cos, gt_width = self.gt
        pred_pos, pred_sin, pred_cos, pred_width = self.pred

        pred_pos = torch.sigmoid(pred_pos)
        loss_pos = self._bin_focal_loss(pred_pos, gt_pos, alpha=0.9) * 10

        pred_sin = torch.sigmoid(pred_sin)
        loss_sin = self._bin_focal_loss(pred_sin, (gt_sin+1)/2, alpha=0.9) * 10

        pred_cos = torch.sigmoid(pred_cos)
        loss_cos = self._bin_focal_loss(pred_cos, (gt_cos+1)/2, alpha=0.9) * 10

        pred_width = torch.sigmoid(pred_width)
        loss_width = self._bin_focal_loss(pred_width, gt_width, alpha=0.9) * 10

        self.loss = loss_pos + loss_sin + loss_cos + loss_width
        self.losses = {
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

    def test(self, val_data, didx, rot, zoom_factor):
        self.forward()
        q_out, ang_out, w_out = self.post_process_output(self.pred['pos'], self.pred['sin'],
                                                         self.pred['cos'], self.pred['width'])
        s = evaluation.calculate_iou_match(q_out, ang_out,
                                           val_data.dataset.get_gtbb(
                                               didx, rot, zoom_factor),
                                           no_grasps=1,
                                           grasp_width=w_out,
                                           )
        return s
