import torch.nn as nn

from .utils import RobustCosineSim, SSIM


class BaseLoss(nn.Module):
    def __init__(self, loss_type, weight=None):
        super(BaseLoss, self).__init__()
        if weight is None:
            weight = {}
        self.weight = weight
        self.final_loss = None
        self.loss_type = loss_type
        self.loss_dic = {
            'l1': nn.L1Loss,
            'l2': nn.MSELoss,
            'smooth_l1': nn.SmoothL1Loss,
            'cross_entropy': nn.CrossEntropyLoss,
            'cosine': RobustCosineSim,
            'ssim': SSIM
        }
        self.loss_func = self.get_loss_func()

    def get_loss_func(self, loss_type=None):
        if loss_type:
            return self.loss_dic.get(loss_type)()
        return self.loss_dic.get(self.loss_type)()

    def backward(self):
        self.final_loss.backward()

    def update_loss(self, value):
        self.final_loss = value

    def forward(self, ):
        raise NotImplementedError
