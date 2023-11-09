import abc
import json
import logging
import os

import torch
from tensorboardX import SummaryWriter
from torch import nn
from torch.optim import Optimizer
from torch.utils.data import DataLoader

from cv_engine.utils import get_logger


class Engine(abc.ABC):
    def __init__(
            self,
            tb_writer: SummaryWriter,
            logger: logging = get_logger('trainer'),
            device='cuda' if torch.cuda.is_available() else 'cpu'
    ):
        self.tb_writer = tb_writer
        self.logger = logger
        self.device = device
        self.state = {}

        self.epoch = 0

    def setup(
            self,
            model: nn.Module,
            train_loader: DataLoader,
            val_loader: DataLoader,
            optimizer: Optimizer,
            criterion: callable,
            epochs=100,
            dir_name='runs',
            eval_freq=None,
            start_epoch=0,
    ):
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.optimizer = optimizer
        self.criterion = criterion
        self.epochs = epochs
        self.dir_name = dir_name
        self.eval_freq = eval_freq
        self.start_epoch = start_epoch

    def run(self):
        for epoch in range(self.start_epoch, self.epochs):
            self.epoch = epoch
            self.before_step()
            self.train_step(epoch)
            self.after_step()
            if self.eval_freq and (epoch + 1) % self.eval_freq == 0:
                self.eval_step(epoch)

    @abc.abstractmethod
    def before_step(self):
        raise NotImplementedError

    @abc.abstractmethod
    def train_step(self, epoch):
        raise NotImplementedError

    def eval_step(self, epoch):
        pass

    def loss_fn(self, pred, target):
        loss = self.criterion(pred, target)
        self.criterion.backward()
        return loss

    def after_step(self):
        pass

    def save_model(self, **kwargs):
        assert os.path.exists(self.dir_name)
        save_path = self.dir_name
        results_to_save = {
            'epoch': self.epoch,
            'model': self.model.state_dict(),
            'optimizer': self.optimizer.state_dict(),
            'state': json.dumps(self.state, indent=4)
        }
        results_to_save.update(**kwargs)
        torch.save(results_to_save, os.path.join(save_path, f'{self.epoch}.pth'))

    def save_model_name(self, filename='last', **kwargs):
        assert os.path.exists(self.dir_name)
        save_path = self.dir_name
        results_to_save = {
            'model': self.model.state_dict(),
            'optimizer': self.optimizer.state_dict(),
            'epoch': self.epoch,
            'state': json.dumps(self.state, indent=4)
        }
        results_to_save.update(**kwargs)
        torch.save(results_to_save, os.path.join(save_path, f'{filename}.pth'))
