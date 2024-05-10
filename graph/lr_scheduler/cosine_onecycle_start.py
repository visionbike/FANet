import math
from torch.optim.lr_scheduler import LRScheduler
from torch.optim.optimizer import Optimizer


__all__ = ["CosineAnnealingOneCycleStartLR"]


class CosineAnnealingOneCycleStartLR(LRScheduler):
    """
    Cosine Annealing LR Scheduler, applying after T_start epochs.
    It has been proposed in `SGDR: Stochastic Gradient Descent with Warm Restarts`.

    Reference:
    - https://arxiv.org/abs/1608.03983.
    - https://github.com/katsura-jp/pytorch-cosine-annealing-with-warmup/blob/master/cosine_annealing_warmup/scheduler.py
    """

    def __init__(self,
                 optimizer: Optimizer,
                 T_max: int,
                 T_start: int = 0,
                 eta_max: float = 1e-3,
                 eta_min: float = 1e-5,
                 last_epoch: int = -1):
        """
        :param optimizer: the wrapped optimizer.
        :param T_max: the number of iterations for cosine annealing.
        :param T_start: the number of iterations before cosine annealing. Default: 0.
        :param eta_max: the maximum learning rate. Default: 1e-3.
        :param eta_min: the minimum learning rate. Default: 1e-5.
        :param last_epoch: the index of the last epoch. Default: -1.
        """
        self.T_max = T_max
        self.eta_max = eta_max
        self.eta_min = eta_min
        self.T_start = T_start
        self.init_lrs = []
        super().__init__(optimizer, last_epoch)
        self.init_lr()

    def init_lr(self):
        for param_group in self.optimizer.param_groups:
            param_group['lr'] = self.eta_max
            self.init_lrs.append(self.eta_max)

    def get_lr(self):
        if self.last_epoch < self.T_start:
            return self.init_lrs
        elif self.T_start <= self.last_epoch < (self.T_start + self.T_max):
            return [
                (self.eta_min + (base_lr - self.eta_min) * (1 + math.cos(math.pi * (self.last_epoch - self.T_start) / self.T_max)) / 2) for base_lr in self.base_lrs
            ]
        return [self.eta_min if (param_group['lr'] < self.eta_min) else param_group['lr'] for param_group in self.optimizer.param_groups]
