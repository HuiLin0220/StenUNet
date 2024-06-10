import math
from torch.optim.lr_scheduler import _LRScheduler,CosineAnnealingWarmRestarts


class PolyLRScheduler(_LRScheduler):
    def __init__(self, optimizer, initial_lr: float, max_steps: int, exponent: float = 0.9, current_step: int = None):
        self.optimizer = optimizer
        self.initial_lr = initial_lr
        self.max_steps = max_steps
        self.exponent = exponent
        self.ctr = 0
        super().__init__(optimizer, current_step if current_step is not None else -1, False)

    def step(self, current_step=None):
        if current_step is None or current_step == -1:
            current_step = self.ctr
            self.ctr += 1

        new_lr = self.initial_lr * (1 - current_step / self.max_steps) ** self.exponent
        for param_group in self.optimizer.param_groups:
            param_group['lr'] = new_lr


class CosineAnnealingWithWarmRestarts(CosineAnnealingWarmRestarts):
    def __init__(self, optimizer, T_0, T_mult=1, eta_min=0, last_epoch=-1):
        self.T_i = T_0
        self.T_mult = T_mult
        self.eta_min = eta_min
        super(CosineAnnealingWithWarmRestarts, self).__init__(optimizer, T_0, T_mult, eta_min, last_epoch)

    def get_lr(self):
        if self.last_epoch == 0:
            return self.base_lrs

        if self.T_i == 1:
            return [self.eta_min + (lr - self.eta_min) * (1 + math.cos(math.pi * self.T_cur / self.T_i)) / 2
                    for lr in self.base_lrs]

        if self.T_cur == self.T_i:
            self.T_cur = 0
            self.T_i *= self.T_mult

        return [self.eta_min + (lr - self.eta_min) * (1 + math.cos(math.pi * self.T_cur / self.T_i)) / 2
                for lr in self.base_lrs]

