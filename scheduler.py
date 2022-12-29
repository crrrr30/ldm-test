import numpy as np

class ScheduledOptim():
    def __init__(self, optimizer, total_steps, base, decay_type, warmup_steps, linear_end=1e-5):
        super().__init__()
        self._optimizer = optimizer
        self.total_steps = total_steps
        self.base = base
        self.decay_type = decay_type
        self.warmup_steps = warmup_steps
        self.linear_end = linear_end
        self.step = 0

    def step_and_update_lr(self):
        self._update_learning_rate()
        self._optimizer.step()

    def zero_grad(self):
        self._optimizer.zero_grad()

    def _update_learning_rate(self):
        self.step += 1
        lr = self.base
        progress = (self.step - self.warmup_steps) / float(self.total_steps - self.warmup_steps)
        progress = np.clip(progress, 0., 1.)
        if self.decay_type == 'linear':
            lr = self.linear_end + (lr - self.linear_end) * (1. - progress)
        elif self.decay_type == 'cosine':
            lr = lr * 0.5 * (1. + np.cos(np.pi * progress))
        else:
            raise ValueError(f'Unknown lr type {self.decay_type}')
        if self.warmup_steps:
            lr = lr * np.minimum(1., self.step / self.warmup_steps)
        for param_group in self._optimizer.param_groups:
            param_group['lr'] = lr
    
    def state_dict(self):
        return {
            "step": self.step,
            "total_steps": self.total_steps,
            "base": self.base,
            "decay_type": self.decay_type,
            "warmup_steps": self.warmup_steps,
            "linear_end": self.linear_end
        }

    def load_state_dict(self, state_dict):
        self.step = state_dict["step"]
        self.total_steps = state_dict["total_steps"]
        self.base = state_dict["base"]
        self.decay_type = state_dict["decay_type"]
        self.warmup_steps = state_dict["warmup_steps"]
        self.linear_end = state_dict["linear_end"]
        