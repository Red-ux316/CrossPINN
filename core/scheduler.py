import torch
import math
from torch.optim.lr_scheduler import LRScheduler

class WarmupCosineLR(LRScheduler):
    """
    Learning Rate Scheduler that combines linear warmup with cosine annealing.
    Ideal for Transformer-based architectures (e.g. TransPINN, TLP).
    """
    def __init__(self, optimizer, warmup_epochs, total_epochs, eta_min=1e-6, last_epoch=-1):
        self.warmup_epochs = warmup_epochs
        self.total_epochs = total_epochs
        self.eta_min = eta_min
        super().__init__(optimizer, last_epoch)

    def get_lr(self):
        if self.last_epoch < self.warmup_epochs:
            # Linear warmup
            alpha = self.last_epoch / max(1, self.warmup_epochs)
            return [self.eta_min + alpha * (base_lr - self.eta_min) for base_lr in self.base_lrs]
        else:
            # Cosine decay
            progress = (self.last_epoch - self.warmup_epochs) / max(1, self.total_epochs - self.warmup_epochs)
            return [self.eta_min + 0.5 * (base_lr - self.eta_min) * (1 + math.cos(math.pi * progress)) for base_lr in self.base_lrs]


def get_scheduler(optimizer, config):
    """
    Factory function to instantiate the learning rate scheduler defined in the config YAML.
    """
    sched_type = config['training'].get('scheduler', 'CosineAnnealing')
    epochs = config['training']['epochs']
    
    if sched_type == 'CosineAnnealing':
        return torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, 
            T_max=epochs, 
            eta_min=1e-6
        )
    elif sched_type == 'WarmupCosine':
        # Default warmup to 10% of total epochs if not specified
        warmup_epochs = config['training'].get('warmup_epochs', int(epochs * 0.1))
        return WarmupCosineLR(
            optimizer,
            warmup_epochs=warmup_epochs,
            total_epochs=epochs,
            eta_min=1e-6
        )
    elif sched_type == 'StepLR':
        step_size = config['training'].get('step_size', epochs // 3)
        return torch.optim.lr_scheduler.StepLR(
            optimizer,
            step_size=max(1, step_size),
            gamma=0.1
        )
    elif sched_type in [None, 'None', 'none']:
        return None
    else:
        raise ValueError(f"Unknown scheduler type: {sched_type}")
