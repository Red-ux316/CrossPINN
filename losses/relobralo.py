# -----------------------------------------------------------------------------
# Attribution and License Notice
# This implementation is adapted from the PyTorch implementation of ReLoBRaLo
# (Relative Loss Balancing with Random Lookback) by Khadrawi.
# Based on the original paper: "Multi-Objective Loss Balancing for 
# Physics-Informed Deep Learning" by Rafael Bischof and Michael Kraus.
# Original Repository: https://github.com/Khadrawi/ReLoBRaLo_PyTorch
# -----------------------------------------------------------------------------

import torch

class ReLoBRaLo:
    def __init__(self, num_losses: int, T: float = 0.1, alpha: float = 0.999,
                 rho: float = 0.999, device: str = 'cuda') -> None:
        """
        Initialization of ReloBRaLo (Relative Loss Balancing with Random Lookback).
        Input:
            num_losses: number of objectives/loss functions to be balanced
            T: temperature parameter
            alpha: exponential decay parameter for the weights of the losses
            rho: expected value of the bernoulli random value 'rho' used for exponential decay
        """
        self.num_losses = num_losses
        self.device = device
        
        # State tracking tensors (L0 for initial epoch, lam for current weights, l for previous losses)
        self.l0 = {f"l0_{i}": torch.tensor([1.0], device=device) for i in range(num_losses)}
        self.lam = {f"lam_{i}": torch.tensor([1.0], device=device) for i in range(num_losses)}
        self.l = {f"l_{i}": torch.tensor([1.0], device=device) for i in range(num_losses)}
        
        self.T = torch.tensor(T, dtype=torch.float32, device=device)
        self.rho_prob = torch.tensor(rho, device=device)
        self.alpha = alpha

    def set_l0(self, loss_list: list[torch.Tensor]) -> None:
        """
        Save the values of the losses at the first epoch of training.
        """
        assert len(loss_list) == self.num_losses, f"Expected {self.num_losses} losses, got {len(loss_list)}"
        assert all(isinstance(x, torch.Tensor) for x in loss_list), "Each loss must be a tensor"

        for i in range(self.num_losses):
            self.l0[f"l0_{i}"] = loss_list[i].detach().clone().reshape(1)

    def __call__(self, loss_list: list[torch.Tensor]) -> torch.Tensor:
        """
        Returns the balanced loss of the multiobjective problem.
        """
        assert len(loss_list) == self.num_losses, f"Expected {self.num_losses} losses, got {len(loss_list)}"
        
        # Sample random lookback variable
        rho = torch.bernoulli(self.rho_prob)
        
        # Compute relative scales
        lambs_hat_raw = torch.cat([loss_list[i].detach() / (self.l[f"l_{i}"] * self.T + 1e-12) for i in range(self.num_losses)])
        lambs0_hat_raw = torch.cat([loss_list[i].detach() / (self.l0[f"l0_{i}"] * self.T + 1e-12) for i in range(self.num_losses)])
        
        # Softmax normalized weighting
        lambs_hat = (torch.softmax(lambs_hat_raw, dim=0) * self.num_losses).detach()
        lambs0_hat = (torch.softmax(lambs0_hat_raw, dim=0) * self.num_losses).detach()
        
        lambs = []
        for i in range(self.num_losses):
            lam_i = (rho * self.alpha * self.lam[f"lam_{i}"] + 
                     (1 - rho) * self.alpha * lambs0_hat[i] + 
                     (1 - self.alpha) * lambs_hat[i])
            lambs.append(lam_i)
            
        # Combine losses
        loss = torch.sum(torch.cat([lambs[i] * loss_list[i].reshape(1) for i in range(self.num_losses)]))
        
        # Update states
        for i in range(self.num_losses):
            self.lam[f"lam_{i}"] = lambs[i].detach().clone()
            self.l[f"l_{i}"] = loss_list[i].detach().clone().reshape(1)
            
        return loss
