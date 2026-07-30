import os
import sys
import torch
import shutil
from torch.utils.tensorboard import SummaryWriter
from pathlib import Path
from losses.relobralo import ReLoBRaLo

class Trainer:
    """
    Core training loop for Physics-Informed Neural Networks.
    Handles dynamic collocation sampling, loss computation, backpropagation, and TensorBoard logging.
    """
    def __init__(self, model, problem, pde_loss, optimizer, config, data_dir, scheduler=None, optimizer_lbfgs=None, wandb_run=None):
        self.model = model
        self.problem = problem
        self.pde_loss = pde_loss
        self.optimizer = optimizer
        self.optimizer_lbfgs = optimizer_lbfgs
        self.scheduler = scheduler
        self.config = config
        self.epochs_adam = config['training'].get('epochs_adam', int(config['training']['epochs'] * 0.8))
        self.wandb_run = wandb_run
        self.device = next(model.parameters()).device
        
        self.data_dir = Path(data_dir)
        self.log_dir = self.data_dir / "logs" / self.config['experiment']['name']
        self.writer = SummaryWriter(log_dir=str(self.log_dir))
        
        self.domain = self.config['data']['domain']
        self.is_2d = isinstance(self.domain[0], list)
        self.batch_size_adam = self.config['data'].get('num_collocation_points_adam', 500)
        self.batch_size_lbfgs = self.config['data'].get('num_collocation_points_lbfgs', 2000)
        self.use_relobralo = self.config['training'].get('use_relobralo', True)
        self.relobralo = None
        
    def train(self, start_epoch=1, trial=None, total_trials=None, current_trial_count=None):
        import time
        from utils.logger import print_info, print_success, print_error, Colors

        def _print_info(*args, **kwargs):
            if trial is None: print_info(*args, **kwargs)
        def _print_success(*args, **kwargs):
            if trial is None: print_success(*args, **kwargs)
        def _print_error(*args, **kwargs):
            if trial is None: print_error(*args, **kwargs)

        epochs = self.config['training']['epochs']
        self.model.train()
        
        _print_info(f"Starting training for {epochs} epochs on device: {self.device}")

        
        patience = self.config['training'].get('early_stopping_patience', 500)
        min_delta = float(self.config['training'].get('early_stopping_min_delta', 1e-5))
        best_mae = float('inf')
        epochs_without_improvement = 0
        force_lbfgs_switch = False

        weights_dir = self.data_dir / "weights"
        weights_dir.mkdir(parents=True, exist_ok=True)
        save_path_best = weights_dir / f"{self.config['experiment']['name']}_best_model.pt"
        save_path_latest = weights_dir / f"{self.config['experiment']['name']}_latest_model.pt"
        
        # Check if the problem defines any boundary or initial conditions.
        has_boundary_conditions = hasattr(self.problem, 'boundary_points')
        
        # Read BC config
        self.num_bc_adam = self.config['data'].get('num_boundary_points_adam', self.batch_size_adam // 8)
        self.num_bc_lbfgs = self.config['data'].get('num_boundary_points_lbfgs', (self.batch_size_lbfgs // 8) if self.batch_size_lbfgs else None)
        
        # Placeholders for static L-BFGS boundary points
        self.x_ic_static = None
        self.x_bc_static = None
        self.u_exact_ic_static = None
        self.u_exact_bc_static = None

        is_hybrid = self.optimizer_lbfgs is not None
        
        # Determine active optimizer if resuming from checkpoint
        if is_hybrid and start_epoch > self.epochs_adam:
            self.optimizer = self.optimizer_lbfgs
            is_lbfgs = True
        else:
            is_lbfgs = isinstance(self.optimizer, torch.optim.LBFGS)
            
        def generate_collocation_points(n_points):
            distribution = self.config['model'].get('support_points_distribution', 'uniform')
            beta_penalty = self.config['model'].get('beta_penalty', 5.0)

            if self.is_2d:
                domain_x, domain_y = self.domain
                if distribution == 'lhs':
                    from scipy.stats import qmc
                    import numpy as np
                    sampler = qmc.LatinHypercube(d=2)
                    sample = sampler.random(n=n_points)
                    x_lhs = sample[:, 0] * (domain_x[1] - domain_x[0]) + domain_x[0]
                    y_lhs = sample[:, 1] * (domain_y[1] - domain_y[0]) + domain_y[0]
                    pts = torch.tensor(np.column_stack((x_lhs, y_lhs)), dtype=torch.float32, device=self.device)
                else:
                    x_rand = (domain_x[1] - domain_x[0]) * torch.rand(n_points, 1, device=self.device) + domain_x[0]
                    y_rand = (domain_y[1] - domain_y[0]) * torch.rand(n_points, 1, device=self.device) + domain_y[0]
                    pts = torch.cat([x_rand, y_rand], dim=1)
            else:
                self.a, self.b = self.domain
                if distribution == 'chebyshev':
                    i = torch.arange(1, n_points + 1, dtype=torch.float32, device=self.device)
                    c_pts = -torch.cos((2 * i - 1) / (2 * n_points) * torch.pi)
                    pts = (c_pts + 1.0) / 2.0 * (self.b - self.a) + self.a
                    pts = pts.view(-1, 1)
                elif distribution == 'beta':
                    m = torch.distributions.beta.Beta(1.0, float(beta_penalty))
                    b_pts = m.sample((n_points,)).to(self.device)
                    b_pts, _ = torch.sort(b_pts)
                    pts = ((b_pts + 1.0) / 2.0) * (self.b - self.a) + self.a
                    pts = pts.view(-1, 1)
                elif distribution == 'lhs':
                    from scipy.stats import qmc
                    sampler = qmc.LatinHypercube(d=1)
                    sample = sampler.random(n=n_points)
                    pts = torch.tensor(sample * (self.b - self.a) + self.a, dtype=torch.float32, device=self.device)
                elif distribution == 'grid':
                    pts = torch.linspace(self.a, self.b, n_points, device=self.device).view(-1, 1)
                else:
                    pts = (self.b - self.a) * torch.rand(n_points, 1, device=self.device) + self.a

            pts.requires_grad_(True)
            return pts

        # [CRITICAL FIX]: L-BFGS maintains a Hessian approximation history. 
        # If we resample points between epochs, the landscape changes and the history is poisoned!
        # Therefore, L-BFGS MUST use a static dataset generated once.
        # Adam/AdamW don't have this issue and benefit from dynamic anti-aliasing resampling.
        if is_lbfgs:
            _print_info("L-BFGS optimizer detected: Using STATIC collocation points to preserve Hessian history.")
            x = generate_collocation_points(self.batch_size_lbfgs)
        else:
            _print_info("Adam/AdamW optimizer detected: Using DYNAMIC collocation resampling every epoch.")
            if is_hybrid:
                _print_info(f"Hybrid mode enabled: Will switch to L-BFGS at epoch {self.epochs_adam + 1}.")

        import sys
        import time
        from tqdm.auto import tqdm
        
        if trial is not None and total_trials is not None:
            display_trial_num = trial.number + 1 if trial is not None else (current_trial_count if current_trial_count is not None else 1)
            desc = f"[Trial {display_trial_num}/{total_trials}]"
        else:
            desc = "Train"
            
        pbar = tqdm(range(start_epoch, epochs + 1), desc=desc, file=sys.stdout, ascii=True, mininterval=1.0, leave=False)
        
        lbfgs_start_time = time.time() if is_lbfgs else None
        
        for epoch in pbar:
            # --- Warm Start: Fine-Tuning Phase (Unfreeze latent basis) ---
            unfreeze_epoch = self.config['training'].get('warm_start_unfreeze_epoch')
            is_in_warmup = getattr(self.model, 'is_in_warmup_phase', False)

            if epoch == 1 and is_in_warmup and not unfreeze_epoch:
                if trial is None: pbar.write(f"{Colors.BLUE}\n--- TLP WARM-START: Model will remain in Phase 1 (frozen prior) for all epochs ('warm_start_unfreeze_epoch' is not set). ---{Colors.RESET}")

            # Check if it's time to transition and if the model is in Phase 1
            if unfreeze_epoch and epoch == unfreeze_epoch and is_in_warmup:
                if trial is None: pbar.write(f"{Colors.BLUE}\n--- TLP WARM-START: TRANSITION TO PHASE 2 (Epoch {epoch}) ---{Colors.RESET}")

                if hasattr(self.model, 'transition_to_free_basis'):
                    self.model.transition_to_free_basis()

            if is_hybrid and not is_lbfgs and (epoch == self.epochs_adam + 1 or force_lbfgs_switch):
                is_lbfgs = True
                self.optimizer = self.optimizer_lbfgs
                force_lbfgs_switch = False
                epochs_without_improvement = 0
                x = generate_collocation_points(self.batch_size_lbfgs)
                lbfgs_start_time = time.time()
                
            if is_lbfgs and lbfgs_start_time is not None:
                if time.time() - lbfgs_start_time > 15 * 60:  # 15 minutes timeout
                    if trial is None: pbar.write(f"{Colors.RED}\n[Timeout] L-BFGS exceeded 15 minutes time limit. Stopping training at epoch {epoch}.{Colors.RESET}")
                    break
                
            if not is_lbfgs:
                # DYNAMIC SAMPLING for Adam/AdamW using target distribution
                x = generate_collocation_points(self.batch_size_adam)
            
            def closure():
                self.optimizer.zero_grad()
                
                # --- Loss Calculation ---
                loss_pde = self.pde_loss(self.model, x)
                
                active_losses = [loss_pde]
                loss_ic_val = torch.tensor(0.0, device=self.device)
                loss_bc_val = torch.tensor(0.0, device=self.device)
                
                if has_boundary_conditions:
                    if is_lbfgs:
                        if self.x_ic_static is None and self.x_bc_static is None:
                            b_dict = self.problem.boundary_points(num_pts=self.num_bc_lbfgs, device=self.device)
                            self.x_ic_static = b_dict.get('ic')
                            self.x_bc_static = b_dict.get('bc')
                            with torch.no_grad():
                                if self.x_ic_static is not None:
                                    self.u_exact_ic_static = self.problem.exact_solution(self.x_ic_static)
                                if self.x_bc_static is not None:
                                    self.u_exact_bc_static = self.problem.exact_solution(self.x_bc_static)
                        
                        x_ic_batch = self.x_ic_static
                        u_exact_ic_batch = getattr(self, 'u_exact_ic_static', None)
                        x_bc_batch = self.x_bc_static
                        u_exact_bc_batch = getattr(self, 'u_exact_bc_static', None)
                    else:
                        b_dict = self.problem.boundary_points(num_pts=self.num_bc_adam, device=self.device)
                        x_ic_batch = b_dict.get('ic')
                        x_bc_batch = b_dict.get('bc')
                        with torch.no_grad():
                            u_exact_ic_batch = self.problem.exact_solution(x_ic_batch) if x_ic_batch is not None else None
                            u_exact_bc_batch = self.problem.exact_solution(x_bc_batch) if x_bc_batch is not None else None
                            
                    if x_ic_batch is not None and x_ic_batch.numel() > 0:
                        u_pred_ic = self.model(x_ic_batch)
                        loss_ic_val = torch.nn.functional.mse_loss(u_pred_ic, u_exact_ic_batch)
                        active_losses.append(loss_ic_val)
                        
                    if x_bc_batch is not None and x_bc_batch.numel() > 0:
                        u_pred_bc = self.model(x_bc_batch)
                        loss_bc_val = torch.nn.functional.mse_loss(u_pred_bc, u_exact_bc_batch)
                        active_losses.append(loss_bc_val)
                
                if self.use_relobralo:
                    if self.relobralo is None:
                        self.relobralo = ReLoBRaLo(num_losses=len(active_losses), device=self.device)
                        self.relobralo.set_l0(active_losses)
                        
                    if not is_lbfgs:
                        loss = self.relobralo(active_losses)
                    else:
                        lambs = [self.relobralo.lam[f"lam_{i}"] for i in range(self.relobralo.num_losses)]
                        loss = sum(lam * l for lam, l in zip(lambs, active_losses))
                else:
                    loss = sum(active_losses)
                    
                loss.backward()
                
                self.last_loss_pde = loss_pde
                self.last_loss_ic = loss_ic_val
                self.last_loss_bc = loss_bc_val

                return loss
            
            # Step the optimizer
            if isinstance(self.optimizer, torch.optim.LBFGS):
                loss = self.optimizer.step(closure)
            else:
                loss = closure()
                self.optimizer.step()

            loss_pde = getattr(self, 'last_loss_pde', loss)
            loss_ic = getattr(self, 'last_loss_ic', torch.tensor(0.0, device=self.device))
            loss_bc = getattr(self, 'last_loss_bc', torch.tensor(0.0, device=self.device))
                
            if self.scheduler is not None and not is_lbfgs:
                self.scheduler.step()
            
            # Use the consistent loss values returned from the closure
            loss_val, loss_pde_val, loss_ic_val, loss_bc_val = loss.item(), loss_pde.item(), loss_ic.item(), loss_bc.item()
            
            if epoch % 10 == 0 or epoch == epochs:
                self.writer.add_scalar('Loss/Train_Total', loss_val, epoch)
                self.writer.add_scalar('Loss/Train_PDE', loss_pde_val, epoch)
                if has_boundary_conditions:
                    if loss_ic_val > 0.0 or getattr(self, 'x_ic_static', None) is not None:
                        self.writer.add_scalar('Loss/Train_IC', loss_ic_val, epoch)
                    if loss_bc_val > 0.0 or getattr(self, 'x_bc_static', None) is not None:
                        self.writer.add_scalar('Loss/Train_BC', loss_bc_val, epoch)
                        
                if self.use_relobralo and self.relobralo is not None:
                    for i in range(self.relobralo.num_losses):
                        self.writer.add_scalar(f'ReLoBRaLo/lam_{i}', self.relobralo.lam[f"lam_{i}"].item(), epoch)
                
                with torch.no_grad():
                    if self.is_2d:
                        domain_x, domain_y = self.domain
                        x_eval_lin = torch.linspace(domain_x[0], domain_x[1], 100, device=self.device)
                        y_eval_lin = torch.linspace(domain_y[0], domain_y[1], 100, device=self.device)
                        gx, gy = torch.meshgrid(x_eval_lin, y_eval_lin, indexing='ij')
                        x_eval = torch.stack([gx, gy], dim=-1).reshape(-1, 2)
                    else:
                        x_eval = torch.linspace(self.a, self.b, 1000, device=self.device).view(-1, 1)
                        
                    u_pred = self.model(x_eval)
                    u_exact = self.problem.exact_solution(x_eval)
                    mae = torch.mean(torch.abs(u_pred - u_exact)).item()
                    self.writer.add_scalar('Metrics/MAE_Exact', mae, epoch)

                    if self.wandb_run:
                        log_dict = {
                            'Loss/Train_Total': loss_val,
                            'Loss/Train_PDE': loss_pde_val,
                            'Metrics/MAE_Exact': mae,
                            'epoch': epoch
                        }
                        if self.scheduler:
                            log_dict['lr'] = self.scheduler.get_last_lr()[0]
                        if has_boundary_conditions:
                            if loss_ic_val > 0.0 or getattr(self, 'x_ic_static', None) is not None:
                                log_dict['Loss/Train_IC'] = loss_ic_val
                            if loss_bc_val > 0.0 or getattr(self, 'x_bc_static', None) is not None:
                                log_dict['Loss/Train_BC'] = loss_bc_val
                        if self.use_relobralo and self.relobralo is not None:
                            for i in range(self.relobralo.num_losses):
                                log_dict[f'ReLoBRaLo/lam_{i}'] = self.relobralo.lam[f"lam_{i}"].item()
                        self.wandb_run.log(log_dict)
                
                # Flush the writer periodically to ensure data is written to disk immediately
                if epoch % 150 == 0:
                    self.writer.flush()
                
                pbar.set_postfix({'Total': f"{loss_val:.2e}", 'MAE': f"{mae:.2e}"})
                
                # Optuna Pruning
                if trial is not None:
                    import optuna
                    trial.report(mae, epoch)
                    
                    # 1. Absolute safety-net pruning (prevents "bad bootstrap" of relative pruners)
                    if epoch >= 1000 and mae >= 0.1:
                        self.writer.close()
                        raise optuna.TrialPruned("Absolute threshold: MAE >= 0.1 at epoch >= 1000")
                    if epoch >= 2000 and mae >= 0.05:
                        self.writer.close()
                        raise optuna.TrialPruned("Absolute threshold: MAE >= 0.05 at epoch >= 2000")
                    
                    # 2. Relative Hyperband pruning
                    if trial.should_prune():
                        self.writer.close()
                        raise optuna.TrialPruned("Pruned by Hyperband algorithm")
                
                # Early Stopping Logic (Based on MAE)
                is_best = (best_mae - mae) > min_delta
                if is_best:
                    best_mae = mae
                    epochs_without_improvement = 0
                else:
                    epochs_without_improvement += 10
                    
                checkpoint = {
                    'epoch': epoch,
                    'state_dict': self.model.state_dict(),
                    'optimizer': self.optimizer.state_dict(),
                    'best_mae': best_mae,
                    'config': self.config
                }
                
                if hasattr(self.model, 'base_mlp_config') and self.model.base_mlp_config is not None:
                    checkpoint['config_mlp'] = self.model.base_mlp_config

                if is_best:
                    # The directory is created at the start of train(), so it should exist.
                    torch.save(checkpoint, str(save_path_best))
                    
                if epochs_without_improvement >= patience:
                    if is_hybrid and not is_lbfgs:
                        if trial is None: pbar.write(f"{Colors.RED}\n[Early Stopping] Adam stagnated for {patience} epochs. Forcing early switch to L-BFGS.{Colors.RESET}")
                        force_lbfgs_switch = True
                        epochs_without_improvement = 0
                    else:
                        if trial is None: pbar.write(f"{Colors.RED}\n[Early Stopping] Training terminated at epoch {epoch} (No MAE improvement for {patience} epochs).{Colors.RESET}")
                        break
                        
            # Save latest model only at the very end of training
            if epoch == epochs:
                checkpoint_latest = {
                    'epoch': epoch,
                    'state_dict': self.model.state_dict(),
                    'optimizer': self.optimizer.state_dict(),
                    'best_mae': best_mae,
                    'config': self.config
                }
                if hasattr(self.model, 'base_mlp_config') and self.model.base_mlp_config is not None:
                    checkpoint_latest['config_mlp'] = self.model.base_mlp_config

                # The directory is created at the start of train(), so it should exist.
                torch.save(checkpoint_latest, str(save_path_latest))
                
        self.writer.close()

        _print_success(f"Training complete. Best Weights saved to {save_path_best}")
        if self.wandb_run:
            self.wandb_run.summary['best_mae'] = best_mae
        return best_mae
