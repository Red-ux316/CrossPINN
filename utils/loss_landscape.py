import argparse
import torch
import numpy as np
import matplotlib.pyplot as plt
import csv
import os
import sys
from pathlib import Path

# Ensure CrossPINN root is in sys.path
current_dir = Path(__file__).resolve().parent
root_dir = current_dir.parent
if str(root_dir) not in sys.path:
    sys.path.append(str(root_dir))

from config.config import load_config
from equations.registry import get_problem
from losses.pde_loss import VolterraPDELoss
from models.factory import build_model
from utils.logger import print_info, print_success, print_error

def main():
    from config.config import add_config_arguments
    from config.path_manager import PathManager
    parser = argparse.ArgumentParser(description="1D Loss Landscape Interpolation for PINN")
    add_config_arguments(parser)
    parser.add_argument('--model_final', type=str, required=False, default=None, help="Path to best/final model weights (.pth o .pt)")
    parser.add_argument('--model_init', type=str, default=None, help="Path to initial model weights (optional, uses random init if missing)")
    parser.add_argument('--alpha_min', type=float, default=-0.5, help="Minimum alpha value")
    parser.add_argument('--alpha_max', type=float, default=1.5, help="Maximum alpha value")
    parser.add_argument('--steps', type=int, default=100, help="Number of interpolation steps")
    parser.add_argument('--num_points', type=int, default=2000, help="Number of spatial collocation points")
    args = parser.parse_args()
    
    pm = PathManager.from_args(args)
    config = load_config(pm)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print_info(f"Using device: {device}")
    
    if args.model_final is None:
        best_ckpt = pm.get_weight_checkpoint_path(checkpoint_type="best")
        args.model_final = str(best_ckpt)
    
    torch.manual_seed(42)
    np.random.seed(42)

    prob_id = config['data']['problem_id']
    a, b = config['data']['domain']
    
    print_info(f"Initializing Problem {prob_id}")
    problem = get_problem(prob_id, a=a, b=b)
    
    model_type = config['model']['type']
    print_info(f"Initializing {model_type}")
    try:
        model = build_model(config, problem, device)
    except ValueError as e:
        print_error(f"Error: {e}")
        sys.exit(1)

    w_init = torch.nn.utils.parameters_to_vector(model.parameters()).detach().clone()
    
    if args.model_init and os.path.exists(args.model_init):
        print_info(f"Loading W_init from {args.model_init}")
        model.load_state_dict(torch.load(args.model_init, map_location=device, weights_only=True))
        w_init = torch.nn.utils.parameters_to_vector(model.parameters()).detach().clone()
    else:
        print_info("Using current random initialization as W_init")

    print_info(f"Loading W_final from {args.model_final}")
    model.load_state_dict(torch.load(args.model_final, map_location=device, weights_only=True))
    w_final = torch.nn.utils.parameters_to_vector(model.parameters()).detach().clone()

    print_info(f"Sampling {args.num_points} collocation points")
    domain = config['data']['domain']
    is_2d = isinstance(domain[0], (list, tuple)) and len(domain) == 2
    if is_2d:
        domain_x, domain_y = domain
        try:
            from pyDOE import lhs
        except ImportError:
            print_error("pyDOE is required for 2D sampling. Install with `pip install pyDOE`.")
            sys.exit(1)
        sample = lhs(2, args.num_points)
        x_lhs = sample[:, 0] * (domain_x[1] - domain_x[0]) + domain_x[0]
        y_lhs = sample[:, 1] * (domain_y[1] - domain_y[0]) + domain_y[0]
        x_static = torch.tensor(np.column_stack((x_lhs, y_lhs)), dtype=torch.float32, device=device)
    else:
        x_static = torch.rand((args.num_points, 1), dtype=torch.float32, device=device) * (b - a) + a
    x_static.requires_grad_(True)
    
    num_bc = config['data'].get('num_boundary_points_adam', 100)
    b_dict = problem.boundary_points(num_pts=num_bc, device=device)
    x_ic = b_dict.get('ic')
    x_bc = b_dict.get('bc')
    
    pde_loss_fn = VolterraPDELoss(problem=problem, num_quadrature_nodes=config['data']['num_quadrature_nodes'], device=device)

    alpha_values = np.linspace(args.alpha_min, args.alpha_max, args.steps)
    pde_losses = []
    bc_losses = []
    total_losses = []

    print_info(f"Starting 1D interpolation ({args.steps} steps from {args.alpha_min} to {args.alpha_max})...")
    
    for idx, alpha in enumerate(alpha_values):
        w_alpha = (1.0 - alpha) * w_init + alpha * w_final
        torch.nn.utils.vector_to_parameters(w_alpha, model.parameters())
        
        model.eval()
        loss_pde_val = pde_loss_fn(model, x_static).item()
        loss_ic_val = 0.0
        loss_bc_val = 0.0
        
        with torch.no_grad():
            if x_ic is not None and x_ic.numel() > 0:
                loss_ic_val = torch.nn.functional.mse_loss(model(x_ic), problem.exact_solution(x_ic)).item()
            if x_bc is not None and x_bc.numel() > 0:
                loss_bc_val = torch.nn.functional.mse_loss(model(x_bc), problem.exact_solution(x_bc)).item()
                
        loss_bc_total = loss_ic_val + loss_bc_val
        loss_total = loss_pde_val + loss_bc_total
        
        pde_losses.append(loss_pde_val)
        bc_losses.append(loss_bc_total)
        total_losses.append(loss_total)
        
        if (idx + 1) % 10 == 0 or idx == 0:
            print(f"Step {idx+1}/{args.steps} | Alpha: {alpha:.3f} | Total Loss: {loss_total:.4e}")

    csv_path = pm.plots_dir / f"{config['experiment']['name']}_loss_landscape.csv"
    print_info(f"Saving raw data to {csv_path}")
    with open(csv_path, mode='w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['alpha', 'total_loss', 'pde_loss', 'bc_loss'])
        for a, t, p, b_l in zip(alpha_values, total_losses, pde_losses, bc_losses):
            writer.writerow([a, t, p, b_l])
            
    plot_path = pm.plots_dir / f"{config['experiment']['name']}_loss_landscape_1d.png"
    print_info(f"Saving plot to {plot_path}")
    
    plt.figure(figsize=(10, 6))
    plt.plot(alpha_values, total_losses, label='Total Loss (L_PDE + L_BC)', color='black', linewidth=2)
    plt.plot(alpha_values, pde_losses, label='L_PDE', color='blue', linestyle='--')
    if any(b > 0 for b in bc_losses):
        plt.plot(alpha_values, bc_losses, label='L_BC/IC', color='red', linestyle='-.')
        
    plt.axvline(x=0.0, color='gray', linestyle=':', label='W_init (Alpha=0)')
    plt.axvline(x=1.0, color='green', linestyle=':', label='W_final (Alpha=1)')
    
    plt.yscale('log')
    plt.xlabel('Alpha (Interpolation Coefficient)')
    plt.ylabel('Loss Value (Log Scale)')
    plt.title(f'1D Loss Landscape Interpolation ({model_type})')
    plt.legend()
    plt.grid(True, which="both", ls="-", alpha=0.2)
    
    plt.tight_layout()
    plt.savefig(plot_path, dpi=300)
    print_success("Done!")

if __name__ == "__main__":
    main()
