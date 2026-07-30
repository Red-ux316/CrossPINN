import sys
import torch
import numpy as np
from pathlib import Path
import yaml
import matplotlib.pyplot as plt
import argparse

# Ensure tools and CrossPINN root directories are importable
current_dir = Path(__file__).resolve().parent
root_dir = current_dir.parent.parent
tools_dir = current_dir.parent
if str(root_dir) not in sys.path:
    sys.path.append(str(root_dir))
if str(tools_dir) not in sys.path:
    sys.path.append(str(tools_dir))

from config.study import Study
from config.path_manager import PathManager
from tool_utils import get_device
from equations.registry import get_problem
from models.factory import build_model
from utils.logger import print_info, print_success, print_error
from tools.visualize.plot_style import create_figure, format_plot, save_and_close_figure, MODEL_COLORS

def run_visualize(study_or_target, quiet=False):
    """
    Generates detailed visualization plots for a single study:
    - Solution profile u_pred(x) vs u_exact(x)
    - Absolute error profile |u_pred - u_exact|
    - Training loss history curves from TensorBoard logs (if available)

    Args:
        study_or_target (Study | PathManager | dict | Namespace): Target study to visualize.
        quiet (bool): Suppress informational messages if True.

    Returns:
        dict: Paths to generated plots and evaluation metrics.
    """
    if isinstance(study_or_target, Study):
        study = study_or_target
    else:
        study = Study.from_args(study_or_target)

    config = study.load_config(quiet=quiet)
    pm = study.pm
    study_name = pm.study_base_name # Define study_name here

    device = get_device()

    prob_id = config['data']['problem_id']
    a, b = config['data']['domain']
    is_2d = isinstance(a, list)

    problem = get_problem(prob_id, a=a, b=b)
    ckpt_path = pm.get_weight_checkpoint_path(checkpoint_type="best")

    if not ckpt_path.exists():
        print_error(f"Cannot visualize study '{pm.study_name}': Weight checkpoint not found at {ckpt_path}")
        return None

    if not quiet:
        print_info(f"\n[Visualize] Loading trained model from: {ckpt_path.name}")

    # Load the full checkpoint dictionary. It contains metadata alongside the model weights.
    # weights_only must be False to correctly unpickle the dictionary structure.
    checkpoint = torch.load(ckpt_path, map_location=device, weights_only=False)

    # If checkpoint contains config, use it as the source of truth for model structure
    # to prevent architecture mismatches.
    if isinstance(checkpoint, dict) and 'config' in checkpoint:
        config = checkpoint['config']
        if not quiet:
            print_info("Using configuration from checkpoint to build the model.")

    # Extract the state_dict from the checkpoint dictionary.
    if isinstance(checkpoint, dict) and 'state_dict' in checkpoint:
        state_dict = checkpoint['state_dict']
    else:
        state_dict = checkpoint  # Fallback for older checkpoints that are just the state_dict

    model = build_model(config, problem, device, quiet=quiet)
    model.load_state_dict(state_dict)
    model.eval()

    plots_dir = pm.plots_dir / study_name # Save plots for a single study in its own folder
    plots_dir.mkdir(parents=True, exist_ok=True)

    results = {
        'metrics': {},
        'plot_paths': {}
    }

    with torch.no_grad():
        if not is_2d:
            x_test = torch.linspace(a, b, 500, device=device).unsqueeze(1)
            u_exact = problem.exact_solution(x_test)
            u_pred = model(x_test)

            x_np = x_test.cpu().numpy().flatten()
            u_exact_np = u_exact.cpu().numpy()
            u_pred_np = u_pred.cpu().numpy()
            
            output_dim = u_exact_np.shape[1] if u_exact_np.ndim > 1 else 1

            err_np = np.abs(u_pred_np - u_exact_np)
            mae = float(np.mean(err_np))
            l2 = float(np.sqrt(np.mean((u_pred_np - u_exact_np) ** 2)))

            model_type = config["model"]["type"]
            model_color = MODEL_COLORS.get(model_type, '#1f77b4')

            # 1. Solution Profile Plot
            fig, ax = create_figure(figsize=(9, 5))
            if output_dim == 1:
                ax.plot(x_np, u_exact_np.flatten(), label='Exact Solution', color=MODEL_COLORS.get('Exact', 'black'), linestyle='--', linewidth=2)
                ax.plot(x_np, u_pred_np.flatten(), label=f'PINN Prediction ({model_type})', color=model_color, alpha=0.85)
            else: # Multi-output 1D problem
                for i in range(output_dim):
                    ax.plot(x_np, u_exact_np[:, i], color=f'C{i}', linestyle='--', linewidth=2, label=f'Exact u{i+1}')
                    ax.plot(x_np, u_pred_np[:, i], color=f'C{i}', alpha=0.85, label=f'Pred u{i+1} ({model_type})')
            format_plot(ax, title=f"{study_name} - Solution Profile (MAE: {mae:.2e})", xlabel='x', ylabel='u(x)')
            sol_path = plots_dir / f"{study_name}_solution.png"
            save_and_close_figure(fig, sol_path)
            results['plot_paths']['solution'] = str(sol_path)

            # 2. Absolute Error Plot
            fig_err, ax_err = create_figure(figsize=(9, 5))
            if output_dim == 1:
                ax_err.plot(x_np, err_np.flatten(), label='Absolute Error |u_pred - u_exact|', color='#d62728', linewidth=1.5)
            else:
                for i in range(output_dim):
                    ax_err.plot(x_np, err_np[:, i], label=f'Error u{i+1}', color=f'C{i}', linewidth=1.5)
            format_plot(ax_err, title=f"{study_name} - Pointwise Absolute Error", xlabel='x', ylabel='Error (Log Scale)', yscale='log')
            err_path = plots_dir / f"{study_name}_error.png"
            save_and_close_figure(fig_err, err_path)
            results['plot_paths']['error'] = str(err_path)

            results['metrics']['mae'] = mae
            results['metrics']['l2'] = l2

        else:
            # 2D Problem Evaluation
            grid_size = 100
            x_line = torch.linspace(a[0], a[1], grid_size, device=device)
            y_line = torch.linspace(b[0], b[1], grid_size, device=device)
            grid_x, grid_y = torch.meshgrid(x_line, y_line, indexing='ij')
            xy_test = torch.stack([grid_x.flatten(), grid_y.flatten()], dim=1)

            u_exact_2d = problem.exact_solution(xy_test).reshape(grid_size, grid_size).cpu().numpy()
            u_pred_2d = model(xy_test).reshape(grid_size, grid_size).cpu().numpy()
            err_2d = np.abs(u_pred_2d - u_exact_2d)

            mae = float(np.mean(err_2d))
            l2 = float(np.sqrt(np.mean((u_pred_2d - u_exact_2d) ** 2)))

            # Create a 1x3 plot for Exact, Prediction, and Error
            fig, axes = create_figure(figsize=(15, 4.5), ncols=3)
            plot_extent = [a[0], a[1], b[0], b[1]]

            # Subplot 1: Exact Solution
            im1 = axes[0].imshow(u_exact_2d.T, extent=plot_extent, origin='lower', cmap='viridis')
            format_plot(axes[0], title="Exact Solution", xlabel='x', ylabel='y', grid=False, legend=False)
            fig.colorbar(im1, ax=axes[0], fraction=0.046, pad=0.04)

            # Subplot 2: PINN Prediction
            im2 = axes[1].imshow(u_pred_2d.T, extent=plot_extent, origin='lower', cmap='viridis')
            format_plot(axes[1], title=f"Prediction ({config['model']['type']})", xlabel='x', ylabel=None, grid=False, legend=False)
            fig.colorbar(im2, ax=axes[1], fraction=0.046, pad=0.04)

            # Subplot 3: Absolute Error
            im3 = axes[2].imshow(err_2d.T, extent=plot_extent, origin='lower', cmap='inferno')
            format_plot(axes[2], title=f"Abs Error (MAE: {mae:.2e})", xlabel='x', ylabel=None, grid=False, legend=False)
            fig.colorbar(im3, ax=axes[2], fraction=0.046, pad=0.04)

            fig.suptitle(f"{study_name} - 2D Evaluation", fontsize=14, y=0.98)

            sol_path = plots_dir / f"{study_name}_2d_evaluation.png"
            save_and_close_figure(fig, sol_path)
            results['plot_paths']['solution_2d'] = str(sol_path)
            results['metrics']['mae'] = mae
            results['metrics']['l2'] = l2

    # 3. Optional TensorBoard Loss History Curves
    log_dir = pm.logs_dir / study_name
    if log_dir.exists():
        try:
            from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
            ea = EventAccumulator(str(log_dir))
            ea.Reload()

            tags = ea.Tags().get('scalars', [])
            loss_tags_present = any(tag.startswith('Loss/') for tag in tags)
            mae_tag_present = 'Metrics/MAE_Exact' in tags

            if loss_tags_present or mae_tag_present:
                fig, ax = create_figure(figsize=(9, 5))

                if 'Loss/Train_Total' in tags:
                    events = ea.Scalars('Loss/Train_Total')
                    ax.plot([e.step for e in events], [e.value for e in events], label='Train Loss (Total)', color='#1f77b4')
                
                if 'Loss/Train_PDE' in tags:
                    events = ea.Scalars('Loss/Train_PDE')
                    ax.plot([e.step for e in events], [e.value for e in events], label='PDE Loss', color='#ff7f0e', linestyle='--')

                if 'Metrics/MAE_Exact' in tags:
                    events = ea.Scalars('Metrics/MAE_Exact')
                    ax.plot([e.step for e in events], [e.value for e in events], label='MAE Exact', color='#2ca02c', linestyle=':')

                format_plot(ax, title=f"{study_name} - Training History", xlabel='Epoch', ylabel='Value (Log Scale)', yscale='log')

                loss_path = plots_dir / f"{study_name}_loss_curves.png"
                save_and_close_figure(fig, loss_path)
                results['plot_paths']['loss_curves'] = str(loss_path)
        except Exception as e:
            if not quiet:
                print_info(f"Note: Could not parse TensorBoard log curves ({e}). Skipping loss history plot.")

    if not quiet:
        print_success(f"[Visualize] Plots saved to: {plots_dir}")

    return results

def visualize_all(debug=False):
    """
    Iterates through all studies defined in studies.yaml and runs run_visualize for each.
    """
    print_info("\n[Visualize All] Starting batch visualization for all studies in studies.yaml...")
    pm = PathManager(debug=debug)
    studies_path = pm.studies_config_path

    if not studies_path.exists():
        print_error(f"Cannot run visualize_all: studies.yaml not found at {studies_path}")
        return

    with open(studies_path, 'r') as f:
        studies_config = yaml.safe_load(f) or {}

    total_studies = sum(len(v) for v in studies_config.values() if v)
    processed_count = 0

    for prob_key, study_names in studies_config.items():
        if not study_names:
            continue
        
        prob_id = int(prob_key.replace('prob', ''))
        print_info(f"\n--- Processing Problem: {prob_id} ---")

        for study_name in study_names:
            processed_count += 1
            print_info(f"[{processed_count}/{total_studies}] Visualizing study: {study_name}")
            
            try:
                # To get the model type, we must load the config from the checkpoint.
                # This respects the DRY principle and MLOps best practices.
                temp_pm = PathManager(prob_id=prob_id, study_name=study_name, debug=debug)
                ckpt_path = temp_pm.get_weight_checkpoint_path(checkpoint_type="best")
                if not ckpt_path.exists():
                    ckpt_path = temp_pm.get_weight_checkpoint_path(checkpoint_type="latest")
                
                if not ckpt_path.exists():
                    print_error(f"  -> Checkpoint not found for '{study_name}'. Skipping.")
                    continue

                checkpoint = torch.load(ckpt_path, map_location='cpu', weights_only=False)
                if not isinstance(checkpoint, dict) or 'config' not in checkpoint:
                    print_error(f"  -> Checkpoint for '{study_name}' is invalid or missing config. Skipping.")
                    continue
                
                model_type = checkpoint['config']['model']['type']

                study = Study(prob_id=prob_id, model=model_type, study_name=study_name, debug=debug)
                run_visualize(study, quiet=True)
                print_success(f"  -> Successfully generated plots for '{study_name}'.")
            except Exception as e:
                print_error(f"  -> An error occurred while visualizing '{study_name}': {e}")
    
    print_info("\n[Visualize All] Batch visualization complete.")

def main():
    parser = argparse.ArgumentParser(description="Visualization tool for PINN models. Use --all or specify a single study.")
    parser.add_argument('--all', action='store_true', help='Visualize all studies defined in studies.yaml.')
    parser.add_argument('--prob_id', type=str, help='Problem ID for a single study (e.g., 1, 34).')
    parser.add_argument('--model', '--method', dest='model', type=str, help='Model type for a single study (e.g., tlp, mlp).')
    parser.add_argument('--study_name', type=str, default=None, help='Optional study name for a single study.')
    parser.add_argument('--debug', action='store_true', help='Use debug directories (data_debug, plots_debug).')
    args, overrides = parser.parse_known_args()

    if args.all:
        visualize_all(debug=args.debug)
    elif args.prob_id and args.model:
        study = Study.from_args(args, overrides=overrides)
        run_visualize(study)
    else:
        parser.print_help()
        print_error("\nError: You must either specify --all, or provide both --prob_id and --model for a single study.")
        sys.exit(1)

if __name__ == "__main__":
    main()
