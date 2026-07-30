from pathlib import Path
import torch
import sys

# Ensure CrossPINN modules are importable
current_dir = Path(__file__).resolve().parent
root_dir = current_dir.parent
if str(root_dir) not in sys.path:
    sys.path.append(str(root_dir))
if str(current_dir) not in sys.path:
    sys.path.append(str(current_dir))

from config.path_manager import PathManager
from utils.logger import print_success, print_info, print_error
from models.factory import build_model
from tool_utils import load_project_environment, get_device

load_project_environment()

def run_evaluate(study_or_target, device=None, force=False):
    import json
    if device is None:
        device = get_device()
    from config.study import Study
    from config.config import load_config

    if isinstance(study_or_target, Study):
        study_obj = study_or_target
    elif isinstance(study_or_target, PathManager):
        study_obj = Study(prob_id=study_or_target.prob_id, model=study_or_target.model, study_name=study_or_target.study_name, debug=study_or_target.debug)
    else:
        study_obj = Study.from_args(study_or_target)

    pm = study_obj.pm
    config = load_config(study_obj, quiet=True)
    model_type = config['model']['type']
    exp_name = config.get('experiment', {}).get('name', pm.study_base_name)

    model_path = pm.get_weight_checkpoint_path(checkpoint_type="best")
    if not model_path.exists():
        latest_path = pm.get_weight_checkpoint_path(checkpoint_type="latest")
        model_path = latest_path if latest_path.exists() else model_path

    eval_metrics_file = pm.logs_dir / exp_name / "eval_metrics.json"

    # Check if cached metrics exist and are up to date with checkpoint
    if eval_metrics_file.exists() and not force:
        try:
            with open(eval_metrics_file, 'r') as f:
                cached = json.load(f)
            checkpoint_mtime = model_path.stat().st_mtime if model_path.exists() else 0
            if cached.get('mtime', 0) >= checkpoint_mtime:
                print_info(f"Loaded cached evaluation metrics for '{exp_name}' from {eval_metrics_file.name}")
                return cached
        except Exception:
            pass

    torch.manual_seed(config['experiment']['seed'])
    
    prob_id = config['data']['problem_id']
    a, b = config['data']['domain']
    is_2d = isinstance(a, list)
    
    from equations.registry import get_problem
    problem = get_problem(prob_id, a=a, b=b)
    
    if not model_path.exists():
        raise FileNotFoundError(f"Weights not found at {model_path}. Please run training first.")
        
    checkpoint = torch.load(model_path, map_location=device, weights_only=False)
    
    kwargs_for_builder = {}

    # --- Checkpoint Analysis & Robust Scaffold Building ---
    if isinstance(checkpoint, dict) and 'config' in checkpoint:
        config = checkpoint['config']
        model_type = config['model']['type']

        print_info("[Evaluate] Checkpoint loaded. Analyzing contents...")
        print_info(f"  - Checkpoint keys: {list(checkpoint.keys())}")
        model_cfg_in_ckpt = checkpoint['config'].get('model', {})
        print_info(f"    - model.type: {model_cfg_in_ckpt.get('type')}")
        print_info(f"    - model.global_skip: {model_cfg_in_ckpt.get('global_skip')}")

        if 'config_mlp' in checkpoint:
            print_success("  - 'config_mlp' key found in checkpoint.")
        else:
            print_error("  - 'config_mlp' key NOT found in checkpoint.")

        state_dict_to_check = checkpoint.get('state_dict', checkpoint)
        base_mlp_keys_in_sd = [k for k in state_dict_to_check.keys() if k.startswith('base_mlp.')]
        if base_mlp_keys_in_sd:
            print_success(f"  - Found {len(base_mlp_keys_in_sd)} 'base_mlp' keys in the state_dict.")
        else:
            print_error("  - NO 'base_mlp' keys found in the state_dict.")
        print_info("-------------------------------------------------")

        if model_type.lower() == 'tlp' and config.get('model', {}).get('global_skip'):
            if 'config_mlp' in checkpoint:
                print_info("[Evaluate] Pre-building base_mlp from 'config_mlp' found in checkpoint.")
                from models.mlp import build_model as build_mlp_model
                config_mlp = checkpoint['config_mlp']
                
                prob_id_eval = config['data']['problem_id']
                a_eval, b_eval = config['data']['domain']
                from equations.registry import get_problem as get_problem_eval
                problem_eval = get_problem_eval(prob_id_eval, a=a_eval, b=b_eval)

                if 'data' not in config_mlp:
                    config_mlp['data'] = config['data']
                    
                base_mlp_scaffold = build_mlp_model(config_mlp, problem_eval, device, quiet=True)
                kwargs_for_builder['base_mlp_scaffold'] = base_mlp_scaffold
                kwargs_for_builder['mlp_config'] = config_mlp
            else:
                print_error("[Evaluate] CRITICAL: TLP uses global_skip, but 'config_mlp' was NOT found in checkpoint.")
                print_error("           -> This will likely cause a size mismatch error.")


        # Disable warm_start during evaluation to prevent loading an external, mismatched MLP.
        if 'training' in config and 'warm_start' in config['training']:
            config['training']['warm_start'] = False

    model = build_model(
        config=config,
        problem=problem,
        device=device,
        debug=study_obj.debug,
        quiet=True,
        **kwargs_for_builder
    )
    
    if isinstance(checkpoint, dict) and 'state_dict' in checkpoint:
        model.load_state_dict(checkpoint['state_dict'], strict=True)
    else:
        model.load_state_dict(checkpoint, strict=True)
    model.eval()
    
    print_info(f"Evaluating {model_type} on Problem {prob_id}")
    
    # --- Compute Physics Losses (Loss_PDE, Loss_BC, Loss_IC) ---
    from losses.pde_loss import VolterraPDELoss
    num_quad = config['data'].get('num_quadrature_nodes', 10 if is_2d else 50)
    pde_loss_fn = VolterraPDELoss(problem, num_quadrature_nodes=num_quad, device=device)

    if not is_2d:
        x_colloc = torch.linspace(a, b, 500, device=device).view(-1, 1)
    else:
        gx = torch.linspace(float(a[0]), float(b[0]), 20, device=device)
        gy = torch.linspace(float(a[1]), float(b[1]), 20, device=device)
        grid_x, grid_y = torch.meshgrid(gx, gy, indexing='ij')
        x_colloc = torch.cat([grid_x.reshape(-1, 1), grid_y.reshape(-1, 1)], dim=1)

    x_colloc.requires_grad_(True)
    loss_pde = pde_loss_fn(model, x_colloc).item()

    loss_ic = 0.0
    loss_bc = 0.0
    if hasattr(problem, 'boundary_points'):
        b_dict = problem.boundary_points(num_pts=100, device=device)
        if 'ic' in b_dict and b_dict['ic'] is not None and b_dict['ic'].numel() > 0:
            x_ic = b_dict['ic']
            u_exact_ic = problem.exact_solution(x_ic)
            with torch.no_grad():
                u_pred_ic = model(x_ic)
            loss_ic = torch.mean((u_pred_ic - u_exact_ic)**2).item()
        if 'bc' in b_dict and b_dict['bc'] is not None and b_dict['bc'].numel() > 0:
            x_bc = b_dict['bc']
            u_exact_bc = problem.exact_solution(x_bc)
            with torch.no_grad():
                u_pred_bc = model(x_bc)
            loss_bc = torch.mean((u_pred_bc - u_exact_bc)**2).item()

    print_info(f"Loss PDE: {loss_pde:.6e}")
    if hasattr(problem, 'boundary_points'):
        print_info(f"Loss BC:  {loss_bc:.6e}")
        print_info(f"Loss IC:  {loss_ic:.6e}")

    # --- Compute MAE and L2 error ---
    with torch.no_grad():
        if not is_2d:
            x_test = torch.linspace(a, b, 500, device=device).unsqueeze(1)
            u_exact = problem.exact_solution(x_test)
            u_pred = model(x_test)
            err = torch.abs(u_pred - u_exact)
            mae = torch.mean(err).item()
            l2_err = torch.sqrt(torch.mean(err**2)).item()
        else:
            # 2D Problem Evaluation
            grid_size = 100
            x_line = torch.linspace(a[0], a[1], grid_size, device=device)
            y_line = torch.linspace(b[0], b[1], grid_size, device=device)
            grid_x, grid_y = torch.meshgrid(x_line, y_line, indexing='ij')
            xy_test = torch.stack([grid_x.flatten(), grid_y.flatten()], dim=1)

            u_exact_2d = problem.exact_solution(xy_test)
            u_pred_2d = model(xy_test)
            err_2d = torch.abs(u_pred_2d - u_exact_2d)
            mae = torch.mean(err_2d).item()
            l2_err = torch.sqrt(torch.mean(err_2d**2)).item()

    print_info(f"\nMean Absolute Error (MAE): {mae:.6e}")
    print_info(f"L2 Error: {l2_err:.6e}")

    eval_results = {
        'mae': mae,
        'l2': l2_err,
        'loss_pde': loss_pde,
        'loss_bc': loss_bc,
        'loss_ic': loss_ic,
        'plot_paths': {}, # Plots are generated by study.visualize()
        'mtime': model_path.stat().st_mtime if model_path.exists() else 0
    }

    eval_metrics_file.parent.mkdir(parents=True, exist_ok=True)
    try:
        with open(eval_metrics_file, 'w') as f:
            json.dump(eval_results, f, indent=2)
        print_success(f"Saved evaluation metrics to {eval_metrics_file}")
    except Exception as e:
        print_info(f"Could not save evaluation metrics: {e}")

    return eval_results

def main():
    from config.config import get_cli_parser
    parser = get_cli_parser(description="Evaluation script for PINN models.")
    args, overrides = parser.parse_known_args()

    from config.study import Study
    study = Study.from_args(args, overrides=overrides)
    study.evaluate()

if __name__ == '__main__':
    main()
