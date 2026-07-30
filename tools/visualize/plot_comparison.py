import argparse
import datetime
import sqlite3
import sys
import shutil
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from pathlib import Path
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
import torch
import yaml

# --- PATH SETUP ---
current_dir = Path(__file__).resolve().parent
root_dir = current_dir.parent.parent
tools_dir = current_dir.parent
if str(root_dir) not in sys.path:
    sys.path.append(str(root_dir))
if str(tools_dir) not in sys.path:
    sys.path.append(str(tools_dir))

from utils.logger import print_error, print_success, print_info
from config.path_manager import PathManager
from config.study import Study
from tool_utils import is_colab, get_data_root
from models.factory import build_model
from equations.registry import PROBLEMS_MAP, get_problem
from losses.pde_loss import VolterraPDELoss
from tools.visualize.plot_style import (
    apply_global_plot_style,
    MODEL_COLORS,
    create_figure,
    format_plot,
    save_and_close_figure
)

# --- MANDATORY BOILERPLATE ---
PathManager.load_project_environment()

PROBLEM_INFO = {
    'prob1': {'name': '1D Volterra', 'exact': r'$x + e^x$', 'integral': r'$u(x) = S(x) + \int_0^x (t-x) u(t) dt$'},
    'prob2': {'name': '1D Volterra (NL)', 'exact': r'$e^x$', 'integral': r'$u(x) = S(x) + \int_0^x u^2(t) dt$'},
    'prob3': {'name': '1D Volt-Fred', 'exact': r'$x + e^x$', 'integral': r'$u(x) = S(x) + \int_0^x (t-x) u(t) dt + \int_0^1 (t-x) u(t) dt$'},
    'prob4': {'name': '1D Volt-Fred (NL)', 'exact': r'$x e^x$', 'integral': r'$u(x) = S(x) + \int_0^x u(t) dt + \int_0^1 x u(t) dt$'},
    'prob5': {'name': '1D Abel', 'exact': r'$x$', 'integral': r'$u(x) = \int_0^x \frac{-1}{\sqrt{x-t}} u(t) dt$'},
    'prob6': {'name': '1D Abel (NL)', 'exact': r'$x$', 'integral': r'$u(x) = \int_0^x \frac{-1}{\sqrt{x-t}} u^3(t) dt$'},
    'prob8': {'name': '2D Volterra System', 'exact': r'$x+y$', 'integral': r'$u(x,y) = S(x,y) + \int_0^y \int_0^x e^{x+y+s+t} u(s,t) ds dt$'},
    'prob11': {'name': '2D Fredholm', 'exact': r'$xy$', 'integral': r'$u(x,y) = S(x,y) + \int_0^1 \int_0^1 (xs) u(s,t) ds dt$'},
    'prob12': {'name': '2D Volterra', 'exact': r'$x+y$', 'integral': r'$u(x,y) = S(x,y) + \int_0^y \int_0^x u(s,t) ds dt$'},
    'prob19': {'name': '1D Volterra IDE System', 'exact': r'$[1+x+x^2, 1-x-x^2]$', 'integral': r'$u_i^{\prime}(x) = S_i(x) + \sum \int_0^x K_{ij} u_j dt$'},
    'prob21': {'name': '1D IDE Fredholm', 'exact': r'$e^x$', 'integral': r'$u^{\prime\prime}(x) = S(x) + \int_0^1 u(t) dt$'},
    'prob22': {'name': '1D IDE Volterra', 'exact': r'$\sin(x)$', 'integral': r'$u^{\prime}(x) = S(x) + \int_0^x u(t) dt$'},
    'prob23': {'name': 'Helmholtz Fredholm', 'exact': r'$\sin(\pi x)$', 'integral': r'$u(x) = S(x) + \int_0^1 \frac{\cos(5|x-t|)}{10} u(t) dt$'},
    'prob31': {'name': '1D Spectral Bias', 'exact': r'$e^{-x}\sin(20\pi x)$', 'integral': r'$u(x) = S(x) + \int_0^x (x-t) u(t) dt$'},
    'prob32': {'name': '1D Transient Trap', 'exact': r'$x$', 'integral': r'$u(x) = S(x) + \int_0^x e^{-50(x-t)} u^3(t) dt$'},
    'prob33': {'name': '1D VIDE Stiffness', 'exact': r'$\sin(\pi x)e^{-t}$', 'integral': r'$u_t - D u_{xx} = \int_0^t e^{-(t-s)} u(x,s) ds + S$'},
    'prob34': {'name': '2D Viscoelastic PDE-IDE', 'exact': r'$\sin(\pi x)e^{-t}$', 'integral': r'$\partial_t u = \alpha \partial_{xx} u + S - \int_0^t e^{-2(t-s)} u ds$'},
    'prob41': {'name': 'Calorimeter Exp1', 'exact': r'$te^{-t}$', 'integral': r'$i(t) = S(t) + \int_0^t e^{-(t-s)} i(s) ds$'},
    'prob42': {'name': 'Calorimeter Exp2', 'exact': r'$t$', 'integral': r'$i(t) = S(t) + \int_0^t K_{stiff} i^2(s) ds$'},
    'prob43': {'name': 'Calorimeter Exp3', 'exact': r'$\sin(\pi x)e^{-t}$', 'integral': r'$u_t - D u_{xx} + v_d u_x = \dots$'},
    'prob44': {'name': 'Calorimeter Exp4', 'exact': r'$\sin(\pi x)e^{-t}$', 'integral': r'$u_t - D u_{xx} = \int_0^t \int_{-1}^1 \cos(\frac{\pi}{2}(x-y)) e^{-(t-s)} u(y,s) dy ds + f$'}
}

COLOR_PALETTE = [
    '#1f77b4', # Steel Blue
    '#ff7f0e', # Orange
    '#2ca02c', # Forest Green
    '#d62728', # Crimson Red
    '#9467bd', # Purple
    '#8c564b', # Brown
    '#e377c2', # Pink
    '#7f7f7f', # Gray
    '#bcbd22', # Olive
    '#17becf', # Cyan
]

def get_study_color(study_name, fallback_idx=0):
    """Returns a unique distinct color for each study from COLOR_PALETTE."""
    return COLOR_PALETTE[fallback_idx % len(COLOR_PALETTE)]

def extract_metrics(log_dir):
    """Extracts and chronologically merges scalar metrics from all TensorBoard event files in log_dir."""
    log_path = Path(log_dir)
    if not log_path.exists():
        return pd.DataFrame()

    event_files = sorted(list(log_path.glob("**/events.out.tfevents*")), key=lambda p: p.stat().st_mtime)
    if not event_files:
        return pd.DataFrame()

    metrics = {}
    expected_tags = ['Loss/Train_Total', 'Loss/Train_PDE', 'Loss/Train_BC', 'Loss/Train_IC', 'Metrics/MAE_Exact']

    for event_file in event_files:
        try:
            ea = EventAccumulator(str(event_file.parent))
            ea.Reload()
            tags = ea.Tags().get('scalars', [])
            for tag in expected_tags:
                if tag in tags:
                    events = ea.Scalars(tag)
                    if not events:
                        continue
                    steps = [e.step for e in events]
                    values = [e.value for e in events]
                    s = pd.Series(data=values, index=steps)
                    if tag not in metrics:
                        metrics[tag] = s
                    else:
                        combined = pd.concat([metrics[tag], s])
                        combined = combined[~combined.index.duplicated(keep='last')]
                        metrics[tag] = combined.sort_index()
        except Exception:
            pass

    if not metrics:
        return pd.DataFrame()

    df = pd.DataFrame(metrics).sort_index()
    
    # Combine IC and BC into a single Loss_ICBC column if present
    ic_col = df['Loss/Train_IC'] if 'Loss/Train_IC' in df.columns else pd.Series(0.0, index=df.index)
    bc_col = df['Loss/Train_BC'] if 'Loss/Train_BC' in df.columns else pd.Series(0.0, index=df.index)
    df['Loss_ICBC'] = ic_col.fillna(0.0) + bc_col.fillna(0.0)
    
    return df

def load_model_and_config(model_path, device, problem):
    """Loads model configuration and weights from checkpoint file."""
    exp_name = model_path.name.replace("_best_model.pt", "").replace("_latest_model.pt", "")
    try:
        checkpoint = torch.load(model_path, map_location=device, weights_only=False)
    except Exception as e:
        print_error(f"Could not load checkpoint file {model_path.name}: {e}")
        return None, None

    if 'config' not in checkpoint:
        print_error(f"CRITICAL: 'config' key not found in checkpoint for experiment '{exp_name}'.")
        return None, None

    config = checkpoint['config']
    print_info(f"Loaded config for '{exp_name}' directly from checkpoint file.")

    try:
        model = build_model(config, problem, device, quiet=True)
    except Exception as e:
        print_error(f"[CRITICAL] Failed to build model for '{exp_name}': {e}")
        return None, config

    try:
        state_dict = checkpoint['state_dict'] if 'state_dict' in checkpoint else checkpoint
        model.load_state_dict(state_dict, strict=False)
        model.eval()
    except Exception as e:
        print_error(f"[CRITICAL] Failed to load weights for {model_path.name}: {e}")
        return None, config

    return model, config

def find_checkpoint_for_study(prob_id_num, study_name, use_debug=False):
    """Finds best or latest checkpoint file for an explicit study name."""
    temp_pm = PathManager(prob_id=prob_id_num, study_name=study_name, debug=use_debug)
    best_pt = temp_pm.get_weight_checkpoint_path(checkpoint_type="best")
    if best_pt.exists():
        return best_pt
    latest_pt = temp_pm.get_weight_checkpoint_path(checkpoint_type="latest")
    if latest_pt.exists():
        return latest_pt

    return None

def extract_model_data(prob_id_num, device, study_names_to_compare, use_debug=False):
    """Finds and loads trained models strictly for the study names listed in studies.yaml."""
    model_data = {}
    problem = get_problem(prob_id_num)

    processed_files = set()
    studies_to_load = []

    for study_name in study_names_to_compare:
        pt_file = find_checkpoint_for_study(prob_id_num, study_name, use_debug=use_debug)
        if pt_file and pt_file not in processed_files:
            processed_files.add(pt_file)
            studies_to_load.append((study_name, pt_file))

    for idx, (study_name, pt_file) in enumerate(studies_to_load):
        model, config = load_model_and_config(pt_file, device, problem)
        if not config or model is None:
            print_error(f"Warning: Failed to load model/config for '{study_name}' ({pt_file.name}). Skipping.")
            continue

        model_type = config.get('model', {}).get('type', 'mlp')
        study_inst = Study(prob_id=prob_id_num, model=model_type, study_name=study_name, debug=use_debug)

        log_dir = study_inst.pm.logs_dir / study_name
        if not log_dir.exists():
            log_dir = study_inst.pm.logs_dir

        df = extract_metrics(log_dir)
        color = get_study_color(study_name, fallback_idx=idx)

        model_data[study_name] = {
            'df': df,
            'config': config,
            'model': model,
            'model_path': pt_file,
            'log_dir': log_dir,
            'study': study_inst,
            'color': color
        }
        print_success(f"Successfully loaded model for study: {study_name} (model type: {model_type}, file: {pt_file.name})")

    return model_data

def count_optuna_trials(study_inst):
    """Returns number of completed Optuna tuning trials from SQLite DB if available."""
    db_path = study_inst.pm.tuning_db_path
    if not db_path.exists():
        tuning_dir = study_inst.pm.tuning_dir
        db_files = list(tuning_dir.glob(f"*{study_inst.study_name}*.db")) if tuning_dir.exists() else []
        if db_files:
            db_path = db_files[0]
        else:
            return "N/A"
    try:
        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()
        cursor.execute("SELECT COUNT(*) FROM trials")
        count = cursor.fetchone()[0]
        conn.close()
        return str(count)
    except Exception:
        return "N/A"

def evaluate_study_metrics(model_data, prob_id_str, problem, is_2d, a, b, device):
    """Extracts test set Loss_PDE, Loss_IC/BC, MAE from study's saved eval_metrics.json, running evaluate() if missing."""
    import json
    table_rows = []

    for exp_name, data in model_data.items():
        study_inst = data['study']
        pt_path = data.get('model_path')

        mod_time_str = "N/A"
        checkpoint_mtime = 0
        if pt_path and pt_path.exists():
            checkpoint_mtime = pt_path.stat().st_mtime
            mod_time_str = datetime.datetime.fromtimestamp(checkpoint_mtime).strftime('%d/%m/%Y %H:%M')

        eval_file = study_inst.pm.logs_dir / exp_name / "eval_metrics.json"
        eval_metrics = None

        if eval_file.exists():
            try:
                with open(eval_file, 'r') as f:
                    cached = json.load(f)
                if cached.get('mtime', 0) >= checkpoint_mtime:
                    eval_metrics = cached
                    print_info(f"Extracted saved metrics for '{exp_name}' from {eval_file.name}")
            except Exception:
                eval_metrics = None

        if eval_metrics is None:
            print_info(f"No valid saved metrics found for '{exp_name}'. Running study.evaluate()...")
            try:
                eval_metrics = study_inst.evaluate(device=device)
            except Exception as e:
                print_error(f"Error evaluating study '{exp_name}': {e}")
                eval_metrics = {}

        loss_pde_val = eval_metrics.get('loss_pde', float('nan'))
        loss_bc_val = eval_metrics.get('loss_bc', 0.0)
        loss_ic_val = eval_metrics.get('loss_ic', 0.0)
        if loss_bc_val is None or np.isnan(loss_bc_val): loss_bc_val = 0.0
        if loss_ic_val is None or np.isnan(loss_ic_val): loss_ic_val = 0.0
        loss_bc_ic = loss_bc_val + loss_ic_val
        loss_total = loss_pde_val + loss_bc_ic if not np.isnan(loss_pde_val) else float('nan')
        mae_val = eval_metrics.get('mae', float('nan'))
        num_trials = count_optuna_trials(study_inst)
        model_type = study_inst.model.upper()

        table_rows.append({
            'Problem ID': prob_id_str.replace('prob', ''),
            'Nome Studio': exp_name,
            'Nome Metodo': model_type,
            'Trial Tuning': num_trials,
            'Data Salvataggio': mod_time_str,
            'Loss_PDE': loss_pde_val,
            'Loss_ICBC': loss_bc_ic,
            'Loss_Total': loss_total,
            'MAE': mae_val
        })

    return pd.DataFrame(table_rows)

def plot_single_loss(model_data, col_name, title, save_path):
    """Plots a single comparative loss metric across epochs for all studies."""
    apply_global_plot_style()
    fig, ax = create_figure(figsize=(9, 5))
    has_positive_data = False

    for exp_name, data in model_data.items():
        df = data['df']
        if not df.empty and col_name in df.columns:
            s = df[col_name].dropna()
            s_pos = s[s > 0]
            if not s_pos.empty:
                ax.plot(s_pos.index, s_pos.values, label=exp_name, color=data['color'], linewidth=1.8, alpha=0.85)
                has_positive_data = True

    if has_positive_data:
        format_plot(ax, title=title, xlabel='Epochs', ylabel=col_name, yscale='log', grid=True, legend=True)
        fig.savefig(save_path, dpi=300, bbox_inches='tight')
        fig.savefig(save_path.with_suffix('.pdf'), bbox_inches='tight')
        return fig
    else:
        plt.close(fig)
        return None

def plot_fit_comparison(model_data, problem, a, b, is_2d, device, save_dir, prob_id_num):
    """Generates fit comparison plot for 1D, Problem 19 (1D coupled), or 2D problems."""
    apply_global_plot_style()
    
    if prob_id_num == 19:
        # Problem 19: 1D Coupled System (2 outputs: u1 and u2)
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
        x_eval = torch.linspace(a, b, 500, device=device).view(-1, 1)
        with torch.no_grad():
            u_exact = problem.exact_solution(x_eval).cpu().numpy()
        x_cpu = x_eval.cpu().numpy().flatten()

        labels_component = [r'$u_1(x) = 1+x+x^2$', r'$u_2(x) = 1-x-x^2$']
        for c_idx in range(2):
            ax = axes[c_idx]
            ax.plot(x_cpu, u_exact[:, c_idx], 'k--', label=f'Exact {labels_component[c_idx]}', linewidth=2.2, zorder=5)
            for exp_name, data in model_data.items():
                model = data['model']
                with torch.no_grad():
                    u_pred = model(x_eval).cpu().numpy()
                ax.plot(x_cpu, u_pred[:, c_idx], label=exp_name, color=data['color'], linewidth=1.8, alpha=0.85)
            format_plot(ax, title=f'Fit Comparison - Component {c_idx+1}', xlabel='x', ylabel=f'u_{c_idx+1}(x)', grid=True, legend=True)

        fig.suptitle('Fit Comparison - Problem 19 (Coupled 1D IDE System)', fontsize=14, y=0.98)
        fig.savefig(save_dir / "fit_comparison.png", dpi=300, bbox_inches='tight')
        fig.savefig(save_dir / "fit_comparison.pdf", bbox_inches='tight')
        return fig

    elif not is_2d:
        fig, ax = create_figure(figsize=(9, 5))
        x_eval = torch.linspace(a, b, 500, device=device).view(-1, 1)
        with torch.no_grad():
            u_exact = problem.exact_solution(x_eval).cpu().numpy().flatten()
        x_cpu = x_eval.cpu().numpy().flatten()

        ax.plot(x_cpu, u_exact, 'k--', label='Exact Ground Truth', linewidth=2.2, zorder=5)
        for exp_name, data in model_data.items():
            model = data['model']
            with torch.no_grad():
                u_pred = model(x_eval).cpu().numpy().flatten()
            ax.plot(x_cpu, u_pred, label=exp_name, color=data['color'], linewidth=1.8, alpha=0.85)

        format_plot(ax, title=f'Fit Comparison - Problem {prob_id_num}', xlabel='x', ylabel='u(x)', grid=True, legend=True)
        fig.savefig(save_dir / "fit_comparison.png", dpi=300, bbox_inches='tight')
        fig.savefig(save_dir / "fit_comparison.pdf", bbox_inches='tight')
        return fig

    else:
        # 2D case: 1D slice + side-by-side heatmaps
        domain_x, domain_y = a, b
        n_pts = 80
        x_lin = torch.linspace(domain_x[0], domain_x[1], n_pts, device=device)
        y_lin = torch.linspace(domain_y[0], domain_y[1], n_pts, device=device)
        gx, gy = torch.meshgrid(x_lin, y_lin, indexing='ij')
        pts_2d = torch.stack([gx.flatten(), gy.flatten()], dim=1)

        with torch.no_grad():
            u_exact_2d = problem.exact_solution(pts_2d).cpu().numpy().reshape(n_pts, n_pts)

        # 1D Slice along y_mid
        y_mid = (domain_y[0] + domain_y[1]) / 2.0
        slice_pts = torch.stack([x_lin, torch.full_like(x_lin, y_mid)], dim=1)
        with torch.no_grad():
            u_exact_slice = problem.exact_solution(slice_pts).cpu().numpy().flatten()

        n_models = len(model_data)
        fig, axes = plt.subplots(1, n_models + 2, figsize=(4.2 * (n_models + 2), 4))

        # Subplot 0: 1D slice
        x_cpu = x_lin.cpu().numpy()
        axes[0].plot(x_cpu, u_exact_slice, 'k--', label='Exact', linewidth=2)
        for exp_name, data in model_data.items():
            with torch.no_grad():
                u_pred_slice = data['model'](slice_pts).cpu().numpy().flatten()
            axes[0].plot(x_cpu, u_pred_slice, label=exp_name, color=data['color'], linewidth=1.8)
        axes[0].set_title(f'1D Slice (y={y_mid:.2f})')
        axes[0].set_xlabel('x')
        axes[0].set_ylabel('u(x, y_mid)')
        axes[0].legend(fontsize=8)
        axes[0].grid(True, alpha=0.3, linestyle='--')

        # Subplot 1: Exact 2D heatmap
        im0 = axes[1].imshow(u_exact_2d.T, origin='lower', extent=[domain_x[0], domain_x[1], domain_y[0], domain_y[1]], cmap='viridis')
        axes[1].set_title('Exact u(x,y)')
        axes[1].set_xlabel('x')
        axes[1].set_ylabel('y')
        plt.colorbar(im0, ax=axes[1], fraction=0.046, pad=0.04)

        # Model heatmaps
        for idx, (exp_name, data) in enumerate(model_data.items()):
            ax_m = axes[idx + 2]
            with torch.no_grad():
                u_pred_2d = data['model'](pts_2d).cpu().numpy().reshape(n_pts, n_pts)
            im_m = ax_m.imshow(u_pred_2d.T, origin='lower', extent=[domain_x[0], domain_x[1], domain_y[0], domain_y[1]], cmap='viridis')
            ax_m.set_title(f'Fit: {exp_name}')
            ax_m.set_xlabel('x')
            plt.colorbar(im_m, ax=ax_m, fraction=0.046, pad=0.04)

        fig.savefig(save_dir / "fit_comparison.png", dpi=300, bbox_inches='tight')
        fig.savefig(save_dir / "fit_comparison.pdf", bbox_inches='tight')
        return fig

def plot_mae_error_distribution(model_data, problem, a, b, is_2d, device, save_dir, prob_id_num):
    """Generates pointwise MAE error distribution plot along domain."""
    apply_global_plot_style()
    
    if prob_id_num == 19:
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
        x_eval = torch.linspace(a, b, 500, device=device).view(-1, 1)
        with torch.no_grad():
            u_exact = problem.exact_solution(x_eval).cpu().numpy()
        x_cpu = x_eval.cpu().numpy().flatten()

        for c_idx in range(2):
            ax = axes[c_idx]
            for exp_name, data in model_data.items():
                model = data['model']
                with torch.no_grad():
                    u_pred = model(x_eval).cpu().numpy()
                err_c = np.abs(u_pred[:, c_idx] - u_exact[:, c_idx])
                ax.plot(x_cpu, err_c, label=exp_name, color=data['color'], linewidth=1.8, alpha=0.85)
            format_plot(ax, title=f'Pointwise MAE Error |u_{c_idx+1} - u_{c_idx+1}^{{exact}}|', xlabel='x', ylabel=f'MAE Component {c_idx+1}', yscale='log', grid=True, legend=True)

        fig.suptitle('MAE Error Distribution - Problem 19 (Coupled 1D IDE System)', fontsize=14, y=0.98)
        fig.savefig(save_dir / "error_mae_domain_comparison.png", dpi=300, bbox_inches='tight')
        fig.savefig(save_dir / "error_mae_domain_comparison.pdf", bbox_inches='tight')
        return fig

    elif not is_2d:
        fig, ax = create_figure(figsize=(9, 5))
        x_eval = torch.linspace(a, b, 500, device=device).view(-1, 1)
        with torch.no_grad():
            u_exact = problem.exact_solution(x_eval).cpu().numpy().flatten()
        x_cpu = x_eval.cpu().numpy().flatten()

        for exp_name, data in model_data.items():
            model = data['model']
            with torch.no_grad():
                u_pred = model(x_eval).cpu().numpy().flatten()
            err = np.abs(u_pred - u_exact)
            ax.plot(x_cpu, err, label=exp_name, color=data['color'], linewidth=1.8, alpha=0.85)

        format_plot(ax, title=f'Pointwise MAE Error |u_pred - u_exact| - Problem {prob_id_num}', xlabel='x', ylabel='Absolute Error (MAE)', yscale='log', grid=True, legend=True)
        fig.savefig(save_dir / "error_mae_domain_comparison.png", dpi=300, bbox_inches='tight')
        fig.savefig(save_dir / "error_mae_domain_comparison.pdf", bbox_inches='tight')
        return fig

    else:
        domain_x, domain_y = a, b
        n_pts = 80
        x_lin = torch.linspace(domain_x[0], domain_x[1], n_pts, device=device)
        y_lin = torch.linspace(domain_y[0], domain_y[1], n_pts, device=device)
        gx, gy = torch.meshgrid(x_lin, y_lin, indexing='ij')
        pts_2d = torch.stack([gx.flatten(), gy.flatten()], dim=1)

        with torch.no_grad():
            u_exact_2d = problem.exact_solution(pts_2d).cpu().numpy().reshape(n_pts, n_pts)

        y_mid = (domain_y[0] + domain_y[1]) / 2.0
        slice_pts = torch.stack([x_lin, torch.full_like(x_lin, y_mid)], dim=1)
        with torch.no_grad():
            u_exact_slice = problem.exact_solution(slice_pts).cpu().numpy().flatten()

        n_models = len(model_data)
        fig, axes = plt.subplots(1, n_models + 1, figsize=(4.2 * (n_models + 1), 4))

        x_cpu = x_lin.cpu().numpy()
        for exp_name, data in model_data.items():
            with torch.no_grad():
                u_pred_slice = data['model'](slice_pts).cpu().numpy().flatten()
            err_slice = np.abs(u_pred_slice - u_exact_slice)
            axes[0].plot(x_cpu, err_slice, label=exp_name, color=data['color'], linewidth=1.8)
        axes[0].set_yscale('log')
        axes[0].set_title(f'Abs Error Slice (y={y_mid:.2f})')
        axes[0].set_xlabel('x')
        axes[0].set_ylabel('Absolute Error')
        axes[0].legend(fontsize=8)
        axes[0].grid(True, which="both", alpha=0.3, linestyle='--')

        for idx, (exp_name, data) in enumerate(model_data.items()):
            ax_m = axes[idx + 1]
            with torch.no_grad():
                u_pred_2d = data['model'](pts_2d).cpu().numpy().reshape(n_pts, n_pts)
            err_2d = np.abs(u_pred_2d - u_exact_2d)
            im_m = ax_m.imshow(err_2d.T, origin='lower', extent=[domain_x[0], domain_x[1], domain_y[0], domain_y[1]], cmap='inferno')
            ax_m.set_title(f'Error: {exp_name}')
            ax_m.set_xlabel('x')
            plt.colorbar(im_m, ax=ax_m, fraction=0.046, pad=0.04)

        fig.savefig(save_dir / "error_mae_domain_comparison.png", dpi=300, bbox_inches='tight')
        fig.savefig(save_dir / "error_mae_domain_comparison.pdf", bbox_inches='tight')
        return fig

def plot_ide_residual_distribution(model_data, problem, a, b, is_2d, device, save_dir, prob_id_num):
    """Generates pointwise IDE physical residual |R(x)| distribution plot along domain."""
    apply_global_plot_style()
    pde_loss_fn = VolterraPDELoss(problem, num_quadrature_nodes=50, device=device)
    
    if prob_id_num == 19:
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
        x_eval = torch.linspace(a, b, 300, device=device).view(-1, 1).requires_grad_(True)
        x_cpu = x_eval.detach().cpu().numpy().flatten()

        for exp_name, data in model_data.items():
            model = data['model']
            is_seq = getattr(model, 'is_sequence_model', False)
            try:
                if is_seq:
                    nodes = getattr(problem, 'get_nodes', lambda x, q: None)(x_eval, pde_loss_fn.quadrature)
                    if nodes is not None:
                        seq = torch.cat([x_eval.unsqueeze(1), nodes], dim=1)
                        U = model(seq)
                        u_pred = U[:, 0, :]
                        u_t = U[:, 1:, :]
                    else:
                        u_pred = model(x_eval.unsqueeze(1)).squeeze(1)
                        u_t = None
                else:
                    u_pred = model(x_eval)
                    u_t = None

                residual = problem.compute_residual(
                    u_pred, u_t, model, x_eval, is_seq, pde_loss_fn.quadrature, pde_loss_fn.compute_derivative
                )
                res_np = np.abs(residual.detach().cpu().numpy())
                for c_idx in range(2):
                    axes[c_idx].plot(x_cpu, res_np[:, c_idx], label=exp_name, color=data['color'], linewidth=1.8, alpha=0.85)
            except Exception as e:
                print_error(f"Failed to compute residual for {exp_name} on Prob 19: {e}")

        for c_idx in range(2):
            format_plot(axes[c_idx], title=f'Pointwise IDE Residual |R_{c_idx+1}(x)|', xlabel='x', ylabel=f'Residual Component {c_idx+1}', yscale='log', grid=True, legend=True)

        fig.suptitle('IDE Physical Residual Distribution - Problem 19 (Coupled 1D IDE System)', fontsize=14, y=0.98)
        fig.savefig(save_dir / "error_ide_domain_comparison.png", dpi=300, bbox_inches='tight')
        fig.savefig(save_dir / "error_ide_domain_comparison.pdf", bbox_inches='tight')
        return fig

    elif not is_2d:
        fig, ax = create_figure(figsize=(9, 5))
        x_eval = torch.linspace(a, b, 300, device=device).view(-1, 1).requires_grad_(True)
        x_cpu = x_eval.detach().cpu().numpy().flatten()

        for exp_name, data in model_data.items():
            model = data['model']
            is_seq = getattr(model, 'is_sequence_model', False)
            try:
                if is_seq:
                    nodes = getattr(problem, 'get_nodes', lambda x, q: None)(x_eval, pde_loss_fn.quadrature)
                    if nodes is not None:
                        seq = torch.cat([x_eval.unsqueeze(1), nodes], dim=1)
                        U = model(seq)
                        u_pred = U[:, 0, :]
                        u_t = U[:, 1:, :]
                    else:
                        u_pred = model(x_eval.unsqueeze(1)).squeeze(1)
                        u_t = None
                else:
                    u_pred = model(x_eval)
                    u_t = None

                residual = problem.compute_residual(
                    u_pred, u_t, model, x_eval, is_seq, pde_loss_fn.quadrature, pde_loss_fn.compute_derivative
                )
                res_np = np.abs(residual.detach().cpu().numpy().flatten())
                ax.plot(x_cpu, res_np, label=exp_name, color=data['color'], linewidth=1.8, alpha=0.85)
            except Exception as e:
                print_error(f"Failed to compute IDE residual for {exp_name}: {e}")

        format_plot(ax, title=f'Pointwise IDE Residual |R(x)| - Problem {prob_id_num}', xlabel='x', ylabel='IDE Residual |R(x)|', yscale='log', grid=True, legend=True)
        fig.savefig(save_dir / "error_ide_domain_comparison.png", dpi=300, bbox_inches='tight')
        fig.savefig(save_dir / "error_ide_domain_comparison.pdf", bbox_inches='tight')
        return fig

    else:
        domain_x, domain_y = a, b
        n_pts = 35
        x_lin = torch.linspace(domain_x[0], domain_x[1], n_pts, device=device)
        y_lin = torch.linspace(domain_y[0], domain_y[1], n_pts, device=device)
        gx, gy = torch.meshgrid(x_lin, y_lin, indexing='ij')
        pts_2d_flat = torch.stack([gx.flatten(), gy.flatten()], dim=1)

        n_models = len(model_data)
        fig, axes = plt.subplots(1, n_models, figsize=(4.2 * n_models, 4))
        if n_models == 1:
            axes = [axes]

        batch_size = 150

        for idx, (exp_name, data) in enumerate(model_data.items()):
            ax_m = axes[idx]
            model = data['model']
            try:
                res_chunks = []
                for b_start in range(0, pts_2d_flat.size(0), batch_size):
                    batch_pts = pts_2d_flat[b_start : b_start + batch_size].clone().detach().to(device).requires_grad_(True)
                    u_pred = model(batch_pts)
                    residual = problem.compute_residual(
                        u_pred, None, model, batch_pts, False, pde_loss_fn.quadrature, pde_loss_fn.compute_derivative
                    )
                    res_chunks.append(np.abs(residual.detach().cpu().numpy()))

                res_2d = np.concatenate(res_chunks, axis=0).reshape(n_pts, n_pts)
                im_m = ax_m.imshow(res_2d.T, origin='lower', extent=[domain_x[0], domain_x[1], domain_y[0], domain_y[1]], cmap='magma')
                ax_m.set_title(f'IDE Residual: {exp_name}')
                ax_m.set_xlabel('x')
                plt.colorbar(im_m, ax=ax_m, fraction=0.046, pad=0.04)
            except Exception as e:
                print_error(f"Failed 2D residual computation for {exp_name}: {e}")

        fig.savefig(save_dir / "error_ide_domain_comparison.png", dpi=300, bbox_inches='tight')
        fig.savefig(save_dir / "error_ide_domain_comparison.pdf", bbox_inches='tight')
        return fig

def plot_summary_table(df_summary, prob_id_str, save_dir):
    """Renders and saves summary table as CSV and PNG, returning the figure for PDF compilation."""
    apply_global_plot_style()
    csv_plots_path = save_dir / f"{prob_id_str}_summary_table.csv"
    df_summary.to_csv(csv_plots_path, index=False)

    fig, ax = plt.subplots(figsize=(14, max(2.5, 0.8 * len(df_summary) + 1.5)))
    ax.axis('off')

    display_headers = ['Problem', 'Studio', 'Metodo', 'Trials', 'Ultimo Salvataggio', 'Loss PDE', 'Loss IC/BC', 'Loss Totale', 'MAE']
    cell_text = []

    best_mae_idx = df_summary['MAE'].idxmin() if not df_summary['MAE'].isna().all() else -1

    for idx, row in df_summary.iterrows():
        prob_id = f"Prob {row['Problem ID']}"
        l_pde = f"{row['Loss_PDE']:.4e}" if not pd.isna(row['Loss_PDE']) else "N/A"
        l_icbc = f"{row['Loss_ICBC']:.4e}" if not pd.isna(row['Loss_ICBC']) else "N/A"
        l_tot = f"{row['Loss_Total']:.4e}" if not pd.isna(row['Loss_Total']) else "N/A"
        mae = f"{row['MAE']:.4e}" if not pd.isna(row['MAE']) else "N/A"
        cell_text.append([prob_id, row['Nome Studio'], row['Nome Metodo'], str(row['Trial Tuning']), row['Data Salvataggio'], l_pde, l_icbc, l_tot, mae])

    table = ax.table(cellText=cell_text, colLabels=display_headers, loc='center', cellLoc='center')
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1, 2.0)

    for i, row in df_summary.iterrows():
        if i == best_mae_idx:
            for j in range(len(display_headers)):
                table[i+1, j].set_facecolor('#d4edda')

    for key, cell in table.get_celld().items():
        row_idx, col_idx = key
        if row_idx == 0:
            cell.set_text_props(weight='bold')
            cell.set_facecolor('#eaeaea')

    plt.title(f"Summary Table - {prob_id_str.upper()}", fontsize=13, pad=12)
    fig.savefig(save_dir / f"{prob_id_str}_summary_table.png", dpi=300, bbox_inches='tight')
    return fig

def compile_problem_pdf_report(figures_list, prob_id_str, save_dir):
    """Compiles all generated plot figures into a single multi-page PDF report."""
    pdf_path = save_dir / f"{prob_id_str}_comparison_report.pdf"
    valid_figs = [f for f in figures_list if f is not None]
    if not valid_figs:
        print_error(f"No valid figures to compile into PDF for {prob_id_str}.")
        return None

    try:
        with PdfPages(pdf_path) as pdf:
            for fig in valid_figs:
                pdf.savefig(fig, bbox_inches='tight')
        print_success(f"Generated comparison PDF report: {pdf_path}")
        return pdf_path
    except Exception as e:
        print_error(f"Failed to compile PDF report for {prob_id_str}: {e}")
        return None

def sync_comparison_outputs_to_drive(prob_plots_dir, central_pdf_dir=None, use_debug=False):
    """Syncs generated comparison plots and PDF report to Google Drive if available."""
    pm = PathManager(debug=use_debug)
    drive_root = pm.drive_root
    if drive_root and drive_root.exists():
        try:
            # Sync problem plots (PNGs & CSVs) to Drive: drive/plots/prob<id>/comparison/
            prob_id_name = prob_plots_dir.parent.name
            drive_prob_dir = drive_root / ("plots_debug" if use_debug else "plots") / prob_id_name / "comparison"
            if drive_prob_dir.resolve() != prob_plots_dir.resolve():
                drive_prob_dir.mkdir(parents=True, exist_ok=True)
                for item in prob_plots_dir.glob("*.*"):
                    shutil.copy(item, drive_prob_dir / item.name)

            # Sync central PDF report to Drive: drive/plots/comparison/
            if central_pdf_dir and central_pdf_dir.exists():
                drive_central_dir = drive_root / ("plots_debug" if use_debug else "plots") / "comparison"
                if drive_central_dir.resolve() != central_pdf_dir.resolve():
                    drive_central_dir.mkdir(parents=True, exist_ok=True)
                    for item in central_pdf_dir.glob("*.pdf"):
                        shutil.copy(item, drive_central_dir / item.name)

            print_success(f"[+] Synced comparison plots and PDF report to Drive.")
        except Exception as e:
            print_error(f"Failed to sync comparison plots to Drive: {e}")

def process_problem(prob_id_input, device, data_dir, generate_plots=True):
    prob_id_str = prob_id_input if str(prob_id_input).startswith('prob') else f"prob{prob_id_input}"
    prob_id_num = int(prob_id_str.replace('prob', ''))

    use_debug = 'debug' in str(data_dir)
    pm = PathManager(prob_id=prob_id_num, debug=use_debug)
    studies_config_path = pm.studies_config_path

    study_names = []
    if studies_config_path.exists():
        try:
            with open(studies_config_path, 'r') as f:
                studies_config = yaml.safe_load(f) or {}
            study_names = studies_config.get(prob_id_str, [])
        except Exception as e:
            print_error(f"Error reading studies config: {e}")

    print_info(f"\nScanning for trained models in problem {prob_id_num}...")
    model_data = extract_model_data(prob_id_num, device, study_names, use_debug=use_debug)

    if not model_data:
        print_error(f"No models found for problem {prob_id_num}. Skipping.")
        return

    if prob_id_num not in PROBLEMS_MAP:
        print_error(f"Error: Problem {prob_id_num} not found in PROBLEMS_MAP. Skipping.")
        return

    first_config = list(model_data.values())[0]['config']
    is_2d = isinstance(first_config['data']['domain'][0], list)
    a, b = first_config['data']['domain']
    problem = get_problem(prob_id_num, a=a, b=b)

    df_summary = evaluate_study_metrics(model_data, prob_id_str, problem, is_2d, a, b, device)

    if generate_plots:
        # 1. Problem-specific plots directory: PINN/plots/prob<id>/comparison/
        prob_plots_dir = pm.plots_dir / "comparison"
        prob_plots_dir.mkdir(parents=True, exist_ok=True)

        # 2. Central plots directory for PDF report ONLY: PINN/plots/comparison/
        central_pdf_dir = pm.plots_root / "comparison"
        central_pdf_dir.mkdir(parents=True, exist_ok=True)

        print_info(f"Generating benchmark visualization outputs for {prob_id_str} in {prob_plots_dir}...")
        collected_figures = []

        # 1. Summary Table as Page 1 of PDF report
        fig_table = plot_summary_table(df_summary, prob_id_str, prob_plots_dir)
        if fig_table is not None:
            collected_figures.append(fig_table)

        # 2. Training metrics vs Epochs
        fig_tot = plot_single_loss(model_data, 'Loss/Train_Total', f'Loss Total vs Epochs - {prob_id_str.upper()}', prob_plots_dir / f"{prob_id_str}_losses_tot.png")
        fig_pde = plot_single_loss(model_data, 'Loss/Train_PDE', f'Loss IDE vs Epochs - {prob_id_str.upper()}', prob_plots_dir / f"{prob_id_str}_losses_ide.png")
        fig_icbc = plot_single_loss(model_data, 'Loss_ICBC', f'Loss IC/BC vs Epochs - {prob_id_str.upper()}', prob_plots_dir / f"{prob_id_str}_losses_icbc.png")
        fig_mae = plot_single_loss(model_data, 'Metrics/MAE_Exact', f'MAE vs Epochs - {prob_id_str.upper()}', prob_plots_dir / f"{prob_id_str}_losses_mae.png")

        for f in [fig_tot, fig_pde, fig_icbc, fig_mae]:
            if f is not None:
                collected_figures.append(f)

        # 3. Fit comparison
        fig_fit = plot_fit_comparison(model_data, problem, a, b, is_2d, device, prob_plots_dir, prob_id_num)
        if fig_fit is not None:
            collected_figures.append(fig_fit)

        # 4. MAE Error distribution along domain
        fig_err_mae = plot_mae_error_distribution(model_data, problem, a, b, is_2d, device, prob_plots_dir, prob_id_num)
        if fig_err_mae is not None:
            collected_figures.append(fig_err_mae)

        # 5. IDE Residual error distribution along domain
        fig_err_ide = plot_ide_residual_distribution(model_data, problem, a, b, is_2d, device, prob_plots_dir, prob_id_num)
        if fig_err_ide is not None:
            collected_figures.append(fig_err_ide)

        # 6. Compile multi-page PDF report ONLY in central plots/comparison folder
        pdf_report_path = compile_problem_pdf_report(collected_figures, prob_id_str, central_pdf_dir)

        # Close all figures
        for f in collected_figures:
            plt.close(f)

        # 7. Sync plots and PDF report to Drive
        sync_comparison_outputs_to_drive(prob_plots_dir, central_pdf_dir, use_debug=use_debug)

def main():
    parser = argparse.ArgumentParser(description="Generate comparison plots and summary tables for PINN studies.")
    parser.add_argument('--prob_id', type=str, default=None, nargs='?', help="Problem ID (e.g. 19, 34, prob19). If omitted, processes all problems.")
    parser.add_argument('--debug', action='store_true', help='Use debug directory (data_debug)')
    args = parser.parse_args()

    # --- MANDATORY BOILERPLATE ---
    IN_COLAB = is_colab()
    device = torch.device('cpu')
    root_data_dir = get_data_root(use_debug=args.debug)

    print_info(f"Running on device: {device}")
    print_info(f"Running in colab: {IN_COLAB}")
    print_success(f"Data directory set to: {root_data_dir}")

    pm = PathManager(debug=args.debug)
    studies_config_path = pm.studies_config_path

    if args.prob_id is None:
        print_info("No problem specified. Processing all problems defined in studies.yaml...")
        if studies_config_path.exists():
            with open(studies_config_path, 'r') as f:
                studies_config = yaml.safe_load(f) or {}
            for prob_id_str in studies_config.keys():
                process_problem(prob_id_str, device, root_data_dir, generate_plots=True)
    else:
        raw_input = str(args.prob_id).strip()
        prob_id_str = raw_input if raw_input.lower().startswith('prob') else f"prob{raw_input}"
        process_problem(prob_id_str, device, root_data_dir, generate_plots=True)

if __name__ == "__main__":
    main()