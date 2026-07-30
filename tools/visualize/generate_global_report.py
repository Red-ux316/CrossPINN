import argparse
import datetime
import sqlite3
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
import torch
import yaml
import shutil

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
from tool_utils import get_device
from equations.registry import PROBLEMS_MAP, get_problem
from tools.visualize.plot_style import (
    apply_global_plot_style,
    create_figure,
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

def _get_study_metrics(study_inst, device):
    """
    Extracts relevant metrics for a single study, running evaluate() if necessary.
    Returns a dictionary of metrics for a table row.
    """
    import json
    
    exp_name = study_inst.pm.study_base_name

    # Try to load checkpoint to get model_type and config
    ckpt_path = study_inst.pm.get_weight_checkpoint_path(checkpoint_type="best")
    if not ckpt_path.exists():
        ckpt_path = study_inst.pm.get_weight_checkpoint_path(checkpoint_type="latest")
    
    if not ckpt_path.exists():
        print_error(f"  -> Checkpoint not found for '{exp_name}'. Skipping metrics extraction.")
        return None

    try:
        checkpoint = torch.load(ckpt_path, map_location='cpu', weights_only=False)
        if not isinstance(checkpoint, dict) or 'config' not in checkpoint:
            print_error(f"  -> Checkpoint for '{exp_name}' is invalid or missing config. Skipping.")
            return None
        
        model_type = checkpoint['config']['model']['type']
        # Re-instantiate the Study object with the correct model type from the checkpoint
        # This ensures PathManager resolves tuning DB paths correctly.
        study_inst = Study(prob_id=study_inst.prob_id, model=model_type, study_name=exp_name, debug=study_inst.debug)
    except Exception as e:
        print_error(f"  -> Error loading checkpoint for '{exp_name}': {e}. Skipping.")
        return None

    mod_time_str = "N/A"
    checkpoint_mtime = 0
    if ckpt_path.exists():
        checkpoint_mtime = ckpt_path.stat().st_mtime
        mod_time_str = datetime.datetime.fromtimestamp(checkpoint_mtime).strftime('%d/%m/%Y %H:%M')

    eval_file = study_inst.pm.logs_dir / exp_name / "eval_metrics.json"
    eval_metrics = None

    if eval_file.exists():
        try:
            with open(eval_file, 'r') as f:
                cached = json.load(f)
            if cached.get('mtime', 0) >= checkpoint_mtime:
                eval_metrics = cached
                print_info(f"  -> Extracted saved metrics for '{exp_name}' from {eval_file.name}")
        except Exception:
            eval_metrics = None

    if eval_metrics is None:
        print_info(f"  -> No valid saved metrics found for '{exp_name}'. Running study.evaluate()...")
        try:
            eval_metrics = study_inst.evaluate(device=device)
        except Exception as e:
            print_error(f"  -> Error evaluating study '{exp_name}': {e}")
            eval_metrics = {}

    loss_pde_val = eval_metrics.get('loss_pde', float('nan'))
    loss_bc_val = eval_metrics.get('loss_bc', 0.0)
    loss_ic_val = eval_metrics.get('loss_ic', 0.0)
    if loss_bc_val is None or np.isnan(loss_bc_val): loss_bc_val = 0.0
    if loss_ic_val is None or np.isnan(loss_ic_val): loss_ic_val = 0.0
    loss_bc_ic = loss_bc_val + loss_ic_val
    mae_val = eval_metrics.get('mae', float('nan'))
    num_trials = count_optuna_trials(study_inst)
    loss_total = loss_pde_val + loss_bc_ic if not np.isnan(loss_pde_val) else float('nan') # Approximate total loss

    return {
        'Problem ID': study_inst.prob_id,
        'Nome Studio': exp_name,
        'Nome Metodo': model_type,
        'Trial Tuning': num_trials,
        'Data Salvataggio': mod_time_str,
        'Loss_PDE': loss_pde_val,
        'Loss_ICBC': loss_bc_ic,
        'Loss_Total': loss_total,
        'MAE': mae_val
    }

def generate_global_summary_report(df_global, save_path):
    """Generates a global summary PDF report from a DataFrame of all studies."""
    apply_global_plot_style()
    
    # Sort by Problem ID then by MAE
    df_global['Problem ID Int'] = pd.to_numeric(df_global['Problem ID'])
    df_global = df_global.sort_values(by=['Problem ID Int', 'MAE'], ascending=[True, True])
    df_global = df_global.drop(columns=['Problem ID Int'])

    fig, ax = plt.subplots(figsize=(20, max(4, 0.4 * len(df_global) + 2)))
    ax.axis('off')

    headers = ['Problem', 'Studio', 'Metodo', 'Trials', 'Ultimo Salvataggio', 'Loss PDE', 'Loss IC/BC', 'Loss Totale', 'MAE']
    cell_text = []
    
    last_prob_id = None
    for idx, row in df_global.iterrows():
        prob_id = row['Problem ID']
        
        # Add a separator for a new problem group
        if prob_id != last_prob_id:
            if last_prob_id is not None:
                cell_text.append([''] * len(headers)) # Add a blank spacer row
            last_prob_id = prob_id

        # Format data for display
        l_pde = f"{row['Loss_PDE']:.3e}" if pd.notna(row['Loss_PDE']) else "N/A"
        l_icbc = f"{row['Loss_ICBC']:.3e}" if pd.notna(row['Loss_ICBC']) else "N/A"
        l_tot = f"{row['Loss_Total']:.3e}" if pd.notna(row['Loss_Total']) else "N/A"
        mae = f"{row['MAE']:.3e}" if pd.notna(row['MAE']) else "N/A"
        
        cell_text.append([
            f"Prob {prob_id}",
            row['Nome Studio'],
            row['Nome Metodo'],
            str(row['Trial Tuning']),
            row['Data Salvataggio'],
            l_pde,
            l_icbc,
            l_tot,
            mae
        ])

    table = ax.table(cellText=cell_text, colLabels=headers, loc='center', cellLoc='left')
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1, 2.2)

    # Style header and highlight best MAE rows
    # Need to map original df_global index to cell_text index, considering spacer rows
    cell_text_idx_map = {}
    current_cell_idx = 0
    current_prob_id = None
    for original_idx, row in df_global.iterrows():
        if row['Problem ID'] != current_prob_id:
            if current_prob_id is not None:
                current_cell_idx += 1 # For the spacer row
            current_prob_id = row['Problem ID']
        cell_text_idx_map[original_idx] = current_cell_idx
        current_cell_idx += 1

    best_mae_indices_original_df = df_global.groupby('Problem ID')['MAE'].idxmin()
    
    for original_idx in best_mae_indices_original_df:
        cell_idx = cell_text_idx_map[original_idx]
        for col_idx in range(len(headers)):
            table[cell_idx + 1, col_idx].set_facecolor('#d4edda') # +1 for header row

    for key, cell in table.get_celld().items():
        row_idx, col_idx = key
        if row_idx == 0:
            cell.set_text_props(weight='bold')
            cell.set_facecolor('#eaeaea')
        cell.set_edgecolor('grey')

    plt.title("Global Studies Summary Report", fontsize=16, pad=20)
    save_and_close_figure(fig, save_path)
    print_success(f"Generated global summary report at: {save_path}")

def main():
    parser = argparse.ArgumentParser(description="Generate a global summary report (CSV and PDF) for all studies.")
    parser.add_argument('--debug', action='store_true', help='Use debug directories (data_debug, plots_debug).')
    args = parser.parse_args()

    device = get_device()
    pm_global = PathManager(debug=args.debug)
    studies_config_path = pm_global.studies_config_path
    root_data_dir = pm_global.data_root

    if not studies_config_path.exists():
        print_error(f"Cannot generate global report: studies.yaml not found at {studies_config_path}")
        return

    with open(studies_config_path, 'r') as f:
        studies_config = yaml.safe_load(f) or {}

    all_study_data = []
    total_studies = sum(len(v) for v in studies_config.values() if v)
    processed_count = 0

    print_info("\n[Global Report] Collecting metrics for all studies...")

    for prob_key, study_names in studies_config.items():
        if not study_names:
            continue
        
        prob_id = int(prob_key.replace('prob', ''))
        print_info(f"\n--- Processing Problem: {prob_id} ---")

        for study_name in study_names:
            processed_count += 1
            print_info(f"[{processed_count}/{total_studies}] Collecting metrics for study: {study_name}")
            
            try:
                # Create a Study object for the current study
                # The model type will be updated inside _get_study_metrics from checkpoint
                study = Study(prob_id=prob_id, model='mlp', study_name=study_name, debug=args.debug)
                metrics = _get_study_metrics(study, device)
                if metrics:
                    all_study_data.append(metrics)
            except Exception as e:
                print_error(f"  -> An error occurred while processing '{study_name}': {e}")
    
    if not all_study_data:
        print_error("No study data collected. Cannot generate global report.")
        return

    df_global = pd.DataFrame(all_study_data)
    
    # 1. Local Paths (PINN/data and PINN/plots inside project repository)
    local_data_dir = pm_global.pinn_root / ("data_debug" if args.debug else "data")
    local_plots_dir = pm_global.pinn_root / ("plots_debug" if args.debug else "plots")
    local_data_dir.mkdir(exist_ok=True, parents=True)
    local_plots_dir.mkdir(exist_ok=True, parents=True)

    local_csv_path = local_data_dir / "global_studies_summary.csv"
    local_pdf_path = local_plots_dir / "global_studies_summary.pdf"

    # Save local CSV
    df_global.to_csv(local_csv_path, index=False)
    print_success(f"Saved local global summary CSV to: {local_csv_path}")

    # Generate local PDF report
    generate_global_summary_report(df_global, local_pdf_path)

    # 2. Sync to Drive (if Drive is mounted and path is different)
    drive_data_dir = pm_global.data_root
    drive_plots_dir = pm_global.plots_root

    try:
        if drive_data_dir.resolve() != local_data_dir.resolve():
            drive_data_dir.mkdir(exist_ok=True, parents=True)
            shutil.copy(local_csv_path, drive_data_dir / local_csv_path.name)
            print_success(f"[+] Global summary CSV synced to Drive: {drive_data_dir / local_csv_path.name}")

        if drive_plots_dir.resolve() != local_plots_dir.resolve():
            drive_plots_dir.mkdir(exist_ok=True, parents=True)
            shutil.copy(local_pdf_path, drive_plots_dir / local_pdf_path.name)
            print_success(f"[+] Global summary PDF synced to Drive: {drive_plots_dir / local_pdf_path.name}")
    except Exception as e:
        print_error(f"Failed to sync global summary to Drive: {e}")
    
    print_info("\n[Global Report] Generation complete.")

if __name__ == "__main__":
    main()