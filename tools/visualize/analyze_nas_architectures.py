import sys
import yaml
import torch
import numpy as np
import pandas as pd
from pathlib import Path
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
import argparse

# Ensure CrossPINN root is in sys.path
current_dir = Path(__file__).resolve().parent
root_dir = current_dir.parent.parent
if str(root_dir) not in sys.path:
    sys.path.append(str(root_dir))

from config.path_manager import PathManager
from utils.logger import print_info, print_success, print_error
from tools.visualize.plot_style import apply_global_plot_style, save_and_close_figure

def load_study_config(prob_id, study_name, debug=False):
    """Loads configuration for a study using PathManager with checkpoint fallback."""
    pm = PathManager(prob_id=prob_id, study_name=study_name, debug=debug)
    
    # 1. Try reading study config YAML directly
    if pm.study_config_path.exists():
        try:
            with open(pm.study_config_path, 'r') as f:
                cfg = yaml.safe_load(f)
                if cfg and isinstance(cfg, dict) and 'model' in cfg:
                    return cfg, pm.study_config_path
        except Exception:
            pass

    # 2. Fallback to reading config from model checkpoint
    best_pt = pm.get_weight_checkpoint_path(checkpoint_type="best")
    if not best_pt.exists():
        best_pt = pm.get_weight_checkpoint_path(checkpoint_type="latest")

    if best_pt.exists():
        try:
            ckpt = torch.load(best_pt, map_location='cpu', weights_only=False)
            if isinstance(ckpt, dict) and 'config' in ckpt:
                return ckpt['config'], best_pt
        except Exception:
            pass

    return None, None

def extract_nas_parameters(studies_by_prob, debug=False):
    """Extracts all NAS architecture and training parameters from search_space.yaml across TLP study configurations."""
    records = []

    for prob_key, study_list in studies_by_prob.items():
        if not study_list:
            continue
        prob_id = int(prob_key.replace('prob', ''))

        for study_name in study_list:
            # Skip non-TLP baseline models (like MLP or RISN)
            if study_name.startswith("MLP_") or study_name.startswith("RISN_"):
                continue

            cfg, source_path = load_study_config(prob_id, study_name, debug=debug)
            if not cfg:
                print_info(f"Could not load config for {study_name} (Prob {prob_id}). Skipping.")
                continue

            model_cfg = cfg.get('model', {})
            training_cfg = cfg.get('training', {})
            data_cfg = cfg.get('data', {})
            exp_cfg = cfg.get('experiment', {})

            global_skip = model_cfg.get('global_skip', False)
            best_mae = exp_cfg.get('best_mae', float('nan'))

            record = {
                'Problem ID': prob_id,
                'Study Name': study_name,
                'Global Skip': global_skip,
                'Activation': model_cfg.get('activation', 'N/A'),
                'Pos Encoder': model_cfg.get('pos_encoder_type', 'N/A'),
                'Siren Omega': f"{model_cfg.get('siren_omega', 0.0):.1f}" if model_cfg.get('pos_encoder_type') == 'siren' and isinstance(model_cfg.get('siren_omega'), (int, float)) else 'N/A',
                'Hidden Feat': model_cfg.get('hidden_features', 'N/A'),
                'Heads': model_cfg.get('num_heads', 'N/A'),
                'Layers': model_cfg.get('num_decoder_layers', 'N/A'),
                'Support Pts': model_cfg.get('num_support_points', 'N/A'),
                'LinNorm': model_cfg.get('LinNorm', False),
                'Residual Norm': model_cfg.get('residual_norm', False),
                'Skip Pretrain': model_cfg.get('skip_use_pretrained_mlp', False) if global_skip else 'N/A',
                'Skip Trainable': model_cfg.get('skip_mlp_trainable', False) if global_skip else 'N/A',
                'Learn Mem Val': model_cfg.get('learn_mem_val', False),
                'Beta Penalty': model_cfg.get('beta_penalty', 'N/A'),
                'Warm Start': training_cfg.get('warm_start', False),
                'Unfreeze Epoch': training_cfg.get('warm_start_unfreeze_epoch', 'N/A') if training_cfg.get('warm_start') else 'N/A',
                'Learning Rate': f"{training_cfg.get('learning_rate', 0.0):.2e}" if isinstance(training_cfg.get('learning_rate'), (int, float)) else 'N/A',
                'Weight Decay': f"{training_cfg.get('weight_decay', 0.0):.2e}" if isinstance(training_cfg.get('weight_decay'), (int, float)) else 'N/A',
                'Warmup Epochs': training_cfg.get('warmup_epochs', 'N/A'),
                'Colloc Pts': data_cfg.get('num_collocation_points_adam', 'N/A'),
                'Boundary Pts': data_cfg.get('num_boundary_points_adam', 'N/A'),
                'Best MAE': best_mae,
                'Config Source': source_path.name if source_path else 'N/A'
            }
            records.append(record)

    return pd.DataFrame(records)

def create_styled_table_figure(df_group, title):
    """Renders a styled Matplotlib summary table figure for all NAS parameters from search_space.yaml."""
    apply_global_plot_style()
    if df_group.empty:
        return None

    display_cols = ['Problem ID', 'Activation', 'Pos Encoder', 'Hidden Feat', 'Heads', 'Layers', 'Support Pts', 'LinNorm', 'Skip Pretrain', 'Learn Mem Val', 'Warm Start', 'Learning Rate', 'Colloc Pts', 'Best MAE']
    display_df = df_group[display_cols].copy()
    
    # Format MAE column for table
    display_df['Best MAE'] = display_df['Best MAE'].apply(lambda val: f"{val:.2e}" if isinstance(val, (int, float)) and not np.isnan(val) else "N/A")

    fig, ax = plt.subplots(figsize=(18, max(2.5, 0.75 * len(display_df) + 1.5)))
    ax.axis('off')

    table = ax.table(
        cellText=display_df.values,
        colLabels=display_cols,
        loc='center',
        cellLoc='center'
    )
    table.auto_set_font_size(False)
    table.set_fontsize(9)
    table.scale(1, 2.0)

    # Header styling
    for key, cell in table.get_celld().items():
        row_idx, col_idx = key
        if row_idx == 0:
            cell.set_text_props(weight='bold', color='white')
            cell.set_facecolor('#2c3e50')
        else:
            if row_idx % 2 == 0:
                cell.set_facecolor('#f8f9fa')
            else:
                cell.set_facecolor('#ffffff')

    plt.title(title, fontsize=14, pad=15, weight='bold', color='#1a252f')
    return fig

def plot_parameter_histograms(df, title_suffix=""):
    """Generates a 3x4 grid of bar charts summarizing frequencies of all search_space.yaml parameters across problems."""
    apply_global_plot_style()
    if df.empty:
        return None

    fig, axes = plt.subplots(3, 4, figsize=(18, 12))
    fig.suptitle(f"Optuna NAS Architecture Search Choice Frequency{title_suffix}", fontsize=16, weight='bold', y=0.99)

    params_to_plot = [
        ('Activation', 'Activation Function', '#3498db'),
        ('Pos Encoder', 'Positional Encoder Type', '#8e44ad'),
        ('Hidden Feat', 'Latent Hidden Features', '#9b59b6'),
        ('Heads', 'Attention Heads', '#2980b9'),
        ('Layers', 'Decoder Layers', '#f39c12'),
        ('Support Pts', 'Support Points', '#d35400'),
        ('LinNorm', 'LinNorm Enabled', '#e74c3c'),
        ('Learn Mem Val', 'Learn Memory Value', '#1abc9c'),
        ('Warm Start', 'Warm Start Enabled', '#2ecc71'),
        ('Skip Pretrain', 'Skip Pretrained MLP', '#27ae60'),
        ('Skip Trainable', 'Skip MLP Trainable', '#16a085'),
        ('Colloc Pts', 'Collocation Points (Adam)', '#e67e22'),
    ]

    for idx, (col_name, plot_title, color) in enumerate(params_to_plot):
        r, c = divmod(idx, 4)
        ax = axes[r, c]
        counts = df[col_name].astype(str).value_counts()
        counts.plot(kind='bar', ax=ax, color=color, edgecolor='black', alpha=0.85)
        ax.set_title(plot_title, fontsize=11, weight='bold')
        ax.set_ylabel("Count")
        ax.grid(axis='y', linestyle='--', alpha=0.7)
        ax.tick_params(axis='x', rotation=30)

    plt.tight_layout(rect=[0, 0.02, 1, 0.96])
    return fig

def run_nas_analysis(debug=False):
    """Executes the NAS architectural analysis and exports tables, charts, and PDF reports."""
    print_info("=== [NAS ANALYSIS] Extracting Optuna Architecture Choices ===")
    pm = PathManager(debug=debug)
    studies_path = pm.studies_config_path

    if not studies_path.exists():
        print_error(f"Cannot run NAS analysis: studies.yaml not found at {studies_path}")
        return

    with open(studies_path, 'r') as f:
        studies_config = yaml.safe_load(f) or {}

    df = extract_nas_parameters(studies_config, debug=debug)

    if df.empty:
        print_error("No TLP architectural records found for analysis.")
        return

    output_dir = pm.plots_root / "nas_analysis"
    output_dir.mkdir(parents=True, exist_ok=True)

    print_success(f"Successfully extracted NAS parameters for {len(df)} studies across problems.")

    # Divide by Global Skip
    df_skip_true = df[df['Global Skip'] == True].copy()
    df_skip_false = df[df['Global Skip'] == False].copy()

    # Save CSVs
    df.to_csv(output_dir / "nas_architecture_all.csv", index=False)
    df_skip_true.to_csv(output_dir / "nas_architecture_skip_true.csv", index=False)
    df_skip_false.to_csv(output_dir / "nas_architecture_skip_false.csv", index=False)

    collected_figs = []

    # 1. Summary Table: Global Skip = True
    fig_table_true = create_styled_table_figure(df_skip_true, "NAS Architecture Choices: C-PINN (Global Skip = True)")
    if fig_table_true:
        fig_table_true.savefig(output_dir / "nas_summary_skip_true.png", dpi=300, bbox_inches='tight')
        collected_figs.append(fig_table_true)

    # 2. Histograms: Global Skip = True
    fig_hist_true = plot_parameter_histograms(df_skip_true, title_suffix=" - C-PINN (Global Skip = True)")
    if fig_hist_true:
        fig_hist_true.savefig(output_dir / "nas_parameter_distributions_skip_true.png", dpi=300, bbox_inches='tight')
        collected_figs.append(fig_hist_true)

    # 3. Summary Table: Global Skip = False
    fig_table_false = create_styled_table_figure(df_skip_false, "NAS Architecture Choices: C-PINN (Global Skip = False)")
    if fig_table_false:
        fig_table_false.savefig(output_dir / "nas_summary_skip_false.png", dpi=300, bbox_inches='tight')
        collected_figs.append(fig_table_false)

    # 4. Histograms: Global Skip = False
    fig_hist_false = plot_parameter_histograms(df_skip_false, title_suffix=" - C-PINN (Global Skip = False)")
    if fig_hist_false:
        fig_hist_false.savefig(output_dir / "nas_parameter_distributions_skip_false.png", dpi=300, bbox_inches='tight')
        collected_figs.append(fig_hist_false)

    # 5. Combined Histograms (All Studies)
    fig_hist_all = plot_parameter_histograms(df, title_suffix=" - All Studies Combined")
    if fig_hist_all:
        fig_hist_all.savefig(output_dir / "nas_parameter_distributions_all.png", dpi=300, bbox_inches='tight')
        collected_figs.append(fig_hist_all)

    # 6. Compile PDF Report
    pdf_path = output_dir / "nas_architectures_report.pdf"
    try:
        with PdfPages(pdf_path) as pdf:
            for fig in collected_figs:
                pdf.savefig(fig, bbox_inches='tight')
        print_success(f"[+] Generated NAS Architecture PDF Report: {pdf_path}")
    except Exception as e:
        print_error(f"Failed to generate PDF report: {e}")

    for fig in collected_figs:
        plt.close(fig)

    # Print summary insights to terminal
    print_info("\n--- [NAS SEARCH INSIGHTS SUMMARY] ---")
    print_info(f"Total TLP Studies Analyzed: {len(df)}")
    print_info(f"Top Activation Choice:       {df['Activation'].mode()[0]} ({ (df['Activation'] == df['Activation'].mode()[0]).mean()*100:.1f}% of studies)")
    print_info(f"LinNorm Disabled (False):    {(df['LinNorm'] == False).mean()*100:.1f}% of studies")
    print_info(f"Warm Start Enabled (True):   {(df['Warm Start'] == True).mean()*100:.1f}% of studies")
    print_info(f"Learn Mem Val Enabled:      {(df['Learn Mem Val'] == True).mean()*100:.1f}% of studies")
    print_info(f"Most Frequent Hidden Feat:   {df['Hidden Feat'].mode()[0]}")
    print_info(f"Outputs saved to: {output_dir}\n")

def main():
    parser = argparse.ArgumentParser(description="Analyze Optuna NAS architecture choices across benchmark problems.")
    parser.add_argument('--debug', action='store_true', help='Use debug directories.')
    args = parser.parse_args()
    run_nas_analysis(debug=args.debug)

if __name__ == '__main__':
    main()
