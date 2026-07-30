import os
import sys
from pathlib import Path
from dotenv import load_dotenv
import torch

# Ensure tools directory is importable when scripts are run directly.
current_dir = Path(__file__).resolve().parent
if str(current_dir) not in sys.path:
    sys.path.append(str(current_dir))


from config.path_manager import PathManager

is_colab = PathManager.is_colab
load_project_environment = PathManager.load_project_environment
get_base_path = PathManager.get_base_path
get_data_root = PathManager.get_data_root
get_plots_root = PathManager.get_plots_root

def get_device():
    return torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def init_wandb_train(args, config, data_dir_or_pm):
    """Initializes a W&B run for training."""
    if not getattr(args, 'wandb', True):
        return None
    from config.path_manager import PathManager
    data_dir = data_dir_or_pm.problem_data_dir if isinstance(data_dir_or_pm, PathManager) else data_dir_or_pm
    import wandb
    from utils.logger import print_success, print_error
    try:
        run = wandb.init(
            project="CrossPINN",
            name=config['experiment']['name'],
            config=config,
            dir=str(data_dir),
            settings=wandb.Settings(silent=True)
        )
        print_success(f"[W&B] Run initialized. Tracking at: {run.url}")
        return run
    except Exception as e:
        print_error(f"[W&B] Failed to initialize Weights & Biases: {e}")
        return None

def init_wandb_tune_trial(trial, args, config, data_dir_or_pm):
    """Initializes a W&B run for a single tuning trial."""
    if not getattr(args, 'wandb', True):
        return None
    from config.path_manager import PathManager
    data_dir = data_dir_or_pm.tuning_dir if isinstance(data_dir_or_pm, PathManager) else data_dir_or_pm
    import wandb
    from utils.logger import print_error, print_success
    try:
        run = wandb.init(
            project="CrossPINN-Tuning",
            group=trial.study.study_name,
            name=f"trial_{trial.number}",
            config=config,
            dir=str(data_dir),
            save_code=False,
            settings=wandb.Settings(silent=True, start_method='thread')
        )
        print_success(f"[W&B] Trial {trial.number} tracking at: {run.url}")
        return run
    except Exception as e:
        print_error(f"[W&B] Failed to initialize W&B for trial {trial.number}: {e}")
        return None

def build_optimizer_and_scheduler(config, model):
    """Builds optimizer and scheduler based on the configuration."""
    from core.scheduler import get_scheduler
    
    opt_type = config['training']['optimizer']
    optimizer_lbfgs = None
    scheduler = None

    optimizer_params = {
        'lr': config['training'].get('learning_rate', 0.001),
        'weight_decay': config['training'].get('weight_decay', 1e-4)
    }

    if opt_type == 'Hybrid':
        optimizer = torch.optim.AdamW(model.parameters(), **optimizer_params)
        scheduler = get_scheduler(optimizer, config)
        optimizer_lbfgs = torch.optim.LBFGS(
            model.parameters(), lr=config['training'].get('learning_rate_lbfgs', 1.0),
            max_iter=1, max_eval=5, tolerance_grad=1e-7, tolerance_change=1e-9, history_size=100,
            line_search_fn="strong_wolfe"
        )
    elif opt_type == 'LBFGS':
        optimizer = torch.optim.LBFGS(
            model.parameters(), lr=config['training'].get('learning_rate', 1.0),
            max_iter=1, max_eval=5, tolerance_grad=1e-7, tolerance_change=1e-9, history_size=100,
            line_search_fn="strong_wolfe"
        )
    elif opt_type in ['AdamW', 'Adam']:
        optimizer = torch.optim.AdamW(model.parameters(), **optimizer_params)
        scheduler = get_scheduler(optimizer, config)
    else:
        raise ValueError(f"Unknown optimizer: {opt_type}")
        
    return optimizer, scheduler, optimizer_lbfgs

def load_states_from_checkpoint(checkpoint, model, optimizer, optimizer_lbfgs, device, config):
    """Loads states from a checkpoint object into the provided model and optimizers."""
    from utils.logger import print_success, print_info, print_error
    
    start_epoch = 1

    state_dict = checkpoint['state_dict'] if 'state_dict' in checkpoint else checkpoint
    model.load_state_dict(state_dict)
    
    if 'epoch' in checkpoint:
        start_epoch = checkpoint['epoch'] + 1

    if 'optimizer' in checkpoint:
        try:
            opt_type = config['training']['optimizer']
            opt_state = checkpoint['optimizer']
            is_lbfgs_ckpt = ('max_iter' in opt_state['param_groups'][0]) if (isinstance(opt_state, dict) and 'param_groups' in opt_state and opt_state['param_groups']) else False

            if opt_type == 'Hybrid':
                if is_lbfgs_ckpt and optimizer_lbfgs is not None:
                    optimizer_lbfgs.load_state_dict(opt_state)
                else:
                    optimizer.load_state_dict(opt_state)
            else:
                if is_lbfgs_ckpt and isinstance(optimizer, torch.optim.LBFGS):
                    optimizer.load_state_dict(opt_state)
                elif not is_lbfgs_ckpt and not isinstance(optimizer, torch.optim.LBFGS):
                    optimizer.load_state_dict(opt_state)
        except Exception as e:
            print_error(f"Warning: Could not load optimizer state: {e}")
        
    if start_epoch > config['training']['epochs']:
        print_info(f"Model already trained for {start_epoch - 1} epochs, which meets or exceeds the target. Skipping additional epochs.")

    return start_epoch

def generate_analysis_plots(study, output_dir, plots_dir=None, target_tuning_pdf_dir=None):
    """
    Generates and saves Optuna analysis plots for a given study.

    Args:
        study (optuna.Study): The completed Optuna study object.
        output_dir (Path): The fallback output directory.
        plots_dir (Path, optional): Explicit target directory for analysis plots.
        target_tuning_pdf_dir (Path, optional): Directory to copy the aggregated slice report PDF.
    """
    from utils.logger import print_error, print_success, print_info
    import matplotlib.pyplot as plt
    import numpy as np
    import optuna
    import warnings
    import subprocess
    import shutil

    print_info("\n[Optuna] Generating analysis plots...")
    try:
        from optuna.visualization.matplotlib import (
            plot_optimization_history,
            plot_slice,
            plot_param_importances,
            plot_parallel_coordinate
        )
        from optuna.importance import get_param_importances
    except ImportError:
        print_error("\n[!] Could not generate analysis plots. Missing libraries ('matplotlib', 'numpy', 'optuna').")
        return

    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=optuna.exceptions.ExperimentalWarning)
        warnings.filterwarnings("ignore", category=UserWarning, message=".*tight_layout.*")
        warnings.filterwarnings("ignore", category=UserWarning, message=".*set identical low and high ylims.*")

        target_plots_dir = Path(plots_dir) if plots_dir else Path(output_dir) / "analysis_plots"
        target_plots_dir.mkdir(exist_ok=True, parents=True)
        plots_dir = target_plots_dir
        print_info(f"Saving analysis plots to: {plots_dir}")

        if len(study.trials) == 0:
            print_info("  -> No trials in study. Skipping plot generation.")
            return

        # --- 1. Generate combined plots ---
        print_info("  -> Generating combined analysis plots...")
        plots_to_generate = {
            "optimization_history": plot_optimization_history,
            "param_importances": plot_param_importances,
            # "slice": plot_slice, # This is now handled by the PDF report and individual PNGs
            "parallel_coordinate": plot_parallel_coordinate,
        }
        for name, plot_func in plots_to_generate.items():
            try:
                plot_obj = plot_func(study)
                if isinstance(plot_obj, np.ndarray):
                    fig = plot_obj.ravel()[0].figure
                elif hasattr(plot_obj, 'figure'):
                    fig = plot_obj.figure
                else:
                    fig = plot_obj
                fig.tight_layout()
                save_path = plots_dir / f"{name}.png"
                fig.savefig(save_path, dpi=150)
                plt.close(fig)
                print_success(f"     -> Saved combined plot: {name}.png")
            except (ValueError, TypeError) as e:
                print_error(f"     -> Could not generate combined plot '{name}': {e}")

        # --- 3. Generate individual slice plots (as PNGs) ---
        print_info("  -> Generating individual slice plots...")
        slices_dir = plots_dir / "slices"
        # Clean up old slices before generating new ones
        if slices_dir.exists():
            shutil.rmtree(slices_dir)
        slices_dir.mkdir(exist_ok=True)

        try:
            completed_trials = study.get_trials(deepcopy=False, states=(optuna.trial.TrialState.COMPLETE,))
            if not completed_trials:
                print_info("     -> No completed trials found. Skipping individual slice plots.")
                return

            # Get all unique parameter names from all completed trials to ensure none are missed.
            all_params_in_study = set()
            for t in completed_trials:
                all_params_in_study.update(t.params.keys())

            if not all_params_in_study:
                print_info("     -> No parameters found in completed trials. Skipping.")
                return

            # Get importances, sort params by importance, then append params with no calculated importance.
            importances = get_param_importances(study, target=lambda t: t.values[0])
            sorted_important_params = sorted(importances.keys(), key=lambda p: importances[p], reverse=True)
            
            zero_importance_params = sorted(list(all_params_in_study - set(sorted_important_params)))
            all_params_to_plot = sorted_important_params + zero_importance_params

            for i, param_name in enumerate(all_params_to_plot):
                ax = plot_slice(study, params=[param_name])
                fig = ax.figure
                fig.set_size_inches(10, 7) # Make individual plots larger
                fig.tight_layout()
                sanitized_name = param_name.replace('/', '_')
                save_path = slices_dir / f"{i+1:02d}_{sanitized_name}.png"
                fig.savefig(save_path, dpi=150)
                plt.close(fig)
            print_success(f"  -> Individual slice plots saved to: {slices_dir}")
        except Exception as e:
            print_error(f"  -> Failed to generate individual slice plots: {e}")

        # --- 4. Generate multi-page PDF for slice plots via LaTeX ---
        print_info("  -> Generating multi-page slice report PDF via LaTeX...")
        slice_pngs = sorted(list(slices_dir.glob("*.png")))

        if not slice_pngs:
            print_info("     -> No slice PNGs found. Skipping PDF report generation.")
            return

        if shutil.which("pdflatex") is None:
            print_error("     -> 'pdflatex' command not found in system PATH. Cannot generate PDF report.")
            print_error("     -> Please install a LaTeX distribution (like MiKTeX) and ensure it's in your PATH.")
            return

        latex_template = r"""
\documentclass[11pt,a4paper]{article}
\usepackage[utf8]{inputenc}
\usepackage[T1]{fontenc}
\usepackage{graphicx}
\usepackage[margin=0.6in, top=0.8in, bottom=0.8in]{geometry}
\usepackage{subcaption}
\usepackage{hyperref}
\usepackage{titling}

\pretitle{\begin{center}\LARGE\bfseries}
\posttitle{\par\end{center}}
\preauthor{\begin{center}\large}
\postauthor{\par\end{center}}
\predate{\begin{center}\large}
\postdate{\par\end{center}}

\begin{document}
\title{Hyperparameter Slices Report}
\author{Optuna Study: %s}
\date{\today}
\maketitle

%s

\end{document}
"""
        content = ""
        params_per_page = 4
        for i in range(0, len(slice_pngs), params_per_page):
            chunk = slice_pngs[i:i + params_per_page]
            content += r"\begin{figure}[p!]" + "\n\\centering\n"
            for j, png_path in enumerate(chunk):
                relative_path = png_path.relative_to(plots_dir).as_posix()
                param_name = png_path.stem[3:].replace('_', r'\_')
                content += r"\begin{subfigure}{0.49\textwidth}\centering" + f"\n\\includegraphics[width=\\textwidth]{{{relative_path}}}" + f"\n\\caption{{{param_name}}}" + r"\end{subfigure}" + ("~" if j % 2 == 0 else "") + "\n"
            content += r"\caption{Slice plots for hyperparameters, ordered by importance.}\end{figure}" + "\n\\clearpage\n\n"

        pdf_stem = f"{study.study_name}_slice_report"
        final_latex_doc = latex_template % (study.study_name.replace('_', r'\_'), content)
        tex_file_path = plots_dir / f"{pdf_stem}.tex"
        with open(tex_file_path, 'w') as f: f.write(final_latex_doc)

        try:
            subprocess.run(['pdflatex', '-interaction=nonstopmode', tex_file_path.name], cwd=plots_dir, check=True, capture_output=True, text=True)
            compiled_pdf = plots_dir / f"{pdf_stem}.pdf"
            print_success(f"     -> Successfully compiled slice report: {compiled_pdf.name}")

            if target_tuning_pdf_dir:
                target_tuning_pdf_dir = Path(target_tuning_pdf_dir)
                target_tuning_pdf_dir.mkdir(parents=True, exist_ok=True)
                shutil.copy(compiled_pdf, target_tuning_pdf_dir / compiled_pdf.name)
                print_success(f"     -> Copied slice report PDF to: {target_tuning_pdf_dir / compiled_pdf.name}")
        except subprocess.CalledProcessError as e:
            print_error(f"     -> pdflatex compilation failed. Log file saved in {plots_dir}")
            with open(plots_dir / 'pdflatex_error.log', 'w') as log_file: log_file.write(e.stdout + "\n" + e.stderr)
        finally:
            for ext in ['.tex', '.aux', '.log', '.out']:
                target_aux = plots_dir / f"{pdf_stem}{ext}"
                if target_aux.exists(): target_aux.unlink()

def sync_file_to_github(file_path, commit_msg):
    """No-op stub for public repository compatibility."""
    pass

