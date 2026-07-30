import sys
import optuna
from pathlib import Path
import torch
import yaml
import argparse

# Ensure tools and CrossPINN root directories are importable
current_dir = Path(__file__).resolve().parent
root_dir = current_dir.parent.parent
tools_dir = current_dir.parent
if str(root_dir) not in sys.path:
    sys.path.append(str(root_dir))
if str(tools_dir) not in sys.path:
    sys.path.append(str(tools_dir))

from tool_utils import load_project_environment, generate_analysis_plots
from utils.logger import print_error, print_success, print_info

def run_analyze(study_or_target, quiet=False):
    """
    Programmatically loads a completed Optuna study from a .db file and generates
    analysis plots for hyperparameter importance, slices, and optimization history.
    """
    from config.study import Study
    from config.path_manager import PathManager
    from config.config import load_config

    if isinstance(study_or_target, Study):
        study_obj = study_or_target
    elif isinstance(study_or_target, PathManager):
        study_obj = Study(prob_id=study_or_target.prob_id, model=study_or_target.model, study_name=study_or_target.study_name, debug=study_or_target.debug)
    else:
        study_obj = Study.from_args(study_or_target)

    load_project_environment()

    pm = study_obj.pm
    try:
        base_c = load_config(study_obj, quiet=quiet)
        optuna_study_name = base_c['experiment']['name']
    except Exception as e:
        if not quiet:
            print_error(f"Error loading configuration for analysis: {e}")
        return None

    db_path = pm.tuning_db_path
    storage_url = f"sqlite:///{db_path.as_posix()}"

    if not db_path.exists():
        if not quiet:
            print_error(f"Optuna database not found at: {db_path}")
            print_error("Ensure you have run tuning for this configuration first.")
        return None

    try:
        study = optuna.load_study(study_name=optuna_study_name, storage=storage_url)
        if not quiet:
            print_success(f"Study '{study.study_name}' loaded successfully ({len(study.trials)} trials) from {db_path.name}.")
    except Exception as e:
        if not quiet:
            print_error(f"Could not load study '{optuna_study_name}' from database: {e}")
        return None

    if len(study.trials) == 0:
        if not quiet:
            print_info("The study contains no trials. No analysis to perform.")
        return study

    # Target centralized tuning PDF dir: plots/tuning/prob<id>/
    target_tuning_dir = pm.plots_root / "tuning" / f"prob{pm.prob_id}"
    generate_analysis_plots(
        study, 
        pm.tuning_dir, 
        plots_dir=pm.tuning_plots_dir / pm.study_base_name,
        target_tuning_pdf_dir=target_tuning_dir
    )
    if not quiet:
        print_success("Analysis complete.")

    return study

def analyze_all(debug=False):
    """
    Iterates through all studies defined in studies.yaml and runs run_analyze for each.
    """
    from config.path_manager import PathManager
    from config.study import Study
    
    print_info("\n[Analyze All] Starting batch analysis for all studies in studies.yaml...")
    pm = PathManager(debug=debug)
    studies_path = pm.studies_config_path

    if not studies_path.exists():
        print_error(f"Cannot run analyze_all: studies.yaml not found at {studies_path}")
        return

    with open(studies_path, 'r') as f:
        studies_config = yaml.safe_load(f) or {}

    total_studies = sum(len(v) for v in studies_config.values() if v)
    processed_count = 0

    for prob_key, study_names in studies_config.items():
        if not study_names:
            continue
        
        prob_id = int(prob_key.replace('prob', ''))
        print_info(f"\n--- Analyzing Problem: {prob_id} ---")

        for study_name in study_names:
            processed_count += 1
            print_info(f"[{processed_count}/{total_studies}] Analyzing study: {study_name}")
            
            try:
                # To get the model type, we must load the config from the checkpoint.
                temp_pm = PathManager(prob_id=prob_id, study_name=study_name, debug=debug)
                ckpt_path = temp_pm.get_weight_checkpoint_path(checkpoint_type="best")
                if not ckpt_path.exists():
                    ckpt_path = temp_pm.get_weight_checkpoint_path(checkpoint_type="latest")
                
                if not ckpt_path.exists():
                    print_error(f"  -> Checkpoint not found for '{study_name}'. Cannot determine model type for analysis. Skipping.")
                    continue

                checkpoint = torch.load(ckpt_path, map_location='cpu', weights_only=False)
                if not isinstance(checkpoint, dict) or 'config' not in checkpoint:
                    print_error(f"  -> Checkpoint for '{study_name}' is invalid or missing config. Skipping.")
                    continue
                
                model_type = checkpoint['config']['model']['type']

                study = Study(prob_id=prob_id, model=model_type, study_name=study_name, debug=debug)
                run_analyze(study, quiet=True)
                print_success(f"  -> Successfully generated analysis plots for '{study_name}'.")
            except Exception as e:
                print_error(f"  -> An error occurred while analyzing '{study_name}': {e}")
    
    print_info("\n[Analyze All] Batch analysis complete.")

def main():
    """
    Loads a completed Optuna study from a .db file and generates analysis plots
    for hyperparameter importance, slices, and optimization history via CLI.
    """
    parser = argparse.ArgumentParser(description="Analyze an Optuna study. Use --all or specify a single study.")
    parser.add_argument('--all', action='store_true', help='Analyze all studies defined in studies.yaml.')
    parser.add_argument('--prob_id', type=str, help='Problem ID for a single study (e.g., 1, 34).')
    parser.add_argument('--model', '--method', dest='model', type=str, help='Model type for a single study (e.g., tlp, mlp).')
    parser.add_argument('--study_name', type=str, default=None, help='Optional study name for a single study.')
    parser.add_argument('--debug', action='store_true', help='Use debug directories (data_debug, plots_debug).')
    args, _ = parser.parse_known_args()

    if args.all:
        analyze_all(debug=args.debug)
    elif args.prob_id and args.model:
        run_analyze(args)
    else:
        parser.print_help()
        print_error("\nError: You must either specify --all, or provide both --prob_id and --model for a single study.")
        sys.exit(1)

if __name__ == "__main__":
    main()
