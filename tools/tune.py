import os
import sys
import optuna
import torch
import yaml
import warnings
import shutil
import time
from pathlib import Path

# Ensure CrossPINN modules are importable
current_dir = Path(__file__).resolve().parent
root_dir = current_dir.parent
if str(root_dir) not in sys.path:
    sys.path.append(str(root_dir))
if str(current_dir) not in sys.path:
    sys.path.append(str(current_dir))

from tool_utils import load_project_environment, get_device, init_wandb_tune_trial, build_optimizer_and_scheduler
from utils.logger import print_error, print_success, print_info

warnings.filterwarnings("ignore", category=FutureWarning)

from core.trainer import Trainer
from models.factory import build_model
from losses.pde_loss import VolterraPDELoss
from equations.registry import get_problem
from config.config import load_config, apply_debug_overrides, apply_cli_overrides

def clear_tuning_artifacts(args, base_c):
    from config.path_manager import PathManager
    pm = PathManager.from_args(args)

    # 1. Remove tuning artifacts
    tuning_dir = pm.tuning_dir
    if tuning_dir.exists():
        shutil.rmtree(tuning_dir)
        print_success(f"[RESET] Removed tuning artifacts: {tuning_dir}")
    else:
        print_info(f"[RESET] No tuning artifacts found at: {tuning_dir}")

    # 2. Remove the local study config file
    specific_config_path = pm.study_config_path
    if specific_config_path.exists():
        try:
            specific_config_path.unlink()
            print_success(f"[RESET] Removed local specific config file: {specific_config_path.name}")
        except Exception as e:
            print_error(f"Failed to remove local config {specific_config_path.name}: {e}")
    else:
        print_info(f"[RESET] Local specific config file not found: {specific_config_path.name}")

def save_best_configuration(trial, value, tuned_config, args, search_space, original_epochs=None, trial_weights_path=None):
    """
    Saves the best-performing configuration and model weights (.pt) to the local study file and Google Drive backup.
    If --export_to_github is enabled, it also pushes the updated config file to Git.
    """
    import copy
    from config.path_manager import PathManager
    pm = PathManager.from_args(args)
    local_study_path = pm.study_config_path

    best_config = copy.deepcopy(tuned_config)
    best_config['experiment']['best_mae'] = float(value)

    # Restore original epochs if they were overridden for tuning.
    if original_epochs is not None:
        if 'training' not in best_config:
            best_config['training'] = {}
        best_config['training']['epochs'] = original_epochs
        print_info(f"Restoring original epochs to {original_epochs} in the saved config file.")

    # --- 1. Save to local repository file (always) ---
    try:
        local_study_path.parent.mkdir(exist_ok=True, parents=True)
        with open(local_study_path, 'w') as f:
            yaml.dump(best_config, f, sort_keys=False)
        print_success(f"\n[+] Saved updated configuration to local file: {local_study_path.name}")
    except Exception as e:
        print_error(f"CRITICAL: Failed to save configuration to local file: {e}")
        raise IOError(f"Failed to write best configuration to {local_study_path}. Check permissions or disk space.") from e

    # --- 2. Save best model weights (.pt) if available ---
    if trial_weights_path and Path(trial_weights_path).exists():
        try:
            target_weights_path = pm.get_weight_checkpoint_path(checkpoint_type="best")
            target_weights_path.parent.mkdir(exist_ok=True, parents=True)
            shutil.copy(trial_weights_path, target_weights_path)
            print_success(f"[+] Saved updated best model weights to: {target_weights_path}")
        except Exception as e:
            print_error(f"Failed to save best model weights: {e}")

    # --- 3. Save backup copy to tuning directory ---
    DATA_DIR = pm.tuning_dir
    try:
        DATA_DIR.mkdir(exist_ok=True, parents=True)
        backup_path = DATA_DIR / local_study_path.name
        shutil.copy(local_study_path, backup_path)
        print_success(f"[+] Saved study configuration backup to: {backup_path}")
    except Exception as e:
        print_error(f"Failed to save configuration backup: {e}")


def suggest_hyperparams(trial, base_config, model_search_space, overrides=None):
    """
    Suggests hyperparameters for a given trial based on the search space,
    updates the config dictionary, handles constraints, and respects CLI overrides.
    """
    import copy
    import optuna
    config = copy.deepcopy(base_config)

    override_config = apply_cli_overrides(copy.deepcopy(base_config), overrides, quiet=True) if overrides else {}

    for section, params in model_search_space.items():
        if section not in config:
            config[section] = {}
            
        for param_name, param_config in params.items():
            # If parameter is explicitly overridden via CLI, fix its value and skip Optuna sampling
            if section in override_config and param_name in override_config.get(section, {}) and override_config[section][param_name] != base_config.get(section, {}).get(param_name):
                config[section][param_name] = override_config[section][param_name]
                continue

            if 'condition' in param_config:
                cond = param_config['condition']
                condition_param_name = cond['param']

                found_section = next((s for s, p_dict in model_search_space.items() if condition_param_name in p_dict), None)
                if not found_section:
                    found_section = next((s for s, p_dict in base_config.items() if isinstance(p_dict, dict) and condition_param_name in p_dict), None)

                if not found_section:
                    raise ValueError(f"Conditional parameter '{condition_param_name}' for '{param_name}' not found in any section of the search space or base config.")

                if config.get(found_section, {}).get(condition_param_name) != cond['value']:
                    continue
                    
            ptype = param_config['type']
            val = None
            
            if ptype == 'float':
                val = trial.suggest_float(param_name, param_config['low'], param_config['high'], log=param_config.get('log', False))
            elif ptype == 'int':
                val = trial.suggest_int(param_name, param_config['low'], param_config['high'])
            elif ptype == 'categorical':
                choices = param_config['choices']
                if hasattr(trial, 'study') and trial.study and trial.study.trials:
                    for prev_t in reversed(trial.study.trials):
                        if prev_t.number != trial.number and param_name in prev_t.distributions:
                            prev_dist = prev_t.distributions[param_name]
                            if isinstance(prev_dist, optuna.distributions.CategoricalDistribution):
                                choices = list(prev_dist.choices)
                                break

                val = trial.suggest_categorical(param_name, choices)
                constraint = param_config.get('constraint')
                if constraint and constraint.endswith('_modulo'):
                    constrained_by_param_name = constraint.replace('_modulo', '')
                    found_section = next((s for s, p in model_search_space.items() if constrained_by_param_name in p), None)
                    if found_section:
                        constrained_by_val = config[found_section].get(constrained_by_param_name)
                        if constrained_by_val is not None and constrained_by_val % val != 0:
                            raise optuna.TrialPruned(f"Constraint violation: {constrained_by_param_name} ({constrained_by_val}) % {param_name} ({val}) != 0")
            if 'prune_if' in param_config and val in param_config['prune_if']:
                raise optuna.TrialPruned(f"Skipping {val} for {param_name} to save memory/compute units.")
            config[section][param_name] = val

    if overrides:
        config = apply_cli_overrides(config, overrides, quiet=True)

    return config

def objective(trial, base_config, search_space, total_trials, current_count, args, overrides=None):
    model_type = base_config['model']['type']

    # Device
    device = get_device()

    # Suggest HPs and get a new config for this trial (respecting CLI overrides)
    config = suggest_hyperparams(trial, base_config, search_space, overrides=overrides)
            
    is_debug = getattr(args, 'debug', False)
    if is_debug:
        config = apply_debug_overrides(config, args)

    # Set up problem
    prob_id = config['data']['problem_id']
    a, b = config['data']['domain']
    problem = get_problem(prob_id, a=a, b=b)

    # Prepare DATA_DIR and paths via PathManager
    from config.path_manager import PathManager
    pm = PathManager.from_args(args)
    DATA_DIR = pm.tuning_dir
    
    # W&B Initialization for the trial
    wandb_run = init_wandb_tune_trial(trial, args, config, DATA_DIR)

    # Set up model
    model = build_model(
        config=config, problem=problem, device=device,
        mlp_save_dir=pm.weights_dir, debug=is_debug, quiet=True
    )

    # Set up Loss
    pde_loss = VolterraPDELoss(
        problem=problem,
        num_quadrature_nodes=config['data']['num_quadrature_nodes'],
        device=device
    )

    # Set up Optimizer and Scheduler dynamically via tool_utils
    optimizer, scheduler, optimizer_lbfgs = build_optimizer_and_scheduler(config, model)

    # Set up Trainer
    trainer = Trainer(
        model=model,
        problem=problem,
        pde_loss=pde_loss,
        optimizer=optimizer,
        config=config,
        data_dir=DATA_DIR,
        scheduler=scheduler,
        optimizer_lbfgs=optimizer_lbfgs,
        wandb_run=wandb_run
    )
    
    import gc
    try:
        best_mae = trainer.train(start_epoch=1, trial=trial, total_trials=total_trials, current_trial_count=current_count)
        if wandb_run:
            wandb_run.log({"final_mae": best_mae})
        return best_mae, config
    except (torch.cuda.OutOfMemoryError, torch.OutOfMemoryError) as e:
        print_error(f"\n[Optuna] CUDA Out of Memory on trial {trial.number}. Pruning trial to teach TPE to avoid this region.")
        if wandb_run:
            wandb_run.summary['state'] = 'oom_pruned'
        del trainer, model, optimizer, scheduler, problem
        torch.cuda.empty_cache()
        gc.collect()
        raise optuna.TrialPruned("CUDA Out of Memory") from None
    except optuna.TrialPruned as e:
        if wandb_run:
            wandb_run.summary['state'] = 'pruned'
        raise e
    finally:
        if wandb_run:
            wandb_run.finish(exit_code=0)
        try:
            del trainer, model, optimizer, scheduler, problem
        except UnboundLocalError:
            pass
        torch.cuda.empty_cache()
        gc.collect()
     
def run_tune(study_or_target, n_trials=50, reset=False, wandb=True, export_to_github=True, continue_tune=False):
    from config.study import Study
    from config.path_manager import PathManager

    if isinstance(study_or_target, Study):
        study_obj = study_or_target
    elif isinstance(study_or_target, PathManager):
        study_obj = Study(prob_id=study_or_target.prob_id, model=study_or_target.model, study_name=study_or_target.study_name, debug=study_or_target.debug)
    else:
        study_obj = Study.from_args(study_or_target)

    class SyntheticArgs:
        def __init__(self, s, n_tr, r, w, ex_gh, c_t):
            self.prob_id = s.prob_id
            self.model = s.model
            self.study_name = s.study_name
            self.debug = s.debug
            self.n_trials = n_tr
            self.reset = r
            self.wandb = w
            self.export_to_github = ex_gh
            self.continue_tune = c_t
            self.config_paths = s.pm

    args = SyntheticArgs(study_obj, n_trials, reset, wandb, export_to_github, continue_tune)
    overrides = study_obj.overrides
    pm = study_obj.pm

    start_time = time.time()
    os.environ['WANDB_SILENT'] = "true"

    load_project_environment()
    base_c = load_config(pm, quiet=True)

    base_c = apply_cli_overrides(base_c, overrides, quiet=True)

    if args.reset:
        print_info("--- RESET MODE ENABLED FOR TUNING: Clearing previous artifacts ---")
        clear_tuning_artifacts(args, base_c)

    original_epochs = None
    if not args.debug:
        original_epochs = base_c.get('training', {}).get('epochs')
        if original_epochs != 10000:
            base_c['training']['epochs'] = 10000
            print_info(f"[Tuning Override] Forcing training epochs to 10000 (original was: {original_epochs}). This will be reverted upon saving the best config.")

    search_space_path = pm.search_space_config_path
    with open(search_space_path, 'r') as f:
        full_search_space = yaml.safe_load(f)

    model_type = base_c['model']['type']
    if model_type not in full_search_space:
        raise ValueError(f"Model type '{model_type}' not found in search_space.yaml")
    model_search_space = full_search_space[model_type]

    if args.debug:
        print_info("--- DEBUG MODE ENABLED FOR OPTUNA ---")

    def wrapped_objective(trial, current_count):
        return objective(trial, base_c, model_search_space, args.n_trials, current_count, args, overrides=overrides)

    DATA_DIR = pm.tuning_dir
    optuna_study_name = base_c['experiment']['name']

    DATA_DIR.mkdir(exist_ok=True, parents=True)
    db_path = pm.tuning_db_path
    storage_url = f"sqlite:///{db_path.as_posix()}"

    if db_path.exists():
        print_success(f"\n[Optuna] Found existing SQLite database at {db_path}.")
        print_success(f"[Optuna] Resuming study '{optuna_study_name}'. Previous trial memory is fully preserved!")
    else:
        print_success(f"\n[Optuna] No SQLite database found at {db_path}.")
        print_success(f"[Optuna] Creating a new study '{optuna_study_name}' from scratch.")

    optuna.logging.set_verbosity(optuna.logging.WARNING)
    study = optuna.create_study(
        study_name=optuna_study_name,
        direction='minimize',
        storage=storage_url,
        load_if_exists=True,
        pruner=optuna.pruners.MedianPruner(
            n_startup_trials=5,
            n_warmup_steps=500,
            interval_steps=100
        )
    )

    if db_path.exists():
        num_past_trials = len(study.trials)
        completed_past_trials = len([t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE])
        print_success(f"[Optuna] Successfully loaded {num_past_trials} previous trials ({completed_past_trials} completed) from database.")
        
        if completed_past_trials == 0 and num_past_trials >= args.n_trials and not args.reset:
            print_error(f"[Optuna] Warning: All {num_past_trials} previous trials in database were pruned/failed (0 completed). Auto-resetting study database.")
            clear_tuning_artifacts(args, base_c)
            study = optuna.create_study(
                study_name=optuna_study_name,
                direction='minimize',
                storage=storage_url,
                load_if_exists=True,
                pruner=optuna.pruners.MedianPruner(
                    n_startup_trials=5,
                    n_warmup_steps=500,
                    interval_steps=100
                )
            )
            num_past_trials = 0
    else:
        num_past_trials = 0

    trials_to_run = args.n_trials
    if not args.continue_tune:
        trials_to_run = max(0, args.n_trials - num_past_trials)
        print_info(f"[Optuna] continue=False: running {trials_to_run} additional trials to reach {args.n_trials} total.")
    else:
        print_info(f"[Optuna] continue=True: running {trials_to_run} additional trials.")

    try:
        completed_trials = 0
        best_mae_to_beat = float('inf')
        local_study_path = pm.study_config_path
        try:
            if local_study_path.exists():
                with open(local_study_path, 'r') as f:
                    old_c = yaml.safe_load(f)
                best_mae_to_beat = old_c.get('experiment', {}).get('best_mae', float('inf'))

                if best_mae_to_beat != float('inf'):
                    print_success(f"\n[i] Loaded historical record from '{local_study_path.name}'. MAE to beat: {best_mae_to_beat:.4e}")
                else:
                    print_info(f"\n[i] Historical config '{local_study_path.name}' found, but no 'best_mae' value. Any result will be a new record.")
            else:
                print_success(f"\n[i] No historical config found at '{local_study_path.name}'. Any result will be a new record.")
        except Exception as e:
            print_error(f"\n[!] Warning while looking for historical config: {e}")
            pass

        initial_best_mae = best_mae_to_beat

        while completed_trials < trials_to_run:
            trial = study.ask()
            try:
                value, tuned_config = wrapped_objective(trial, completed_trials + 1)
                study.tell(trial, value)
                completed_trials += 1

                if value < best_mae_to_beat:
                    best_mae_to_beat = value
                    print_info(f"\n[+] New global best MAE: {value:.4e} at trial {completed_trials} (Study Trial ID: {trial.number}). Saving configuration and best model weights...")
                    trial_weights_path = DATA_DIR / "weights" / f"{tuned_config['experiment']['name']}_best_model.pt"
                    save_best_configuration(trial, value, tuned_config, args, model_search_space, original_epochs, trial_weights_path=trial_weights_path)
            except optuna.TrialPruned:
                study.tell(trial, state=optuna.trial.TrialState.PRUNED)
                completed_trials += 1
    except KeyboardInterrupt:
        print_error("Optimization interrupted by user.")

    end_time = time.time()
    duration_seconds = end_time - start_time
    duration_str = time.strftime("%H:%M:%S", time.gmtime(duration_seconds))

    print_info("\n=======================================================")
    print_success(f"Study '{optuna_study_name}' Summary")
    print_info(f"  - Total duration: {duration_str}")
    if 'completed_trials' in locals():
        print_info(f"  - Trials performed in this session: {completed_trials} / {trials_to_run}")
    print_info(f"  - Total trials in database: {len(study.trials)}")
    print_info("-------------------------------------------------------")
    try:
        trial = study.best_trial
        print_info("Best trial found in study:")
        print_info(f"  MAE: {trial.value}")
        print_info("  Params: ")
        for key, value in trial.params.items():
            print_info(f"    {key}: {value}")
        if trial.value >= initial_best_mae:
            print_info(f"\n[-] Tuning completed, but no configurations beat the historical record ({initial_best_mae:.4e}). The saved configuration was retained.")
        else:
            print_success(f"\n[+] Tuning completed successfully! The historical record was beaten ({initial_best_mae:.4e} -> {trial.value:.4e}).")
    except ValueError:
        print_error("\n[!] Could not determine best trial. This usually happens if all trials were pruned.")
        if initial_best_mae != float('inf'):
            print_info(f"    No new configurations beat the historical record of {initial_best_mae:.4e}.")
        else:
            print_info("    No configuration was saved as no trials completed successfully.")

    return study

def main():
    from config.config import get_cli_parser
    parser = get_cli_parser(description="Hyperparameter tuning script using Optuna.")
    parser.add_argument('--n_trials', type=int, default=50, help='Number of Optuna trials')
    parser.add_argument('--continue', dest='continue_tune', action='store_true', help='Continue tuning after n_trials. By default (False), tuning stops when the total number of trials reaches n_trials.')
    parser.add_argument('--no_github', dest='export_to_github', action='store_false', help='Disable syncing the best config to GitHub (enabled by default).')
    args, overrides = parser.parse_known_args()

    from config.study import Study
    study = Study.from_args(args, overrides=overrides)
    study.tune(n_trials=args.n_trials, reset=args.reset, wandb=args.wandb, export_to_github=getattr(args, 'export_to_github', True), continue_tune=args.continue_tune)

if __name__ == '__main__':
    main()
