from pathlib import Path
import torch
import yaml
import shutil
import sys

# Ensure CrossPINN modules are importable
current_dir = Path(__file__).resolve().parent
root_dir = current_dir.parent
if str(root_dir) not in sys.path:
    sys.path.append(str(root_dir))
if str(current_dir) not in sys.path:
    sys.path.append(str(current_dir))

from tool_utils import load_project_environment, get_device, is_colab, init_wandb_train, build_optimizer_and_scheduler, load_states_from_checkpoint
from utils.logger import print_error, print_success, print_info

load_project_environment()
IN_COLAB = is_colab()
device = get_device()

from core.trainer import Trainer
from models.factory import build_model
from losses.pde_loss import VolterraPDELoss
from equations.registry import get_problem

def run_train(study_or_target, epochs=None, reset=False, wandb=True, continue_train=False):
    from config.study import Study
    from config.path_manager import PathManager
    from config.config import load_config, print_config

    if isinstance(study_or_target, Study):
        study_obj = study_or_target
    elif isinstance(study_or_target, PathManager):
        study_obj = Study(prob_id=study_or_target.prob_id, model=study_or_target.model, study_name=study_or_target.study_name, debug=study_or_target.debug)
    else:
        study_obj = Study.from_args(study_or_target)

    class SyntheticArgs:
        def __init__(self, s, r, w, c_t):
            self.prob_id = s.prob_id
            self.model = s.model
            self.study_name = s.study_name
            self.debug = s.debug
            self.reset = r
            self.wandb = w
            self.continue_train = c_t

    args = SyntheticArgs(study_obj, reset, wandb, continue_train)
    overrides = study_obj.overrides
    pm = study_obj.pm

    # load_config automatically merges 3 tiers, applies overrides, and applies debug overrides
    local_config = load_config(study_obj, quiet=True)
    config = local_config
    checkpoint = None

    best_path = pm.get_weight_checkpoint_path(checkpoint_type="best")
    latest_path = pm.get_weight_checkpoint_path(checkpoint_type="latest")
    load_path = latest_path if latest_path.exists() else (best_path if best_path.exists() else None)

    if not args.reset and load_path:
        print_success(f"Found checkpoint at {load_path}. Checking compatibility...")
        checkpoint = torch.load(load_path, map_location=device)
        if 'config' in checkpoint:
            from config.config import _fill_defaults
            config_from_checkpoint = checkpoint['config']
            
            # Check if model architecture parameters match
            checkpoint_model_cfg = config_from_checkpoint.get('model', {})
            local_model_cfg = local_config.get('model', {})
            
            arch_keys = ['type', 'hidden_features', 'num_blocks', 'hidden_layers', 'num_decoder_layers', 'num_heads', 'activation', 'pos_encoder_type', 'global_skip', 'output_dim', 'out_features']
            arch_mismatch = any(
                local_model_cfg.get(k) != checkpoint_model_cfg.get(k)
                for k in arch_keys
                if (k in local_model_cfg or k in checkpoint_model_cfg)
            )
            
            if arch_mismatch:
                print_error(f"[!] Warning: Checkpoint model architecture {checkpoint_model_cfg} differs from local config {local_model_cfg}.")
                print_info("[!] Ignoring stale checkpoint and training fresh model with updated architecture.")
                checkpoint = None
            else:
                if yaml.dump(local_config, sort_keys=True) != yaml.dump(config_from_checkpoint, sort_keys=True):
                    print_error("Warning: Configuration from checkpoint differs from local .yaml files.")
                print_info("Using configuration from checkpoint as the source of truth.")
                _fill_defaults(config_from_checkpoint, local_config)
                config = config_from_checkpoint

    if epochs is not None:
        if 'training' not in config:
            config['training'] = {}
        config['training']['epochs'] = int(epochs)

    past_epochs = 0
    if checkpoint and not args.reset:
        if 'epoch' in checkpoint:
            past_epochs = checkpoint['epoch']

    target_epochs = config.get('training', {}).get('epochs', 10000)

    if args.continue_train:
        total_target_epochs = past_epochs + target_epochs
        config['training']['epochs'] = total_target_epochs
        print_info(f"[Training] continue=True: running {target_epochs} additional epochs starting from epoch {past_epochs + 1} (total target: {total_target_epochs}).")
    else:
        epochs_to_run = max(0, target_epochs - past_epochs)
        print_info(f"[Training] continue=False: running {epochs_to_run} additional epochs to reach {target_epochs} total (past epochs: {past_epochs}).")

    print_config(config)

    if args.reset:
        print_info("--- RESET FLAG DETECTED: Deleting old checkpoints and logs ---")
        if best_path.exists():
            best_path.unlink()
            print_success(f"[RESET] Removed best model checkpoint: {best_path.name}")
        if latest_path.exists():
            latest_path.unlink()
            print_success(f"[RESET] Removed latest model checkpoint: {latest_path.name}")
        
        study_log_dir = pm.logs_dir / pm.study_base_name
        if study_log_dir.exists():
            shutil.rmtree(study_log_dir)
            print_success(f"[RESET] Removed study log directory: {study_log_dir}")

    wandb_run = init_wandb_train(args, config, pm.problem_data_dir)
    torch.manual_seed(config['experiment']['seed'])

    prob_id = config['data']['problem_id']
    a, b = config['data']['domain']

    from equations.registry import get_problem
    problem = get_problem(prob_id, a=a, b=b)

    model = build_model(
        config=config,
        problem=problem,
        device=device,
        mlp_save_dir=pm.weights_dir,
        debug=args.debug,
        quiet=False
    )

    pde_loss = VolterraPDELoss(
        problem=problem,
        num_quadrature_nodes=config['data']['num_quadrature_nodes'],
        device=device
    )

    optimizer, scheduler, optimizer_lbfgs = build_optimizer_and_scheduler(config, model)

    start_epoch = 1
    if checkpoint and not args.reset:
        start_epoch = load_states_from_checkpoint(checkpoint, model, optimizer, optimizer_lbfgs, device, config)

    trainer = Trainer(
        model=model,
        problem=problem,
        pde_loss=pde_loss,
        optimizer=optimizer,
        config=config,
        data_dir=pm.problem_data_dir,
        scheduler=scheduler,
        optimizer_lbfgs=optimizer_lbfgs,
        wandb_run=wandb_run
    )

    try:
        trainer.train(start_epoch=start_epoch)
    finally:
        if wandb_run:
            wandb_run.finish()
            print_success("[W&B] Run finished.")
    return trainer

def main():
    from config.config import get_cli_parser
    parser = get_cli_parser(description="Main training script for PINN models.")
    parser.add_argument('--epochs', type=int, default=None, help='Number of training epochs (overrides config)')
    parser.add_argument('--continue', dest='continue_train', action='store_true', help='Continue training after reaching target epochs. By default (False), training stops when total epochs reach target.')
    args, overrides = parser.parse_known_args()

    from config.study import Study
    study = Study.from_args(args, overrides=overrides)
    study.train(epochs=args.epochs, reset=args.reset, wandb=args.wandb, continue_train=args.continue_train)

if __name__ == '__main__':
    main()
