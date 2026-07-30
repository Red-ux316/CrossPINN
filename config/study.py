from config.path_manager import PathManager
from config.config import load_config

class Study:
    """
    Declarative Experiment Study object encapsulating problem ID, model type, 
    study name, debug mode, and hyperparameter overrides.
    
    Provides programmatic instance methods:
        study.tune(n_trials=30, reset=False)
        study.train(reset=False)
        study.evaluate()
    """
    def __init__(self, prob_id, model, study_name=None, debug=False, overrides=None, **kwargs):
        self.prob_id = str(prob_id)
        self.model = str(model).lower()
        self.study_name = study_name
        self.debug = debug

        # Parse overrides passed either as list/dict or as direct kwargs
        self.overrides = self._format_overrides(overrides, kwargs)
        
        # Instantiate centralized PathManager
        self.pm = PathManager(
            prob_id=self.prob_id,
            model=self.model,
            study_name=self.study_name,
            debug=self.debug
        )

        # Automatically register non-debug study in studies.yaml
        if not self.debug:
            self.register_in_studies_manifest()

    def register_in_studies_manifest(self):
        """
        Registers the current study name under the problem ID key in studies.yaml
        if not running in debug mode.
        """
        if self.debug or self.study_name is None:
            return

        manifest_path = self.pm.studies_config_path
        prob_key = f"prob{self.prob_id}"
        target_name = self.pm.study_base_name

        import yaml
        manifest = {}
        if manifest_path.exists():
            with open(manifest_path, 'r') as f:
                manifest = yaml.safe_load(f) or {}

        if prob_key not in manifest or not isinstance(manifest.get(prob_key), list):
            manifest[prob_key] = []

        if target_name not in manifest[prob_key]:
            manifest[prob_key].append(target_name)
            with open(manifest_path, 'w') as f:
                yaml.dump(manifest, f, sort_keys=False, default_flow_style=False)


    def _format_overrides(self, overrides, kwargs):
        formatted = []
        if isinstance(overrides, list):
            formatted.extend([str(o) for o in overrides])
        elif isinstance(overrides, dict):
            for k, v in overrides.items():
                formatted.append(f"{k}={v}")

        for k, v in kwargs.items():
            formatted.append(f"{k}={v}")
            
        return formatted

    def load_config(self, quiet=True):
        """Loads the merged 3-tier configuration for this study with overrides and debug settings applied."""
        return load_config(self, quiet=quiet)

    def tune(self, n_trials=50, reset=False, wandb=True, export_to_github=True, continue_tune=False):
        """Programmatically launches Optuna hyperparameter tuning for this study."""
        from tools.tune import run_tune
        optuna_study = run_tune(self, n_trials=n_trials, reset=reset, wandb=wandb, export_to_github=export_to_github, continue_tune=continue_tune)
        self.analyze(quiet=False)
        return optuna_study

    def train(self, epochs=None, reset=False, wandb=True, continue_train=False):
        """Programmatically launches main model training for this study."""
        from tools.train import run_train
        return run_train(self, epochs=epochs, reset=reset, wandb=wandb, continue_train=continue_train)

    def evaluate(self, device=None):
        """Programmatically launches model evaluation and plotting for this study."""
        from tools.evaluate import run_evaluate
        eval_results = run_evaluate(self, device=device)
        self.visualize(quiet=False)
        return eval_results

    def visualize(self, quiet=False):
        """Programmatically generates detailed visualization plots for this study."""
        from tools.visualize.visualize import run_visualize
        return run_visualize(self, quiet=quiet)

    def plot(self, quiet=False):
        """Alias for visualize()."""
        return self.visualize(quiet=quiet)

    def analyze(self, quiet=False):
        """Programmatically analyzes the Optuna study and generates HP importance plots for this study."""
        from tools.visualize.analyze_study import run_analyze
        return run_analyze(self, quiet=quiet)

    def analyze_study(self, quiet=False):
        """Alias for analyze()."""
        return self.analyze(quiet=quiet)

    @classmethod
    def from_args(cls, args, overrides=None):
        """Constructs a Study instance from CLI argparse Namespace object."""
        return cls(
            prob_id=getattr(args, 'prob_id', 1),
            model=getattr(args, 'model', 'mlp'),
            study_name=getattr(args, 'study_name', None),
            debug=getattr(args, 'debug', False),
            overrides=overrides
        )

    def __repr__(self):
        return f"<Study(prob_id={self.prob_id}, model='{self.model}', study_name='{self.pm.study_base_name}', debug={self.debug})>"
