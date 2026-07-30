import os
from pathlib import Path
from dotenv import load_dotenv

class PathManager:
    """
    Centralized Path Resolution Manager for CrossPINN.
    Encapsulates all path definitions for configs, checkpoints, logs, plots, and tuning artifacts.
    Handles debug mode overrides, local data roots, and 3-tier config hierarchy.
    """
    def __init__(self, prob_id=None, model=None, study_name=None, debug=False):
        self.prob_id = str(prob_id) if prob_id is not None else None
        self.model = str(model).lower() if model is not None else None
        self.study_name = study_name
        self.debug = bool(debug)
        self.root_dir = Path(__file__).resolve().parent.parent

    @staticmethod
    def is_colab():
        return 'COLAB_RELEASE_TAG' in os.environ

    @staticmethod
    def load_project_environment():
        local_env_path = Path(__file__).resolve().parent.parent / '.env'
        if local_env_path.exists():
            load_dotenv(dotenv_path=local_env_path)

    @classmethod
    def get_base_path(cls):
        cls.load_project_environment()
        return Path(__file__).resolve().parent.parent

    @classmethod
    def get_data_root(cls, use_debug=False):
        base_path = cls.get_base_path()
        data_dir_name = 'data_debug' if use_debug else os.getenv('DATA_DIR', 'data')
        data_path = base_path / data_dir_name
        data_path.mkdir(exist_ok=True, parents=True)
        return data_path

    @classmethod
    def get_plots_root(cls, use_debug=False):
        base_path = cls.get_base_path()
        plots_dir_name = 'plots_debug' if use_debug else 'plots'
        plots_path = base_path / plots_dir_name
        plots_path.mkdir(exist_ok=True, parents=True)
        return plots_path

    @classmethod
    def from_args(cls, args):
        """Constructs PathManager directly from CLI argparse Namespace or dict."""
        if isinstance(args, dict):
            return cls(
                prob_id=args.get('prob_id'),
                model=args.get('model'),
                study_name=args.get('study_name'),
                debug=args.get('debug', False)
            )
        return cls(
            prob_id=getattr(args, 'prob_id', None),
            model=getattr(args, 'model', None),
            study_name=getattr(args, 'study_name', None),
            debug=getattr(args, 'debug', False)
        )

    @property
    def drive_root(self) -> Path:
        """Returns base repository path."""
        return self.get_base_path()

    @property
    def data_root(self) -> Path:
        """Returns data root directory (data_debug or data)."""
        return self.get_data_root(use_debug=self.debug)

    @property
    def plots_root(self) -> Path:
        """Returns plots root directory (plots_debug or plots)."""
        return self.get_plots_root(use_debug=self.debug)

    @property
    def base_models_config_path(self) -> Path:
        """Path to global base models config: config/yamls/models/models.yaml"""
        return self.root_dir / 'config' / 'yamls' / 'models' / 'models.yaml'

    @property
    def model_category_config_path(self) -> Path:
        """Path to model category config: config/yamls/models/<mlp_type|attention_type>.yaml"""
        if not self.model:
            raise ValueError("model is required to resolve model category config path.")
        category = "mlp_type.yaml" if self.model in ["mlp", "risn"] else "attention_type.yaml"
        return self.root_dir / 'config' / 'yamls' / 'models' / category

    @property
    def base_problems_config_path(self) -> Path:
        """Path to global base problems config: equations/configs/problems.yaml"""
        return self.root_dir / 'equations' / 'configs' / 'problems.yaml'

    @property
    def studies_config_path(self) -> Path:
        """Path to studies manifest config: config/yamls/studies.yaml"""
        return self.root_dir / 'config' / 'yamls' / 'studies.yaml'

    @property
    def search_space_config_path(self) -> Path:
        """Path to Optuna search space config: config/yamls/search_space.yaml"""
        return self.root_dir / 'config' / 'yamls' / 'search_space.yaml'

    @property
    def tuning_db_path(self) -> Path:
        """Path to Optuna study SQLite DB: data/tuning/prob<id>/<MODEL>/<study_base_name>.db"""
        return self.tuning_dir / f"{self.study_base_name}.db"

    @property
    def model_config_path(self) -> Path:
        """Path to base model config: config/yamls/models/<model>.yaml"""
        if not self.model:
            raise ValueError("model is required to resolve model config path.")
        path = self.root_dir / 'config' / 'yamls' / 'models' / f"{self.model}.yaml"
        if not path.exists():
            raise FileNotFoundError(f"Model configuration file not found at: {path}")
        return path

    @property
    def problem_config_path(self) -> Path:
        """Path to base problem config: equations/configs/problem_<id>/config_prob<id>.yaml"""
        if not self.prob_id:
            raise ValueError("prob_id is required to resolve problem config path.")
        path = self.root_dir / 'equations' / 'configs' / f"problem_{self.prob_id}" / f"config_prob{self.prob_id}.yaml"
        if not path.exists():
            raise FileNotFoundError(f"Problem configuration file for prob{self.prob_id} not found at: {path}")
        return path

    @property
    def study_base_name(self) -> str:
        """Base name for the study/experiment."""
        if self.study_name:
            return self.study_name
        if self.prob_id and self.model:
            return f"prob{self.prob_id}_{self.model}"
        raise ValueError("Either study_name or both prob_id and model must be set to determine study_base_name.")

    @property
    def study_config_path(self) -> Path:
        """Path to local study config: equations/configs/problem_<id>/<study_name>[_debug].yaml"""
        if not self.prob_id:
            raise ValueError("prob_id is required to resolve study config path.")
        problem_dir = self.root_dir / 'equations' / 'configs' / f"problem_{self.prob_id}"
        debug_suffix = "_debug" if self.debug else ""
        return problem_dir / f"{self.study_base_name}{debug_suffix}.yaml"

    @property
    def problem_data_dir(self) -> Path:
        """Path to problem data dir: data/prob<id>/"""
        if not self.prob_id:
            raise ValueError("prob_id is required to resolve problem data dir.")
        return self.data_root / f"prob{self.prob_id}"

    @property
    def weights_dir(self) -> Path:
        """Path to weights dir: data/prob<id>/weights/"""
        dir_path = self.problem_data_dir / "weights"
        dir_path.mkdir(exist_ok=True, parents=True)
        return dir_path

    @property
    def plots_dir(self) -> Path:
        """Path to evaluation/results plots dir: plots/prob<id>/results/"""
        if not self.prob_id:
            raise ValueError("prob_id is required to resolve plots dir.")
        dir_path = self.plots_root / f"prob{self.prob_id}"
        dir_path.mkdir(exist_ok=True, parents=True)
        return dir_path

    @property
    def tuning_plots_dir(self) -> Path:
        """Path to Optuna tuning plots dir: plots/prob<id>/tuning/<MODEL>/"""
        if not self.prob_id:
            raise ValueError("prob_id is required to resolve tuning plots dir.")
        model_folder = self.model.upper() if self.model else "UNKNOWN"
        dir_path = self.plots_root / f"prob{self.prob_id}" / "tuning" / model_folder
        dir_path.mkdir(exist_ok=True, parents=True)
        return dir_path

    @property
    def logs_dir(self) -> Path:
        """Path to logs dir: data/prob<id>/logs/"""
        dir_path = self.problem_data_dir / "logs"
        dir_path.mkdir(exist_ok=True, parents=True)
        return dir_path

    @property
    def tuning_dir(self) -> Path:
        """Path to tuning dir: data/tuning/prob<id>/<MODEL>/"""
        if not self.prob_id:
            raise ValueError("prob_id is required to resolve tuning dir.")
        model_folder = self.model.upper() if self.model else "UNKNOWN"
        dir_path = self.data_root / "tuning" / f"prob{self.prob_id}" / model_folder
        dir_path.mkdir(exist_ok=True, parents=True)
        return dir_path

    def get_weight_checkpoint_path(self, exp_name=None, checkpoint_type="best") -> Path:
        """Returns full path for model checkpoint: weights/<exp_name>_<checkpoint_type>_model.pt"""
        name = exp_name if exp_name else self.study_base_name
        if self.debug and not name.endswith("_debug"):
            name = f"{name}_debug"
        return self.weights_dir / f"{name}_{checkpoint_type}_model.pt"

    def to_paths_dict(self) -> dict:
        """Returns dictionary of paths matching legacy resolve_config_path output."""
        return {
            'study': self.study_config_path,
            'problem': self.problem_config_path,
            'models': self.model_config_path,
            'explicit_model': self.model,
            'manager': self
        }
