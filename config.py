import os
from dataclasses import dataclass, field
from typing import Tuple, Dict, Optional
from datetime import datetime

@dataclass
class Config:
    """Configuration for HSGSP pruning method"""
    # ========== DATA CONFIGURATION ==========
    task: str = 'cifar10' # 'cifar10' or 'cifar100'
    validation_split: float = 0.2
    data_augmentation: bool = True
    batch_size: int = 128
    # Dataset specific
    num_classes_cifar10: int = 10
    num_classes_cifar100: int = 100
    input_shape_cifar10: Tuple[int, int, int] = (3, 32, 32)
    input_shape_cifar100: Tuple[int, int, int] = (3, 32, 32)
    # ========== TRAINING CONFIGURATION ==========
    default_epochs: int = 200
    initial_lr: float = 7e-4
    pruned_growth_lr: float = 1e-4 # default: 1e-4
    min_lr: float = 1e-5 # default: 1e-5
    momentum: float = 0.9 # for SGD
    optimizer: str = 'adamw' # 'adam', 'adamw', 'sgd', 'rmsprop'
    # Learning rate schedule
    lr_schedule: str = 'cosine' # 'cosine', 'exponential', 'step'
    lr_warmup_epochs: int = 5
    lr_decay_rate: float = 0.2
    lr_decay_steps: int = 60
    # Early Stopping (updated)
    early_stopping_patience: int = 7
    early_stopping_min_delta: float = 1e-4
    reduce_lr_patience: int = 10
    reduce_lr_factor: float = 0.5
    reduce_lr_min_delta: float = 1e-3
    fine_tune_lr_schedule: str = 'cosine' # 'plateau', 'exponential', 'cosine', 'linear', 'step'
    fine_tune_exp_decay: float = 0.8
    fine_tune_cosine_min_factor: float = 0.05 # | 0.05 by default
    fine_tune_linear_end_factor: float = 0.1
    fine_tune_step_decay_rate: float = 0.5
    fine_tune_step_decay_epochs: int = 5
    # Regularization
    l2_regularization: float = 1e-4
    batch_norm_momentum: float = 0.1
    dropout_rate: float = 0.3 # convolutional dropout strength
    use_spatial_dropout: bool = True
    spatial_dropout_rate: float = 0.25
    fc_dropout_rate1: float = 0.4
    fc_dropout_rate2: float = 0.3
    weight_decay: float = 1e-4
    # Label Smoothing
    label_smoothing: float = 0.1
    # ========== DISTILLATION CONFIGURATION ==========
    distill_alpha: float = 0.2
    distill_temperature: float = 2.5
    # ========== PRUNING CONFIGURATION ==========
    # Frequency analysis
    frequency_bands: Dict[str, tuple] = None
    # Comlexity weights
    complexity_weights: Dict[str, float] = None
    max_global_pruning_ratio: float = 0.85 # Hard cap on global pruning ratio
    min_global_keep: float = 0.3 # Ensure at least this fraction of channels remain
    max_accuracy_drop: float = 0.01 # Target maximum accuracy drop when estimating pruning ratio
    accuracy_guard_center: float = 0.92 # Center of accuracy guard margin
    accuracy_guard_sharpness: float = 0.5 # Sharpness of accuracy guard sigmoid
    # ========== PRUNING CONFIGURATION ==========
    simple_finetune_epochs: int = 20
    simple_finetune_lr: float = 1e-5
    # ========== AUGMENTATION CONFIGURATION ==========
    use_mixup: bool = True
    mixup_alpha: float = 0.4
    mixup_prob: float = 0.5
    # ========== HYBRID BASELINE CONFIGURATION ==========
    hybrid_iterations: int = 20
    hybrid_prune_fraction: float = 0.07
    hybrid_alpha: float = 0.5
    hybrid_kappa_beta: float = 0.1
    hybrid_initial_kappa_ratio: float = 0.5
    hybrid_mode: str = 'original' # 'frequency' or 'original'
    hybrid_min_filters: int = 8
    hybrid_finetune_epochs: int = 20
    hybrid_warmup_epochs: int = 0
    hybrid_warmup_lr: float = 2e-4
    hybrid_regrow_fraction: float = 0.15 # 0.15 by default
    frequency_regularization_layers: int = 4 # 4 | Number of layers to apply frequency regularization
    frequency_entropy_beta: float = 0.05 # 0.05 | Weight for frequency entropy loss
    frequency_entropy_target_batches: int = 8
    frequency_entropy_refresh_interval: int = 3
    frequency_entropy_layer_weights: Dict[str, float] = field(default_factory=dict)
  
    frn_validation_split: float = 0.2
    frn_plot_training: bool = True
    frn_feature_count: Optional[int] = 3
    frn_hidden_units: Tuple[int, ...] = (64, 32)
    frn_activation_batches: int = 512
    frn_min_validation_samples: int = 128
    frn_dropout_rate: float = 0.05 # 0.02 by default
    frn_use_batchnorm: bool = False
    frn_use_activation_features: bool = False
    frn_low_vs_rest: bool = False
    frn_architecture: str = "dense" # 'dense' or 'residual'
    frn_ema_beta: float = 0.8
    frn_sharpen_gamma: float = 2.0
    frn_initial_lr: float = 1e-4
    frn_min_lr: float = 1e-5
    frn_epochs: int = 15
    frn_cosine_min_factor: float = 0.1
    frn_batch_size: int = 256
    frn_weight_clip: float = 2.0
    # ========== PATHs CONFIGURATION ==========
    run_id: str = field(default_factory=lambda: datetime.now().strftime("%d%m%Y_%H%M%S"))
    tensorboard_dir: str = field(init=False)
    results_dir: str = field(init=False)
    logs_dir: str = field(init=False)
    models_dir: str = field(init=False)
    plots_dir: str = field(init=False)
    def __post_init__(self):
        if self.frequency_bands is None:
            self.frequency_bands = {
                'low': (0.0, 0.25),
                'mid': (0.25, 0.5),
                'high': (0.5, 1.0)
            }
        # Build directories using the provided task and a timestamp run_id
        exp_root = f"./EXPERIMENT/{self.run_id}_{self.task}"
        self.tensorboard_dir = os.path.join(exp_root, "tensorboard_logs")
        self.results_dir = os.path.join(exp_root, "results")
        self.logs_dir = os.path.join(exp_root, "logs")
        self.models_dir = os.path.join(exp_root, "models")
        self.plots_dir = os.path.join(exp_root, "plots")
        # Create directories
        os.makedirs(self.tensorboard_dir, exist_ok=True)
        os.makedirs(self.results_dir, exist_ok=True)
        os.makedirs(self.logs_dir, exist_ok=True)
        os.makedirs(self.models_dir, exist_ok=True)
        os.makedirs(self.plots_dir, exist_ok=True)