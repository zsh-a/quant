"""Evaluation metrics, screening, validation, and GPU acceleration."""

from .metrics import compute_forward_returns, compute_ic_metrics, compute_rank_ic
from .fast_screen import fast_screen_ic
from .validation import CPCVValidator, ValidationFold
from .gpu_metrics import compute_ic_metrics_gpu, compute_rank_ic_batch_gpu, compute_rank_ic_gpu
from .gpu_ops import TRITON_AVAILABLE
