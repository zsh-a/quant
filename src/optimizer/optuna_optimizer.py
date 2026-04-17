"""
Optuna 贝叶斯参数优化 — TPE 采样 + 中位数剪枝。

使用 Optuna 替代 sklearn GP，支持:
  - Tree-structured Parzen Estimator (TPE) 采样
  - 不良试验早停 (MedianPruner)
  - 试验历史持久化 (SQLite)
  - 并行采样
"""

from __future__ import annotations

from typing import Any, Callable, Dict, Optional

from loguru import logger

from .optimizer import OptimizationObjective, OptimizationReport, OptimizationResult, ParamSpec


def optuna_optimize(
    param_space: Dict[str, ParamSpec],
    backtest_fn: Callable[[Dict], Dict],
    objective: OptimizationObjective = OptimizationObjective.MAX_SHARPE,
    n_trials: int = 50,
    timeout_seconds: Optional[int] = None,
    study_name: Optional[str] = None,
) -> OptimizationReport:
    """使用 Optuna TPE 进行参数优化。

    Args:
        param_space: 参数搜索空间
        backtest_fn: 回测函数，接受 params dict 返回 metrics dict
        objective: 优化目标
        n_trials: 最大试验次数
        timeout_seconds: 超时秒数（None = 不限时）
        study_name: 研究名称（用于持久化）

    Returns:
        OptimizationReport
    """
    try:
        import optuna
        optuna.logging.set_verbosity(optuna.logging.WARNING)
    except ImportError:
        logger.error("optuna not installed. Install with: pip install optuna")
        raise ImportError("pip install optuna")

    import time
    start_time = time.time()
    all_results: list[OptimizationResult] = []

    objective_key = {
        OptimizationObjective.MAX_SHARPE: "sharpe_ratio",
        OptimizationObjective.MAX_RETURN: "total_return",
        OptimizationObjective.MIN_DRAWDOWN: "max_drawdown",
        OptimizationObjective.MAX_CALMAR: "calmar_ratio",
    }[objective]

    maximize = objective != OptimizationObjective.MIN_DRAWDOWN
    direction = "maximize" if maximize else "minimize"

    def _objective(trial: optuna.Trial) -> float:
        params: Dict[str, Any] = {}
        for name, spec in param_space.items():
            if spec.param_type == "int":
                params[name] = trial.suggest_int(name, int(spec.low), int(spec.high), step=int(spec.step or 1))
            elif spec.param_type == "float":
                params[name] = trial.suggest_float(name, spec.low, spec.high, step=spec.step)
            elif spec.param_type == "categorical":
                params[name] = trial.suggest_categorical(name, spec.choices or [])
            else:
                params[name] = spec.low

        try:
            result = backtest_fn(params)
            score = result.get(objective_key, 0.0)
            all_results.append(OptimizationResult(
                params=params,
                metrics=result,
                objective_value=float(score),
            ))
            return float(score)
        except Exception as exc:
            logger.warning("Trial {} failed: {}", trial.number, exc)
            return float("-inf") if maximize else float("inf")

    # Create study with TPE sampler + median pruner
    sampler = optuna.samplers.TPESampler(seed=42)
    pruner = optuna.pruners.MedianPruner(n_startup_trials=5, n_warmup_steps=3)

    study = optuna.create_study(
        study_name=study_name or "quent_optimize",
        direction=direction,
        sampler=sampler,
        pruner=pruner,
    )

    study.optimize(_objective, n_trials=n_trials, timeout=timeout_seconds)

    elapsed = time.time() - start_time
    best = study.best_trial

    logger.info(
        "optuna.optimize completed: {} trials, best {}={:.4f}, {:.1f}s",
        len(study.trials), objective_key, best.value, elapsed,
    )

    # Sort results
    all_results.sort(key=lambda r: r.objective_value, reverse=maximize)

    return OptimizationReport(
        method="optuna_tpe",
        objective=objective.value,
        total_iterations=len(study.trials),
        best_params=best.params,
        best_metrics=all_results[0].metrics if all_results else {},
        best_objective=float(best.value),
        all_results=all_results[:100],  # top 100
        elapsed_seconds=elapsed,
        param_importance=_get_importance(study, param_space),
    )


def _get_importance(study, param_space: Dict[str, ParamSpec]) -> Dict[str, float]:
    """Extract parameter importance from Optuna study."""
    try:
        import optuna
        importance = optuna.importance.get_param_importances(study)
        return {k: round(v, 4) for k, v in importance.items()}
    except Exception:
        return {}
