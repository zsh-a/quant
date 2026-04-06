from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from .compiler import BytecodeProgram

try:
    import torch
except Exception:  # pragma: no cover - torch is optional for this scaffold
    torch = None

try:
    from ..eval.gpu_ops import (
        TRITON_AVAILABLE as _TRITON_OK,
        rolling_mean_std as _triton_rolling_mean_std,
        rolling_reduce as _triton_rolling_reduce,
        parallel_ema as _triton_parallel_ema,
        rolling_corr_cov as _triton_rolling_corr_cov,
        cs_rank as _triton_cs_rank,
        decay_linear as _triton_decay_linear,
    )
except Exception:  # pragma: no cover
    _TRITON_OK = False


ArrayLike = Any


@dataclass
class TensorStore:
    fields: dict[str, ArrayLike]

    def get_field(self, name: str) -> ArrayLike:
        if name not in self.fields:
            raise KeyError(f"Field not found in tensor store: {name}")
        return self.fields[name]

    def shape(self) -> tuple[int, ...]:
        if not self.fields:
            raise ValueError("TensorStore is empty")
        first = next(iter(self.fields.values()))
        return tuple(first.shape)

    def uses_torch(self) -> bool:
        first = next(iter(self.fields.values()), None)
        return bool(torch is not None and isinstance(first, torch.Tensor))


def to_numpy(value: ArrayLike) -> np.ndarray:
    """Safely convert any array-like (including CUDA tensors) to numpy."""
    if torch is not None and isinstance(value, torch.Tensor):
        return value.detach().cpu().numpy()
    return np.asarray(value)


class SubexprCache:
    """Thread-safe LRU cache for subexpression results.

    Cache keys include the dataset shape so that results from different
    fold splits (train vs valid vs test) never collide — even when
    evaluated concurrently by ThreadPoolExecutor.
    """

    def __init__(self, max_entries: int = 4096):
        import threading
        from collections import OrderedDict
        self._cache: OrderedDict[tuple[Any, ...], ArrayLike] = OrderedDict()
        self._max = max_entries
        self._lock = threading.Lock()
        self.hits = 0
        self.misses = 0

    def get(self, key: tuple[Any, ...], shape: tuple[int, ...]) -> ArrayLike | None:
        full_key = (shape, *key)
        with self._lock:
            val = self._cache.get(full_key)
            if val is not None:
                self._cache.move_to_end(full_key)
                self.hits += 1
                return val
            self.misses += 1
            return None

    def put(self, key: tuple[Any, ...], shape: tuple[int, ...], value: ArrayLike) -> None:
        full_key = (shape, *key)
        with self._lock:
            self._cache[full_key] = value
            self._cache.move_to_end(full_key)
            while len(self._cache) > self._max:
                self._cache.popitem(last=False)


class StackVM:
    def __init__(self, device: str | None = None, prefer_torch: bool = True, use_triton: bool = True):
        self.prefer_torch = prefer_torch
        self.device = self._resolve_device(device)
        self.backend = "torch" if self.device is not None else "numpy"
        self.use_triton = (
            use_triton
            and self.backend == "torch"
            and _TRITON_OK
            and self.device is not None
            and self.device.type == "cuda"
        )
        self.persistent_cache: SubexprCache | None = None

    def enable_persistent_cache(self, max_entries: int = 4096) -> None:
        """Enable cross-batch subexpression cache for repeated evaluations."""
        self.persistent_cache = SubexprCache(max_entries=max_entries)

    def run(self, program: BytecodeProgram, store: TensorStore) -> ArrayLike:
        prepared_store = self._prepare_store(store)
        result = self._run_program(program, prepared_store)
        return self._restore_output(result, store)

    def run_batch(self, programs: list[BytecodeProgram], store: TensorStore) -> list[ArrayLike]:
        prepared_store = self._prepare_store(store)
        shared_cache: dict[tuple[Any, ...], ArrayLike] = {}
        outputs = [self._run_program(program, prepared_store, shared_cache) for program in programs]
        return [self._restore_output(output, store) for output in outputs]

    def _run_program(
        self,
        program: BytecodeProgram,
        store: TensorStore,
        shared_cache: dict[tuple[Any, ...], ArrayLike] | None = None,
    ) -> ArrayLike:
        registers: dict[int, ArrayLike] = {}
        register_keys: dict[int, tuple[Any, ...]] = {}
        output_shape = store.shape()

        for ins in program.instructions:
            if ins.opcode == "push_field":
                cache_key = ("field", ins.value)
                registers[ins.dst] = self._cache_lookup_or_compute(
                    cache_key, shared_cache, output_shape,
                    lambda: store.get_field(ins.value),
                )
                register_keys[ins.dst] = cache_key
                continue
            if ins.opcode == "push_const":
                cache_key = ("const", float(ins.value), output_shape)
                registers[ins.dst] = self._cache_lookup_or_compute(
                    cache_key, shared_cache, output_shape,
                    lambda: self._broadcast_const(ins.value, output_shape, store),
                )
                register_keys[ins.dst] = cache_key
                continue
            args = [registers[idx] for idx in ins.args]
            cache_key = (ins.opcode, *(register_keys[idx] for idx in ins.args))
            registers[ins.dst] = self._cache_lookup_or_compute(
                cache_key, shared_cache, output_shape,
                lambda: self._execute(ins.opcode, args),
            )
            register_keys[ins.dst] = cache_key

        return registers[program.output_register]

    def _resolve_device(self, device: str | None) -> Any:
        if not self.prefer_torch or torch is None:
            return None
        if device is not None:
            return torch.device(device)
        if torch.cuda.is_available():
            return torch.device("cuda")
        return torch.device("cpu")

    def _prepare_store(self, store: TensorStore) -> TensorStore:
        if self.backend != "torch":
            return store
        fields = {
            name: self._as_torch_tensor(value)
            for name, value in store.fields.items()
        }
        return TensorStore(fields)

    def _restore_output(self, value: ArrayLike, original_store: TensorStore) -> ArrayLike:
        if self.backend != "torch" or original_store.uses_torch():
            return value
        return value.detach().cpu().numpy()

    def _cache_lookup_or_compute(
        self,
        cache_key: tuple[Any, ...],
        shared_cache: dict[tuple[Any, ...], ArrayLike] | None,
        shape: tuple[int, ...],
        compute: Any,
    ) -> ArrayLike:
        # Check persistent cache (shape-keyed, thread-safe)
        pc = self.persistent_cache
        if pc is not None:
            val = pc.get(cache_key, shape)
            if val is not None:
                if shared_cache is not None:
                    shared_cache[cache_key] = val
                return val

        # Per-batch shared cache (single-thread within one run_batch call)
        if shared_cache is None:
            result = compute()
        elif cache_key not in shared_cache:
            shared_cache[cache_key] = compute()
            result = shared_cache[cache_key]
        else:
            result = shared_cache[cache_key]

        if pc is not None:
            pc.put(cache_key, shape, result)
        return result

    def _broadcast_const(self, value: Any, shape: tuple[int, ...], store: TensorStore) -> ArrayLike:
        if self.backend == "torch":
            first = next(iter(store.fields.values()))
            return torch.full(shape, float(value), dtype=first.dtype, device=first.device)
        return np.full(shape, float(value), dtype=float)

    def _execute(self, opcode: str, args: list[ArrayLike]) -> ArrayLike:
        if self.backend == "torch":
            return self._execute_torch(opcode, args)
        return self._execute_numpy(opcode, args)

    # ------------------------------------------------------------------
    # numpy backend
    # ------------------------------------------------------------------

    def _execute_numpy(self, opcode: str, args: list[ArrayLike]) -> ArrayLike:
        if opcode == "add":
            return args[0] + args[1]
        if opcode == "sub":
            return args[0] - args[1]
        if opcode == "mul":
            return args[0] * args[1]
        if opcode == "div":
            return args[0] / (args[1] + 1e-12)
        if opcode == "max":
            return np.maximum(args[0], args[1])
        if opcode == "min":
            return np.minimum(args[0], args[1])
        if opcode == "pow":
            return np.power(args[0], args[1])
        if opcode == "abs":
            return np.abs(args[0])
        if opcode == "log":
            return np.log(np.abs(args[0]) + 1e-12)
        if opcode == "sign":
            return np.sign(args[0])
        if opcode == "sqrt":
            return np.sqrt(np.abs(args[0]))
        if opcode == "sigmoid":
            return 1.0 / (1.0 + np.exp(-np.clip(args[0], -20, 20)))
        if opcode == "neg":
            return -args[0]
        if opcode == "not":
            return np.logical_not(args[0])
        if opcode == "gt":
            return args[0] > args[1]
        if opcode == "ge":
            return args[0] >= args[1]
        if opcode == "lt":
            return args[0] < args[1]
        if opcode == "le":
            return args[0] <= args[1]
        if opcode == "eq":
            return args[0] == args[1]
        if opcode == "ne":
            return args[0] != args[1]
        if opcode == "and":
            return np.logical_and(args[0], args[1])
        if opcode == "or":
            return np.logical_or(args[0], args[1])
        if opcode == "clip":
            return np.clip(args[0], self._scalar(args[1]), self._scalar(args[2]))
        if opcode == "fillna":
            return np.where(np.isnan(args[0]), args[1], args[0])
        if opcode == "where":
            return np.where(args[0], args[1], args[2])
        if opcode == "power":
            p = self._scalar(args[1])
            return np.sign(args[0]) * np.power(np.abs(args[0]) + 1e-12, p)
        if opcode == "delay":
            return self._delay_numpy(args[0], int(self._scalar(args[1])))
        if opcode == "delta":
            delayed = self._delay_numpy(args[0], int(self._scalar(args[1])))
            return args[0] - delayed
        if opcode == "returns_n":
            delayed = self._delay_numpy(args[0], int(self._scalar(args[1])))
            return args[0] / (delayed + 1e-12) - 1.0
        if opcode == "log_return":
            delayed = self._delay_numpy(args[0], int(self._scalar(args[1])))
            return np.log((np.abs(args[0]) + 1e-12) / (np.abs(delayed) + 1e-12))
        if opcode == "ts_mean":
            return self._rolling_numpy(args[0], int(self._scalar(args[1])), np.nanmean)
        if opcode == "ts_std":
            return self._rolling_numpy(args[0], int(self._scalar(args[1])), np.nanstd)
        if opcode == "ts_sum":
            return self._rolling_numpy(args[0], int(self._scalar(args[1])), np.nansum)
        if opcode == "ts_max":
            return self._rolling_numpy(args[0], int(self._scalar(args[1])), np.nanmax)
        if opcode == "ts_min":
            return self._rolling_numpy(args[0], int(self._scalar(args[1])), np.nanmin)
        if opcode == "ts_rank":
            return self._ts_rank_numpy(args[0], int(self._scalar(args[1])))
        if opcode == "ts_zscore":
            window = int(self._scalar(args[1]))
            mean = self._rolling_numpy(args[0], window, np.nanmean)
            std = self._rolling_numpy(args[0], window, np.nanstd)
            return (args[0] - mean) / (std + 1e-12)
        if opcode == "ts_corr":
            return self._rolling_pair_numpy(args[0], args[1], int(self._scalar(args[2])), reducer="corr")
        if opcode == "ts_cov":
            return self._rolling_pair_numpy(args[0], args[1], int(self._scalar(args[2])), reducer="cov")
        if opcode == "decay_linear":
            return self._decay_linear_numpy(args[0], int(self._scalar(args[1])))
        if opcode == "ts_argmax":
            return self._ts_argextreme_numpy(args[0], int(self._scalar(args[1])), mode="max")
        if opcode == "ts_argmin":
            return self._ts_argextreme_numpy(args[0], int(self._scalar(args[1])), mode="min")
        if opcode == "ts_ema":
            return self._ts_ema_numpy(args[0], int(self._scalar(args[1])))
        if opcode == "ts_winsorize":
            return self._ts_winsorize_numpy(args[0], int(self._scalar(args[1])), self._scalar(args[2]))
        if opcode == "cs_rank":
            return self._cs_rank_numpy(args[0])
        if opcode == "cs_scale":
            denom = np.nansum(np.abs(args[0]), axis=1, keepdims=True)
            return args[0] / (denom + 1e-12)
        if opcode == "cs_zscore":
            mean = np.nanmean(args[0], axis=1, keepdims=True)
            std = np.nanstd(args[0], axis=1, keepdims=True)
            return (args[0] - mean) / (std + 1e-12)
        if opcode == "cs_demean":
            mean = np.nanmean(args[0], axis=1, keepdims=True)
            return args[0] - mean
        if opcode == "oi_delta":
            delayed = self._delay_numpy(args[0], int(self._scalar(args[1])))
            return args[0] - delayed
        if opcode == "funding_delta":
            delayed = self._delay_numpy(args[0], int(self._scalar(args[1])))
            return args[0] - delayed
        if opcode == "spread_ratio":
            return args[0] / (np.abs(args[1]) + 1e-12)
        if opcode == "adv_n":
            return self._rolling_numpy(args[0], int(self._scalar(args[1])), np.nanmean)
        if opcode == "amihud":
            illiquidity = np.abs(self._execute_numpy("log_return", [args[0], args[2]])) / (np.abs(args[1]) + 1e-12)
            return self._rolling_numpy(illiquidity, int(self._scalar(args[2])), np.nanmean)
        if opcode == "hlc3":
            return (args[0] + args[1] + args[2]) / 3.0
        if opcode == "ohlc4":
            return (args[0] + args[1] + args[2] + args[3]) / 4.0
        if opcode == "true_range":
            prev_close = self._delay_numpy(args[2], 1)
            return np.maximum(
                np.maximum(args[0] - args[1], np.abs(args[0] - prev_close)),
                np.abs(args[1] - prev_close),
            )
        if opcode == "atr_n":
            true_range = self._execute_numpy("true_range", [args[0], args[1], args[2]])
            return self._rolling_numpy(true_range, int(self._scalar(args[3])), np.nanmean)
        if opcode == "volatility_n":
            returns = args[0] / (self._delay_numpy(args[0], 1) + 1e-12) - 1.0
            return self._rolling_numpy(returns, int(self._scalar(args[1])), np.nanstd)
        raise ValueError(f"Unsupported opcode: {opcode}")

    # ------------------------------------------------------------------
    # torch backend
    # ------------------------------------------------------------------

    def _execute_torch(self, opcode: str, args: list[ArrayLike]) -> ArrayLike:
        if opcode == "add":
            return args[0] + args[1]
        if opcode == "sub":
            return args[0] - args[1]
        if opcode == "mul":
            return args[0] * args[1]
        if opcode == "div":
            return args[0] / (args[1] + 1e-12)
        if opcode == "max":
            return torch.maximum(args[0], args[1])
        if opcode == "min":
            return torch.minimum(args[0], args[1])
        if opcode == "pow":
            return torch.pow(args[0], args[1])
        if opcode == "abs":
            return torch.abs(args[0])
        if opcode == "log":
            return torch.log(torch.abs(args[0]) + 1e-12)
        if opcode == "sign":
            return torch.sign(args[0])
        if opcode == "sqrt":
            return torch.sqrt(torch.abs(args[0]))
        if opcode == "sigmoid":
            return torch.sigmoid(torch.clamp(args[0], -20, 20))
        if opcode == "neg":
            return -args[0]
        if opcode == "not":
            return torch.logical_not(args[0])
        if opcode == "gt":
            return args[0] > args[1]
        if opcode == "ge":
            return args[0] >= args[1]
        if opcode == "lt":
            return args[0] < args[1]
        if opcode == "le":
            return args[0] <= args[1]
        if opcode == "eq":
            return args[0] == args[1]
        if opcode == "ne":
            return args[0] != args[1]
        if opcode == "and":
            return torch.logical_and(args[0], args[1])
        if opcode == "or":
            return torch.logical_or(args[0], args[1])
        if opcode == "clip":
            return torch.clamp(args[0], min=self._scalar(args[1]), max=self._scalar(args[2]))
        if opcode == "fillna":
            return torch.where(torch.isnan(args[0]), args[1], args[0])
        if opcode == "where":
            return torch.where(args[0], args[1], args[2])
        if opcode == "power":
            p = self._scalar(args[1])
            return torch.sign(args[0]) * torch.pow(torch.abs(args[0]) + 1e-12, p)
        if opcode == "delay":
            return self._delay_torch(args[0], int(self._scalar(args[1])))
        if opcode == "delta":
            delayed = self._delay_torch(args[0], int(self._scalar(args[1])))
            return args[0] - delayed
        if opcode == "returns_n":
            delayed = self._delay_torch(args[0], int(self._scalar(args[1])))
            return args[0] / (delayed + 1e-12) - 1.0
        if opcode == "log_return":
            delayed = self._delay_torch(args[0], int(self._scalar(args[1])))
            return torch.log((torch.abs(args[0]) + 1e-12) / (torch.abs(delayed) + 1e-12))
        if opcode == "ts_mean":
            window = int(self._scalar(args[1]))
            if self.use_triton:
                mean, _ = _triton_rolling_mean_std(args[0], window)
                return mean
            return self._rolling_torch(args[0], window, reducer="mean")
        if opcode == "ts_std":
            window = int(self._scalar(args[1]))
            if self.use_triton:
                _, std = _triton_rolling_mean_std(args[0], window)
                return std
            return self._rolling_torch(args[0], window, reducer="std")
        if opcode == "ts_sum":
            window = int(self._scalar(args[1]))
            if self.use_triton:
                return _triton_rolling_reduce(args[0], window, "sum")
            return self._rolling_torch(args[0], window, reducer="sum")
        if opcode == "ts_max":
            window = int(self._scalar(args[1]))
            if self.use_triton:
                return _triton_rolling_reduce(args[0], window, "max")
            return self._rolling_torch(args[0], window, reducer="max")
        if opcode == "ts_min":
            window = int(self._scalar(args[1]))
            if self.use_triton:
                return _triton_rolling_reduce(args[0], window, "min")
            return self._rolling_torch(args[0], window, reducer="min")
        if opcode == "ts_rank":
            return self._ts_rank_torch(args[0], int(self._scalar(args[1])))
        if opcode == "ts_zscore":
            window = int(self._scalar(args[1]))
            if self.use_triton:
                mean, std = _triton_rolling_mean_std(args[0], window)
                return (args[0] - mean) / (std + 1e-12)
            mean = self._rolling_torch(args[0], window, reducer="mean")
            std = self._rolling_torch(args[0], window, reducer="std")
            return (args[0] - mean) / (std + 1e-12)
        if opcode == "ts_corr":
            window = int(self._scalar(args[2]))
            if self.use_triton:
                return _triton_rolling_corr_cov(args[0], args[1], window, "corr")
            return self._rolling_pair_torch(args[0], args[1], window, reducer="corr")
        if opcode == "ts_cov":
            window = int(self._scalar(args[2]))
            if self.use_triton:
                return _triton_rolling_corr_cov(args[0], args[1], window, "cov")
            return self._rolling_pair_torch(args[0], args[1], window, reducer="cov")
        if opcode == "decay_linear":
            window = int(self._scalar(args[1]))
            if self.use_triton:
                return _triton_decay_linear(args[0], window)
            return self._decay_linear_torch(args[0], window)
        if opcode == "ts_argmax":
            return self._ts_argextreme_torch(args[0], int(self._scalar(args[1])), mode="max")
        if opcode == "ts_argmin":
            return self._ts_argextreme_torch(args[0], int(self._scalar(args[1])), mode="min")
        if opcode == "ts_ema":
            window = int(self._scalar(args[1]))
            if self.use_triton:
                return _triton_parallel_ema(args[0], window)
            return self._ts_ema_torch(args[0], window)
        if opcode == "ts_winsorize":
            window = int(self._scalar(args[1]))
            n_std = self._scalar(args[2])
            if self.use_triton:
                mean, std = _triton_rolling_mean_std(args[0], window)
                return torch.clamp(torch.clamp(args[0], min=mean - n_std * std), max=mean + n_std * std)
            return self._ts_winsorize_torch(args[0], window, n_std)
        if opcode == "cs_rank":
            if self.use_triton:
                return _triton_cs_rank(args[0])
            return self._cs_rank_torch(args[0])
        if opcode == "cs_scale":
            denom = self._torch_nansum(torch.abs(args[0]), dim=1, keepdim=True)
            return args[0] / (denom + 1e-12)
        if opcode == "cs_zscore":
            mean = self._torch_nanmean(args[0], dim=1, keepdim=True)
            std = self._torch_nanstd(args[0], dim=1, keepdim=True)
            return (args[0] - mean) / (std + 1e-12)
        if opcode == "cs_demean":
            mean = self._torch_nanmean(args[0], dim=1, keepdim=True)
            return args[0] - mean
        if opcode == "oi_delta":
            delayed = self._delay_torch(args[0], int(self._scalar(args[1])))
            return args[0] - delayed
        if opcode == "funding_delta":
            delayed = self._delay_torch(args[0], int(self._scalar(args[1])))
            return args[0] - delayed
        if opcode == "spread_ratio":
            return args[0] / (torch.abs(args[1]) + 1e-12)
        if opcode == "adv_n":
            window = int(self._scalar(args[1]))
            if self.use_triton:
                mean, _ = _triton_rolling_mean_std(args[0], window)
                return mean
            return self._rolling_torch(args[0], window, reducer="mean")
        if opcode == "amihud":
            window = int(self._scalar(args[2]))
            illiquidity = torch.abs(self._execute_torch("log_return", [args[0], args[2]])) / (torch.abs(args[1]) + 1e-12)
            if self.use_triton:
                mean, _ = _triton_rolling_mean_std(illiquidity, window)
                return mean
            return self._rolling_torch(illiquidity, window, reducer="mean")
        if opcode == "hlc3":
            return (args[0] + args[1] + args[2]) / 3.0
        if opcode == "ohlc4":
            return (args[0] + args[1] + args[2] + args[3]) / 4.0
        if opcode == "true_range":
            prev_close = self._delay_torch(args[2], 1)
            return torch.maximum(
                torch.maximum(args[0] - args[1], torch.abs(args[0] - prev_close)),
                torch.abs(args[1] - prev_close),
            )
        if opcode == "atr_n":
            true_range = self._execute_torch("true_range", [args[0], args[1], args[2]])
            window = int(self._scalar(args[3]))
            if self.use_triton:
                mean, _ = _triton_rolling_mean_std(true_range, window)
                return mean
            return self._rolling_torch(true_range, window, reducer="mean")
        if opcode == "volatility_n":
            returns = args[0] / (self._delay_torch(args[0], 1) + 1e-12) - 1.0
            window = int(self._scalar(args[1]))
            if self.use_triton:
                _, std = _triton_rolling_mean_std(returns, window)
                return std
            return self._rolling_torch(returns, window, reducer="std")
        raise ValueError(f"Unsupported opcode: {opcode}")

    # ------------------------------------------------------------------
    # helpers
    # ------------------------------------------------------------------

    def _scalar(self, value: ArrayLike) -> float:
        if torch is not None and isinstance(value, torch.Tensor):
            return float(value.reshape(-1)[0].item())
        if np.isscalar(value):
            return float(value)
        flat = np.asarray(value).reshape(-1)
        return float(flat[0])

    def _as_torch_tensor(self, value: ArrayLike) -> Any:
        if torch is None:
            raise RuntimeError("Torch backend requested but torch is not available")
        if isinstance(value, torch.Tensor):
            tensor = value.to(device=self.device)
            if tensor.dtype == torch.float64:
                return tensor.to(dtype=torch.float32)
            return tensor
        array = np.asarray(value)
        if array.dtype == np.bool_:
            return torch.as_tensor(array, device=self.device, dtype=torch.bool)
        return torch.as_tensor(array, device=self.device, dtype=torch.float32)

    # --- delay ---

    def _delay_numpy(self, arr: ArrayLike, periods: int) -> ArrayLike:
        result = np.full_like(arr, np.nan, dtype=float)
        if periods <= 0:
            return np.asarray(arr, dtype=float)
        if periods >= arr.shape[0]:
            return result
        result[periods:] = np.asarray(arr[:-periods], dtype=float)
        return result

    def _delay_torch(self, arr: ArrayLike, periods: int) -> ArrayLike:
        data = arr if isinstance(arr, torch.Tensor) else self._as_torch_tensor(arr)
        result = torch.full_like(data, torch.nan)
        if periods <= 0:
            return data.clone()
        if periods >= data.shape[0]:
            return result
        result[periods:] = data[:-periods]
        return result

    # --- rolling ---

    def _rolling_numpy(self, arr: ArrayLike, window: int, reducer) -> ArrayLike:
        data = np.asarray(arr, dtype=float)
        result = np.full_like(data, np.nan, dtype=float)
        if window <= 0 or data.shape[0] < window:
            return result
        windows = np.lib.stride_tricks.sliding_window_view(data, window_shape=window, axis=0)
        reduced = reducer(windows, axis=-1)
        result[window - 1:] = reduced
        return result

    def _rolling_torch(self, arr: ArrayLike, window: int, reducer: str) -> ArrayLike:
        data = arr if isinstance(arr, torch.Tensor) else self._as_torch_tensor(arr)
        result = torch.full_like(data, torch.nan)
        if window <= 0 or data.shape[0] < window:
            return result
        windows = data.unfold(0, window, 1)
        if reducer == "mean":
            reduced = self._torch_nanmean(windows, dim=-1)
        elif reducer == "std":
            reduced = self._torch_nanstd(windows, dim=-1)
        elif reducer == "sum":
            reduced = self._torch_nansum(windows, dim=-1)
        elif reducer == "max":
            reduced = self._torch_nanmax(windows, dim=-1)
        elif reducer == "min":
            reduced = self._torch_nanmin(windows, dim=-1)
        else:
            raise ValueError(f"Unsupported rolling reducer: {reducer}")
        result[window - 1:] = reduced
        return result

    # --- rolling pair ---

    def _rolling_pair_numpy(self, left: ArrayLike, right: ArrayLike, window: int, reducer: str) -> ArrayLike:
        x = np.asarray(left, dtype=float)
        y = np.asarray(right, dtype=float)
        result = np.full_like(x, np.nan, dtype=float)
        if window <= 1 or x.shape[0] < window:
            return result
        x_windows = np.lib.stride_tricks.sliding_window_view(x, window_shape=window, axis=0)
        y_windows = np.lib.stride_tricks.sliding_window_view(y, window_shape=window, axis=0)
        valid = ~np.isnan(x_windows) & ~np.isnan(y_windows)
        count = valid.sum(axis=-1)
        x_safe = np.where(valid, x_windows, 0.0)
        y_safe = np.where(valid, y_windows, 0.0)
        x_mean = x_safe.sum(axis=-1) / np.maximum(count, 1)
        y_mean = y_safe.sum(axis=-1) / np.maximum(count, 1)
        centered_x = np.where(valid, x_windows - x_mean[..., None], 0.0)
        centered_y = np.where(valid, y_windows - y_mean[..., None], 0.0)
        cov = (centered_x * centered_y).sum(axis=-1) / np.maximum(count, 1)
        if reducer == "cov":
            reduced = cov
        elif reducer == "corr":
            x_std = np.sqrt((centered_x * centered_x).sum(axis=-1) / np.maximum(count, 1))
            y_std = np.sqrt((centered_y * centered_y).sum(axis=-1) / np.maximum(count, 1))
            reduced = cov / (x_std * y_std + 1e-12)
        else:
            raise ValueError(f"Unsupported rolling pair reducer: {reducer}")
        reduced[count == 0] = np.nan
        result[window - 1:] = reduced
        return result

    def _rolling_pair_torch(self, left: ArrayLike, right: ArrayLike, window: int, reducer: str) -> ArrayLike:
        x = left if isinstance(left, torch.Tensor) else self._as_torch_tensor(left)
        y = right if isinstance(right, torch.Tensor) else self._as_torch_tensor(right)
        result = torch.full_like(x, torch.nan)
        if window <= 1 or x.shape[0] < window:
            return result
        x_windows = x.unfold(0, window, 1)
        y_windows = y.unfold(0, window, 1)
        valid = (~torch.isnan(x_windows)) & (~torch.isnan(y_windows))
        count = valid.sum(dim=-1)
        x_safe = torch.where(valid, x_windows, torch.zeros_like(x_windows))
        y_safe = torch.where(valid, y_windows, torch.zeros_like(y_windows))
        x_mean = x_safe.sum(dim=-1) / count.clamp(min=1).to(dtype=x.dtype)
        y_mean = y_safe.sum(dim=-1) / count.clamp(min=1).to(dtype=y.dtype)
        centered_x = torch.where(valid, x_windows - x_mean.unsqueeze(-1), torch.zeros_like(x_windows))
        centered_y = torch.where(valid, y_windows - y_mean.unsqueeze(-1), torch.zeros_like(y_windows))
        cov = (centered_x * centered_y).sum(dim=-1) / count.clamp(min=1).to(dtype=x.dtype)
        if reducer == "cov":
            reduced = cov
        elif reducer == "corr":
            x_std = torch.sqrt((centered_x * centered_x).sum(dim=-1) / count.clamp(min=1).to(dtype=x.dtype))
            y_std = torch.sqrt((centered_y * centered_y).sum(dim=-1) / count.clamp(min=1).to(dtype=y.dtype))
            reduced = cov / (x_std * y_std + 1e-12)
        else:
            raise ValueError(f"Unsupported rolling pair reducer: {reducer}")
        reduced = torch.where(count > 0, reduced, torch.full_like(reduced, torch.nan))
        result[window - 1:] = reduced
        return result

    # --- decay linear ---

    def _decay_linear_numpy(self, arr: ArrayLike, window: int) -> ArrayLike:
        data = np.asarray(arr, dtype=float)
        result = np.full_like(data, np.nan, dtype=float)
        if window <= 0 or data.shape[0] < window:
            return result
        windows = np.lib.stride_tricks.sliding_window_view(data, window_shape=window, axis=0)
        weights = np.arange(1, window + 1, dtype=float)
        valid = ~np.isnan(windows)
        weighted = np.where(valid, windows * weights, 0.0)
        denom = np.where(valid, weights, 0.0).sum(axis=-1)
        reduced = weighted.sum(axis=-1) / np.maximum(denom, 1e-12)
        reduced[denom <= 0] = np.nan
        result[window - 1:] = reduced
        return result

    def _decay_linear_torch(self, arr: ArrayLike, window: int) -> ArrayLike:
        data = arr if isinstance(arr, torch.Tensor) else self._as_torch_tensor(arr)
        result = torch.full_like(data, torch.nan)
        if window <= 0 or data.shape[0] < window:
            return result
        windows = data.unfold(0, window, 1)
        weights = torch.arange(1, window + 1, device=data.device, dtype=data.dtype)
        valid = ~torch.isnan(windows)
        weighted = torch.where(valid, windows * weights, torch.zeros_like(windows))
        denom = torch.where(valid, weights, torch.zeros_like(windows)).sum(dim=-1)
        reduced = weighted.sum(dim=-1) / denom.clamp(min=1e-12)
        reduced = torch.where(denom > 0, reduced, torch.full_like(reduced, torch.nan))
        result[window - 1:] = reduced
        return result

    # --- ts_ema (NEW) ---

    def _ts_ema_numpy(self, arr: ArrayLike, window: int) -> ArrayLike:
        data = np.asarray(arr, dtype=float)
        alpha = 2.0 / (window + 1)
        result = np.full_like(data, np.nan, dtype=float)
        if data.shape[0] == 0:
            return result
        result[0] = data[0]
        for i in range(1, data.shape[0]):
            prev = result[i - 1]
            cur = data[i]
            nan_prev = np.isnan(prev)
            nan_cur = np.isnan(cur)
            result[i] = np.where(
                nan_cur,
                prev,
                np.where(nan_prev, cur, alpha * cur + (1 - alpha) * prev),
            )
        return result

    def _ts_ema_torch(self, arr: ArrayLike, window: int) -> ArrayLike:
        data = arr if isinstance(arr, torch.Tensor) else self._as_torch_tensor(arr)
        alpha = 2.0 / (window + 1)
        result = torch.full_like(data, torch.nan)
        if data.shape[0] == 0:
            return result
        result[0] = data[0]
        for i in range(1, data.shape[0]):
            prev = result[i - 1]
            cur = data[i]
            nan_prev = torch.isnan(prev)
            nan_cur = torch.isnan(cur)
            result[i] = torch.where(
                nan_cur,
                prev,
                torch.where(nan_prev, cur, alpha * cur + (1 - alpha) * prev),
            )
        return result

    # --- ts_winsorize (NEW) ---

    def _ts_winsorize_numpy(self, arr: ArrayLike, window: int, n_std: float) -> ArrayLike:
        data = np.asarray(arr, dtype=float)
        mean = self._rolling_numpy(data, window, np.nanmean)
        std = self._rolling_numpy(data, window, np.nanstd)
        upper = mean + n_std * std
        lower = mean - n_std * std
        return np.clip(data, lower, upper)

    def _ts_winsorize_torch(self, arr: ArrayLike, window: int, n_std: float) -> ArrayLike:
        data = arr if isinstance(arr, torch.Tensor) else self._as_torch_tensor(arr)
        mean = self._rolling_torch(data, window, reducer="mean")
        std = self._rolling_torch(data, window, reducer="std")
        upper = mean + n_std * std
        lower = mean - n_std * std
        return torch.clamp(torch.clamp(data, min=lower), max=upper)

    # --- ts_argextreme ---

    def _ts_argextreme_numpy(self, arr: ArrayLike, window: int, mode: str) -> ArrayLike:
        data = np.asarray(arr, dtype=float)
        result = np.full_like(data, np.nan, dtype=float)
        if window <= 0 or data.shape[0] < window:
            return result
        windows = np.lib.stride_tricks.sliding_window_view(data, window_shape=window, axis=0)
        valid = ~np.isnan(windows)
        if mode == "max":
            safe = np.where(valid, windows, -np.inf)
        elif mode == "min":
            safe = np.where(valid, windows, np.inf)
        else:
            raise ValueError(f"Unsupported argextreme mode: {mode}")
        indices = np.argmax(safe, axis=-1) if mode == "max" else np.argmin(safe, axis=-1)
        counts = valid.sum(axis=-1)
        scaled = indices.astype(float) / max(window - 1, 1)
        scaled[counts == 0] = np.nan
        result[window - 1:] = scaled
        return result

    def _ts_argextreme_torch(self, arr: ArrayLike, window: int, mode: str) -> ArrayLike:
        data = arr if isinstance(arr, torch.Tensor) else self._as_torch_tensor(arr)
        result = torch.full_like(data, torch.nan)
        if window <= 0 or data.shape[0] < window:
            return result
        windows = data.unfold(0, window, 1)
        valid = ~torch.isnan(windows)
        if mode == "max":
            safe = torch.where(valid, windows, torch.full_like(windows, -torch.inf))
            indices = torch.argmax(safe, dim=-1)
        elif mode == "min":
            safe = torch.where(valid, windows, torch.full_like(windows, torch.inf))
            indices = torch.argmin(safe, dim=-1)
        else:
            raise ValueError(f"Unsupported argextreme mode: {mode}")
        counts = valid.sum(dim=-1)
        scaled = indices.to(dtype=data.dtype) / max(window - 1, 1)
        scaled = torch.where(counts > 0, scaled, torch.full_like(scaled, torch.nan))
        result[window - 1:] = scaled
        return result

    # --- ts_rank ---

    def _ts_rank_numpy(self, arr: ArrayLike, window: int) -> ArrayLike:
        data = np.asarray(arr, dtype=float)
        result = np.full_like(data, np.nan, dtype=float)
        if window <= 1 or data.shape[0] < window:
            return result
        windows = np.lib.stride_tricks.sliding_window_view(data, window_shape=window, axis=0)
        last = windows[..., -1:]
        valid = ~np.isnan(windows)
        counts = valid[..., :-1].sum(axis=-1)
        better = np.where(valid[..., :-1], last > windows[..., :-1], False).sum(axis=-1)
        ranked = better / np.maximum(counts, 1)
        ranked[counts == 0] = np.nan
        result[window - 1:] = ranked
        return result

    def _ts_rank_torch(self, arr: ArrayLike, window: int) -> ArrayLike:
        data = arr if isinstance(arr, torch.Tensor) else self._as_torch_tensor(arr)
        result = torch.full_like(data, torch.nan)
        if window <= 1 or data.shape[0] < window:
            return result
        windows = data.unfold(0, window, 1)
        history = windows[..., :-1]
        last = windows[..., -1:].expand_as(history)
        valid = (~torch.isnan(history)) & (~torch.isnan(last))
        counts = valid.sum(dim=-1)
        better = (valid & (last > history)).sum(dim=-1)
        ranked = better.to(dtype=data.dtype) / counts.clamp(min=1).to(dtype=data.dtype)
        ranked = torch.where(counts > 0, ranked, torch.full_like(ranked, torch.nan))
        result[window - 1:] = ranked
        return result

    # --- cs_rank ---

    def _cs_rank_numpy(self, arr: ArrayLike) -> ArrayLike:
        data = np.asarray(arr, dtype=float)
        nan_mask = np.isnan(data)
        safe = np.where(nan_mask, np.inf, data)
        order = np.argsort(safe, axis=1, kind="mergesort")
        ranks = np.argsort(order, axis=1, kind="mergesort") + 1
        denom = (~nan_mask).sum(axis=1, keepdims=True)
        scaled = ranks / np.maximum(denom, 1)
        scaled[nan_mask] = np.nan
        return scaled

    def _cs_rank_torch(self, arr: ArrayLike) -> ArrayLike:
        data = arr if isinstance(arr, torch.Tensor) else self._as_torch_tensor(arr)
        nan_mask = torch.isnan(data)
        safe = torch.where(nan_mask, torch.full_like(data, torch.inf), data)
        order = torch.argsort(safe, dim=1, stable=True)
        ranks = torch.argsort(order, dim=1, stable=True).to(dtype=data.dtype) + 1.0
        denom = (~nan_mask).sum(dim=1, keepdim=True).to(dtype=data.dtype)
        scaled = ranks / denom.clamp(min=1.0)
        scaled = torch.where(nan_mask, torch.full_like(scaled, torch.nan), scaled)
        return scaled

    # --- torch nan-safe helpers ---

    def _torch_nansum(self, tensor: Any, dim: int, keepdim: bool = False) -> Any:
        mask = ~torch.isnan(tensor)
        safe = torch.where(mask, tensor, torch.zeros_like(tensor))
        return safe.sum(dim=dim, keepdim=keepdim)

    def _torch_nanmean(self, tensor: Any, dim: int, keepdim: bool = False) -> Any:
        mask = ~torch.isnan(tensor)
        safe = torch.where(mask, tensor, torch.zeros_like(tensor))
        count = mask.sum(dim=dim, keepdim=keepdim)
        mean = safe.sum(dim=dim, keepdim=keepdim) / count.clamp(min=1).to(dtype=tensor.dtype)
        return torch.where(count > 0, mean, torch.full_like(mean, torch.nan))

    def _torch_nanstd(self, tensor: Any, dim: int, keepdim: bool = False) -> Any:
        mean = self._torch_nanmean(tensor, dim=dim, keepdim=True)
        mask = ~torch.isnan(tensor)
        centered = torch.where(mask, tensor - mean, torch.zeros_like(tensor))
        count = mask.sum(dim=dim, keepdim=keepdim)
        variance = (centered * centered).sum(dim=dim, keepdim=keepdim) / count.clamp(min=1).to(dtype=tensor.dtype)
        std = torch.sqrt(variance)
        return torch.where(count > 0, std, torch.full_like(std, torch.nan))

    def _torch_nanmax(self, tensor: Any, dim: int, keepdim: bool = False) -> Any:
        mask = ~torch.isnan(tensor)
        safe = torch.where(mask, tensor, torch.full_like(tensor, -torch.inf))
        values = safe.max(dim=dim, keepdim=keepdim).values
        all_nan = (~mask).all(dim=dim, keepdim=keepdim)
        return torch.where(all_nan, torch.full_like(values, torch.nan), values)

    def _torch_nanmin(self, tensor: Any, dim: int, keepdim: bool = False) -> Any:
        mask = ~torch.isnan(tensor)
        safe = torch.where(mask, tensor, torch.full_like(tensor, torch.inf))
        values = safe.min(dim=dim, keepdim=keepdim).values
        all_nan = (~mask).all(dim=dim, keepdim=keepdim)
        return torch.where(all_nan, torch.full_like(values, torch.nan), values)
