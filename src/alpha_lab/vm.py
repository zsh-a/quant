from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from .compiler import BytecodeProgram

try:
    import torch
except Exception:  # pragma: no cover - torch is optional for this scaffold
    torch = None


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


class StackVM:
    def __init__(self, device: str | None = None, prefer_torch: bool = True):
        self.prefer_torch = prefer_torch
        self.device = self._resolve_device(device)
        self.backend = "torch" if self.device is not None else "numpy"

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
                    cache_key,
                    shared_cache,
                    lambda: store.get_field(ins.value),
                )
                register_keys[ins.dst] = cache_key
                continue
            if ins.opcode == "push_const":
                cache_key = ("const", float(ins.value), output_shape)
                registers[ins.dst] = self._cache_lookup_or_compute(
                    cache_key,
                    shared_cache,
                    lambda: self._broadcast_const(ins.value, output_shape, store),
                )
                register_keys[ins.dst] = cache_key
                continue
            args = [registers[idx] for idx in ins.args]
            cache_key = (ins.opcode, *(register_keys[idx] for idx in ins.args))
            registers[ins.dst] = self._cache_lookup_or_compute(
                cache_key,
                shared_cache,
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
        if store.uses_torch():
            fields = {
                name: self._as_torch_tensor(value)
                for name, value in store.fields.items()
            }
            return TensorStore(fields)
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
        compute: Any,
    ) -> ArrayLike:
        if shared_cache is None:
            return compute()
        if cache_key not in shared_cache:
            shared_cache[cache_key] = compute()
        return shared_cache[cache_key]

    def _broadcast_const(self, value: Any, shape: tuple[int, ...], store: TensorStore) -> ArrayLike:
        if self.backend == "torch":
            first = next(iter(store.fields.values()))
            return torch.full(shape, float(value), dtype=first.dtype, device=first.device)
        return np.full(shape, float(value), dtype=float)

    def _execute(self, opcode: str, args: list[ArrayLike]) -> ArrayLike:
        if self.backend == "torch":
            return self._execute_torch(opcode, args)
        return self._execute_numpy(opcode, args)

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
        if opcode == "gt":
            return args[0] > args[1]
        if opcode == "ge":
            return args[0] >= args[1]
        if opcode == "lt":
            return args[0] < args[1]
        if opcode == "le":
            return args[0] <= args[1]
        if opcode == "and":
            return np.logical_and(args[0], args[1])
        if opcode == "or":
            return np.logical_or(args[0], args[1])
        if opcode == "where":
            return np.where(args[0], args[1], args[2])
        if opcode == "delay":
            return self._delay_numpy(args[0], int(self._scalar(args[1])))
        if opcode == "delta":
            delayed = self._delay_numpy(args[0], int(self._scalar(args[1])))
            return args[0] - delayed
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
        if opcode == "cs_rank":
            return self._cs_rank_numpy(args[0])
        if opcode == "cs_scale":
            denom = np.nansum(np.abs(args[0]), axis=1, keepdims=True)
            return args[0] / (denom + 1e-12)
        if opcode == "volatility_n":
            returns = args[0] / (self._delay_numpy(args[0], 1) + 1e-12) - 1.0
            return self._rolling_numpy(returns, int(self._scalar(args[1])), np.nanstd)
        raise ValueError(f"Unsupported opcode: {opcode}")

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
        if opcode == "gt":
            return args[0] > args[1]
        if opcode == "ge":
            return args[0] >= args[1]
        if opcode == "lt":
            return args[0] < args[1]
        if opcode == "le":
            return args[0] <= args[1]
        if opcode == "and":
            return torch.logical_and(args[0], args[1])
        if opcode == "or":
            return torch.logical_or(args[0], args[1])
        if opcode == "where":
            return torch.where(args[0], args[1], args[2])
        if opcode == "delay":
            return self._delay_torch(args[0], int(self._scalar(args[1])))
        if opcode == "delta":
            delayed = self._delay_torch(args[0], int(self._scalar(args[1])))
            return args[0] - delayed
        if opcode == "ts_mean":
            return self._rolling_torch(args[0], int(self._scalar(args[1])), reducer="mean")
        if opcode == "ts_std":
            return self._rolling_torch(args[0], int(self._scalar(args[1])), reducer="std")
        if opcode == "ts_sum":
            return self._rolling_torch(args[0], int(self._scalar(args[1])), reducer="sum")
        if opcode == "ts_max":
            return self._rolling_torch(args[0], int(self._scalar(args[1])), reducer="max")
        if opcode == "ts_min":
            return self._rolling_torch(args[0], int(self._scalar(args[1])), reducer="min")
        if opcode == "ts_rank":
            return self._ts_rank_torch(args[0], int(self._scalar(args[1])))
        if opcode == "cs_rank":
            return self._cs_rank_torch(args[0])
        if opcode == "cs_scale":
            denom = self._torch_nansum(torch.abs(args[0]), dim=1, keepdim=True)
            return args[0] / (denom + 1e-12)
        if opcode == "volatility_n":
            returns = args[0] / (self._delay_torch(args[0], 1) + 1e-12) - 1.0
            return self._rolling_torch(returns, int(self._scalar(args[1])), reducer="std")
        raise ValueError(f"Unsupported opcode: {opcode}")

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

    def _rolling_numpy(self, arr: ArrayLike, window: int, reducer) -> ArrayLike:
        data = np.asarray(arr, dtype=float)
        result = np.full_like(data, np.nan, dtype=float)
        if window <= 0 or data.shape[0] < window:
            return result
        windows = np.lib.stride_tricks.sliding_window_view(data, window_shape=window, axis=0)
        reduced = reducer(windows, axis=-1)
        result[window - 1 :] = reduced
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
        result[window - 1 :] = reduced
        return result

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
        result[window - 1 :] = ranked
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
        ranked = torch.where(
            counts > 0,
            ranked,
            torch.full_like(ranked, torch.nan),
        )
        result[window - 1 :] = ranked
        return result

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
