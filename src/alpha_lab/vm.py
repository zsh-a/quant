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


class StackVM:
    def run(self, program: BytecodeProgram, store: TensorStore) -> ArrayLike:
        registers: dict[int, ArrayLike] = {}
        output_shape = store.shape()

        for ins in program.instructions:
            if ins.opcode == "push_field":
                registers[ins.dst] = store.get_field(ins.value)
                continue
            if ins.opcode == "push_const":
                registers[ins.dst] = self._broadcast_const(ins.value, output_shape, store)
                continue
            args = [registers[idx] for idx in ins.args]
            registers[ins.dst] = self._execute(ins.opcode, args)

        return registers[program.output_register]

    def run_batch(self, programs: list[BytecodeProgram], store: TensorStore) -> list[ArrayLike]:
        return [self.run(program, store) for program in programs]

    def _broadcast_const(self, value: Any, shape: tuple[int, ...], store: TensorStore) -> ArrayLike:
        first = next(iter(store.fields.values()))
        if torch is not None and isinstance(first, torch.Tensor):
            return torch.full(shape, float(value), dtype=first.dtype, device=first.device)
        return np.full(shape, float(value), dtype=float)

    def _execute(self, opcode: str, args: list[ArrayLike]) -> ArrayLike:
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
            return self._delay(args[0], int(self._scalar(args[1])))
        if opcode == "delta":
            delayed = self._delay(args[0], int(self._scalar(args[1])))
            return args[0] - delayed
        if opcode == "ts_mean":
            return self._rolling(args[0], int(self._scalar(args[1])), np.nanmean)
        if opcode == "ts_std":
            return self._rolling(args[0], int(self._scalar(args[1])), np.nanstd)
        if opcode == "ts_sum":
            return self._rolling(args[0], int(self._scalar(args[1])), np.nansum)
        if opcode == "ts_max":
            return self._rolling(args[0], int(self._scalar(args[1])), np.nanmax)
        if opcode == "ts_min":
            return self._rolling(args[0], int(self._scalar(args[1])), np.nanmin)
        if opcode == "ts_rank":
            return self._ts_rank(args[0], int(self._scalar(args[1])))
        if opcode == "cs_rank":
            return self._cs_rank(args[0])
        if opcode == "cs_scale":
            denom = np.nansum(np.abs(args[0]), axis=1, keepdims=True)
            return args[0] / (denom + 1e-12)
        if opcode == "volatility_n":
            returns = args[0] / (self._delay(args[0], 1) + 1e-12) - 1.0
            return self._rolling(returns, int(self._scalar(args[1])), np.nanstd)
        raise ValueError(f"Unsupported opcode: {opcode}")

    def _scalar(self, value: ArrayLike) -> float:
        if np.isscalar(value):
            return float(value)
        flat = np.asarray(value).reshape(-1)
        return float(flat[0])

    def _delay(self, arr: ArrayLike, periods: int) -> ArrayLike:
        result = np.full_like(arr, np.nan, dtype=float)
        if periods <= 0:
            return np.asarray(arr, dtype=float)
        result[periods:] = np.asarray(arr[:-periods], dtype=float)
        return result

    def _rolling(self, arr: ArrayLike, window: int, reducer) -> ArrayLike:
        data = np.asarray(arr, dtype=float)
        result = np.full_like(data, np.nan, dtype=float)
        if window <= 0 or data.shape[0] < window:
            return result
        windows = np.lib.stride_tricks.sliding_window_view(data, window_shape=window, axis=0)
        reduced = reducer(windows, axis=-1)
        result[window - 1 :] = reduced
        return result

    def _ts_rank(self, arr: ArrayLike, window: int) -> ArrayLike:
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

    def _cs_rank(self, arr: ArrayLike) -> ArrayLike:
        data = np.asarray(arr, dtype=float)
        nan_mask = np.isnan(data)
        safe = np.where(nan_mask, np.inf, data)
        order = np.argsort(safe, axis=1, kind="mergesort")
        ranks = np.argsort(order, axis=1, kind="mergesort") + 1
        denom = (~nan_mask).sum(axis=1, keepdims=True)
        scaled = ranks / np.maximum(denom, 1)
        scaled[nan_mask] = np.nan
        return scaled
