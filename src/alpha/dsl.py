from __future__ import annotations

import ast
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Iterable

if TYPE_CHECKING:
    from .operators import OperatorRegistry


@dataclass
class ASTNode:
    kind: str
    value: Any = None
    children: list["ASTNode"] = field(default_factory=list)
    inferred_kind: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "kind": self.kind,
            "value": self.value,
            "inferred_kind": self.inferred_kind,
            "children": [child.to_dict() for child in self.children],
        }


@dataclass(frozen=True)
class TensorSchema:
    fields: frozenset[str]
    masks: frozenset[str] = frozenset()

    @classmethod
    def default_market_schema(cls) -> "TensorSchema":
        return cls(
            fields=frozenset(
                {
                    "open",
                    "high",
                    "low",
                    "close",
                    "volume",
                    "turnover",
                    "vwap",
                    "funding_rate",
                    "open_interest",
                    "bid_ask_spread",
                }
            ),
            masks=frozenset({"liquidity_mask", "session_mask"}),
        )

    @classmethod
    def default_stock_schema(cls) -> "TensorSchema":
        return cls(
            fields=frozenset(
                {
                    "open",
                    "high",
                    "low",
                    "close",
                    "volume",
                    "amount",
                    "vwap",
                }
            ),
        )


@dataclass
class ValidationReport:
    ok: bool
    normalized_formula: str
    errors: list[str]
    ast_hash: str | None = None
    ast_tree: dict[str, Any] | None = None


class FormulaParser:
    def __init__(self, registry: "OperatorRegistry"):
        self.registry = registry

    def parse(self, formula: str) -> ASTNode:
        try:
            parsed = ast.parse(formula.strip(), mode="eval")
        except SyntaxError as exc:
            raise ValueError(f"Formula syntax error: {exc.msg}") from exc
        return self._convert(parsed.body)

    def _convert(self, node: ast.AST) -> ASTNode:
        if isinstance(node, ast.Name):
            return ASTNode(kind="name", value=node.id.lower())
        if isinstance(node, ast.Constant):
            return ASTNode(kind="const", value=node.value)
        if isinstance(node, ast.Call):
            if not isinstance(node.func, ast.Name):
                raise ValueError("Only direct function calls are allowed in DSL")
            func_name = self.registry.normalize_name(node.func.id)
            if not self.registry.has(func_name):
                raise ValueError(f"Unsupported operator: {node.func.id}")
            args = [self._convert(arg) for arg in node.args]
            return ASTNode(kind="call", value=func_name, children=args)
        if isinstance(node, ast.BinOp):
            opcode = {
                ast.Add: "add",
                ast.Sub: "sub",
                ast.Mult: "mul",
                ast.Div: "div",
                ast.Pow: "pow",
            }.get(type(node.op))
            if not opcode:
                raise ValueError(f"Unsupported binary operator: {type(node.op).__name__}")
            return ASTNode(kind="call", value=opcode, children=[self._convert(node.left), self._convert(node.right)])
        if isinstance(node, ast.UnaryOp):
            opcode = {
                ast.USub: "neg",
                ast.Not: "not",
            }.get(type(node.op))
            if not opcode:
                raise ValueError(f"Unsupported unary operator: {type(node.op).__name__}")
            return ASTNode(kind="call", value=opcode, children=[self._convert(node.operand)])
        if isinstance(node, ast.Compare):
            if len(node.ops) != 1 or len(node.comparators) != 1:
                raise ValueError("Chained comparisons are not allowed")
            opcode = {
                ast.Gt: "gt",
                ast.GtE: "ge",
                ast.Lt: "lt",
                ast.LtE: "le",
                ast.Eq: "eq",
                ast.NotEq: "ne",
            }.get(type(node.ops[0]))
            if not opcode:
                raise ValueError(f"Unsupported comparison operator: {type(node.ops[0]).__name__}")
            return ASTNode(kind="call", value=opcode, children=[self._convert(node.left), self._convert(node.comparators[0])])
        if isinstance(node, ast.BoolOp):
            opcode = {
                ast.And: "and",
                ast.Or: "or",
            }.get(type(node.op))
            if not opcode:
                raise ValueError(f"Unsupported logical operator: {type(node.op).__name__}")
            values = [self._convert(v) for v in node.values]
            current = values[0]
            for nxt in values[1:]:
                current = ASTNode(kind="call", value=opcode, children=[current, nxt])
            return current
        raise ValueError(f"Unsupported AST node: {type(node).__name__}")


class TypeChecker:
    def __init__(self, registry: "OperatorRegistry"):
        self.registry = registry
        self.field_aliases = {
            "oi": "open_interest",
            "openinterest": "open_interest",
            "fundingrate": "funding_rate",
            "bidaskspread": "bid_ask_spread",
            "bid_ask_spread": "bid_ask_spread",
        }

    def infer(self, ast_node: ASTNode, schema: TensorSchema) -> ASTNode:
        if ast_node.kind == "name":
            compact = str(ast_node.value).replace("_", "")
            ast_node.value = self.field_aliases.get(compact, ast_node.value)
            if ast_node.value in schema.fields:
                ast_node.inferred_kind = "tensor"
                return ast_node
            if ast_node.value in schema.masks:
                ast_node.inferred_kind = "mask"
                return ast_node
            raise ValueError(f"Unknown field or mask: {ast_node.value}")
        if ast_node.kind == "const":
            if not isinstance(ast_node.value, (int, float, bool)):
                raise ValueError(f"Unsupported constant type: {type(ast_node.value).__name__}")
            ast_node.inferred_kind = "scalar"
            return ast_node
        if ast_node.kind != "call":
            raise ValueError(f"Unsupported AST node kind: {ast_node.kind}")

        spec = self.registry.get(ast_node.value)
        child_kinds = [self.infer(child, schema).inferred_kind for child in ast_node.children]
        if not spec.min_args <= len(child_kinds) <= spec.max_args:
            raise ValueError(
                f"Operator {spec.name} expects between {spec.min_args} and {spec.max_args} args, "
                f"received {len(child_kinds)}"
            )

        self._validate_signature(spec.name, child_kinds)
        ast_node.inferred_kind = spec.output_kind
        return ast_node

    def _validate_signature(self, name: str, child_kinds: Iterable[str]) -> None:
        kinds = list(child_kinds)
        if name in {"add", "sub", "mul", "div", "max", "min", "pow"}:
            if not all(kind in {"tensor", "scalar"} for kind in kinds):
                raise ValueError(f"Operator {name} only accepts tensor/scalar inputs")
        elif name in {"gt", "ge", "lt", "le", "eq", "ne"}:
            if not all(kind in {"tensor", "scalar"} for kind in kinds):
                raise ValueError(f"Comparison {name} only accepts tensor/scalar inputs")
        elif name in {"and", "or"}:
            if not all(kind == "mask" for kind in kinds):
                raise ValueError(f"Logical operator {name} only accepts mask inputs")
        elif name == "not":
            if kinds[0] != "mask":
                raise ValueError("Logical operator not only accepts a mask input")
        elif name == "where":
            if kinds[0] != "mask":
                raise ValueError("where(condition, x, y) requires a mask as the first argument")
            if kinds[1] not in {"tensor", "scalar"} or kinds[2] not in {"tensor", "scalar"}:
                raise ValueError("where(condition, x, y) requires tensor/scalar branches")
        elif name in {"clip"}:
            if not all(kind in {"tensor", "scalar"} for kind in kinds):
                raise ValueError("clip(x, lo, hi) requires tensor/scalar inputs")
        elif name in {"fillna"}:
            if kinds[0] not in {"tensor", "scalar"} or kinds[1] not in {"tensor", "scalar"}:
                raise ValueError("fillna(x, value) requires tensor/scalar inputs")
        elif name == "power":
            if kinds[0] not in {"tensor", "scalar"} or kinds[1] != "scalar":
                raise ValueError("power(x, p) requires (tensor/scalar, scalar)")
        elif name in {
            "delay",
            "delta",
            "returns_n",
            "log_return",
            "ts_mean",
            "ts_std",
            "ts_sum",
            "ts_max",
            "ts_min",
            "ts_rank",
            "ts_zscore",
            "decay_linear",
            "ts_argmax",
            "ts_argmin",
            "ts_ema",
            "oi_delta",
            "funding_delta",
            "adv_n",
            "volatility_n",
        }:
            if kinds[0] != "tensor" or kinds[1] != "scalar":
                raise ValueError(f"Operator {name} requires (tensor, scalar_window)")
        elif name == "ts_winsorize":
            if kinds[0] != "tensor" or kinds[1] != "scalar" or kinds[2] != "scalar":
                raise ValueError("ts_winsorize(x, window, n_std) requires (tensor, scalar, scalar)")
        elif name in {"ts_corr", "ts_cov"}:
            if kinds[0] != "tensor" or kinds[1] != "tensor" or kinds[2] != "scalar":
                raise ValueError(f"Operator {name} requires (tensor, tensor, scalar_window)")
        elif name in {"spread_ratio"}:
            if kinds[0] != "tensor" or kinds[1] != "tensor":
                raise ValueError("spread_ratio(spread, mid_price) requires (tensor, tensor)")
        elif name in {"amihud"}:
            if kinds[0] != "tensor" or kinds[1] != "tensor" or kinds[2] != "scalar":
                raise ValueError("amihud(close, turnover, n) requires (tensor, tensor, scalar_window)")
        elif name in {"hlc3", "true_range"}:
            if any(kind != "tensor" for kind in kinds):
                raise ValueError(f"Operator {name} requires tensor inputs")
        elif name in {"ohlc4", "atr_n"}:
            if name == "ohlc4":
                if any(kind != "tensor" for kind in kinds):
                    raise ValueError("ohlc4(open, high, low, close) requires tensor inputs")
            else:
                if kinds[0] != "tensor" or kinds[1] != "tensor" or kinds[2] != "tensor" or kinds[3] != "scalar":
                    raise ValueError("atr_n(high, low, close, n) requires (tensor, tensor, tensor, scalar_window)")
        elif name in {"cs_rank", "cs_scale", "cs_zscore", "cs_demean", "abs", "log", "sign", "sqrt", "sigmoid", "neg"}:
            if kinds[0] not in {"tensor", "scalar"}:
                raise ValueError(f"Operator {name} requires tensor/scalar input")


def normalize_formula(ast_node: ASTNode) -> str:
    if ast_node.kind == "name":
        return str(ast_node.value)
    if ast_node.kind == "const":
        return repr(ast_node.value)
    args = ",".join(normalize_formula(child) for child in ast_node.children)
    return f"{ast_node.value}({args})"
