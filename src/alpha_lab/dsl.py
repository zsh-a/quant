from __future__ import annotations

import ast
import hashlib
from dataclasses import dataclass, field
from typing import Any, Iterable


@dataclass(frozen=True)
class OperatorSpec:
    name: str
    min_args: int
    max_args: int
    output_kind: str
    category: str


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


@dataclass
class ValidationReport:
    ok: bool
    normalized_formula: str
    errors: list[str]
    ast_hash: str | None = None
    ast_tree: dict[str, Any] | None = None


class DSLRegistry:
    def __init__(self):
        self._operators = self._build_defaults()
        self._aliases = self._build_aliases()

    def _build_defaults(self) -> dict[str, OperatorSpec]:
        specs = [
            OperatorSpec("abs", 1, 1, "tensor", "unary"),
            OperatorSpec("log", 1, 1, "tensor", "unary"),
            OperatorSpec("sign", 1, 1, "tensor", "unary"),
            OperatorSpec("sqrt", 1, 1, "tensor", "unary"),
            OperatorSpec("sigmoid", 1, 1, "tensor", "unary"),
            OperatorSpec("neg", 1, 1, "tensor", "unary"),
            OperatorSpec("add", 2, 2, "tensor", "binary"),
            OperatorSpec("sub", 2, 2, "tensor", "binary"),
            OperatorSpec("mul", 2, 2, "tensor", "binary"),
            OperatorSpec("div", 2, 2, "tensor", "binary"),
            OperatorSpec("max", 2, 2, "tensor", "binary"),
            OperatorSpec("min", 2, 2, "tensor", "binary"),
            OperatorSpec("pow", 2, 2, "tensor", "binary"),
            OperatorSpec("where", 3, 3, "tensor", "conditional"),
            OperatorSpec("gt", 2, 2, "mask", "comparison"),
            OperatorSpec("ge", 2, 2, "mask", "comparison"),
            OperatorSpec("lt", 2, 2, "mask", "comparison"),
            OperatorSpec("le", 2, 2, "mask", "comparison"),
            OperatorSpec("and", 2, 2, "mask", "logical"),
            OperatorSpec("or", 2, 2, "mask", "logical"),
            OperatorSpec("delay", 2, 2, "tensor", "time_series"),
            OperatorSpec("delta", 2, 2, "tensor", "time_series"),
            OperatorSpec("ts_mean", 2, 2, "tensor", "time_series"),
            OperatorSpec("ts_std", 2, 2, "tensor", "time_series"),
            OperatorSpec("ts_sum", 2, 2, "tensor", "time_series"),
            OperatorSpec("ts_max", 2, 2, "tensor", "time_series"),
            OperatorSpec("ts_min", 2, 2, "tensor", "time_series"),
            OperatorSpec("ts_rank", 2, 2, "tensor", "time_series"),
            OperatorSpec("cs_rank", 1, 1, "tensor", "cross_sectional"),
            OperatorSpec("cs_scale", 1, 1, "tensor", "cross_sectional"),
            OperatorSpec("volatility_n", 2, 2, "tensor", "domain"),
        ]
        return {spec.name: spec for spec in specs}

    def list_operators(self) -> list[OperatorSpec]:
        return sorted(self._operators.values(), key=lambda spec: spec.name)

    def has(self, name: str) -> bool:
        return self.normalize_name(name) in self._operators

    def get(self, name: str) -> OperatorSpec:
        return self._operators[self.normalize_name(name)]

    def normalize_name(self, name: str) -> str:
        lowered = name.lower()
        if lowered in self._operators:
            return lowered
        compact = lowered.replace("_", "")
        return self._aliases.get(compact, lowered)

    def _build_aliases(self) -> dict[str, str]:
        aliases: dict[str, str] = {}
        for canonical in self._operators:
            aliases[canonical.replace("_", "")] = canonical
        aliases.update(
            {
                "csrank": "cs_rank",
                "csscale": "cs_scale",
                "volatility": "volatility_n",
                "tsmax": "ts_max",
                "tsmin": "ts_min",
                "tsmean": "ts_mean",
                "tsstd": "ts_std",
                "tssum": "ts_sum",
                "tsrank": "ts_rank",
            }
        )
        return aliases

    def validate_formula(
        self,
        formula: str,
        schema: TensorSchema | None = None,
    ) -> ValidationReport:
        parser = FormulaParser(self)
        checker = TypeChecker(self)
        try:
            parsed = parser.parse(formula)
            typed = checker.infer(parsed, schema or TensorSchema.default_market_schema())
            normalized = normalize_formula(parsed)
            ast_hash = hashlib.sha256(normalized.encode("utf-8")).hexdigest()
            return ValidationReport(
                ok=True,
                normalized_formula=normalized,
                errors=[],
                ast_hash=ast_hash,
                ast_tree=typed.to_dict(),
            )
        except ValueError as exc:
            return ValidationReport(
                ok=False,
                normalized_formula=formula.strip(),
                errors=[str(exc)],
            )


class FormulaParser:
    def __init__(self, registry: DSLRegistry):
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
    def __init__(self, registry: DSLRegistry):
        self.registry = registry

    def infer(self, ast_node: ASTNode, schema: TensorSchema) -> ASTNode:
        if ast_node.kind == "name":
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
        elif name in {"gt", "ge", "lt", "le"}:
            if not all(kind in {"tensor", "scalar"} for kind in kinds):
                raise ValueError(f"Comparison {name} only accepts tensor/scalar inputs")
        elif name in {"and", "or"}:
            if not all(kind == "mask" for kind in kinds):
                raise ValueError(f"Logical operator {name} only accepts mask inputs")
        elif name == "where":
            if kinds[0] != "mask":
                raise ValueError("where(condition, x, y) requires a mask as the first argument")
            if kinds[1] not in {"tensor", "scalar"} or kinds[2] not in {"tensor", "scalar"}:
                raise ValueError("where(condition, x, y) requires tensor/scalar branches")
        elif name in {"delay", "delta", "ts_mean", "ts_std", "ts_sum", "ts_max", "ts_min", "ts_rank", "volatility_n"}:
            if kinds[0] != "tensor" or kinds[1] != "scalar":
                raise ValueError(f"Operator {name} requires (tensor, scalar_window)")
        elif name in {"cs_rank", "cs_scale", "abs", "log", "sign", "sqrt", "sigmoid", "neg"}:
            if kinds[0] not in {"tensor", "scalar"}:
                raise ValueError(f"Operator {name} requires tensor/scalar input")


def normalize_formula(ast_node: ASTNode) -> str:
    if ast_node.kind == "name":
        return str(ast_node.value)
    if ast_node.kind == "const":
        return repr(ast_node.value)
    args = ",".join(normalize_formula(child) for child in ast_node.children)
    return f"{ast_node.value}({args})"
