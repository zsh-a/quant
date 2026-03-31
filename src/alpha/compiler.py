from __future__ import annotations

import hashlib
from dataclasses import dataclass, field
from typing import Any

from .dsl import ASTNode, FormulaParser, TensorSchema, TypeChecker, normalize_formula
from .operators import OperatorRegistry


@dataclass
class Instruction:
    opcode: str
    dst: int
    args: tuple[int, ...] = ()
    value: Any = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "opcode": self.opcode,
            "dst": self.dst,
            "args": list(self.args),
            "value": self.value,
        }


@dataclass
class BytecodeProgram:
    instructions: list[Instruction]
    output_register: int
    expr_hash: str
    normalized_formula: str
    max_register: int = field(default=0)

    def to_dict(self) -> dict[str, Any]:
        return {
            "expr_hash": self.expr_hash,
            "normalized_formula": self.normalized_formula,
            "output_register": self.output_register,
            "max_register": self.max_register,
            "instructions": [ins.to_dict() for ins in self.instructions],
        }


class FormulaCompiler:
    def __init__(
        self,
        registry: OperatorRegistry | None = None,
        parser: FormulaParser | None = None,
        checker: TypeChecker | None = None,
    ):
        self.registry = registry or OperatorRegistry()
        self.parser = parser or FormulaParser(self.registry)
        self.checker = checker or TypeChecker(self.registry)
        self._next_register = 0
        self._instructions: list[Instruction] = []

    def compile(self, formula: str, schema: TensorSchema | None = None) -> BytecodeProgram:
        self._next_register = 0
        self._instructions = []
        schema = schema or TensorSchema.default_market_schema()
        parsed = self.parser.parse(formula)
        typed = self.checker.infer(parsed, schema)
        output_register = self._emit_node(typed)
        normalized = normalize_formula(typed)
        expr_hash = hashlib.sha256(normalized.encode("utf-8")).hexdigest()
        return BytecodeProgram(
            instructions=list(self._instructions),
            output_register=output_register,
            expr_hash=expr_hash,
            normalized_formula=normalized,
            max_register=max((ins.dst for ins in self._instructions), default=output_register),
        )

    def _alloc(self) -> int:
        reg = self._next_register
        self._next_register += 1
        return reg

    def _emit_node(self, node: ASTNode) -> int:
        if node.kind == "name":
            dst = self._alloc()
            self._instructions.append(Instruction("push_field", dst, value=node.value))
            return dst
        if node.kind == "const":
            dst = self._alloc()
            self._instructions.append(Instruction("push_const", dst, value=node.value))
            return dst
        if node.kind != "call":
            raise ValueError(f"Unsupported node kind during compilation: {node.kind}")
        arg_regs = tuple(self._emit_node(child) for child in node.children)
        dst = self._alloc()
        self._instructions.append(Instruction(node.value, dst, args=arg_regs))
        return dst
