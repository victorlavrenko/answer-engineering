from __future__ import annotations

import ast
from pathlib import Path


def _is_protocol_base(base: ast.expr) -> bool:
    if isinstance(base, ast.Name):
        return base.id == "Protocol"
    if isinstance(base, ast.Attribute):
        return base.attr == "Protocol"
    return False


def _is_abc_base(base: ast.expr) -> bool:
    if isinstance(base, ast.Name):
        return base.id == "ABC"
    if isinstance(base, ast.Attribute):
        return base.attr == "ABC"
    return False


def _is_abstractmethod_decorator(decorator: ast.expr) -> bool:
    if isinstance(decorator, ast.Name):
        return decorator.id == "abstractmethod"
    if isinstance(decorator, ast.Attribute):
        return decorator.attr == "abstractmethod"
    return False


def test_protocols_do_not_use_abstractmethod_or_abc_hybrids() -> None:
    src_root = (
        Path(__file__).resolve().parents[2] / "src" / "answer_engineering"
    )
    violations: list[str] = []

    for file_path in src_root.rglob("*.py"):
        module = ast.parse(file_path.read_text(), filename=str(file_path))
        relative_path = file_path.relative_to(src_root).as_posix()

        for node in ast.walk(module):
            if not isinstance(node, ast.ClassDef):
                continue

            has_protocol_base = any(
                _is_protocol_base(base) for base in node.bases
            )
            if not has_protocol_base:
                continue

            if any(_is_abc_base(base) for base in node.bases):
                violations.append(
                    f"{relative_path}:{node.lineno} "
                    f"class {node.name} mixes Protocol and ABC"
                )

            for member in node.body:
                if not isinstance(
                    member, (ast.FunctionDef, ast.AsyncFunctionDef)
                ):
                    continue
                if any(
                    _is_abstractmethod_decorator(decorator)
                    for decorator in member.decorator_list
                ):
                    violations.append(
                        f"{relative_path}:{member.lineno} "
                        f"{node.name}.{member.name} uses @abstractmethod"
                    )

    assert violations == []
