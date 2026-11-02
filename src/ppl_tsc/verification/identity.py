"""Reject known process-dependent and stateful primitives in generated rules.

This is a targeted syntax gate, not a proof of determinism of arbitrary Python.
Direct calls, imports and simple aliases are checked. Finite source replay also
checks repeatability; neither check proves determinism of arbitrary Python.
"""

import ast

FORBIDDEN = {"hash", "id", "builtins.hash", "builtins.id"}
UNSTABLE_MODULES = {"random", "time", "datetime", "secrets", "uuid"}
DYNAMIC_CALLS = {
    "eval",
    "exec",
    "compile",
    "__import__",
    "globals",
    "locals",
    "vars",
    "setattr",
    "delattr",
}


def forbidden_call(path):
    if not path:
        return False
    return (
        path in FORBIDDEN
        or path.removeprefix("builtins.") in DYNAMIC_CALLS
        or path.split(".")[0] in UNSTABLE_MODULES
        or path == "os.urandom"
        or path.startswith("numpy.random.")
        or path in {"numpy.random", "numpy.seterr", "numpy.errstate"}
    )


def check_process_identity(code):
    try:
        tree = ast.parse(code)
    except SyntaxError as exc:
        return dict(
            valid=False,
            violations=[f"Invalid Python syntax: {exc}"],
            message="Cannot inspect policy primitives",
        )
    aliases = {name: name for name in FORBIDDEN | DYNAMIC_CALLS if "." not in name}
    aliases.update(builtins="builtins", __builtins__="builtins", np="numpy", numpy="numpy")
    violations = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for name in node.names:
                aliases[name.asname or name.name.split(".")[0]] = (
                    name.name if name.asname else name.name.split(".")[0]
                )
        elif isinstance(node, ast.ImportFrom):
            for name in node.names:
                if name.name == "*":
                    violations.append(
                        f"Line {node.lineno}: use explicit imports so utility dependencies can be checked."
                    )
                else:
                    aliases[name.asname or name.name] = (node.module or "") + "." + name.name
        elif isinstance(node, (ast.Global, ast.Nonlocal)):
            violations.append(
                f"Line {node.lineno}: global/nonlocal state is not a traffic input. "
                "Keep scratch variables local to each utility call."
            )
        elif isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            defaults = [*node.args.defaults, *[d for d in node.args.kw_defaults if d is not None]]
            if any(
                isinstance(
                    part, (ast.List, ast.Dict, ast.Set, ast.ListComp, ast.DictComp, ast.SetComp)
                )
                for default in defaults
                for part in ast.walk(default)
            ):
                violations.append(
                    f"Line {node.lineno}: mutable function defaults persist between calls. "
                    "Use local scratch variables or fixed read-only constants."
                )

    def resolve(node):
        if isinstance(node, ast.Name):
            return aliases.get(node.id)
        if isinstance(node, ast.Attribute):
            base = resolve(node.value)
            return base + "." + node.attr if base else None
        if isinstance(node, ast.Subscript) and isinstance(node.slice, ast.Constant):
            if resolve(node.value) == "builtins" and node.slice.value in {"hash", "id"}:
                return "builtins." + node.slice.value
        if (
            isinstance(node, ast.Call)
            and isinstance(node.func, ast.Name)
            and node.func.id == "getattr"
        ):
            if (
                len(node.args) >= 2
                and resolve(node.args[0]) == "builtins"
                and isinstance(node.args[1], ast.Constant)
            ):
                if node.args[1].value in {"hash", "id"}:
                    return "builtins." + node.args[1].value
            if (
                len(node.args) >= 2
                and isinstance(node.args[1], ast.Constant)
                and isinstance(node.args[1].value, str)
            ):
                base = resolve(node.args[0])
                return base + "." + node.args[1].value if base else None
        return None

    # Resolve simple aliases h=hash; j=h. This deliberately does not execute code.
    for _ in range(len(list(ast.walk(tree))) + 1):
        changed = False
        for node in ast.walk(tree):
            if isinstance(node, ast.Assign):
                targets, value = node.targets, node.value
            elif isinstance(node, ast.AnnAssign):
                targets, value = [node.target], node.value
            else:
                continue
            resolved = resolve(value)
            if resolved:
                for target in targets:
                    if isinstance(target, ast.Name) and target.id not in aliases:
                        aliases[target.id] = resolved
                        changed = True
        if not changed:
            break
    for node in ast.walk(tree):
        if isinstance(node, ast.Call) and forbidden_call(resolve(node.func)):
            primitive = resolve(node.func)
            violations.append(
                f"Line {node.lineno}: {primitive} is not a reproducible traffic utility primitive. "
                "Use declared traffic inputs and fixed source statistics; avoid random numbers, clocks, "
                "object identity, dynamic code and numerical error suppression."
            )
    return dict(
        valid=not violations,
        violations=violations,
        message=(
            "Non-reproducible utility primitive rejected"
            if violations
            else "No known stateful utility primitive"
        ),
    )


def require_no_process_identity(code):
    check = check_process_identity(code)
    if not check["valid"]:
        raise ValueError(check["violations"][0])
