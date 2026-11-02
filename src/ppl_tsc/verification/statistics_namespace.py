"""Identify proven reads from frozen statistics; never whitelist raw state fields.

This is an AST-only namespace check. It does not execute candidate programs.
Unknown bindings are handled conservatively by the existing feature checker.
"""

import ast

ROOT = ("root", frozenset())
EMPTY = ("empty", frozenset())


def mask_statistic_keys(code, statistics):
    tree = ast.parse(code)
    literals = set()
    violations = []

    def merge(a, b):
        if a == b:
            return a
        if a == EMPTY and b and b[0] in {"root", "entry"}:
            return b
        if b == EMPTY and a and a[0] in {"root", "entry"}:
            return a
        if a and b and a[0] == b[0] == "entry":
            return ("entry", a[1] | b[1])
        return None

    def access(container, key):
        if not container or not isinstance(key, ast.Constant) or not isinstance(key.value, str):
            return None
        name = key.value
        if container[0] == "root":
            if name in statistics:
                literals.add((key.lineno, key.col_offset, key.end_lineno, key.end_col_offset))
                return ("entry", frozenset([name]))
            violations.append(
                f"Statistics alias '{name}' is absent from frozen feature_stats; available aliases: {sorted(statistics)}"
            )
        elif container[0] == "entry":
            allowed = set().union(*(set(statistics[k]) for k in container[1]))
            if name in allowed:
                literals.add((key.lineno, key.col_offset, key.end_lineno, key.end_col_offset))
            else:
                violations.append(
                    f"Statistics field '{name}' is absent from the source entries {sorted(container[1])}; available fields: {sorted(allowed)}"
                )
        return None

    def expr(node, env):
        if node is None:
            return None
        if isinstance(node, ast.Name):
            return env.get(node.id)
        if isinstance(node, ast.Dict) and not node.keys:
            return EMPTY
        if isinstance(node, ast.Constant) and node.value is None:
            return EMPTY
        if isinstance(node, ast.IfExp):
            expr(node.test, env)
            return merge(expr(node.body, env), expr(node.orelse, env))
        if isinstance(node, ast.BoolOp):
            values = [expr(x, env) for x in node.values]
            if isinstance(node.op, ast.Or):
                result = values[0]
                for value in values[1:]:
                    result = merge(result, value)
                return result
            return None
        if isinstance(node, ast.Subscript):
            container = expr(node.value, env)
            expr(node.slice, env)
            return access(container, node.slice)
        if isinstance(node, ast.Call):
            if isinstance(node.func, ast.Attribute):
                container = expr(node.func.value, env)
                for argument in node.args:
                    expr(argument, env)
                for argument in node.keywords:
                    expr(argument.value, env)
                if node.func.attr == "get" and node.args:
                    return access(container, node.args[0])
                if node.func.attr == "copy" and not node.args:
                    return container
                return None
            expr(node.func, env)
            for argument in node.args:
                expr(argument, env)
            for argument in node.keywords:
                expr(argument.value, env)
            return None
        for child in ast.iter_child_nodes(node):
            expr(child, env)
        return None

    def bind(target, value, env):
        if isinstance(target, ast.Name):
            env[target.id] = value
        elif isinstance(target, (ast.Tuple, ast.List)):
            for child in target.elts:
                bind(child, None, env)
        else:
            expr(target, env)

    def join_envs(a, b):
        return {name: merge(a.get(name), b.get(name)) for name in a.keys() | b.keys()}

    def known_truth(node, env):
        """Resolve only conditions proved by the supplied frozen dictionaries."""
        if isinstance(node, ast.Name):
            binding = env.get(node.id)
            if binding == ROOT:
                return bool(statistics)
            if binding == EMPTY:
                return False
            if binding and binding[0] == "entry":
                values = {bool(statistics[key]) for key in binding[1]}
                if len(values) == 1:
                    return values.pop()
        if isinstance(node, ast.UnaryOp) and isinstance(node.op, ast.Not):
            value = known_truth(node.operand, env)
            return None if value is None else not value
        if isinstance(node, ast.BoolOp):
            values = [known_truth(value, env) for value in node.values]
            if isinstance(node.op, ast.Or):
                if True in values:
                    return True
                if all(value is False for value in values):
                    return False
            elif isinstance(node.op, ast.And):
                if False in values:
                    return False
                if all(value is True for value in values):
                    return True
        if isinstance(node, ast.Compare) and len(node.ops) == 1 and len(node.comparators) == 1:
            a, b = node.left, node.comparators[0]
            if isinstance(a, ast.Constant) and a.value is None:
                a, b = b, a
            binding = env.get(a.id) if isinstance(a, ast.Name) else None
            if (
                binding
                and binding[0] in {"root", "entry"}
                and isinstance(b, ast.Constant)
                and b.value is None
            ):
                if isinstance(node.ops[0], (ast.Is, ast.Eq)):
                    return False
                if isinstance(node.ops[0], (ast.IsNot, ast.NotEq)):
                    return True
        return None

    def block(nodes, env):
        for node in nodes:
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                local = dict(env)
                for argument in [*node.args.posonlyargs, *node.args.args, *node.args.kwonlyargs]:
                    local[argument.arg] = None
                if node.name == "compute_utility":
                    local["feature_stats"] = ROOT
                block(node.body, local)
                env[node.name] = None
            elif isinstance(node, ast.Assign):
                value = expr(node.value, env)
                for target in node.targets:
                    bind(target, value, env)
            elif isinstance(node, ast.AnnAssign):
                bind(node.target, expr(node.value, env), env)
            elif isinstance(node, ast.AugAssign):
                expr(node.value, env)
                bind(node.target, None, env)
            elif isinstance(node, ast.If):
                expr(node.test, env)
                # Runtime supplies the frozen dictionaries. Unreachable None/
                # empty fallbacks must not erase their namespace provenance.
                known = known_truth(node.test, env)
                if known is not None:
                    block(node.body if known else node.orelse, env)
                    continue
                left, right = dict(env), dict(env)
                block(node.body, left)
                block(node.orelse, right)
                env.clear()
                env.update(join_envs(left, right))
            elif isinstance(node, (ast.For, ast.AsyncFor, ast.While)):
                expr(node.iter if isinstance(node, (ast.For, ast.AsyncFor)) else node.test, env)
                prior, body = dict(env), dict(env)
                if isinstance(node, (ast.For, ast.AsyncFor)):
                    bind(node.target, None, body)
                block(node.body, body)
                block(node.orelse, body)
                env.clear()
                env.update(join_envs(prior, body))
            elif isinstance(node, ast.Try):
                paths = []
                normal = dict(env)
                block(node.body, normal)
                block(node.orelse, normal)
                paths.append(normal)
                for handler in node.handlers:
                    branch = dict(env)
                    block(handler.body, branch)
                    paths.append(branch)
                joined = paths[0]
                for path in paths[1:]:
                    joined = join_envs(joined, path)
                env.clear()
                env.update(joined)
                block(node.finalbody, env)
            else:
                for child in ast.iter_child_nodes(node):
                    expr(child, env)
        return env

    block(tree.body, {})
    # AST offsets are UTF-8 byte offsets; preserve the original code everywhere
    # except the proven statistics field string literals in this checker copy.
    lines = code.encode("utf-8").splitlines(keepends=True)
    starts = [0]
    for line in lines:
        starts.append(starts[-1] + len(line))
    masked = code.encode("utf-8")
    for line, col, end_line, end_col in sorted(literals, reverse=True):
        begin, end = starts[line - 1] + col, starts[end_line - 1] + end_col
        masked = masked[:begin] + b"'std'" + masked[end:]
    return masked.decode("utf-8"), sorted(set(violations)), len(literals)
