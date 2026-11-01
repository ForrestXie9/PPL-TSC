"""Reject process-dependent hash/object-identity primitives in generated rules.

This is a targeted syntax gate, not a proof of determinism of arbitrary Python.
Ordinary variables named id/hash and strings/comments are allowed. Direct calls,
import aliases and simple callable aliases of the built-ins are checked.
"""
import ast

FORBIDDEN = {'hash', 'id', 'builtins.hash', 'builtins.id'}


def check_process_identity(code):
    try:
        tree = ast.parse(code)
    except SyntaxError as exc:
        return dict(valid=False, violations=[f'Invalid Python syntax: {exc}'], message='Cannot inspect policy primitives')
    aliases = {'hash': 'hash', 'id': 'id', 'builtins': 'builtins', '__builtins__': 'builtins'}
    # Track explicit imports first. Other libraries are not rewritten.
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for name in node.names:
                if name.name == 'builtins':
                    aliases[name.asname or name.name] = 'builtins'
        elif isinstance(node, ast.ImportFrom) and node.module == 'builtins':
            for name in node.names:
                if name.name in {'hash', 'id'}:
                    aliases[name.asname or name.name] = 'builtins.' + name.name

    def resolve(node):
        if isinstance(node, ast.Name):
            return aliases.get(node.id)
        if isinstance(node, ast.Attribute):
            base = resolve(node.value)
            return base + '.' + node.attr if base else None
        if isinstance(node, ast.Subscript) and isinstance(node.slice, ast.Constant):
            if resolve(node.value) == 'builtins' and node.slice.value in {'hash', 'id'}:
                return 'builtins.' + node.slice.value
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Name) and node.func.id == 'getattr':
            if len(node.args) >= 2 and resolve(node.args[0]) == 'builtins' and isinstance(node.args[1], ast.Constant):
                if node.args[1].value in {'hash', 'id'}:
                    return 'builtins.' + node.args[1].value
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
            if resolved in FORBIDDEN or resolved == 'builtins':
                for target in targets:
                    if isinstance(target, ast.Name) and aliases.get(target.id) != resolved:
                        aliases[target.id] = resolved
                        changed = True
        if not changed:
            break
    violations = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Call) and resolve(node.func) in FORBIDDEN:
            primitive = resolve(node.func)
            violations.append(f'Line {node.lineno}: {primitive} is not a reproducible traffic feature. '
                'String hashes change across interpreter processes and object identities depend on allocation. '
                'Remove hash/identity-based utility weights; use declared traffic measurements and fixed source statistics.')
    return dict(valid=not violations, violations=violations,
                message='Process-dependent hash/identity primitive rejected' if violations else 'No known hash/identity call')


def require_no_process_identity(code):
    check = check_process_identity(code)
    if not check['valid']:
        raise ValueError(check['violations'][0])
