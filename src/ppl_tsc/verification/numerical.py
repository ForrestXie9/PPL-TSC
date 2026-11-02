"""Conservative scalar-division checks and deterministic source-independent probes.

This is a candidate admission check, not a proof of arbitrary Python programs.
Unknown signed denominators require an explicit guard. Sampled traffic ranges
never establish that an unseen denominator is positive.
"""

from __future__ import annotations
import ast
import copy
import math

UNKNOWN = (-math.inf, math.inf)


def key(node):
    return ast.dump(node, include_attributes=False)


def call_name(node):
    if isinstance(node, ast.Name):
        return node.id
    if isinstance(node, ast.Attribute):
        return call_name(node.value) + "." + node.attr
    return ""


class DivisionCheck:
    def __init__(self, tree):
        self.violations = []
        self.shadowed = {
            n.id for n in ast.walk(tree) if isinstance(n, ast.Name) and isinstance(n.ctx, ast.Store)
        }
        self.shadowed.update(n.arg for n in ast.walk(tree) if isinstance(n, ast.arg))
        self.import_aliases = {}
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                for alias in node.names:
                    self.import_aliases.setdefault(
                        alias.asname or alias.name.split(".")[0], set()
                    ).add(alias.name)
            elif isinstance(node, ast.ImportFrom) and node.module:
                for alias in node.names:
                    self.import_aliases.setdefault(alias.asname or alias.name, set()).add(
                        node.module + "." + alias.name
                    )
        self.shadowed.update(set(self.import_aliases) & {"abs", "max", "min", "len"})

    def numeric_call_names(self, node):
        """Resolve imported aliases for hazard detection, never as safety proofs.

        Multiple bindings conservatively retain every possible imported name.
        Arbitrary dynamic rebinding still needs executed numerical validation.
        """
        name = call_name(node)
        root, dot, tail = name.partition(".")
        return {name} | {
            resolved + (dot + tail if dot else "") for resolved in self.import_aliases.get(root, ())
        }

    def bounds(self, node, env):
        if isinstance(node, ast.Constant) and type(node.value) in (int, float):
            try:
                x = float(node.value)
                return (x, x) if math.isfinite(x) else UNKNOWN
            except OverflowError:
                return UNKNOWN
        if isinstance(node, ast.Name):
            return env.get(node.id, UNKNOWN)
        if isinstance(node, ast.UnaryOp):
            lo, hi = self.bounds(node.operand, env)
            if isinstance(node.op, ast.USub):
                return -hi, -lo
            if isinstance(node.op, ast.UAdd):
                return lo, hi
        if isinstance(node, ast.BinOp):
            a, b = self.bounds(node.left, env)
            c, d = self.bounds(node.right, env)
            if isinstance(node.op, ast.Add):
                result = (a + c, b + d)
            elif isinstance(node.op, ast.Sub):
                result = (a - d, b - c)
            elif isinstance(node.op, ast.Mult):
                values = [a * c, a * d, b * c, b * d]
                if any(math.isnan(x) for x in values):
                    return UNKNOWN
                result = (min(values), max(values))
            else:
                return UNKNOWN
            return UNKNOWN if any(math.isnan(x) for x in result) else result
        if isinstance(node, ast.Call):
            name = call_name(node.func)
            if name.split(".")[0] in self.shadowed:
                return UNKNOWN
            values = [self.bounds(x, env) for x in node.args]
            if name in ("abs", "np.abs", "numpy.abs") and len(values) == 1:
                lo, hi = values[0]
                return (0.0 if lo <= 0 <= hi else min(abs(lo), abs(hi)), max(abs(lo), abs(hi)))
            if name in ("max", "np.maximum", "numpy.maximum") and len(values) >= 2:
                return max(x[0] for x in values), max(x[1] for x in values)
            if name in ("min", "np.minimum", "numpy.minimum") and len(values) >= 2:
                return min(x[0] for x in values), min(x[1] for x in values)
            if name in ("len",) and len(values) == 1:
                return 0.0, math.inf
        return UNKNOWN

    def nonzero(self, node, env, guards):
        if key(node) in guards:
            return True
        lo, hi = self.bounds(node, env)
        return lo > 0 or hi < 0

    def branch_guards(self, test, truth=True):
        if isinstance(test, ast.UnaryOp) and isinstance(test.op, ast.Not):
            return self.branch_guards(test.operand, not truth)
        if isinstance(test, ast.BoolOp):
            groups = [self.branch_guards(v, truth) for v in test.values]
            all_required = (isinstance(test.op, ast.And) and truth) or (
                isinstance(test.op, ast.Or) and not truth
            )
            return set().union(*groups) if all_required else set.intersection(*groups)
        if not isinstance(test, ast.Compare) or len(test.ops) != 1:
            return set()
        left, right, op = test.left, test.comparators[0], test.ops[0]
        r = self.bounds(right, {})
        if r[0] != r[1] or not math.isfinite(r[0]):
            return set()
        x = r[0]
        safe = (
            (isinstance(op, ast.NotEq) and x == 0 and truth)
            or (isinstance(op, ast.Eq) and x == 0 and not truth)
            or (isinstance(op, ast.Gt) and ((truth and x >= 0) or (not truth and x < 0)))
            or (isinstance(op, ast.GtE) and ((truth and x > 0) or (not truth and x <= 0)))
            or (isinstance(op, ast.Lt) and ((truth and x <= 0) or (not truth and x > 0)))
            or (isinstance(op, ast.LtE) and ((truth and x < 0) or (not truth and x >= 0)))
        )
        if not safe:
            return set()
        result = {key(left)}
        if (
            isinstance(left, ast.Call)
            and call_name(left.func) in ("abs", "np.abs", "numpy.abs")
            and len(left.args) == 1
        ):
            if call_name(left.func).split(".")[0] not in self.shadowed:
                # abs(x) being nonzero establishes that x is nonzero as well.
                result.add(key(left.args[0]))
        return result

    def check_divisor(self, denominator, env, guards, line):
        if not self.nonzero(denominator, env, guards):
            self.violations.append(
                f"Line {line}: denominator may be zero: {ast.unparse(denominator)}. "
                "Pressure and derived sums can be negative; adding 1 or epsilon is not a guard. "
                "Use max(abs(d), 1e-8), or divide only inside an explicit abs(d) > 1e-8 branch."
            )

    def expression(self, node, env, guards):
        if isinstance(node, ast.IfExp):
            self.expression(node.test, env, guards)
            self.expression(node.body, env, guards | self.branch_guards(node.test))
            self.expression(node.orelse, env, guards | self.branch_guards(node.test, False))
            return
        if isinstance(node, ast.BinOp) and isinstance(node.op, (ast.Div, ast.FloorDiv, ast.Mod)):
            self.check_divisor(node.right, env, guards, node.lineno)
        if isinstance(node, ast.BinOp) and isinstance(node.op, ast.Pow):
            exponent = self.bounds(node.right, env)
            if exponent[1] < 0:
                self.check_divisor(node.left, env, guards, node.lineno)
        if isinstance(node, ast.Call):
            numpy_divisions = {"divide", "true_divide", "floor_divide", "remainder", "mod", "fmod"}
            names = self.numeric_call_names(node.func)
            if names & {f"{module}.{op}" for module in ("np", "numpy") for op in numpy_divisions}:
                if len(node.args) < 2:
                    self.violations.append(
                        f"Line {node.lineno}: use explicit scalar guarded division"
                    )
                else:
                    self.check_divisor(node.args[1], env, guards, node.lineno)
            elif names & {"np.reciprocal", "numpy.reciprocal"}:
                if node.args:
                    self.check_divisor(node.args[0], env, guards, node.lineno)
                else:
                    self.violations.append(
                        f"Line {node.lineno}: use an explicit guarded reciprocal argument"
                    )
            elif names & {
                "pow",
                "np.power",
                "numpy.power",
                "np.float_power",
                "numpy.float_power",
                "math.pow",
            }:
                # pow(x, y, modulus) has separate integer semantics; do not
                # interpret it as scalar inverse power.
                if "pow" in names and (len(node.args) > 2 or node.keywords):
                    self.violations.append(
                        f"Line {node.lineno}: use an explicit two-argument scalar power"
                    )
                elif len(node.args) < 2:
                    self.violations.append(
                        f"Line {node.lineno}: use explicit scalar power arguments"
                    )
                elif self.bounds(node.args[1], env)[1] < 0:
                    self.check_divisor(node.args[0], env, guards, node.lineno)
        for child in ast.iter_child_nodes(node):
            self.expression(child, env, guards)

    @staticmethod
    def invalidate(names, env, guards):
        for name in names:
            env.pop(name, None)
        # AST keys contain the names they reference. Invalidating extra guards
        # is conservative; never let a previous assignment justify a new one.
        guards.difference_update(
            g for g in list(guards) if any(f"id='{name}'" in g for name in names)
        )

    def block(self, statements, env=None, guards=None):
        env = dict(env or {})
        guards = set(guards or ())
        for stmt in statements:
            if isinstance(stmt, (ast.FunctionDef, ast.AsyncFunctionDef)):
                self.block(stmt.body)
            elif isinstance(stmt, ast.If):
                self.expression(stmt.test, env, guards)
                yes, gy = self.block(stmt.body, env, guards | self.branch_guards(stmt.test))
                no, gn = self.block(stmt.orelse, env, guards | self.branch_guards(stmt.test, False))
                env = {
                    n: (min(yes[n][0], no[n][0]), max(yes[n][1], no[n][1]))
                    for n in yes.keys() & no.keys()
                }
                guards = gy & gn
            elif isinstance(stmt, (ast.For, ast.While, ast.AsyncFor)):
                names = {
                    n.id
                    for n in ast.walk(stmt)
                    if isinstance(n, ast.Name) and isinstance(n.ctx, ast.Store)
                }
                self.invalidate(names, env, guards)
                self.expression(
                    stmt.iter if isinstance(stmt, (ast.For, ast.AsyncFor)) else stmt.test,
                    env,
                    guards,
                )
                self.block(stmt.body, env, guards)
                self.block(stmt.orelse, env, guards)
            elif isinstance(stmt, (ast.Assign, ast.AnnAssign, ast.AugAssign)):
                value = stmt.value
                if value is not None:
                    self.expression(value, env, guards)
                if isinstance(stmt, ast.AugAssign) and isinstance(
                    stmt.op, (ast.Div, ast.FloorDiv, ast.Mod)
                ):
                    self.check_divisor(value, env, guards, stmt.lineno)
                if isinstance(stmt, ast.AugAssign) and isinstance(stmt.op, ast.Pow):
                    if self.bounds(value, env)[1] < 0:
                        self.check_divisor(stmt.target, env, guards, stmt.lineno)
                targets = stmt.targets if isinstance(stmt, ast.Assign) else [stmt.target]
                names = {n.id for t in targets for n in ast.walk(t) if isinstance(n, ast.Name)}
                bound = (
                    self.bounds(value, env)
                    if value is not None and not isinstance(stmt, ast.AugAssign)
                    else UNKNOWN
                )
                self.invalidate(names, env, guards)
                for target in targets:
                    if isinstance(target, ast.Name):
                        env[target.id] = bound
            elif isinstance(stmt, (ast.Try, ast.With, ast.AsyncWith)):
                # Complex control flow is checked without inherited proofs.
                self.expression(stmt, {}, set())
                env.clear()
                guards.clear()
            else:
                self.expression(stmt, env, guards)
        return env, guards


def check_division_safety(code):
    try:
        tree = ast.parse(code)
    except SyntaxError as exc:
        return {"valid": False, "violations": [str(exc)], "message": "Invalid Python"}
    checker = DivisionCheck(tree)
    checker.block(tree.body)
    violations = list(dict.fromkeys(checker.violations))
    return {
        "valid": not violations,
        "violations": violations,
        "message": (
            "Division checks passed"
            if not violations
            else "Explicit nonzero denominator guards required"
        ),
    }


def signed_boundary_states(template, phase_id, keys):
    """Small deterministic combinations, independent of observed OOD results."""
    controlled = list(
        template.get("controlled_directions") or template.get("features_by_direction", {})
    )
    if not controlled:
        controlled = ["boundary_direction"]
    pressure_keys = [k for k in keys if "pressure" in k.lower()]
    if not pressure_keys:
        return
    for queue in (0.0, 1.0, 2.0, 10.0):
        for pressure in (-1000.0, -11.0, -3.0, -2.0, -1.0, -1e-8, 0.0, 1e-8, 1.0, 1000.0):
            for active in (True, False):
                state = copy.deepcopy(template)
                state["current_phase"] = phase_id if active else -1
                state["controlled_directions"] = controlled
                features = {k: 0.0 for k in keys}
                for k in keys:
                    if k in pressure_keys:
                        features[k] = pressure / len(controlled)
                    elif "num_waiting_vehicle" in k or k == "num_vehicle":
                        features[k] = queue / len(controlled)
                    elif "waiting_time" in k:
                        features[k] = 5.0 * queue / len(controlled)
                state["features_by_direction"] = {d: dict(features) for d in controlled}
                yield f"signed_pressure={pressure},queue={queue},active={active}", state
