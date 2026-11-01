"""Proved scalar bounds; unknown and signed zero-crossing divisors remain rejected."""
import ast,math
from . import numerical_safety as original

class ProvedDivisionCheck(original.DivisionCheck):
    def __init__(self,tree):
        super().__init__(tree)
        self.mutated_modules=set()
        self.dynamic_mutation=False
        for node in ast.walk(tree):
            if isinstance(node,ast.Attribute) and isinstance(node.ctx,(ast.Store,ast.Del)):
                self.mutated_modules.add(original.call_name(node).split('.')[0])
            if isinstance(node,ast.Call) and original.call_name(node.func) in {'setattr','delattr','globals','locals','vars','exec','eval'}:
                self.dynamic_mutation=True

    def bounds(self,node,env):
        if not isinstance(node,ast.Call) or len(node.args)!=1 or node.keywords:
            return super().bounds(node,env)
        name=original.call_name(node.func);root=name.split('.')[0]
        if root in self.shadowed or self.dynamic_mutation:
            return super().bounds(node,env)
        # Preserve scalar bounds only for the unshadowed Python builtin.
        if name=='float' and name not in self.import_aliases:
            return self.bounds(node.args[0],env)
        bindings=self.import_aliases.get(root,set())
        resolved={b+name[len(root):] for b in bindings}
        if len(resolved)!=1 or root in self.mutated_modules:
            return super().bounds(node,env)
        resolved=next(iter(resolved))
        if resolved not in {'math.log1p','numpy.log1p','math.log','numpy.log','math.sqrt','numpy.sqrt'}:
            return super().bounds(node,env)
        lo,hi=self.bounds(node.args[0],env)
        if resolved.endswith('.log1p') and lo>-1:
            return math.log1p(lo),math.log1p(hi)
        if resolved.endswith('.log') and lo>0:
            return math.log(lo),math.log(hi)
        if resolved.endswith('.sqrt') and lo>=0:
            return math.sqrt(lo),math.sqrt(hi)
        return original.UNKNOWN


def check_division_safety(code):
    try:tree=ast.parse(code)
    except SyntaxError as exc:return dict(valid=False,violations=[str(exc)],message='Invalid Python')
    checker=ProvedDivisionCheck(tree);checker.block(tree.body)
    violations=list(dict.fromkeys(checker.violations))
    return dict(valid=not violations,violations=violations,message='Division checks passed' if not violations else 'Explicit nonzero denominator guards required')


DivisionCheck=ProvedDivisionCheck
