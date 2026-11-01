"""Teacher agents loaded on demand; unused backends need no optional dependencies."""
from importlib import import_module

_MODULES = {'BaseAgent': 'base_agent', 'BaseDRL': 'drl_based.base_drl', 'CoLight': 'drl_based.colight', 'EcoLight': 'drl_based.ecolight', 'FRAP': 'drl_based.frap', 'MPLight': 'drl_based.mplight', 'TinyLight': 'tiny_light.tiny_light'}

def __getattr__(name):
    if name not in _MODULES:
        raise AttributeError(f"Unsupported agent: {name}")
    value = getattr(import_module("agent." + _MODULES[name]), name)
    globals()[name] = value
    return value
