"""Teacher agents loaded on demand; unused backends need no optional dependencies."""
from importlib import import_module

_MODULES = {'BaseAgent': 'base_agent', 'BaseDRL': 'drl_based.base_drl', 'AttendLight': 'drl_based.attendlight', 'CoLight': 'drl_based.colight', 'EcoLight': 'drl_based.ecolight', 'FRAP': 'drl_based.frap', 'MPLight': 'drl_based.mplight', 'AdvancedMPLight': 'drl_based.advanced_mplight', 'AdvancedCoLight': 'drl_based.advanced_colight', 'PressLight': 'drl_based.presslight', 'TinyLight': 'tiny_light.tiny_light', 'RandomPath': 'tiny_light.random_path', 'MaxPressure': 'rule_based.max_pressure', 'FixedTime': 'rule_based.fixed_time', 'SOTL': 'rule_based.sotl', 'PiLight': 'pi_light.pi_light', 'MCTS_synthesizer': 'pi_light.MCTS'}

def __getattr__(name):
    if name not in _MODULES:
        raise AttributeError(f"Unsupported agent: {name}")
    value = getattr(import_module("agent." + _MODULES[name]), name)
    globals()[name] = value
    return value
