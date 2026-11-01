"""Uniform V8 source-only prompt contract, validated before default adoption."""
import json
from .concise_contract import render as v7_render,raw_keys,ALIASES

def namespace_note(generator):
    if generator._decision_mode()=='binary_keep_switch':
        return ('INPUT LOOKUPS: green=state.get("binary_features",{}).get("green_vehicle",0.0); '
                'green_scale=(feature_stats or {}).get("green_vehicle",{}).get("p95",1.0). '
                'The first reads current traffic; the second reads a frozen SOURCE statistic. '
                'red_vehicle follows the same two separate paths. No direction-feature fields exist in this interface.')
    keys=raw_keys(generator);key=keys[0];alias=ALIASES[key]+'_raw'
    pairs={k:ALIASES[k]+'_raw' for k in keys}
    return ('INPUT LOOKUPS: direction_data=state.get("features_by_direction",{}).get(direction,{}); '
            f'value=direction_data.get("{key}",0.0); scale=(feature_stats or {{}}).get("{alias}",{{}}).get("p95",1.0). '
            'These are DIFFERENT dictionaries. Runtime direction_data keys are ONLY '+json.dumps(keys,separators=(',',':'))+'. '
            'SOURCE statistic lookup mapping: '+json.dumps(pairs,sort_keys=True,separators=(',',':'))+'. '
            'Statistic names such as vehicles_raw/pressure_raw/p95 and phase_active are not direction_data inputs. '
            'Compute active=float(phase_id==state.get("current_phase")); phase_active statistics only describe its binary support. '
            'Local variable names may be arbitrary; dictionary lookups must follow the declared paths.')


def render(generator,stage,parent=None,explanation=None,feedback=None):
    text=v7_render(generator,stage,parent,explanation,feedback)
    marker='CONSTRAINTS:';assert text.count(marker)==1
    text=text.replace(marker,namespace_note(generator)+'\n'+marker,1)
    old='Guard transformed inputs and keep code readable.'
    new=('Guard transformed inputs and keep code readable. A guarded base does not make every transformed denominator nonzero: '
         'log(1), log1p(0) and sqrt(0) are zero. Compute scale=max(abs(float(value)),1e-8) when a positive magnitude scale is intended; '
         'for a signed ratio use an explicit abs(d)>1e-8 branch. Preserve monotonicity of the WHOLE utility when a denominator depends on runtime demand.')
    assert old in text;text=text.replace(old,new,1)
    if stage=='repair':
        extra='REPAIR FOCUS: use the reported failing line or numeric counterexample to identify the affected lookup/arithmetic; recheck the complete function against all existing rules. Multiple changes are allowed; preserve the declared input/action interface and do not merely catch exceptions or return a constant.'
        needle='Output: one Python code block'
        assert needle in text;text=text.replace(needle,extra+'\n'+needle,1)
    return text

def install():
    from LLM_SR_DISTILLATION_method_v2.evolution import Evolution
    if getattr(Evolution,'_generation_yield_v8_installed',False):return
    Evolution._generation_yield_v8_installed=True
    from .lean_feedback import render as parent_render
    from .feedback_helpers import cached_diagnostic
    original=Evolution._build_parent_info
    def parent(self,individual):
        original(self,individual)
        return parent_render(self,individual,cached_diagnostic(self,individual))
    Evolution._build_parent_info=parent
    from LLM_SR_DISTILLATION.llm_generator import LLMGenerator
    names={'_build_initialization_prompt':'initialization','_build_evolution_mutation_prompt':'evolution','_build_constraint_fix_prompt':'repair','_build_constraint_retry_prompt':'repair'}
    for name,stage in names.items():
        def builder(self,*args,_stage=stage,**kwargs):
            code=kwargs.get('parent_function_code',args[0] if args else None)
            feedback=kwargs.get('parent_performance_info',kwargs.get('constraint_violations',args[1] if len(args)>1 else None))
            explanation=kwargs.get('parent_explanation',args[2] if len(args)>2 else None)
            return render(self,_stage,code,explanation,feedback)
        setattr(LLMGenerator,name,builder)
