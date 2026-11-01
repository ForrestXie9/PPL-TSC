"""Mode-specific prompt contracts. No controller, verifier or statistics edits."""
import copy,json

ALIASES={'num_vehicle':'vehicles','num_waiting_vehicle':'waiting_veh',
         'waiting_time':'waiting_time','sum_waiting_time':'waiting_time','pressure':'pressure'}

def raw_keys(generator):
    keys=sorted(set(generator._get_state_keys_for_teacher())-{'current_phase'})
    unknown=set(keys)-set(ALIASES)
    if unknown:raise ValueError('Contract requires explicit semantics for raw fields: '+str(sorted(unknown)))
    return keys

def statistics(generator,binary):
    """Deduplicate only documented aliases; never merge unrelated equal distributions."""
    original=generator.feature_stats
    if original is None:return dict(values={},aliases={},available=[],supplied=False)
    wanted={'green_vehicle','red_vehicle'} if binary else {'phase_active'}|{ALIASES[k]+suffix for k in raw_keys(generator) for suffix in ['_raw','_norm']}
    values={k:copy.deepcopy(v) for k,v in original.items() if k in wanted};aliases={}
    for key in sorted(list(values)):
        if key.endswith('_norm'):
            canonical=key[:-5]+'_raw'
            if canonical in values and values[key]==values[canonical]:
                aliases[key]=canonical;del values[key]
    return dict(values=values,aliases=aliases,available=sorted(set(values)|set(aliases)),supplied=True)

from .short_contract import build

def render(generator,stage,parent=None,explanation=None,feedback=None):
    from ppl_tsc_current.paper_prompts import TEMPLATES
    # Fixed manuscript text is unchanged. New information lives in placeholders.
    parent_payload='\n```python\n'+(parent or '')+'\n```\n'
    if stage!='evolution' and explanation:parent_payload+='Design rationale: '+explanation+'\n'
    fields=dict(features='\n'+build(generator)+'\n',parent=parent_payload,evaluation='\n'+(feedback or 'No performance metrics available')+'\n',violations='\n'+(feedback or 'Correct the reported verifier failure.')+'\n',syntactic='syntactic correctness (the declared executable interface)',mathematical='mathematical validity (finite arithmetic with guarded operations)',semantic='semantic consistency (the declared decision and verifier properties)')
    return TEMPLATES[stage].format_map(fields)

def install():
    from LLM_SR_DISTILLATION.llm_generator import LLMGenerator
    if getattr(LLMGenerator,'_concise_contract_v4_installed',False):return
    LLMGenerator._concise_contract_v4_installed=True
    names={'_build_initialization_prompt':'initialization','_build_evolution_mutation_prompt':'evolution','_build_constraint_fix_prompt':'repair','_build_constraint_retry_prompt':'repair'}
    for name,stage in names.items():
        def builder(self,*args,_stage=stage,**kwargs):
            parent=kwargs.get('parent_function_code',args[0] if args else None)
            feedback=kwargs.get('parent_performance_info',kwargs.get('constraint_violations',args[1] if len(args)>1 else None))
            explanation=kwargs.get('parent_explanation',args[2] if len(args)>2 else None)
            return render(self,_stage,parent,explanation,feedback)
        setattr(LLMGenerator,name,builder)
