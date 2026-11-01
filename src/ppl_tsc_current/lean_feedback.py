"""Minimal evolution-only parent feedback; opt-in experiment overlay."""
import json
from .feedback_helpers import number, fmt, cached_diagnostic, one_example

def display_example(example,mode):
    """Round only displayed diagnostic numbers, never execution inputs or metrics."""
    import copy
    result=copy.deepcopy(example)
    if mode=='phase_argmax' and isinstance(result.get('state'),dict):
        result['state'].pop('binary_features',None)
    def rounded(value):
        if isinstance(value,float):return float(format(value,'.6g'))
        if isinstance(value,dict):return {k:rounded(v) for k,v in value.items()}
        if isinstance(value,list):return [rounded(v) for v in value]
        return value
    result=rounded(result)
    # These three quantities are algebraically redundant with the displayed
    # raw utilities and offsets. Keep state, actions and actual offsets.
    for key in ['adjusted_utilities','raw_pair_gap','matched_bias_advantage']:result.pop(key,None)
    result['display_note']='Rounded display; relevant actions only.'
    return result

def render(evolution, parent, diagnostic=None):
    m=parent.metrics or {}
    if m.get('error'):
        return 'SOURCE EVALUATION FAILED: '+str(m['error'])+'\nNo valid ATT or ABF is inferred.'
    student=m.get('student_travel_time');teacher=m.get('teacher_travel_time')
    gain=100*(teacher-student)/teacher if number(student) and number(teacher) and student>0 and teacher>0 else None
    if not number(gain):gain=None
    lines=[f"ATT(s): student={fmt(student,3)}, teacher={fmt(teacher,3)}; gain={fmt(gain,2)}% (+better); ABF={fmt(m.get('behavioral_fidelity'))}."]
    state=m.get('constraint_satisfied')
    lines.append('Verifier: '+('passed' if state is True else 'failed' if state is False else 'not evaluated')+'.')
    if state is False:
        for item in ((m.get('constraint_result') or {}).get('violations') or [])[:3]:
            lines.append('Violation: '+str(item))
    history=[max(x.score for x in generation) for generation in getattr(evolution,'generation_history',[]) if generation]
    stagnant=0
    for previous,current in reversed(list(zip(history,history[1:]))):
        if current>previous+1e-10:break
        stagnant+=1
    if stagnant:lines.append(f'Search score unchanged for {stagnant} completed generations.')
    if diagnostic:
        lines.append(f"Source sample n={diagnostic['samples']}: agreement raw→calibrated {fmt(diagnostic['raw_ABF'])}→{fmt(diagnostic['calibrated_ABF'])}; changed actions {100*diagnostic['changed_fraction']:.1f}%. Sample agreement is not full-source ABF.")
    cp=m.get('control_performance')
    if number(cp) and cp<1:
        example=one_example(m,diagnostic,getattr(evolution,'action_mode',None))
        if example:
            example=display_example(example,getattr(evolution,'action_mode',None))
            lines.append('ONE SOURCE DIAGNOSTIC:\n'+json.dumps(example,sort_keys=True,separators=(',',':'),allow_nan=False))
    return '\n'.join(lines)
