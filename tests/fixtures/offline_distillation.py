"""Run the real pipeline with deterministic candidate text; forbid paid/network calls."""
import argparse,json,runpy,socket,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2]
p=argparse.ArgumentParser();p.add_argument('--config',required=True);p.add_argument('--variant',required=True);p.add_argument('--record',required=True);p.add_argument('--resume',action='store_true');a=p.parse_args()
sys.path[:0]=[str(ROOT/'src'),str(ROOT)]
import LLM_SR_DISTILLATION.llm_generator as generator
import agent.imitation_based.llm_integration as api
calls=[]
def response(prompt,**kwargs):
    calls.append({'index':len(calls),'prompt_bytes':len(prompt.encode())})
    coefficient=1+(len(calls)%5)*.1
    if a.variant=='EcoLight-native':
        body="return float(state['binary_features']['red_vehicle'] if phase_id == 1 else state['binary_features']['green_vehicle'])"
    else:
        key='pressure' if a.variant=='MPLight' else 'num_vehicle'
        body=f"return {coefficient} * float(sum(state['features_by_direction'][d]['{key}'] for d in state['controlled_directions']))"
    return '```python\ndef compute_utility(state, phase_id=None, feature_stats=None):\n    '+body+'\n```\nDeterministic audit fixture; not an LLM result.'
def forbidden(*args,**kwargs):raise AssertionError('Network requests are forbidden in this audit fixture')
generator.run_llm=response;api.run_llm=response;api._open_http_connection=forbidden
socket.create_connection=forbidden;socket.socket.connect=forbidden
sys.argv=[str(ROOT/'run.py'),'--config',a.config,'--offline','--no-api-profile']+(['--resume'] if a.resume else [])
record={'status':'running','API_calls':0}
try:
    runpy.run_path(str(ROOT/'run.py'),run_name='__main__')
except SystemExit as exc:
    if exc.code not in (None,0):record['status']='failed';raise
    record['status']='passed'
except BaseException:
    record['status']='failed';raise
else:record['status']='passed'
finally:
    record['fixture_calls']=len(calls)
    Path(a.record).write_text(json.dumps(record,indent=2)+'\n')
