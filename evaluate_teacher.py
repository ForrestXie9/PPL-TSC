#!/usr/bin/env python3
"""Evaluate an ordinary teacher checkpoint using explicit original-family transfer rules."""
import argparse,hashlib,json,sys,math
from pathlib import Path
ROOT=Path(__file__).resolve().parent
sys.path.insert(0,str(ROOT/'src'))

def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()

def resolve_seed(explicit_seed,audit):
 if explicit_seed is not None:return explicit_seed
 if 'seed' not in audit:raise ValueError('Training audit lacks seed; pass --seed explicitly')
 return int(audit['seed'])

def load_source_zero(env,config,checkpoint):
 import torch
 import agent.evaluation as evaluation
 payload=torch.load(checkpoint,map_location='cpu',weights_only=False)
 if payload.get('parameter_sharing') is True or 'shared_local' in payload:
  raise ValueError('Experimental Shared checkpoint is not an original-family checkpoint')
 local=payload['agent_0_local'];target=payload.get('agent_0_target',local)
 env.n_agent=[]
 for i in range(env.n):env.n_agent.append(evaluation.get_agent(config['cur_agent'])(config,env,i))
 # Validate all shapes before loading any tensor. Never truncate/pad weights or use strict=False.
 for i,agent in enumerate(env.n_agent):
  for model,state in [(agent.network_local,local),(agent.network_target,target)]:
   expected=model.state_dict()
   if expected.keys()!=state.keys():raise ValueError(f'Target intersection {i}: checkpoint keys differ')
   differences={k:{'source':list(state[k].shape),'target':list(expected[k].shape)} for k in expected if state[k].shape!=expected[k].shape}
   if differences:raise ValueError(f'Target intersection {i}: incompatible tensor shapes: {differences}')
 for agent in env.n_agent:
  agent.network_local.load_state_dict(local,strict=True);agent.network_target.load_state_dict(target,strict=True)
  agent.network_local.eval();agent.network_target.eval()
 return {'agents':env.n_agent,'latest_observations':None}

def main():
 p=argparse.ArgumentParser(description=__doc__)
 p.add_argument('--checkpoint',required=True,type=Path)
 p.add_argument('--training-audit',required=True,type=Path,help='audit.json emitted by train_teacher.py')
 p.add_argument('--scenario',required=True,choices=sorted(x.name for x in (ROOT/'data').iterdir() if x.is_dir()))
 p.add_argument('--flow',required=True,type=int,choices=range(5))
 p.add_argument('--seed',type=int,help='Defaults to the source training audit seed')
 p.add_argument('--protocol',required=True,choices=['same_network','source_agent0'])
 p.add_argument('--output',required=True,type=Path)
 a=p.parse_args()
 if a.output.resolve() in {a.checkpoint.resolve(),a.training_audit.resolve()}:p.error('Output must not overwrite checkpoint or training audit')
 from ppl_tsc.artifacts import evaluation_record
 with evaluation_record(a.output, {'checkpoint_path':str(a.checkpoint.resolve()),'scenario':a.scenario,'flow':a.flow,'protocol':a.protocol}) as commit_result:
  audit=json.loads(a.training_audit.read_text());teacher=audit['teacher']
  if teacher not in {'CoLight','EcoLight','FRAP','MPLight','TinyLight'}:p.error('Unsupported teacher in training audit')
  if audit.get('scenario') not in {path.name for path in (ROOT/'data').iterdir() if path.is_dir()}:p.error('Unsupported source scenario in training audit')
  from collect_teacher import validate_checkpoint_audit
  validate_checkpoint_audit(a.checkpoint,a.training_audit,teacher,ROOT/'data'/audit['scenario']/'roadnet.json',
                            traffic_config=ROOT/'configs/traffic.json',environment_semantics='per_decision')
  a.seed=resolve_seed(a.seed,audit)
  import agent.evaluation as evaluation
  horizon={'Atlanta':900,'LosAngeles':1800}.get(a.scenario,3600)
  config=evaluation.build_config(ROOT/'configs/traffic.json',teacher,a.scenario,a.flow,a.seed,horizon,ROOT/'data'/a.scenario)
  config.update(waiting_time_semantics='per_decision',direction_feature_semantics='compatible_shared_lane_mean_v1')
  if a.protocol=='same_network':
   if audit['roadnet_sha256']!=sha(ROOT/'data'/a.scenario/'roadnet.json'):p.error('same_network requires identical roadnet')
   setup=lambda env:evaluation.load_teacher_agents(env,config,a.checkpoint)
  else:
   if teacher=='TinyLight':p.error('PI-Light baseline-transfer script does not establish a TinyLight transfer protocol; use same_network')
   setup=lambda env:load_source_zero(env,config,a.checkpoint)
  record={'teacher':teacher,'protocol':a.protocol,'source_scenario':audit['scenario'],'source_flow':audit['flow'],'target_scenario':a.scenario,'target_flow':a.flow,'checkpoint_sha256':sha(a.checkpoint),'training_audit_sha256':sha(a.training_audit),'traffic_config_sha256':sha(ROOT/'configs/traffic.json'),'environment_semantics':config['waiting_time_semantics'],'target_roadnet_sha256':sha(ROOT/'data'/a.scenario/'roadnet.json'),'target_flow_sha256':sha(ROOT/'data'/a.scenario/f'flow_{a.flow}.json'),'simulation_seed':a.seed,'horizon':horizon,'target_training':False,'source_agent':0 if a.protocol=='source_agent0' else 'each original intersection'}
  result=evaluation.run_episode(config,None,evaluation.teacher_action_builder,setup)
  att=result['metrics']['ATT']
  if not math.isfinite(att) or att<=0:raise ValueError('Teacher evaluation produced invalid ATT')
  record.update(result=result)
  commit_result(record)
  print(json.dumps(result['metrics'],allow_nan=False))
if __name__=='__main__':main()
