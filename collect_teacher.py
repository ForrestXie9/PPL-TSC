#!/usr/bin/env python3
"""Collect aligned teacher trajectories with the same feature builder used at deployment."""
from __future__ import annotations
import argparse
import gzip
import hashlib
import json
import math
import os
from pathlib import Path
import sys

ROOT=Path(__file__).resolve().parent
sys.path.insert(0,str(ROOT/'src'))
for key in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','NUMEXPR_NUM_THREADS'):os.environ[key]='1'


def validate_checkpoint_audit(checkpoint, audit_path, teacher, roadnet, allow_unaudited=False,
                              *, traffic_config=None, environment_semantics=None):
    """Collection is same-network replay, not implicit cross-network transfer."""
    if not audit_path.is_file():
        if not allow_unaudited:
            raise ValueError('Training audit missing; provide --training-audit or explicitly use --allow-unaudited-checkpoint for a legacy model')
        return {'status':'unaudited_legacy_checkpoint'}
    audit=json.loads(audit_path.read_text())
    if audit.get('status')!='complete':raise ValueError('Teacher training audit is not complete')
    if audit.get('teacher')!=teacher:raise ValueError('Checkpoint teacher does not match requested teacher')
    digest=lambda path:hashlib.sha256(path.read_bytes()).hexdigest()
    if audit.get('checkpoint_sha256')!=digest(checkpoint):raise ValueError('Checkpoint hash does not match training audit')
    if audit.get('roadnet_sha256')!=digest(roadnet):raise ValueError('Collection requires the original source roadnet; use explicit teacher transfer evaluation for another roadnet')
    if traffic_config is not None and audit.get('traffic_config_sha256')!=digest(traffic_config):
        raise ValueError('Traffic configuration differs from the teacher training audit')
    if environment_semantics is not None and audit.get('environment_semantics')!=environment_semantics:
        raise ValueError('Environment semantics differ from the teacher training audit')
    return {'status':'verified','training_audit_sha256':digest(audit_path),
            'source_scenario':audit['scenario'],'source_flow':audit['flow'],'source_seed':audit['seed']}


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--teacher',choices=['CoLight','EcoLight','FRAP','MPLight','TinyLight'],required=True)
    parser.add_argument('--scenario',required=True,choices=sorted(p.name for p in (ROOT/'data').iterdir() if p.is_dir()))
    parser.add_argument('--flow',type=int,choices=range(5),required=True)
    parser.add_argument('--checkpoint',type=Path,required=True)
    parser.add_argument('--training-audit',type=Path,help='Defaults to audit.json beside checkpoint')
    parser.add_argument('--allow-unaudited-checkpoint',action='store_true',help='Explicitly allow legacy models without a training audit; never bypass a mismatching audit')
    parser.add_argument('--direction-feature-semantics',choices=['legacy_first_match','compatible_shared_lane_mean_v1'],default='compatible_shared_lane_mean_v1')
    parser.add_argument('--seed',type=int)
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args()
    audit_path=args.training_audit or args.checkpoint.parent/'audit.json'
    if args.output.resolve() in {args.checkpoint.resolve(),audit_path.resolve()}:
        parser.error("Output must not overwrite the teacher checkpoint")
    if args.output.exists():parser.error('Trajectory output already exists; use a new path')
    provenance=validate_checkpoint_audit(args.checkpoint,audit_path,args.teacher,ROOT/'data'/args.scenario/'roadnet.json',args.allow_unaudited_checkpoint,
                                         traffic_config=ROOT/'configs/traffic.json',environment_semantics='per_decision')
    import teacher_evaluation as evaluation
    import torch
    from LLM_SR_DISTILLATION.simulator import StudentSimulator
    from env.ood_env import OODTSCEnv
    seed=args.seed if args.seed is not None else [992832,284765,905873,776383,198876][args.flow]
    horizon={'Atlanta':900,'LosAngeles':1800}.get(args.scenario,3600)
    config=evaluation.build_config(ROOT/'configs/traffic.json',args.teacher,args.scenario,args.flow,seed,horizon,ROOT/'data'/args.scenario)
    config['waiting_time_semantics']='per_decision'
    config['direction_feature_semantics']=args.direction_feature_semantics
    evaluation.seed_everything(seed);env=OODTSCEnv(config)
    controller=evaluation.load_teacher_agents(env,config,args.checkpoint);agents=controller['agents']
    features=config[args.teacher]['observation_feature_list'];helper=StudentSimulator(config,features)
    observations=env.reset();rows=[]
    for index,step in enumerate(range(0,horizon,config['action_interval'])):
        config['current_episode_step_idx']=step
        states=[helper.build_phase_state_dict(env,i) for i in range(env.n)]
        q_values=[]
        for i,agent in enumerate(agents):
            agent.network_local.eval()
            with torch.no_grad():q=agent.network_local(observations if args.teacher=='CoLight' else observations[i])[0].cpu().tolist()
            if not all(math.isfinite(v) for v in q):raise ValueError('Nonfinite teacher Q value')
            q_values.append(q)
        actions=[int(agent.pick_action(observations,False)) for agent in agents]
        observations,_,done,info=env.step(actions)
        for i in range(env.n):
            phases=[]
            for phase,state in states[i].items():
                cell={'phase_id':phase,'state':state,'selected':phase==actions[i]}
                if args.teacher!='EcoLight':cell.update(Q_value=q_values[i][phase],logit=q_values[i][phase])
                phases.append(cell)
            row={'intersection_id':i,'timestep':index,'current_phase':int(states[i][0]['current_phase']),
                 'teacher_action':actions[i],'phases':phases,'metrics':evaluation.extract_metrics(info,env)}
            if args.teacher=='EcoLight':
                binary=int(q_values[i][1]>q_values[i][0]);expected=(row['current_phase']+binary)%len(phases)
                if expected!=actions[i]:raise ValueError('Native binary action/phase mismatch')
                row.update(teacher_binary_action=binary,teacher_binary_q_values=q_values[i])
            rows.append(row)
        if all(done) and step+config['action_interval']<horizon:raise ValueError('Episode ended before requested horizon')
    metrics=evaluation.extract_metrics(info,env)
    header={'meta':{'model':args.teacher,'dataset':args.scenario,'flow_idx':args.flow,'seed':seed,
                    'checkpoint_sha256':hashlib.sha256(args.checkpoint.read_bytes()).hexdigest(),
                    'roadnet_sha256':hashlib.sha256((ROOT/'data'/args.scenario/'roadnet.json').read_bytes()).hexdigest(),
                    'flow_sha256':hashlib.sha256((ROOT/'data'/args.scenario/f'flow_{args.flow}.json').read_bytes()).hexdigest(),
                    'teacher_variant':args.teacher,'checkpoint_provenance':provenance,
                    'collector':'deployment-feature-aligned-v2',
                    'environment_semantics':config['waiting_time_semantics'],
                    'direction_feature_semantics':args.direction_feature_semantics},
            'config':{'episode_seeds':[seed],'n_episodes':1,'num_step':horizon,
                      'intersections':[{'id':i,'num_phases':len(inter.n_phase)} for i,inter in enumerate(env.n_intersection)]},
            'stats':{'avg_travel_time':metrics['ATT'],'total_samples':horizon//config['action_interval']},
            'schema':{'timestep':'zero-based decision index','binary_interface':args.teacher=='EcoLight'}}
    from ppl_tsc_current.artifact_io import new_gzip_jsonl
    new_gzip_jsonl(args.output, [header,*rows])
    print(f'Collected {len(rows)} records, ATT={metrics["ATT"]:.6f}: {args.output}')
    return 0


if __name__=='__main__':raise SystemExit(main())
