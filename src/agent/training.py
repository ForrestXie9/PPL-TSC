#!/usr/bin/env python3
"""Train original DRL teachers with deterministic checkpointing and resume."""
from __future__ import annotations
import argparse
import hashlib
import json
import os
from pathlib import Path
import random
import sys
import time

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT / 'src'), str(ROOT)]
for name in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','NUMEXPR_NUM_THREADS'):
    os.environ[name] = '1'


def sha(path): return hashlib.sha256(path.read_bytes()).hexdigest()


def write_json(path, value):
    temporary=path.with_suffix('.tmp')
    temporary.write_text(json.dumps(value,indent=2,allow_nan=False)+'\n');temporary.replace(path)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--teacher',choices=['CoLight','EcoLight','FRAP','MPLight','TinyLight'],required=True)
    parser.add_argument('--scenario',required=True,choices=sorted(p.name for p in (ROOT/'data').iterdir() if p.is_dir()))
    parser.add_argument('--flow',type=int,choices=range(5),required=True)
    parser.add_argument('--episodes',type=int,default=128)
    parser.add_argument('--seed',type=int)
    parser.add_argument('--selection',choices=['last','best_training_att'],default='best_training_att')
    parser.add_argument('--device',default='cpu')
    parser.add_argument('--output-dir',type=Path,required=True)
    parser.add_argument('--resume',action='store_true')
    parser.add_argument('--checkpoint-every',type=int,default=5)
    parser.add_argument('--stop-after',type=int,help='Stop at this episode boundary, retaining the full planned protocol')
    args=parser.parse_args()
    if args.episodes<1 or args.checkpoint_every<1:parser.error('Episodes and checkpoint interval must be positive')
    if args.stop_after is not None and args.stop_after<1:parser.error('--stop-after must be a positive episode number')
    import fcntl
    import numpy as np
    import torch
    from env import TSCEnv
    from utilities.utils import get_config,get_agent,set_seed,set_thread
    from utilities.snippets import run_an_episode
    from ppl_tsc import cli as runner
    args.output_dir.mkdir(parents=True,exist_ok=True)
    lock=(args.output_dir/'.lock').open('w');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    if args.device.startswith('cuda') and not torch.cuda.is_available():raise RuntimeError('CUDA is unavailable')
    seed=args.seed if args.seed is not None else [992832,284765,905873,776383,198876][args.flow]
    horizon={'Atlanta':900,'LosAngeles':1800}.get(args.scenario,3600)
    protocol={'teacher':args.teacher,'scenario':args.scenario,'flow':args.flow,'seed':seed,
              'episodes':args.episodes,'selection':args.selection,'device':args.device,'horizon':horizon,
              'source_hash':runner._sha256_source_tree(),'torch':torch.__version__,'numpy':np.__version__,
              'traffic_config_sha256':sha(ROOT/'configs/traffic.json'),
              'roadnet_sha256':sha(ROOT/'data'/args.scenario/'roadnet.json'),
              'flow_sha256':sha(ROOT/'data'/args.scenario/f'flow_{args.flow}.json'),
              'environment_semantics':'per_decision','epsilon_decay_episodes':55,
              'tiny_light_architecture':{'learn_architecture_weights':True,'prune_architecture':False} if args.teacher=='TinyLight' else None}
    p=args.output_dir/'protocol.json'
    if p.exists() and json.loads(p.read_text())!=protocol:raise ValueError('Protocol changed; use a new output directory')
    write_json(p,protocol)
    audit_path=args.output_dir/'audit.json';checkpoint_path=args.output_dir/'checkpoint.pth'
    if args.resume and audit_path.exists() and checkpoint_path.exists():
        audit=json.loads(audit_path.read_text())
        if audit['status']=='complete' and audit['checkpoint_sha256']==sha(checkpoint_path):
            print('Verified complete training; skipped',flush=True);return 0
    config=get_config(str(ROOT/'configs/traffic.json'))
    config.update(cur_agent=args.teacher,inter_name=args.scenario,flow_idx=args.flow,seed=seed,num_step=horizon,
                  device=args.device,saveReplay=False,save_result=False,render=False,
                  dir=str(ROOT/'data'/args.scenario)+'/',flowFile=f'flow_{args.flow}.json',
                  current_episode_idx=0,current_episode_step_idx=0)
    set_seed(seed);set_thread(1)
    env=TSCEnv(config);env.n_agent=[]
    for idx in range(env.n):env.n_agent.append(get_agent(args.teacher)(config,env,idx))
    def snapshot():
        return {f'agent_{i}_{side}':{k:v.detach().cpu().clone() for k,v in getattr(a,'network_'+side).state_dict().items()}
                for i,a in enumerate(env.n_agent) for side in ('local','target')}
    def restore(weights):
        for i,a in enumerate(env.n_agent):
            for side in ('local','target'):getattr(a,'network_'+side).load_state_dict(weights[f'agent_{i}_{side}'],strict=True)
    initial=snapshot();best=None;best_att=float('inf');best_episode=None;history=[];episode=0
    initial_epsilon=float(config[args.teacher]['epsilon']);started=time.monotonic();previous_seconds=0.
    resume_path=args.output_dir/'training_state.pth'
    if resume_path.exists():
        if not args.resume:raise ValueError('Existing training state: use --resume')
        state=torch.load(resume_path,map_location='cpu',weights_only=False)
        if state['protocol']!=protocol:raise ValueError('Resume protocol mismatch')
        restore(state['weights']);initial=state['initial'];best=state['best'];best_att=state['best_att']
        best_episode=state['best_episode'];history=state['history'];episode=state['episode'];previous_seconds=state['seconds']
        for a,saved in zip(env.n_agent,state['agents']):
            a.network_optim.load_state_dict(saved['network_optim'])
            if hasattr(a,'alpha_optim'):a.alpha_optim.load_state_dict(saved['alpha_optim'])
            if hasattr(a,'network_lr_scheduler'):a.network_lr_scheduler.load_state_dict(saved['scheduler'])
            b=a.replay_buffer;n=saved['buffer_size']
            for key,values in saved['buffer'].items():
                if isinstance(values,list):
                    for dest,src in zip(b.memory[key],values):dest[:n]=src
                else:b.memory[key][:n]=values
            b.current_size=n;b._cur_idx=saved['buffer_index']
        random.setstate(state['rng_python']);np.random.set_state(state['rng_numpy']);torch.set_rng_state(state['rng_torch'])
        if args.device.startswith('cuda'):torch.cuda.set_rng_state_all(state['rng_cuda'])
        print(f'Resuming at episode {episode}',flush=True)
    def save_training():
        agents=[]
        for a in env.n_agent:
            b=a.replay_buffer;n=b.current_size
            data={k:[v[:n].copy() for v in val] if isinstance(val,list) else val[:n].copy() for k,val in b.memory.items()}
            d={'network_optim':a.network_optim.state_dict(),'buffer':data,'buffer_size':n,'buffer_index':b._cur_idx}
            if hasattr(a,'alpha_optim'):d['alpha_optim']=a.alpha_optim.state_dict()
            if hasattr(a,'network_lr_scheduler'):d['scheduler']=a.network_lr_scheduler.state_dict()
            agents.append(d)
        state={'protocol':protocol,'weights':snapshot(),'initial':initial,'best':best,'best_att':best_att,
               'best_episode':best_episode,'history':history,'episode':episode,'agents':agents,
               'rng_python':random.getstate(),'rng_numpy':np.random.get_state(),'rng_torch':torch.get_rng_state(),
               'rng_cuda':torch.cuda.get_rng_state_all() if args.device.startswith('cuda') else None,
               'seconds':previous_seconds+time.monotonic()-started}
        temp=resume_path.with_suffix('.tmp');torch.save(state,temp);temp.replace(resume_path)
    limit=min(args.episodes,args.stop_after) if args.stop_after is not None else args.episodes
    if args.stop_after is not None and limit<episode:
        raise ValueError('--stop-after precedes the resumed episode; select a later boundary')
    for episode in range(episode+1,limit+1):
        tick=time.monotonic();config['current_episode_idx']=episode
        config[args.teacher]['epsilon']=max(0.,initial_epsilon*(1-episode/55))
        info=run_an_episode(env,config,True,True,True);att=float(info['world_2_average_travel_time'][0])
        if not np.isfinite(att) or att<=0:raise ValueError('Invalid training ATT')
        history.append({'episode':episode,'ATT':att,'seconds':time.monotonic()-tick})
        if att<best_att:best_att=att;best=snapshot();best_episode=episode
        write_json(args.output_dir/'progress.json',{'status':'training','pid':os.getpid(),'episode':episode,
                   'total_episodes':args.episodes,'ATT':att,'best_training_ATT':best_att,'best_episode':best_episode,
                   'elapsed_seconds':previous_seconds+time.monotonic()-started,'device':args.device})
        print(f'\nEpisode {episode}/{args.episodes}: ATT={att:.6f}, seconds={history[-1]["seconds"]:.3f}',flush=True)
        if episode%args.checkpoint_every==0 or episode==limit:save_training()
    if limit<args.episodes:
        print('Stopped at requested episode boundary with resumable state',flush=True);return 0
    if args.selection=='best_training_att':restore(best)
    checkpoint=snapshot()
    if not all(torch.isfinite(v).all() for s in checkpoint.values() for v in s.values()):raise ValueError('Nonfinite learned weights')
    updates=sum(not torch.equal(v,initial[name][key]) for name,s in checkpoint.items() for key,v in s.items())
    if updates==0:raise RuntimeError('No learning updates occurred')
    metrics=run_an_episode(env,config,False,False,False)
    temp=checkpoint_path.with_suffix('.tmp');torch.save(checkpoint,temp);temp.replace(checkpoint_path)
    audit={**protocol,'status':'complete','changed_tensors':updates,'evaluation_ATT':float(metrics['world_2_average_travel_time'][0]),
           'checkpoint_sha256':sha(checkpoint_path),'history':history,'best_episode':best_episode,
           'wall_seconds':previous_seconds+time.monotonic()-started}
    write_json(audit_path,audit);write_json(args.output_dir/'progress.json',{'status':'complete','episode':episode,'total_episodes':args.episodes})
    print(f'Completed {args.teacher} {args.scenario} flow{args.flow}: {checkpoint_path}',flush=True);return 0


if __name__=='__main__':raise SystemExit(main())
