#!/usr/bin/env python3
"""Create a current-method config for a newly collected teacher trajectory (offline)."""
import argparse,gzip,json,hashlib,os,tempfile
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
def phase_projection(source, destination):
 """Preserve physical teacher actions; remove native-only supervision fields."""
 source=Path(source);destination=Path(destination)
 if source.resolve()==destination.resolve() or destination.exists():
  raise ValueError('Projection output must be new and different from the source')
 digest=hashlib.sha256(source.read_bytes()).hexdigest()
 opener=gzip.open if source.suffix=='.gz' else open
 fd,name=tempfile.mkstemp(prefix='.projection-',dir=destination.parent);os.close(fd)
 temporary=Path(name)
 try:
  index=-1
  with opener(source,'rt') as inp, temporary.open('wb') as raw, gzip.GzipFile(filename='',mode='wb',fileobj=raw,mtime=0) as out:
   for index,line in enumerate(inp):
    row=json.loads(line)
    if index==0:
     row['meta'].update(action_interface='physical_phase_selection_projection',native_trajectory_sha256=digest)
     row['meta'].pop('binary_supervision',None)
     row.setdefault('schema',{})['binary_interface']=False
    else:
     phases=row['phases'];action=row['teacher_action']
     if [p['phase_id'] for p in phases if p.get('selected')]!=[action]:
      raise ValueError('Physical phase labels are inconsistent')
     for key in ['teacher_binary_action','teacher_binary_q_values']:row.pop(key,None)
     for phase in phases:
      for key in ['binary_features','exact_binary_features']:phase['state'].pop(key,None)
    out.write((json.dumps(row,allow_nan=False)+'\n').encode())
  if index < 1:raise ValueError('Trajectory must contain a header and observations')
  # Publish only a complete file, without replacing another file created concurrently.
  os.link(temporary,destination)
 finally:
  temporary.unlink(missing_ok=True)

def main():
 p=argparse.ArgumentParser(description=__doc__)
 p.add_argument('--variant',required=True,choices=[x.stem for x in (ROOT/'configs/examples').glob('*.json')])
 p.add_argument('--trajectory',required=True,type=Path)
 p.add_argument('--output-dir',required=True,type=Path)
 p.add_argument('--seed',type=int,default=2026103101)
 p.add_argument('--endpoint',default='api.n1n.ai')
 p.add_argument('--model',default='gemini-3.1-flash-lite')
 a=p.parse_args();trajectory=a.trajectory.resolve()
 opener=gzip.open if trajectory.suffix=='.gz' else open
 with opener(trajectory,'rt') as f:header=json.loads(next(f))
 c=json.loads((ROOT/'configs/examples'/f'{a.variant}.json').read_text())
 if header['meta']['model']!=c['data']['teacher_model']:p.error('Trajectory teacher does not match variant')
 out=a.output_dir.resolve();out.mkdir(parents=True,exist_ok=True);dest=out/'config.json'
 if dest.exists():p.error('Config already exists; choose a new output directory')
 if a.variant in {'EcoLight-phase-selection', 'EcoLight-naive'}:
  projected=out/'phase_trajectory.jsonl.gz';phase_projection(trajectory,projected);trajectory=projected
 c['experiment'].update(name=a.variant+'__'+header['meta']['dataset']+'__flow'+str(header['meta']['flow_idx']),output_dir=str(out),random_seed=a.seed)
 c['data'].update(teacher_data=str(trajectory),traffic_config=str(ROOT/'configs/traffic_current.json'))
 c['evaluation']['simulation_seed']=header['meta']['seed']
 c['llm'].update(endpoint=a.endpoint,model=a.model,cache_dir=str(out/'llm_cache'))
 dest.write_text(json.dumps(c,indent=2)+'\n');print(dest)
if __name__=='__main__':main()
