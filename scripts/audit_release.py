#!/usr/bin/env python3
"""Sequential offline release audit. Writes all trained/test artifacts outside source.

Uses real two-episode teacher training and fixed-budget searches with deterministic
LLM text fixtures. No provider calls; this checks execution, not learned quality.
"""
import argparse,hashlib,json,os,signal,subprocess,sys,time
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
TEACHERS=['CoLight','FRAP','MPLight','TinyLight','EcoLight']
VARIANTS=['CoLight','FRAP','MPLight','TinyLight','EcoLight-native','EcoLight-naive','EcoLight-phase-selection']
STAGES=['environment','unit_modules','fresh_teachers','collection','distillation','student_replays','teacher_transfer','resume','source_integrity']
def snapshot():
    return {str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(ROOT.rglob('*')) if p.is_file() and p.suffix in {'.py','.json','.sh','.txt','.cpp','.h','.hpp'} and not any(part in {'__pycache__','.venv','outputs','.cache','build'} for part in p.relative_to(ROOT).parts) and p.parts[len(ROOT.parts)]!='docs'}
def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--output-dir',required=True,type=Path);parser.add_argument('--timeout',type=int,default=600);args=parser.parse_args()
    out=args.output_dir.resolve()
    if out==ROOT or ROOT in out.parents:parser.error('Audit output must be outside the source directory')
    if out.exists():parser.error('Use a new audit output directory')
    out.mkdir(parents=True);(out/'logs').mkdir()
    env={k:v for k,v in os.environ.items() if not k.startswith(('LLM_','PPL_','METHOD_')) and k!='PYTHONPATH'}
    env.update(PYTHONDONTWRITEBYTECODE='1',OMP_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',MKL_NUM_THREADS='1',CUDA_VISIBLE_DEVICES='')
    report={'status':'running','API_calls':0,'stages':{s:'not_run' for s in STAGES},'checks':[],'scope':{'teacher_episodes':2,'source_scenarios':['Hangzhou1','Atlanta'],'LLM':'deterministic text fixture; network forbidden','search_population':5,'search_generations':5,'CMA_population':20,'CMA_max_iterations':20},'not_covered':['Live provider availability and generation quality','128-episode convergence','GPU execution','All-input formal safety','Unlisted topology/protocol combinations']}
    frozen=snapshot();(out/'source_snapshot.json').write_text(json.dumps(frozen,indent=2))
    def save():
        temp=out/'audit_report.tmp';temp.write_text(json.dumps(report,indent=2)+'\n');temp.replace(out/'audit_report.json')
    def start(stage):
        report['current_stage']=stage;report['stages'][stage]='running';save();print('STAGE '+stage,flush=True)
    def finish(stage):report['stages'][stage]='passed';save()
    def call(name,arguments,expected_failure=False):
        log=out/'logs'/(name+'.log');tick=time.monotonic();row={'name':name,'stage':report['current_stage'],'status':'running','log':str(log.relative_to(out))};report['checks'].append(row);save()
        with log.open('w') as stream:
            proc=subprocess.Popen([sys.executable,*map(str,arguments)],cwd=ROOT,env=env,stdout=stream,stderr=subprocess.STDOUT,start_new_session=True)
            try:code=proc.wait(timeout=args.timeout)
            except subprocess.TimeoutExpired:
                os.killpg(proc.pid,signal.SIGKILL);proc.wait();row.update(status='failed',error='timeout');save();raise
        passed=(code!=0) if expected_failure else (code==0)
        row.update(status='passed' if passed else 'failed',exit_code=code,expected_failure=expected_failure,seconds=round(time.monotonic()-tick,3));save()
        if not passed:raise RuntimeError(f'{name} failed; inspect {log}')
        print('PASS '+name,flush=True)
    def check(name,condition):
        report['checks'].append({'name':name,'stage':report['current_stage'],'status':'passed' if condition else 'failed'});save()
        if not condition:raise RuntimeError(name)
    def teacher_dir(t,city):return out/'teachers'/t/city
    try:
        start('environment');call('dependencies',['-m','pip','check']);call('environment',['scripts/check_environment.py']);call('package',['scripts/check_package.py']);finish('environment')
        start('unit_modules')
        for test in sorted((ROOT/'tests').glob('test_*.py')):call(test.stem,['-m','unittest','discover','-s','tests','-p',test.name,'-v'])
        call('all_tests',['-m','unittest','discover','-s','tests','-v']);finish('unit_modules')
        start('fresh_teachers')
        for city in ['Hangzhou1','Atlanta']:
            for t in TEACHERS:
                call('train_'+t+'_'+city,['train_teacher.py','--teacher',t,'--scenario',city,'--flow','0','--episodes','2','--output-dir',teacher_dir(t,city)])
                audit=json.loads((teacher_dir(t,city)/'audit.json').read_text());check('learned_weights_'+t+'_'+city,audit['changed_tensors']>0 and audit['status']=='complete')
        finish('fresh_teachers')
        start('collection')
        for city in ['Hangzhou1','Atlanta']:
            for t in TEACHERS:
                call('collect_'+t+'_'+city,['collect_teacher.py','--teacher',t,'--scenario',city,'--flow','0','--checkpoint',teacher_dir(t,city)/'checkpoint.pth','--output',out/'trajectories'/(t+'_'+city+'.jsonl.gz')])
        finish('collection')
        start('distillation');deployments={};configs={}
        for v in VARIANTS:
            t=v.split('-')[0];directory=out/'students'/v
            call('prepare_'+v,['scripts/prepare_distillation.py','--variant',v,'--trajectory',out/'trajectories'/(t+'_Hangzhou1.jsonl.gz'),'--output-dir',directory])
            config=directory/'config.json';configs[v]=config
            if v=='MPLight':
                cfg=json.loads(config.read_text());trajectory=cfg['data'].pop('teacher_data');cfg['data'].pop('teacher_model',None)
                matrix=directory/'matrix.json';matrix.write_text(json.dumps([{'teacher_model':t,'teacher_data_path':trajectory,'random_seed':cfg['experiment']['random_seed']}]))
                cfg['data']['teacher_manifest']=str(matrix);config.write_text(json.dumps(cfg,indent=2))
            call('preflight_'+v,['run.py','--config',config,'--dry-run','--no-api-profile'])
            call('distill_'+v,['tests/fixtures/offline_distillation.py','--variant',v,'--config',config,'--record',directory/'fixture_calls.json'])
            found=list(directory.rglob('deployment_model*.json'));check('one_policy_'+v,len(found)==1);deployments[v]=found[0]
            d=json.loads(found[0].read_text());cfg=d['run_config'];history=json.loads(next(found[0].parent.glob('results_*.json')).read_text())['generation_history']
            check('frozen_budget_'+v,cfg['population_size']==cfg['n_generations']==5 and cfg['calibration']['population_size']==cfg['calibration']['max_iterations']==20)
            check('best_fitness_nondecreasing_'+v,all(b['best_score']>=a['best_score'] for a,b in zip(history,history[1:])))
        finish('distillation')
        start('student_replays')
        for v,d in deployments.items():
            for flow in [0,1]:
                result=out/'evaluation'/(v+'_flow'+str(flow)+'.json')
                call('replay_'+v+'_flow'+str(flow),['evaluate.py','--deployment',d,'--scenario','Hangzhou1','--flow',flow,'--output',result])
                if flow==0:check('search_export_ATT_equal_'+v,abs(json.loads(result.read_text())['metrics']['avg_travel_time']-json.loads(d.read_text())['metrics']['student_travel_time'])<1e-6)
            target=out/'transfers'/(v+'.json');call('export_'+v,['scripts/export_formula_transfer.py','--deployment',d,'--output',target])
            original=json.loads(d.read_text());export=json.loads(target.read_text());check('formula_stats_preserved_'+v,export['function_code']==original['function_code'] and export['feature_stats']==original['feature_stats'] and export['per_intersection_bias']=={})
            call('transfer_student_'+v,['evaluate.py','--deployment',target,'--scenario','Atlanta','--flow','0','--steps','900','--output',out/'evaluation'/(v+'_Atlanta.json')])
        finish('student_replays')
        start('teacher_transfer')
        for t in TEACHERS:
            b=teacher_dir(t,'Hangzhou1')
            common=['evaluate_teacher.py','--checkpoint',b/'checkpoint.pth','--training-audit',b/'audit.json']
            call('teacher_flow_'+t,common+['--scenario','Hangzhou1','--flow','1','--protocol','same_network','--output',out/'evaluation'/(t+'_teacher_flow1.json')])
            target=out/'evaluation'/(t+'_teacher_Atlanta.json')
            call('teacher_topology_'+t,common+['--scenario','Atlanta','--flow','0','--protocol','source_agent0','--output',target],expected_failure=t in ['TinyLight','CoLight'])
            check('teacher_protocol_status_'+t,json.loads(target.read_text())['status']==('failed' if t in ['TinyLight','CoLight'] else 'complete'))
        finish('teacher_transfer')
        start('resume')
        v='MPLight';before=hashlib.sha256(deployments[v].read_bytes()).hexdigest();record=out/'resume_fixture.json'
        call('search_resume',['tests/fixtures/offline_distillation.py','--variant',v,'--config',configs[v],'--record',record,'--resume'])
        check('resume_zero_candidates',json.loads(record.read_text())['fixture_calls']==0 and hashlib.sha256(deployments[v].read_bytes()).hexdigest()==before)
        base=['train_teacher.py','--teacher','MPLight','--scenario','Atlanta','--flow','0','--episodes','2','--output-dir',out/'resumed_teacher']
        call('teacher_interrupt',base+['--stop-after','1']);call('teacher_resume',base+['--resume'])
        import torch
        a=torch.load(out/'resumed_teacher/checkpoint.pth',map_location='cpu',weights_only=False);b=torch.load(teacher_dir('MPLight','Atlanta')/'checkpoint.pth',map_location='cpu',weights_only=False)
        check('resumed_weights_equal_continuous',all(torch.equal(value,b[key][name]) for key,state in a.items() for name,value in state.items()));finish('resume')
        start('source_integrity');check('source_unchanged_during_tests',snapshot()==frozen);finish('source_integrity')
        report['status']='passed';report.pop('current_stage',None);save();print('AUDIT PASSED: '+str(out/'audit_report.json'),flush=True)
    except BaseException as exc:
        report['status']='failed';report['stages'][report.get('current_stage',STAGES[0])]='failed';report['error']=type(exc).__name__+': '+str(exc);save();raise
if __name__=='__main__':main()
