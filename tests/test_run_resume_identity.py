import copy,importlib.util,json,tempfile,unittest
from pathlib import Path
from unittest.mock import patch
ROOT=Path(__file__).resolve().parents[1]
spec=importlib.util.spec_from_file_location('release_resume_runner',ROOT/'run.py')
runner=importlib.util.module_from_spec(spec);spec.loader.exec_module(runner)

class ResumeIdentityTests(unittest.TestCase):
    def fixture(self, root):
        road=root/'data'/'city';road.mkdir(parents=True)
        (road/'roadnet.json').write_text('{}');(road/'flow_0.json').write_text('[]')
        trajectory=root/'teacher.jsonl';trajectory.write_text(json.dumps({'meta':{'dataset':'city','flow_idx':0}})+'\n')
        traffic=root/'traffic.json';traffic.write_text('{}')
        return {'experiment':{'random_seed':42},'llm':{'model':'fixture'}},{'method_project_root':str(root),'traffic_config':str(traffic)},[trajectory],root/'out'
    def test_same_inputs_resume_but_changed_seed_rejected(self):
        with tempfile.TemporaryDirectory() as d,patch.object(runner,'_sha256_source_tree',return_value='source'):
            cfg,res,paths,out=self.fixture(Path(d))
            with runner._run_output_guard(cfg,res,paths,out):pass
            before=(out/'run_identity.json').read_bytes()
            with runner._run_output_guard(cfg,res,paths,out):pass
            changed=copy.deepcopy(cfg);changed['experiment']['random_seed']=43
            with self.assertRaisesRegex(ValueError,'changed'):
                with runner._run_output_guard(changed,res,paths,out):pass
            self.assertEqual((out/'run_identity.json').read_bytes(),before)
    def test_changed_trajectory_topology_and_implementation_rejected(self):
        for field in ['trajectory','roadnet','implementation']:
            with tempfile.TemporaryDirectory() as d,patch.object(runner,'_sha256_source_tree',return_value='source') as digest:
                cfg,res,paths,out=self.fixture(Path(d))
                with runner._run_output_guard(cfg,res,paths,out):pass
                if field=='trajectory':paths[0].write_text(paths[0].read_text()+'{}\n')
                elif field=='roadnet':(Path(d)/'data/city/roadnet.json').write_text('{"changed":true}')
                else:digest.return_value='new source'
                with self.assertRaisesRegex(ValueError,'changed'):
                    with runner._run_output_guard(cfg,res,paths,out):pass
    def test_concurrent_writer_rejected_and_lock_released(self):
        with tempfile.TemporaryDirectory() as d,patch.object(runner,'_sha256_source_tree',return_value='source'):
            args=self.fixture(Path(d))
            with runner._run_output_guard(*args):
                with self.assertRaisesRegex(ValueError,'Another run'):
                    with runner._run_output_guard(*args):pass
            with self.assertRaises(RuntimeError):
                with runner._run_output_guard(*args):raise RuntimeError('simulated failure')
            with runner._run_output_guard(*args):pass
    def test_legacy_results_not_silently_reused(self):
        with tempfile.TemporaryDirectory() as d,patch.object(runner,'_sha256_source_tree',return_value='source'):
            args=self.fixture(Path(d));out=args[-1];out.mkdir();old=out/'deployment_model_old.json';old.write_text('{}')
            with self.assertRaisesRegex(ValueError,'no run identity'):
                with runner._run_output_guard(*args):pass
            self.assertEqual(old.read_text(),'{}');self.assertFalse((out/'run_identity.json').exists())
    def test_manifest_alias_has_same_artifact_paths(self):
        with tempfile.TemporaryDirectory() as d:
            root=Path(d);cfg,res,paths,out=self.fixture(root)
            common={'teacher_model':'FRAP'}
            kwargs=dict(output_dir=out,population_size=5,generations=5,simulation_seed=42)
            self.assertEqual(runner._result_artifacts({**common,'teacher_data_path':str(paths[0])},**kwargs),runner._result_artifacts({**common,'teacher_data':str(paths[0])},**kwargs))
    def test_different_search_seeds_cannot_overwrite_same_path(self):
        with tempfile.TemporaryDirectory() as d:
            root=Path(d);cfg,res,paths,out=self.fixture(root)
            manifest=root/'matrix.json';manifest.write_text(json.dumps([{'teacher_model':'FRAP','teacher_data_path':str(paths[0]),'random_seed':seed} for seed in [1,2]]))
            cfg={'experiment':{'output_dir':str(out)},'data':{'teacher_manifest':str(manifest),'traffic_config':res['traffic_config']},'llm':{'endpoint':'fixture.invalid','model':'fixture'},'search':{},'evaluation':{}}
            with patch.object(runner,'ROOT',root):
                with self.assertRaisesRegex(ValueError,'overwrite the same result path'):
                    runner._build_internal_argv(cfg,offline=True,disable_cache=False,validate_only=True)
    def test_completed_artifacts_must_describe_same_policy(self):
        with tempfile.TemporaryDirectory() as d:
            paths=tuple(Path(d)/name for name in ['results.json','deployment.json','policy.py'])
            code='def compute_utility(state, phase_id=None, feature_stats=None):\n    return 1.0\n'
            metrics={'student_travel_time':10.,'teacher_travel_time':12.}
            result={'best_individual':{'score':1.,'metrics':metrics,'function_code':code},'run_config':{'seed':1},'per_intersection_bias':{'0':{'0':0.1}}}
            deployment={'score':1.,'metrics':metrics,'function_code':code,'run_config':{'seed':1},'per_intersection_bias':{'0':{'0':0.1}}}
            paths[0].write_text(json.dumps(result));paths[2].write_text(code)
            paths[1].write_text(json.dumps(deployment));self.assertTrue(runner._artifacts_are_complete(paths))
            for key,value in [('function_code',code.replace('1.0','2.0')),('run_config',{'seed':2}),('per_intersection_bias',{})]:
                paths[1].write_text(json.dumps({**deployment,key:value}))
                self.assertFalse(runner._artifacts_are_complete(paths),key)
if __name__=='__main__':unittest.main()
