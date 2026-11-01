import gzip,importlib.util,json,math,subprocess,sys,tempfile,unittest
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))
from ppl_tsc_current.artifact_io import atomic_json,evaluation_record,new_gzip_jsonl
spec=importlib.util.spec_from_file_location('evaluation_contract_guard',ROOT/'evaluate.py');evaluator=importlib.util.module_from_spec(spec);spec.loader.exec_module(evaluator)
class ArtifactFailureTests(unittest.TestCase):
    def test_failed_retry_cannot_leave_old_success(self):
        with tempfile.TemporaryDirectory() as d:
            path=Path(d)/'result.json';atomic_json(path,{'status':'complete','metrics':{'ATT':10}})
            with self.assertRaisesRegex(RuntimeError,'injected'):
                with evaluation_record(path,{'flow':1}):
                    self.assertEqual(json.loads(path.read_text())['status'],'running')
                    raise RuntimeError('injected')
            result=json.loads(path.read_text());self.assertEqual(result['status'],'failed');self.assertNotIn('metrics',result)
    def test_success_failure_missing_commit_and_lock_release(self):
        with tempfile.TemporaryDirectory() as d:
            path=Path(d)/'result.json'
            with self.assertRaisesRegex(RuntimeError,'without a complete'):
                with evaluation_record(path,{}):pass
            with evaluation_record(path,{}) as complete:
                with self.assertRaisesRegex(ValueError,'Another evaluation'):
                    with evaluation_record(path,{}):pass
                complete({'metrics':{'ATT':12}})
            self.assertEqual(json.loads(path.read_text())['status'],'complete')
    def test_nonfinite_result_and_interrupt_are_recorded_as_failure(self):
        with tempfile.TemporaryDirectory() as d:
            path=Path(d)/'result.json'
            with self.assertRaises(ValueError):
                with evaluation_record(path,{}) as complete:complete({'ATT':math.nan})
            self.assertEqual(json.loads(path.read_text())['status'],'failed')
            with self.assertRaises(KeyboardInterrupt):
                with evaluation_record(path,{}):raise KeyboardInterrupt()
            self.assertEqual(json.loads(path.read_text())['error_type'],'KeyboardInterrupt')
    def test_atomic_json_never_truncates_on_serialization_error(self):
        with tempfile.TemporaryDirectory() as d:
            path=Path(d)/'result.json';path.write_text('original')
            with self.assertRaises(ValueError):atomic_json(path,{'bad':math.inf})
            self.assertEqual(path.read_text(),'original')
    def test_trajectory_publishes_only_complete_new_files(self):
        with tempfile.TemporaryDirectory() as d:
            path=Path(d)/'trajectory.gz'
            with self.assertRaises(ValueError):new_gzip_jsonl(path,[{'meta':{}},{'bad':math.nan}])
            self.assertEqual(list(Path(d).iterdir()),[])
            new_gzip_jsonl(path,[{'meta':{}},{'value':1}]);before=path.read_bytes()
            with self.assertRaises(ValueError):new_gzip_jsonl(path,[{}])
            self.assertEqual(before,path.read_bytes())
            with gzip.open(path,'rt') as f:self.assertEqual([json.loads(x) for x in f],[{'meta':{}},{'value':1}])
    def test_unknown_feature_contract_cannot_silently_change_features(self):
        base={'action_mode':'phase_argmax','feature_stats':{}}
        for key,value in [('feature_contract_version','normalized_typo'),('direction_feature_semantics','unknown')]:
            with self.assertRaises(ValueError):evaluator.validate_deployment({**base,'run_config':{key:value}},'hash')
        for contract in ['raw_direction_features_v1','normalized_phase_features_v1','normalized_phase_features_v2']:
            self.assertEqual(evaluator.validate_deployment({**base,'run_config':{'feature_contract_version':contract}},'hash'),{})
    def test_cli_bad_input_replaces_stale_success_with_failed_status(self):
        with tempfile.TemporaryDirectory() as d:
            d=Path(d);source=d/'malformed.json';source.write_text('bad JSON');out=d/'result.json'
            for args in [['evaluate.py','--deployment',str(source)],['evaluate_teacher.py','--checkpoint',str(source),'--training-audit',str(source),'--protocol','same_network']]:
                out.write_text('{"status":"complete"}')
                run=subprocess.run([sys.executable,str(ROOT/args[0]),*args[1:],'--scenario','Hangzhou1','--flow','0','--output',str(out)],capture_output=True,text=True)
                self.assertNotEqual(run.returncode,0);self.assertEqual(json.loads(out.read_text())['status'],'failed')
                self.assertEqual(source.read_text(),'bad JSON')
if __name__=='__main__':unittest.main()
