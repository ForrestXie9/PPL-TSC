import hashlib, importlib.util, json, subprocess, sys, tempfile, unittest
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
spec=importlib.util.spec_from_file_location('collector_provenance',ROOT/'collect_teacher.py')
collector=importlib.util.module_from_spec(spec);spec.loader.exec_module(collector)
class ProvenanceTests(unittest.TestCase):
    def test_audit_matches_and_mismatches(self):
        with tempfile.TemporaryDirectory() as d:
            d=Path(d);weights=d/'weights';road=d/'road';audit=d/'audit.json'
            weights.write_bytes(b'weights');road.write_bytes(b'road')
            valid=dict(status='complete',teacher='FRAP',checkpoint_sha256=hashlib.sha256(weights.read_bytes()).hexdigest(),roadnet_sha256=hashlib.sha256(road.read_bytes()).hexdigest(),scenario='s',flow=0,seed=42)
            audit.write_text(json.dumps(valid))
            self.assertEqual(collector.validate_checkpoint_audit(weights,audit,'FRAP',road)['status'],'verified')
            for field,value in [('teacher','MPLight'),('status','training'),('checkpoint_sha256','wrong'),('roadnet_sha256','wrong')]:
                audit.write_text(json.dumps({**valid,field:value}))
                for legacy in [False,True]:
                    with self.assertRaises(ValueError):collector.validate_checkpoint_audit(weights,audit,'FRAP',road,legacy)
    def test_missing_audit_requires_explicit_legacy_mode(self):
        with tempfile.TemporaryDirectory() as d:
            p=Path(d)/'missing'
            with self.assertRaises(ValueError):collector.validate_checkpoint_audit(p,p,'FRAP',p)
            self.assertEqual(collector.validate_checkpoint_audit(p,p,'FRAP',p,True)['status'],'unaudited_legacy_checkpoint')
    def test_invalid_training_boundary_rejected_before_outputs(self):
        with tempfile.TemporaryDirectory() as d:
            output=Path(d)/'training'
            for stop in ['0','-1']:
                result=subprocess.run([sys.executable,str(ROOT/'train_teacher.py'),'--teacher','MPLight','--scenario','Hangzhou1','--flow','0','--output-dir',str(output),'--stop-after',stop],capture_output=True,text=True)
                self.assertEqual(result.returncode,2,result.stderr);self.assertFalse(output.exists())
if __name__=='__main__':unittest.main()
