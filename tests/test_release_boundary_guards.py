import gzip, importlib.util, json, tempfile, unittest, subprocess, sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
def module(name,path):
    spec=importlib.util.spec_from_file_location(name,ROOT/path)
    m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m);return m
projection=module('projection_guard','scripts/prepare_distillation.py')
evaluate=module('deployment_guard','evaluate.py')
teacher=module('teacher_guard','evaluate_teacher.py')
class BoundaryTests(unittest.TestCase):
    def test_source_seed_and_explicit_override(self):
        self.assertEqual(teacher.resolve_seed(None,{'seed':284765}),284765)
        self.assertEqual(teacher.resolve_seed(42,{'seed':284765}),42)
        with self.assertRaises(ValueError):teacher.resolve_seed(None,{})
    def test_projection_never_overwrites(self):
        with tempfile.TemporaryDirectory() as d:
            source=Path(d)/'source.gz';source.write_bytes(b'original')
            for output in [source,Path(d)/'existing.gz']:
                output.write_bytes(b'original')
                with self.assertRaises(ValueError):projection.phase_projection(source,output)
                self.assertEqual(output.read_bytes(),b'original')
    def test_failed_projection_publishes_nothing(self):
        for content in ['',json.dumps({'meta':{}})+'\n',json.dumps({'meta':{}})+'\n{bad json']:
            with tempfile.TemporaryDirectory() as d:
                source=Path(d)/'source.gz';output=Path(d)/'output.gz'
                with gzip.open(source,'wt') as f:f.write(content)
                before=source.read_bytes()
                with self.assertRaises((ValueError,KeyError)):projection.phase_projection(source,output)
                self.assertFalse(output.exists());self.assertEqual(source.read_bytes(),before)
                self.assertEqual(list(Path(d).iterdir()),[source])
    def test_cli_refuses_input_as_output(self):
        with tempfile.TemporaryDirectory() as d:
            source=Path(d)/'input.json';source.write_text('{"teacher":"FRAP"}')
            commands=[['evaluate.py','--deployment',str(source),'--scenario','Hangzhou1','--flow','0'],
                      ['collect_teacher.py','--teacher','FRAP','--checkpoint',str(source),'--scenario','Hangzhou1','--flow','0'],
                      ['evaluate_teacher.py','--checkpoint',str(source),'--training-audit',str(source),'--protocol','same_network','--scenario','Hangzhou1','--flow','0']]
            for command in commands:
                result=subprocess.run([sys.executable,str(ROOT/command[0]),*command[1:],'--output',str(source)],capture_output=True,text=True)
                self.assertEqual(result.returncode,2,result.stderr)
                self.assertIn('must not overwrite',result.stderr)
                self.assertEqual(source.read_text(),'{"teacher":"FRAP"}')
    def test_reject_ambiguous_bias_indices(self):
        for key in ['01','-1','1.0',1.2,True]:
            with self.assertRaises(ValueError):evaluate.index_key(key)
        self.assertEqual(evaluate.index_key('1'),1)
    def test_bias_cannot_be_silently_ignored(self):
        evaluate.validate_bias_support({0:{0:1,1:0},1:{2:1}},[2,3])
        for bias in [{2:{0:1}},{0:{2:1}}]:
            with self.assertRaises(ValueError):evaluate.validate_bias_support(bias,[2,3])
if __name__=='__main__':unittest.main()
