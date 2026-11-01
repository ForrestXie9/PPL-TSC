import gzip,hashlib,importlib.util,json,tempfile,unittest
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
spec=importlib.util.spec_from_file_location('prepare',ROOT/'scripts/prepare_distillation.py');mod=importlib.util.module_from_spec(spec);spec.loader.exec_module(mod)
class ProjectionTests(unittest.TestCase):
 def test_preserves_physical_labels_and_source(self):
  with tempfile.TemporaryDirectory() as directory:
   source=Path(directory)/'native.jsonl.gz';output=Path(directory)/'phase.jsonl.gz'
   rows=[{'meta':{'model':'EcoLight'},'schema':{'binary_interface':True}}, {'teacher_action':2,'teacher_binary_action':1,'teacher_binary_q_values':[0,1],'phases':[{'phase_id':2,'selected':True,'state':{'features_by_direction':{'NT':{'num_vehicle':3}},'binary_features':{'green_vehicle':1,'red_vehicle':3}}}]}]
   with gzip.open(source,'wt') as f:
    for r in rows:f.write(json.dumps(r)+'\n')
   before=source.read_bytes();mod.phase_projection(source,output)
   with gzip.open(output,'rt') as f:result=[json.loads(line) for line in f]
   self.assertEqual(source.read_bytes(),before)
   self.assertEqual(result[0]['meta']['native_trajectory_sha256'],hashlib.sha256(before).hexdigest())
   self.assertFalse(result[0]['schema']['binary_interface'])
   self.assertEqual(result[1]['teacher_action'],2)
   self.assertEqual(result[1]['phases'][0]['state']['features_by_direction'],rows[1]['phases'][0]['state']['features_by_direction'])
   self.assertNotIn('teacher_binary_action',result[1]);self.assertNotIn('binary_features',result[1]['phases'][0]['state'])
if __name__=='__main__':unittest.main()
