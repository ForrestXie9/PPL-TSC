import importlib.util,tempfile,unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch
import torch
ROOT=Path(__file__).resolve().parents[1]
spec=importlib.util.spec_from_file_location('release_teacher',ROOT/'evaluate_teacher.py');runner=importlib.util.module_from_spec(spec);spec.loader.exec_module(runner)
import evaluation
class TransferTests(unittest.TestCase):
 def test_source_zero_replicates_even_if_other_source_has_different_weights(self):
  source=torch.nn.Linear(2,2);other=torch.nn.Linear(2,2)
  def agent(config,env,i):return SimpleNamespace(network_local=torch.nn.Linear(2,2),network_target=torch.nn.Linear(2,2),topology=i)
  with tempfile.TemporaryDirectory() as d:
   path=Path(d)/'model.pth';torch.save({'agent_0_local':source.state_dict(),'agent_1_local':other.state_dict()},path)
   env=SimpleNamespace(n=3)
   with patch.object(evaluation,'get_agent',return_value=agent):controller=runner.load_source_zero(env,{'cur_agent':'FRAP'},path)
   for i,a in enumerate(controller['agents']):
    self.assertEqual(a.topology,i)
    self.assertTrue(torch.equal(a.network_local.weight,source.weight))
 def test_shape_mismatch_is_rejected(self):
  with tempfile.TemporaryDirectory() as d:
   path=Path(d)/'model.pth';torch.save({'agent_0_local':torch.nn.Linear(2,2).state_dict()},path)
   def agent(*args):return SimpleNamespace(network_local=torch.nn.Linear(3,2),network_target=torch.nn.Linear(3,2))
   with patch.object(evaluation,'get_agent',return_value=agent):
    with self.assertRaisesRegex(ValueError,'incompatible tensor shapes'):runner.load_source_zero(SimpleNamespace(n=1),{'cur_agent':'CoLight'},path)
 def test_added_shared_training_checkpoint_is_rejected(self):
  with tempfile.TemporaryDirectory() as d:
   path=Path(d)/'model.pth';torch.save({'shared_local':{},'parameter_sharing':True},path)
   with self.assertRaisesRegex(ValueError,'Shared checkpoint'):runner.load_source_zero(SimpleNamespace(n=1),{'cur_agent':'FRAP'},path)
if __name__=='__main__':unittest.main()
