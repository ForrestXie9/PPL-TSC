"""The historical phase-feature head must keep its cyclic binary action set."""
import json,sys,unittest
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))
from LLM_SR_DISTILLATION.function_utility import FunctionUtility
class NaiveInterfaceTests(unittest.TestCase):
 def policy(self,mode):
  u=object.__new__(FunctionUtility);u.action_mode=mode;u.feature_stats={}
  u.compute=lambda state,**kwargs:state['demand']
  return u
 def states(self,current,values):
  return {i:{'current_phase':current,'demand':v} for i,v in zip([2,5,9],values)}
 def test_next_only_not_global_argmax(self):
  states=self.states(2,[1,2,100])
  self.assertEqual(self.policy('keep_switch').select_action(states),5)
  self.assertEqual(self.policy('phase_argmax').select_action(states),9)
 def test_wrap_and_tie_keep(self):
  u=self.policy('keep_switch')
  self.assertEqual(u.select_action(self.states(9,[3,100,1])),2)
  self.assertEqual(u.select_action(self.states(2,[2,2,100])),2)
 def test_bias_is_physical_phase_bias(self):
  self.assertEqual(self.policy('keep_switch').select_action(self.states(2,[1,2,100]),phase_bias={2:2,5:0}),2)
 def test_prompt_contract_names_the_cyclic_interface(self):
  from LLM_SR_DISTILLATION.llm_generator import LLMGenerator
  from ppl_tsc_current.short_contract import build
  generator=LLMGenerator('unused','unused','unused',['num_vehicle'],3,'EcoLight',action_mode='keep_switch',feature_stats={'num_vehicle':{'min':0,'max':10,'mean':3,'std':2}})
  contract=build(generator)
  self.assertIn('cyclic successor',contract)
  self.assertIn('Do not select other phases',contract)
  self.assertNotIn('then argmax',contract)
 def test_config_differs_only_by_interface_and_paths(self):
  a=json.loads((ROOT/'configs/examples/EcoLight-naive.json').read_text())
  b=json.loads((ROOT/'configs/examples/EcoLight-phase-selection.json').read_text())
  self.assertEqual(a['search'].pop('action_mode'),'keep_switch');b['search'].pop('action_mode')
  a.pop('experiment');b.pop('experiment');a['llm'].pop('cache_dir');b['llm'].pop('cache_dir')
  self.assertEqual(a,b)
if __name__=='__main__':unittest.main()
