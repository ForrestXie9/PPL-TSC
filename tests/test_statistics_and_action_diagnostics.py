import sys,unittest
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'src'))
from LLM_SR_DISTILLATION.statistics_namespace import mask_statistic_keys
from LLM_SR_DISTILLATION.policy_diagnostics import PolicyDecisionDiagnostics
class StatisticsTests(unittest.TestCase):
 stats={'green_vehicle':{'p95':32.,'max':40.}}
 def check(self,code):return mask_statistic_keys(code,self.stats)[1]
 def test_none_fallback_does_not_hide_bad_nested_key(self):
  code="def compute_utility(state, phase_id=None, feature_stats=None):\n if feature_stats is None:\n  feature_stats={'green_vehicle':{'num_vehicle':32}}\n return feature_stats.get('green_vehicle',{}).get('num_vehicle',32)"
  self.assertTrue(any('num_vehicle' in x for x in self.check(code)))
  self.assertFalse(self.check(code.replace(".get('num_vehicle',32)",".get('p95',32)")))
 def test_alias_and_or_fallback(self):
  code="def compute_utility(state, phase_id=None, feature_stats=None):\n fs=feature_stats or {}\n item=fs.get('green_vehicle',{})\n return item.get('wrong',32)"
  self.assertTrue(self.check(code));self.assertFalse(self.check(code.replace("'wrong'","'p95'")))
 def test_empty_and_combined_fallbacks_preserve_source_namespace(self):
  for condition in ['not fs', 'fs is None or not fs', 'fs == None', 'not (fs is not None)']:
   with self.subTest(condition=condition):
    code="def compute_utility(state, phase_id=None, feature_stats=None):\n fs=feature_stats\n if "+condition+":\n  fs={'green_vehicle':{'num_vehicle':32}}\n return fs.get('green_vehicle',{}).get('num_vehicle',32)"
    self.assertTrue(any('num_vehicle' in x for x in self.check(code)))
    self.assertFalse(self.check(code.replace(".get('num_vehicle',32)",".get('p95',32)")))
 def test_unknown_condition_is_not_assumed_false(self):
  code="def compute_utility(state, phase_id=None, feature_stats=None):\n fs=feature_stats\n if state.get('replace',False):\n  fs={}\n return fs.get('green_vehicle',{}).get('num_vehicle',32)"
  self.assertTrue(self.check(code))
 def test_raw_feature_is_separate_namespace(self):
  self.assertFalse(self.check("def compute_utility(state, phase_id=None, feature_stats=None):\n return state['num_vehicle']"))
 def test_direct_validator_checks_nested_fields(self):
  from LLM_SR_DISTILLATION.constraint_validator import ConstraintValidator
  v=object.__new__(ConstraintValidator);v.feature_stats=self.stats
  self.assertFalse(v.check_feature_stats_key_constraints("def compute_utility(state, phase_id=None, feature_stats=None):\n return feature_stats['green_vehicle'].get('num_vehicle',0)")['valid'])
class ActionDiagnosticsTests(unittest.TestCase):
 def test_flags_are_per_intersection_and_descriptive(self):
  d=PolicyDecisionDiagnostics()
  for _ in range(25):d.record(0,0,0,0);d.record(1,0,1,1)
  s=d.summary();self.assertEqual(s['switch_rate'],.5)
  self.assertEqual({x['kind'] for x in s['diagnostic_flags']},{'near_always_hold','near_always_switch'})
 def test_short_and_empty_rollouts_do_not_flag(self):
  d=PolicyDecisionDiagnostics();self.assertEqual(d.summary()['diagnostic_flags'],[])
  for _ in range(19):d.record(0,0,0,0)
  self.assertEqual(d.summary()['diagnostic_flags'],[])
if __name__=='__main__':unittest.main()
