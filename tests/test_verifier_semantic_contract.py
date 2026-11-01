"""Executable positive and negative controls for the documented verifier semantics."""
import copy,sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/"src"))
from types import SimpleNamespace
import unittest
from LLM_SR_DISTILLATION.constraint_validator import ConstraintValidator
from LLM_SR_DISTILLATION.semantic_checks import phase_monotonicity,binary_monotonicity
from LLM_SR_DISTILLATION.active_phase_protection import check

def fixture(binary=False):
    states={p:dict(current_phase=0,controlled_directions=['a','b'],features_by_direction={'a':dict(pressure=-5.,num_vehicle=2.),'b':dict(pressure=2.,num_vehicle=4.)},binary_features=dict(green_vehicle=2.,red_vehicle=3.)) for p in range(2)}
    loader=SimpleNamespace(intersection_ids=[0],get_action_consistency_data=lambda *a,**k:[(states,0)])
    validator=ConstraintValidator('EcoLight' if binary else 'FRAP',[],'unused',data_loader=loader,actual_feature_keys=['pressure','num_vehicle'],feature_stats={'pressure_raw':{'min':-10.,'max':10.},'vehicles_raw':{'min':0.,'max':10.}},action_mode='binary_keep_switch' if binary else 'phase_argmax')
    return validator,states

def utility(fn):return SimpleNamespace(compute=lambda state,phase_id=None,**kw:fn(state,phase_id))

class SemanticContractTests(unittest.TestCase):
    def test_signed_pressure_positive_and_negative_controls(self):
        v,states=fixture();before=copy.deepcopy(states)
        for sign,expected in [(1,True),(-1,False)]:
            candidate=utility(lambda s,p:sign*sum(s['features_by_direction'][d]['pressure'] for d in s['controlled_directions']))
            self.assertEqual(phase_monotonicity(v,candidate)[0],expected)
        self.assertEqual(states,before)
    def test_active_bonus_and_active_penalty(self):
        v,_=fixture()
        for coefficient,expected in [(0,'passed'),(100,'passed'),(-1,'failed')]:
            result=check(v,utility(lambda s,p:sum(f['num_vehicle'] for f in s['features_by_direction'].values())+coefficient*(p==s['current_phase'])),stop_on_failure=False)
            self.assertEqual(result['status'],expected);self.assertTrue(result['complete'])
    def test_binary_preference_directions(self):
        v,states=fixture(True);before=copy.deepcopy(states)
        for sign,expected in [(1,True),(-1,False)]:
            candidate=SimpleNamespace(_native_binary_context=lambda states:(states[0],0,1),compute_binary_action_values=lambda states,**kw:(0,1,sign*states[0]['binary_features']['green_vehicle'],sign*states[0]['binary_features']['red_vehicle']))
            self.assertEqual(binary_monotonicity(v,candidate)[0],expected)
        self.assertEqual(states,before)
    def test_empty_source_is_not_a_pass(self):
        v,_=fixture();v.data_loader.get_action_consistency_data=lambda *a,**kw:[]
        self.assertFalse(phase_monotonicity(v,utility(lambda s,p:0.))[0])
        self.assertEqual(check(v,utility(lambda s,p:0.))['status'],'failed')
if __name__=='__main__':unittest.main()
