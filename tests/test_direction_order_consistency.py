"""Reject observed position weighting without rejecting legitimate aggregation."""
import copy
import math
import sys
import unittest
from pathlib import Path
from types import SimpleNamespace

sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'src'))
from LLM_SR_DISTILLATION.constraint_validator import ConstraintValidator
from LLM_SR_DISTILLATION.direction_order_checks import direction_order_consistency
from LLM_SR_DISTILLATION_method_v2.evolution import Evolution


def fixture(directions=('a','b','c'), empty=False, binary=False):
    state=dict(current_phase=0,controlled_directions=list(directions),
               features_by_direction={d:{'num_vehicle':float(i+1)} for i,d in enumerate(directions)})
    states={0:state,1:copy.deepcopy(state)}
    loader=SimpleNamespace(intersection_ids=[0],get_action_consistency_data=lambda *a,**kw:[] if empty else [(states,0)])
    v=ConstraintValidator('FRAP',[],'unused',data_loader=loader,
        actual_feature_keys=['num_vehicle'],feature_stats={'vehicles_raw':{'min':0.,'max':10.}},
        action_mode='binary_keep_switch' if binary else 'phase_argmax')
    return v,states


def u(fn):return SimpleNamespace(compute=lambda s,phase_id=None,**kw:fn(s,phase_id))


class DirectionOrderConsistencyTests(unittest.TestCase):
    def test_positional_weighting_rejected(self):
        v,states=fixture();before=copy.deepcopy(states)
        candidate=u(lambda s,p:sum((1+.05*(i%3))*s['features_by_direction'][d]['num_vehicle'] for i,d in enumerate(s['controlled_directions'])))
        r=direction_order_consistency(v,candidate)
        self.assertFalse(r['satisfied']);self.assertIn('Q ',r['details'][0]);self.assertEqual(states,before)

    def test_feature_and_named_movement_weights_and_holding_allowed(self):
        v,_=fixture()
        candidate=u(lambda s,p:sum((2. if d=='a' else 1.)*s['features_by_direction'][d]['num_vehicle'] for d in s['controlled_directions'])+100*(p==s['current_phase']))
        self.assertTrue(direction_order_consistency(v,candidate)['satisfied'])

    def test_enumerate_for_running_mean_is_allowed(self):
        v,_=fixture()
        def score(s,p):
            mean=0.
            for i,d in enumerate(s['controlled_directions']):mean+=(s['features_by_direction'][d]['num_vehicle']-mean)/(i+1)
            return mean
        self.assertTrue(direction_order_consistency(v,u(score))['satisfied'])

    def test_rotation_finds_dependency_hidden_by_reversal(self):
        v,states=fixture()
        for state in states.values():state['features_by_direction']['c']['num_vehicle']=1.
        r=direction_order_consistency(v,u(lambda s,p:s['features_by_direction'][s['controlled_directions'][0]]['num_vehicle']))
        self.assertFalse(r['satisfied'])
        self.assertTrue(any(p['order_after']==['b','c','a'] and p['Q_before']!=p['Q_after'] for p in r['probes']))

    def test_sampling_reaches_later_intersections_and_unequal_values(self):
        v,states=fixture();zeros=copy.deepcopy(states)
        for state in zeros.values():
            for f in state['features_by_direction'].values():f['num_vehicle']=0.
        v.data_loader=SimpleNamespace(intersection_ids=list(range(40)),get_action_consistency_data=lambda inter,**kw:[(zeros,0),(states if inter==37 else zeros,0)])
        r=direction_order_consistency(v,u(lambda s,p:s['features_by_direction'][s['controlled_directions'][0]]['num_vehicle']),max_checks=4)
        self.assertFalse(r['satisfied']);self.assertLessEqual(r['checked_pairs'],4)
        self.assertTrue(any(p['intersection']==37 for p in r['probes']))

    def test_empty_data_nonfinite_output_and_native_scope(self):
        v,_=fixture(empty=True);self.assertFalse(direction_order_consistency(v,u(lambda s,p:0.))['satisfied'])
        v,_=fixture();self.assertFalse(direction_order_consistency(v,u(lambda s,p:math.nan))['satisfied'])
        v,_=fixture(binary=True);r=direction_order_consistency(v,SimpleNamespace())
        self.assertFalse(r['applicable']);self.assertEqual(r['checked_pairs'],0)
        v,_=fixture(directions=('a',));self.assertFalse(direction_order_consistency(v,u(lambda s,p:1.))['applicable'])

    def test_counterexample_reaches_existing_llm_retry_feedback(self):
        v,_=fixture()
        candidate=u(lambda s,p:sum((1+.05*(i%3))*s['features_by_direction'][d]['num_vehicle'] for i,d in enumerate(s['controlled_directions'])))
        code='def compute_utility(state, phase_id=None, feature_stats=None):\n    return 0.\n'
        r=v.validate_hybrid(code,candidate)
        self.assertFalse(r['valid']);self.assertTrue(r['constraints']['monotonicity']['satisfied'])
        e=Evolution.__new__(Evolution);e.repair_rejected_candidates=False
        feedback=e._build_constraint_feedback(r)
        for text in ('order ', 'unchanged traffic', 'Q ', 'Fix priority derived from list position'):
            self.assertIn(text,feedback)


if __name__=='__main__':unittest.main()
