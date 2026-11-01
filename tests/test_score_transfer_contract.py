import importlib.util,math,sys,unittest
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'src'))
from LLM_SR_DISTILLATION_method_v2.evaluator import Evaluator
spec=importlib.util.spec_from_file_location('transfer_contract_runner',ROOT/'evaluate.py');runner=importlib.util.module_from_spec(spec);spec.loader.exec_module(runner)
class ScoreTransferContractTests(unittest.TestCase):
    def test_capped_control_score_contract(self):
        e=Evaluator(None,100,None)
        self.assertEqual(e._compute_performance_score(0),1.)
        self.assertEqual(e._compute_performance_score(.5),1.)
        self.assertAlmostEqual(e._compute_performance_score(-1),0.36787944117144233)
        self.assertGreater(e._compute_relative_performance_factor(80),0)
        self.assertLess(e._compute_relative_performance_factor(120),0)
    def test_invalid_teacher_and_objective_rejected(self):
        for value in [0,-1,math.nan,math.inf]:
            with self.assertRaises(ValueError):Evaluator(None,value,None)
        for weights in [(0,0),(-1,1),(math.nan,1)]:
            with self.assertRaises(ValueError):Evaluator(None,100,None,action_weight=weights[0],performance_weight=weights[1])
    def test_source_offsets_preserved_only_on_matching_topology(self):
        d={'action_mode':'phase_argmax','feature_stats':{},'run_config':{'source_roadnet_sha256':'source'},'per_intersection_bias':{'0':{'0':.3,'1':-.2}}}
        self.assertEqual(runner.validate_deployment(d,'source'),{0:{0:.3,1:-.2}})
        with self.assertRaises(ValueError):runner.validate_deployment(d,'target')
        self.assertEqual(d['per_intersection_bias']['0']['0'],.3)
if __name__=='__main__':unittest.main()
