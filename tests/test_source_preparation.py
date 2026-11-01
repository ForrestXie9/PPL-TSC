import copy,unittest,sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/"src"))
from LLM_SR_DISTILLATION.source_preparation import SourcePreparationError,check_rows,check_statistics,moments,road_schema

def fixture(binary=False):
 schema={n:{0:['a'],1:['b']} for n in range(2)}
 header=dict(config=dict(num_step=40,intersections=[dict(id=n,num_phases=2) for n in range(2)]),schema=dict(binary_interface=binary),stats=dict(avg_travel_time=100.,total_samples=2))
 rows=[]
 for n in range(2):
  for t in range(2):
   phases=[]
   for p,d in [(0,'a'),(1,'b')]:
    state=dict(current_phase=0,controlled_directions=[d],features_by_direction={d:dict(num_vehicle=float(t),pressure=float(t-5))})
    if binary:state['binary_features']=dict(green_vehicle=float(t),red_vehicle=1.)
    phases.append(dict(phase_id=p,state=state,selected=p==0,Q_value=1.))
   row=dict(intersection_id=n,timestep=t,current_phase=0,teacher_action=0,phases=phases)
   if binary:row['teacher_binary_action']=0
   rows.append(row)
 return header,rows,schema

def audit(header,rows,schema,binary=False):return check_rows(header,rows,schema,['num_vehicle','pressure'],'binary_keep_switch' if binary else 'phase_argmax',40)

class SourceContractTests(unittest.TestCase):
 def test_complete_source_with_signed_pressure_passes(self):
  h,r,s=fixture();a=audit(h,r,s);self.assertEqual(a['source_bundles'],4);self.assertEqual(a['physical_phase_states'],8);self.assertLess(a['direction_statistics']['pressure']['min'],0.)
 def test_missing_node_or_time_or_duplicate_is_preparation_error(self):
  h,r,s=fixture()
  for rows in [r[:2],r[:-1],r+[r[0]]]:
   with self.assertRaises(SourcePreparationError):audit(h,rows,s)
 def test_declared_missing_phase_cannot_become_not_applicable(self):
  h,r,s=fixture()
  for row in r:row['phases']=row['phases'][:1]
  with self.assertRaises(SourcePreparationError):audit(h,r,s)
  for inter in h['config']['intersections']:inter['num_phases']=1
  with self.assertRaises(SourcePreparationError):audit(h,r,s)
 def test_missing_required_feature(self):
  h,r,s=fixture();r[0]['phases'][0]['state']['features_by_direction']['a'].pop('pressure')
  with self.assertRaises(SourcePreparationError):audit(h,r,s)
 def test_wrong_direction_and_invalid_current_phase(self):
  for case in ['direction','current']:
   h,r,s=fixture()
   if case=='direction':r[0]['phases'][0]['state']['controlled_directions']=['wrong']
   else:r[0]['current_phase']=7
   with self.assertRaises(SourcePreparationError):audit(h,r,s)
 def test_negative_count_and_nan_are_preparation_errors(self):
  for value in [-1.,float('nan'),float('inf')]:
   h,r,s=fixture();r[0]['phases'][0]['state']['features_by_direction']['a']['num_vehicle']=value
   with self.assertRaises(SourcePreparationError):audit(h,r,s)
 def test_labels_must_match_selected_physical_phase(self):
  h,r,s=fixture();r[0]['teacher_action']=1
  with self.assertRaises(SourcePreparationError):audit(h,r,s)
 def test_binary_labels_cyclic_successor_and_shared_input(self):
  h,r,s=fixture(True);self.assertEqual(audit(h,r,s,True)['status'],'passed')
  r[0]['teacher_binary_action']=1
  with self.assertRaises(SourcePreparationError):audit(h,r,s,True)
  h,r,s=fixture(True);r[0]['phases'][1]['state']['binary_features']['red_vehicle']=100.
  with self.assertRaises(SourcePreparationError):audit(h,r,s,True)
 def test_constant_zero_ranges_and_percentiles_are_valid(self):
  stats=moments([0.,0.,0.]);self.assertEqual(stats['std'],0.);self.assertEqual(stats['p50'],stats['p99']);self.assertEqual(check_statistics({'x':stats},{'x':stats})['status'],'passed')
 def test_statistical_mismatch_not_candidate_violation(self):
  reference={'x':moments([0.,1.,2.])};wrong=copy.deepcopy(reference);wrong['x']['mean']=3.
  with self.assertRaises(SourcePreparationError):check_statistics(wrong,reference)
 def test_phase_support_is_authoritative_not_observed_frequency(self):
  expected={'phase_active':dict(min=0.,max=1.,mean=.5,std=.5)}
  self.assertEqual(check_statistics(expected,expected)['status'],'passed')
 def test_header_counts_global_frames_not_node_rows(self):
  h,r,s=fixture();self.assertEqual(audit(h,r,s)['source_bundles'],4)
  h['stats']['total_samples']=4
  with self.assertRaises(SourcePreparationError):audit(h,r,s)
 def test_roadnet_physical_phases_exclude_all_red(self):
  net=dict(roads=[dict(id='e',points=[dict(x=0,y=0),dict(x=1,y=0)]),dict(id='n',points=[dict(x=0,y=0),dict(x=0,y=1)])],intersections=[dict(virtual=False,trafficLight=dict(lightphases=[dict(availableRoadLinks=[]),dict(availableRoadLinks=[0]),dict(availableRoadLinks=[1])]),roadLinks=[dict(startRoad='e',type='go_straight'),dict(startRoad='n',type='turn_left')])])
  self.assertEqual(road_schema(net),{0:{0:['ET'],1:['NL']}})

if __name__=='__main__':unittest.main()
