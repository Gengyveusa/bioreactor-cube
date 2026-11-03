"""Independent conservation, null, discretization, provenance and tool-boundary checks."""
import json,unittest
from sim.cube.journey import simulate,parameters,challenge,evidence,context
from sim.cube.journey_spatial import branch,metrics,diffusivity
from sim.cube.journey_agent import run_journey_agent,SESSIONS
from test_journey_standalone import FakeClient,CTX

class JourneyV2Tests(unittest.TestCase):
 def test_zero_coupling_and_cd44_controls(self):
  for p in ({'coupling':0},{'cd44':False}):
   r=simulate(p);self.assertEqual(r['baseline'],r['intervention'])
 def test_ha_size_does_not_invent_efficacy(self):
  a,b=simulate({'ha_kda':27}),simulate({'ha_kda':1000})
  self.assertEqual(a['intervention'],b['intervention']);self.assertNotEqual(a['run_id'],b['run_id'])
 def test_size_physics_has_inverse_diameter(self):
  self.assertAlmostEqual(diffusivity(100),2*diffusivity(200))
  a,b=simulate({'diameter_nm':100}),simulate({'diameter_nm':200})
  self.assertNotEqual(a['baseline'][10]['profile'],b['baseline'][10]['profile'])
 def test_profile_closes_tissue_ledger_and_plaque_is_a_sink(self):
  r=simulate({})
  for name in ('baseline','intervention'):
   for f in r[name]:self.assertAlmostEqual(sum(f['profile']),f['tissue'],places=13)
   self.assertEqual(r[name][0]['plaque'],0)
   self.assertGreater(r[name][-1]['plaque'],0)
  self.assertTrue(all(f['plaque']==0 for f in simulate({'plaque_capture':0})['baseline']))
 def test_positivity_at_extreme_parameters(self):
  for p in ({'diameter_nm':50,'hindrance':1,'velocity':2,'clearance':2,'plaque_capture':2},
            {'diameter_nm':300,'hindrance':.05,'velocity':.1,'clearance':.25,'plaque_capture':0}):
   r=simulate(p);self.assertTrue(r['checks']['mass_conserved']);self.assertTrue(r['checks']['nonnegative'])
 def test_time_refinement_preserves_conclusion(self):
  p=parameters({});a=metrics(branch(p,True,20));b=metrics(branch(p,True,40));c=metrics(branch(p,True,80))
  for k in ('blood_auc','plaque_retained','tissue_auc'):
   self.assertLess(abs(b[k]-c[k]),abs(a[k]-b[k]))
   self.assertLess(abs(a[k]-c[k])/max(abs(c[k]),1e-8),.005)
 def test_metrics_are_integrals_not_final_frames(self):
  r=simulate({});b,t=r['metrics']['baseline'],r['metrics']['intervention']
  self.assertLess(t['blood_auc'],b['blood_auc']);self.assertGreater(t['final_inflammation'],b['final_inflammation'])
  self.assertAlmostEqual(t['blood_auc'],sum((a['blood']+b['blood'])*5 for a,b in zip(r['intervention'],r['intervention'][1:])))
 def test_sensitivity_null_controls_and_origin(self):
  c=challenge({});self.assertEqual(c['run_id'],simulate({})['run_id']);self.assertEqual(len(c['rows']),8)
  for row in c['rows'][:2]:self.assertEqual(row['blood_auc_delta'],0);self.assertEqual(row['plaque_retained_delta'],0)
  self.assertTrue(all(r['checks']['mass_conserved'] for r in c['rows']))
 def test_all_pathway_sources_resolve(self):
  e=evidence();self.assertEqual(len(e['pathways']),16)
  for p in e['pathways']:
   self.assertTrue(p['context']);self.assertTrue(p['measure'])
   for source in p['sources']:self.assertIn(source,e['sources'])
  self.assertIn('single-particle',e['vesicle']['properties'][0]['note'])
 def test_agent_challenge_and_bounded_session_memory(self):
  SESSIONS.clear();sid='test-session-00000001'
  client=FakeClient([('inspect_vesicle',{}),('set_parameters',{'cd44':False}),('challenge_model',{})])
  r=run_journey_agent('Compare',CTX,client,session_id=sid)
  self.assertEqual([x['status'] for x in r['receipts']],['success']*3)
  self.assertEqual(r['experiment']['baseline'],r['experiment']['intervention'])
  self.assertEqual(r['actions'][-1]['type'],'challenge')
  next_client=FakeClient([]);run_journey_agent('What changed?',CTX,next_client,session_id=sid)
  self.assertIn('Untrusted previous session notes',json.dumps(next_client.payloads))
 def test_agent_timeline_and_view_controls(self):
  r=run_journey_agent('Compare and rewind',CTX,FakeClient([('set_view',{'control':'compare','enabled':True}),('seek_time',{'step':40,'playing':False})]))
  self.assertEqual([x['status'] for x in r['receipts']],['success']*2)
  self.assertEqual(r['context']['step'],40);self.assertEqual(r['actions'][1]['step'],40)
 def test_no_parameter_injection_or_memory_in_context(self):
  with self.assertRaises(ValueError):context({**CTX,'memory':'fake result'})
  for p in ([],False,{'cd44':1},{'velocity':0},{'ha_kda':float('nan')},{'coupling':True}):
   with self.assertRaises(ValueError):simulate(p)
  r=run_journey_agent('test',CTX,FakeClient([('set_parameters',{'strength':1}),('run_intervention',{'plaque_capture':0})]))
  self.assertEqual([x['status'] for x in r['receipts']],['error','error'])

if __name__=='__main__':unittest.main()
