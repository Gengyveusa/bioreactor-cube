import json,unittest,hashlib
from pathlib import Path
import numpy as np
from sim.cube.learning_models import Network,features,constrain_trajectory,OUTPUTS
from sim.cube.learning_teacher import generate
from sim.cube.journey import parameters
from sim.cube.journey_spatial import branch
from sim.cube.learning import inspect,plan,assets
D=Path(__file__).resolve().parents[1]/'data/learning'
class LearningTests(unittest.TestCase):
 def test_analytic_gradients(self):
  rng=np.random.default_rng(61);x=rng.uniform(.1,.9,(5,7));y=rng.uniform(.1,.9,(5,8));y[:,:6]/=y[:,:6].sum(1,keepdims=True)
  for kind in ('mlp','mps'):
   m=Network(kind,seed=3);_,grad=m.gradient(x,y)
   for j,w in enumerate(m.w):
    ix=tuple(int(v//2) for v in w.shape);v=w[ix];h=1e-6
    w[ix]=v+h;a=np.mean((m.forward(x)-y)**2);w[ix]=v-h;b=np.mean((m.forward(x)-y)**2);w[ix]=v
    self.assertAlmostEqual(grad[j][ix],(a-b)/(2*h),places=7)
 def test_born_normalization_and_global_phase(self):
  m=Network('mps');y,(z,_)=m.forward(np.random.default_rng(2).random((16,7)),True)
  a=z[:,:6].astype(complex)*np.exp(1j*.819);p=np.abs(a)**2+1e-10;p/=p.sum(1,keepdims=True)
  np.testing.assert_allclose(p,y[:,:6],atol=1e-14);np.testing.assert_allclose(p.sum(1),1,atol=1e-14)
 def test_checkpoint_roundtrip(self):
  for kind in ('mps','mlp'):
   m=Network(kind);x=np.random.default_rng(2).random((20,7));n=Network.restore(json.loads(json.dumps(m.serial())))
   np.testing.assert_array_equal(m.forward(x),n.forward(x))
 def test_batched_teacher_random_parity(self):
  p=parameters(dict(start=19,diameter_nm=91.3,hindrance=.71,velocity=1.73,clearance=.88,plaque_capture=1.3))
  y=generate([p],(0,19,20,64,120))
  for b in (0,1):
   ref=branch(p,bool(b));z=[[ref[t][k] for k in OUTPUTS] for t in (0,19,20,64,120)]
   np.testing.assert_allclose(y[0,b],z,rtol=1e-12,atol=1e-14)
 def test_case_splits_and_ha_feature_exclusion(self):
  d=json.loads((D/'dataset_manifest.json').read_text());sets=[set(v) for v in d['ids'].values()]
  self.assertEqual(sum(map(len,sets)),len(set.union(*sets)))
  self.assertFalse(set(d['retired_test_ids'])&sets[3])
  p=parameters({});q={**p,'ha_kda':27};np.testing.assert_array_equal(features(p,True,[0,80]),features(q,True,[0,80]))
  # Whole trajectory and both branches are the first two dataset dimensions.
  with np.load(D/'synthetic_journeys.npz') as a:
   for s,n in assets()[0]['protocol']['splits'].items():self.assertEqual(a[s+'_y'].shape,(n,2,16,8))
 def test_null_and_preintervention_exact(self):
  for p in [parameters({'coupling':0}),parameters({'cd44':False})]:
   r=inspect(p);self.assertEqual(r['predicted_frames']['baseline'],r['predicted_frames']['intervention'])
  r=inspect({});self.assertEqual(r['predicted_frames']['baseline'][:3],r['predicted_frames']['intervention'][:3])
 def test_conservative_postprocessor(self):
  y=np.array([[.8,0,0,0,.1,.1,.6,.3],[.7,0,0,0,.05,.25,.6,.3],[.1,0,0,0,.8,.1,.6,.3]])
  z,c=constrain_trajectory(y);self.assertGreater(c,0);np.testing.assert_allclose(z[:,:6].sum(1),1)
  self.assertTrue((np.diff(z[:,4:6],axis=0)>=-1e-14).all());self.assertTrue((z[:,:6]>=0).all())
 def test_ood_abstention_and_calibration(self):
  r=inspect({'velocity':.1});self.assertIn('velocity',r['abstention_reasons']);self.assertIsNone(r['final']['baseline']['blood']['synthetic_90pct_band'])
  r=inspect({});self.assertFalse(r['abstention_reasons']);self.assertIsNotNone(r['final']['baseline']['blood']['synthetic_90pct_band'])
 def test_report_hash_and_constraints(self):
  c,r,_=assets();self.assertEqual(r['checkpoint_sha256'],hashlib.sha256((D/'checkpoint.json').read_bytes()).hexdigest())
  self.assertEqual(r['protocol_sha256'],hashlib.sha256((D/'protocol.json').read_bytes()).hexdigest())
  for n,v in r['results']['test'].items():
   if isinstance(v,dict):self.assertEqual(v['cumulative_monotonicity_violation_fraction'],0);self.assertLess(v['mass_error'],1e-10)
 def test_planner_is_reproducible_and_solver_verified(self):
  r=plan({});s=plan({});self.assertEqual(r,s);self.assertEqual(len(r['candidates']),9)
  self.assertIn('full simulator',r['provenance']);self.assertEqual(len(r['hypotheses']),3)
  a=r['candidates'];self.assertTrue(all(x['score']>=y['score'] for x,y in zip(a,a[1:])))
  self.assertTrue(all(0<=x['expected_information_nats']<=np.log(3)+.05 for x in a))
 def test_meta_learning_and_scale_tools(self):
  from test_journey_standalone import FakeClient,CTX
  from sim.cube.journey_agent import run_journey_agent
  r=run_journey_agent('Compare models and choose measurements',CTX,FakeClient([('inspect_learning',{}),('predict_transport',{}),('plan_measurements',{})]))
  self.assertEqual([x['status'] for x in r['receipts']],['success']*3)
  self.assertEqual([x['kind'] for x in r['actions']],['inspect_learning','predict_transport','plan_measurements'])
  r=run_journey_agent('Explain information energy terrain time',CTX,FakeClient([('inspect_system_contract',{}),('explore_scale',{'scale':'ecosystem'})]))
  self.assertEqual(r['receipts'][0]['result']['contract']['schema'],'information-energy-terrain-time/v1')
  self.assertEqual(r['actions'],[{'type':'scale','scale':'ecosystem'}])
 def test_learning_http_contract(self):
  import http.client,threading
  from http.server import ThreadingHTTPServer
  from sim.cube.lab_server import Handler
  server=ThreadingHTTPServer(('127.0.0.1',0),Handler);worker=threading.Thread(target=server.serve_forever,kwargs={'poll_interval':.01},daemon=True);worker.start()
  try:
   conn=http.client.HTTPConnection('127.0.0.1',server.server_port,timeout=20)
   conn.request('GET','/api/journey/learning');r=conn.getresponse();d=json.loads(r.read());self.assertEqual(r.status,200);self.assertEqual(d['selected'],'mlp')
   for path in ('/api/journey/predict','/api/journey/plan'):
    conn.request('POST',path,'{}',{'Content-Type':'application/json'});r=conn.getresponse();d=json.loads(r.read());self.assertEqual(r.status,200);self.assertIn('run_id',d)
    conn.request('POST',path,'{"untrusted":42}',{'Content-Type':'application/json'});r=conn.getresponse();r.read();self.assertEqual(r.status,400)
   conn.close()
  finally:server.shutdown();server.server_close();worker.join()
if __name__=='__main__':unittest.main()
