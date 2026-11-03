"""Run without optional dependencies: python -m unittest discover -s tests -p test_journey_standalone.py."""
import http.client,json,threading,unittest
from http.server import ThreadingHTTPServer
from sim.cube.journey import simulate,context
from sim.cube.journey_agent import run_journey_agent
from sim.cube.lab_server import Handler

CTX={'parameters':{'strength':.7,'start':20},'region':'barrier','step':60,'branch':'intervention','movement':'fly'}
class FakeClient:
    def __init__(self,calls):self.calls=calls;self.payloads=[]
    def create_response(self,payload):
        self.payloads.append(json.loads(json.dumps(payload)))
        if self.calls:
            name,args=self.calls.pop(0)
            output=[{'type':'function_call','call_id':str(len(self.payloads)),'status':'completed','name':name,'arguments':json.dumps(args)}]
        else:output=[{'type':'message','role':'assistant','content':[{'type':'output_text','text':'Compared the returned model states.'}]}]
        return {'status':'completed','output':output}

class JourneyTests(unittest.TestCase):
    def test_null_intervention_matches_entire_baseline(self):
        r=simulate({'strength':0,'start':20});self.assertEqual(r['baseline'],r['intervention'])
    def test_mass_and_bounds_across_all_presets(self):
        for strength in (0,.35,.7,1):
            for start in (0,20,40,80):
                r=simulate({'strength':strength,'start':start})
                self.assertTrue(r['checks']['mass_conserved']);self.assertTrue(r['checks']['nonnegative'])
                self.assertTrue(r['checks']['identical_before_intervention'])
                for f in r['baseline']+r['intervention']:
                    self.assertTrue(0<=f['integrity']<=1 and 0<=f['inflammation']<=1)
    def test_start_delays_causal_difference(self):
        r=simulate({'strength':1,'start':40})
        self.assertEqual(r['baseline'][:41],r['intervention'][:41])
        self.assertGreater(r['intervention'][80]['integrity'],r['baseline'][80]['integrity'])
        self.assertLess(r['intervention'][80]['cumulative'],r['baseline'][80]['cumulative'])
    def test_reproducible_and_distinct_runs(self):
        self.assertEqual(simulate({}),simulate({}))
        self.assertNotEqual(simulate({})['run_id'],simulate({'strength':0})['run_id'])
    def test_reject_invalid_inputs(self):
        for p in ({'strength':True},{'strength':float('nan')},{'strength':1.1},{'start':20.5},{'start':False},{'dose':2}):
            with self.assertRaises(ValueError):simulate(p)
    def test_context_cannot_supply_fabricated_outcomes(self):
        with self.assertRaises(ValueError):context({**CTX,'blood':99})
        _,obs=context(CTX);self.assertEqual(obs['baseline'],simulate(CTX['parameters'])['baseline'][60])
    def test_meta_navigates_and_runs_real_paired_model(self):
        c=FakeClient([('navigate',{'region':'circulation'}),('set_movement',{'mode':'walk'}),('run_intervention',{'strength':1,'start':40})])
        r=run_journey_agent('Compare',CTX,c)
        self.assertEqual([x['status'] for x in r['receipts']],['success']*3)
        self.assertEqual(r['experiment'],simulate({'strength':1,'start':40}))
        self.assertEqual(r['actions'],[{'type':'navigate','region':'circulation'},{'type':'movement','mode':'walk'}])
        self.assertEqual(r['context']['region'],'circulation')
        self.assertEqual(r['receipts'][0]['result']['movement'],'fly')
        self.assertIn('function_call_output',json.dumps(c.payloads[-1]))
    def test_disallows_arbitrary_tools_and_doses(self):
        r=run_journey_agent('test',CTX,FakeClient([('execute_code',{'code':'x'}),('run_intervention',{'drug':'HA','dose':30})]))
        self.assertEqual([x['status'] for x in r['receipts']],['error','error']);self.assertEqual(r['actions'],[])
    def test_http_surface(self):
        server=ThreadingHTTPServer(('127.0.0.1',0),Handler)
        worker=threading.Thread(target=server.serve_forever,kwargs={'poll_interval':.01},daemon=True);worker.start()
        try:
            conn=http.client.HTTPConnection('127.0.0.1',server.server_port,timeout=10)
            conn.request('GET','/journey');res=conn.getresponse();body=res.read()
            self.assertEqual(res.status,200);self.assertIn(b'THE CROSSING',body)
            conn.request('POST','/api/journey',json.dumps({'strength':.7,'start':20}),{'Content-Type':'application/json'})
            res=conn.getresponse();result=json.loads(res.read());self.assertEqual(res.status,200)
            self.assertEqual(result,simulate({'strength':.7,'start':20}))
            conn.request('POST','/api/journey',json.dumps({'strength':2}),{'Content-Type':'application/json'})
            res=conn.getresponse();res.read();self.assertEqual(res.status,400)
            conn.request('GET','/journey',headers={'Host':'evil.example'})
            res=conn.getresponse();res.read();self.assertEqual(res.status,403)
            conn.close()
        finally:server.shutdown();server.server_close();worker.join()

if __name__=='__main__':unittest.main()
