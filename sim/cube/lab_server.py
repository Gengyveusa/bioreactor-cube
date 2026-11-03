"""Run a local-only Cube research workspace: python -m sim.cube.lab_server."""
import argparse
import json
import os
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from sim.cube.research_lab import BASELINE, run_experiment

WEB = Path(__file__).resolve().parents[2] / 'web' / 'cube-lab.html'
WORLD_WEB = WEB.with_name('cube-world.html')
LOCK = threading.Lock()

class Handler(BaseHTTPRequestHandler):
    def log_message(self, *args):
        pass  # Do not log research questions or credentials.

    def respond(self, status, value, content_type='application/json'):
        raw = value if isinstance(value,bytes) else json.dumps(value,allow_nan=False).encode()
        self.send_response(status)
        self.send_header('Content-Type',content_type)
        self.send_header('Content-Length',str(len(raw)))
        self.send_header('Cache-Control','no-store')
        self.send_header('X-Content-Type-Options','nosniff')
        self.send_header('Content-Security-Policy',"default-src 'self'; script-src 'self' 'unsafe-inline'; style-src 'self' 'unsafe-inline'; img-src 'self' data:; connect-src 'self'; frame-ancestors 'none'")
        self.end_headers();self.wfile.write(raw)

    def permitted(self):
        hosts={f'127.0.0.1:{self.server.server_port}',f'localhost:{self.server.server_port}'}
        if self.headers.get('Host') not in hosts:
            return False
        origin=self.headers.get('Origin')
        return not origin or origin in {'http://'+h for h in hosts}

    def do_GET(self):
        if not self.permitted(): return self.respond(403,{'error':'Local origin required'})
        if self.path=='/api/status':
            return self.respond(200,{'meta_configured':bool(os.environ.get('MODEL_API_KEY')),
                'model':os.environ.get('META_MODEL','muse-spark-1.3'),'baseline':BASELINE,
                'features':['sensor_csv_v1','world_v1','journey_v2','learning_v1']})
        if self.path=='/':
            return self.respond(200,WEB.read_bytes(),'text/html; charset=utf-8')
        if self.path=='/api/journey/learning':
            from sim.cube.learning import summary
            return self.respond(200,summary())
        if self.path=='/journey':
            return self.respond(200,WEB.with_name('cube-journey.html').read_bytes(),'text/html; charset=utf-8')
        if self.path=='/world':
            return self.respond(200,WORLD_WEB.read_bytes(),'text/html; charset=utf-8')
        return self.respond(404,{'error':'Not found'})

    def do_POST(self):
        if not self.permitted(): return self.respond(403,{'error':'Local origin required'})
        if self.path not in ('/api/experiment','/api/agent','/api/sensor','/api/world/agent','/api/journey','/api/journey/agent','/api/journey/challenge','/api/journey/predict','/api/journey/plan'):
            return self.respond(404,{'error':'Not found'})
        if self.headers.get('Content-Type','').split(';')[0]!='application/json':
            return self.respond(415,{'error':'JSON required'})
        try: size=int(self.headers.get('Content-Length','0'))
        except ValueError: return self.respond(400,{'error':'Invalid length'})
        limit=1024*1024 if self.path=='/api/sensor' else 8192
        if not 0<size<=limit: return self.respond(413,{'error':f'Request must be 1–{limit} bytes'})
        if not LOCK.acquire(blocking=False): return self.respond(429,{'error':'An experiment is already running'})
        try:
            self.connection.settimeout(10)
            data=json.loads(self.rfile.read(size))
            if self.path=='/api/experiment': result=run_experiment(data)
            elif self.path=='/api/journey':
                from sim.cube.journey import simulate
                result=simulate(data)
            elif self.path=='/api/journey/challenge':
                from sim.cube.journey import challenge
                result=challenge(data)
            elif self.path in ('/api/journey/predict','/api/journey/plan'):
                from sim.cube.learning import inspect,plan
                result=(inspect if self.path.endswith('/predict') else plan)(data)
            elif self.path=='/api/journey/agent':
                if not isinstance(data,dict) or set(data) not in ({'question','context'},{'question','context','session_id'}):
                    raise ValueError('Expected question and context')
                from sim.cube.journey_agent import run_journey_agent
                result=run_journey_agent(data['question'],data['context'],session_id=data.get('session_id'))
            elif self.path=='/api/sensor':
                from sim.cube.sensor_intake import inspect_sensor, SensorValidationError
                try: result=inspect_sensor(data)
                except SensorValidationError as e:
                    return self.respond(400,{'error':str(e)})
            elif self.path=='/api/world/agent':
                if not isinstance(data,dict) or set(data)!={'question','context'} or data['context'] is None:
                    raise ValueError('Expected a question and world context')
                from sim.cube.meta_agent import run_agent
                result=run_agent(data['question'],world_context=data['context'])
            else:
                if not isinstance(data,dict) or set(data)!={'question'}:
                    raise ValueError('Expected a question')
                from sim.cube.meta_agent import run_agent
                result=run_agent(data['question'])
            return self.respond(200,result)
        except (ValueError, TypeError):
            return self.respond(400,{'error':'Invalid request or experiment parameters'})
        except RuntimeError as e:
            return self.respond(503,{'error':str(e)})
        except Exception:
            return self.respond(500,{'error':'Experiment failed. No successful result was recorded.'})
        finally: LOCK.release()

def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--port',type=int,default=8765)
    p.add_argument('--meta',action='store_true',help='Prompt privately for a Meta key if MODEL_API_KEY is unset')
    args=p.parse_args()
    if not 1024<=args.port<=65535: p.error('Port must be between 1024 and 65535')
    if args.meta and not os.environ.get('MODEL_API_KEY'):
        import getpass
        import sys
        if not sys.stdin.isatty(): p.error('--meta requires an interactive terminal')
        key=getpass.getpass('Meta API key (hidden; retained only for this process): ').strip()
        if not key or any(c.isspace() for c in key): p.error('A valid Meta API key is required')
        os.environ['MODEL_API_KEY']=key
    print(f'Cube Lab: http://127.0.0.1:{args.port}',flush=True)
    ThreadingHTTPServer(('127.0.0.1',args.port),Handler).serve_forever()

if __name__=='__main__': main()
