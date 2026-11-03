"""Bundle deterministic paired experiments and source references for offline exploration."""
import json
import sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from sim.cube.journey import simulate, challenge
from sim.cube.learning import inspect,plan

def main():
    snapshots={f'{s}:{start}':simulate({'strength':s,'start':start})
               for s in (0,.35,.7,1) for start in (0,20,40)}
    extra='<script>window.JOURNEY_DEFAULT_ID='+json.dumps(snapshots['0.7:20']['run_id'])+';window.JOURNEY_CHALLENGE='+json.dumps(challenge({}),separators=(',',':'),allow_nan=False).replace('<','\\u003c')+';</script>\n'
    learning={r['run_id']:inspect(r['parameters']) for r in snapshots.values()}
    extra+='<script>window.JOURNEY_LEARNING='+json.dumps(learning,separators=(',',':')).replace('<','\\u003c')+';window.JOURNEY_PLAN='+json.dumps(plan({}),separators=(',',':')).replace('<','\\u003c')+';</script>\n'
    data=json.dumps(snapshots,separators=(',',':'),allow_nan=False).replace('<','\\u003c')
    html=(ROOT/'web/cube-journey.html').read_text()
    html=html.replace('<script>',extra+'<script>window.JOURNEY_SNAPSHOTS='+data+';</script>\n<script>',1)
    (ROOT/'demo/cube-journey-recorded.html').write_text(html)
    print(json.dumps({'file':'demo/cube-journey-recorded.html','paired_runs':len(snapshots),'all_checks_passed':all(all(v for k,v in r['checks'].items() if k!='mass_error') for r in snapshots.values())}))
if __name__=='__main__':main()
