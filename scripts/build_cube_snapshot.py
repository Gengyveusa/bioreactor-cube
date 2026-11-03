"""Create a portable recorded-run view, with execution explicitly disabled."""
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from sim.cube.research_lab import run_experiment


def main():
    result = run_experiment({})
    source = (ROOT / 'web' / 'cube-lab.html').read_text()
    data = json.dumps(result, allow_nan=False, separators=(',', ':')).replace('<', '\\u003c')
    snapshot = source.replace('<script>', '<script>window.CUBE_SNAPSHOT=' + data + ';</script>\n<script>', 1)
    snapshot = snapshot.replace('href="/journey"', 'href="cube-journey-recorded.html"')
    snapshot = snapshot.replace('href="/world"', 'href="cube-world-recorded.html"')
    dest = ROOT / 'demo' / 'cube-recorded.html'
    dest.parent.mkdir(exist_ok=True)
    dest.write_text(snapshot)
    print(json.dumps({'file': str(dest), 'run_id': result['run_id'],
                      'checks_passed': all(c['passed'] for c in result['checks'])}))
    world = (ROOT / 'web' / 'cube-world.html').read_text()
    world = world.replace('<script>', '<script>window.CUBE_WORLD_SNAPSHOT=' + data + ';</script>\n<script>', 1)
    world = world.replace('href="/journey"', 'href="cube-journey-recorded.html"')
    world = world.replace('href="/"', 'href="cube-recorded.html"')
    world_dest = ROOT / 'demo' / 'cube-world-recorded.html'
    world_dest.write_text(world)
    print(json.dumps({'file': str(world_dest), 'run_id': result['run_id'],
                      'mode': 'recorded_synthetic_world'}))


if __name__ == '__main__':
    main()
