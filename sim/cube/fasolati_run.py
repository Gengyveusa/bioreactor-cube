"""Run the existing 3-D oral Cube and export a source-linked Fasolati readout.

python -m sim.cube.fasolati_run --dim 12 --steps 20 --out output/fasolati
"""
from __future__ import annotations
import argparse
from dataclasses import asdict
import hashlib
import json
from pathlib import Path
import numpy as np
from sim.cube.fasolati import PANEL_PATH, load_panel, read_panel
from sim.cube.fused import CubeEpoch4
from sim.cube.grid import CubeGrid, CHEMISTRY_FIELDS
from sim.cube.tissues import ORAL_TISSUE
from sim.cube.sample import BLEEDING_POCKET

UPSTREAM = '2a0c800b38aaaf49a8b5428cffafcb61a713e0d8'

def run(out, dim=12, steps=20, dt=0.1):
    if not 4 <= dim <= 64 or not 1 <= steps <= 1000:
        raise ValueError('Use dim 4..64 and steps 1..1000 for this bounded prototype')
    if not np.isfinite(dt) or not 0 < dt <= 0.1:
        raise ValueError('dt must be finite and in (0, 0.1] model time units')
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    grid = CubeGrid(dim, dim, dim, species_ids=ORAL_TISSUE.cell_ids)
    eng = CubeEpoch4(grid, ORAL_TISSUE)
    eng.seed_uniform()
    panel = load_panel()
    trajectory = [{'step': 0, 'model_time': eng.t, 'targets': read_panel(grid, panel=panel)}]
    for step in range(1, steps+1):
        eng.step(dt)
        if step % max(1, steps//10) == 0 or step == steps:
            trajectory.append({'step': step, 'model_time': eng.t,
                               'targets': read_panel(grid, panel=panel)})
    final = trajectory[-1]['targets']
    arrays = {name: grid.get_field(name) for name in CHEMISTRY_FIELDS}
    arrays['composition'] = grid.composition
    if not all(np.isfinite(a).all() for a in arrays.values()):
        raise ValueError('Non-finite engine output; do not publish this run')
    np.savez_compressed(out/'cube-fields.npz', **arrays)
    (out/'readout.json').write_text(json.dumps(trajectory, indent=2, allow_nan=False))
    sha = lambda path: hashlib.sha256(Path(path).read_bytes()).hexdigest()
    manifest = {
        'engine': 'Existing CubeEpoch4, ORAL_TISSUE', 'upstream_commit': UPSTREAM,
        'shape': list(grid.shape), 'voxels': grid.n_cells,
        'cell_size_um': grid.cell_size_um, 'geometry_status': 'legacy_assumption_not_calibrated',
        'steps': steps, 'dt_model_units': dt, 'elapsed_model_units': eng.t,
        'physical_elapsed_seconds': None, 'physical_clock_status': 'not_anchored',
        'panel_sha256': sha(PANEL_PATH), 'target_count': len(final),
        'modeled_proxy_count': sum(r['state']=='modeled_proxy' for r in final),
        'sensor_measurements': 0, 'not_implemented_count': sum(r['state']=='not_implemented' for r in final),
        'boundary_conditions': asdict(BLEEDING_POCKET),
        'boundary_source': 'sim/cube/faces.py and sim/cube/tissues.py',
        'evidence_class': 'hypothesis', 'multiverse_backend': 'not_connected',
        'artifacts': {f: sha(out/f) for f in ['cube-fields.npz','readout.json']},
        'source_sha256': {str(p.relative_to(Path(__file__).resolve().parents[2])): sha(p)
                          for p in sorted((Path(__file__).resolve().parents[2]/'sim').rglob('*.py'))},
        'limits': ['No physical-time prediction', 'No clinical validation',
                   'No Gengyve efficacy inference', 'No claimed 115-molecule sensor coverage',
                   'Legacy convergence and falsification failures remain unresolved'],
    }
    (out/'manifest.json').write_text(json.dumps(manifest,indent=2,allow_nan=False))
    print(json.dumps({k:manifest[k] for k in ['shape','voxels','target_count','modeled_proxy_count',
                                             'not_implemented_count','sensor_measurements','physical_clock_status']}))
    return manifest

if __name__ == '__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--dim',type=int,default=12)
    p.add_argument('--steps',type=int,default=20)
    p.add_argument('--dt',type=float,default=0.1)
    p.add_argument('--out',default='output/fasolati')
    a=p.parse_args()
    run(a.out,a.dim,a.steps,a.dt)

