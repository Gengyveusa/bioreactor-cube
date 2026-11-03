"""Read-only bridge from the synthetic Fasolati registry to existing Cube fields.

These are model proxies, never sensor readings. Unsupported targets remain null.
No clinical interpretation, unit conversion, inferred proteins or state assimilation.
"""
from __future__ import annotations

import json
from pathlib import Path
import numpy as np

PANEL_PATH = Path(__file__).resolve().parents[2] / 'data/fasolati/panel-115-v0.1.json'
MODEL_PROXIES = {
    'L-lactate': 'lactate', 'Urea': 'urea', 'Ammonia': 'NH3',
    'Hydrogen sulfide': 'H2S', 'Acetate': 'acetate',
    'Propionate': 'propionate', 'Butyrate': 'butyrate',
}


def load_panel(path=PANEL_PATH):
    panel = json.loads(Path(path).read_text())['panel']
    if len(panel) != 115 or len({r['analyte_id'] for r in panel}) != 115:
        raise ValueError('Expected exactly 115 unique target IDs')
    if len({r['name'] for r in panel}) != 115:
        raise ValueError('Target names must be unique')
    return panel


def read_panel(grid, position=None, *, panel=None):
    """Read a voxel or a whole-cube mean; neither is a saliva specimen measurement.

    Raw legacy field scales are retained. Even lactate is not stereospecific here.
    This aggregate is a volume mean, not a surface flux or sensor response.
    """
    if position is not None:
        position = tuple(position)
        if len(position) != 3 or any(not isinstance(p, (int, np.integer)) or p < 0 or p >= n
                                     for p, n in zip(position, grid.shape)):
            raise ValueError('Position must be an in-bounds integer voxel address')
    records = []
    for target in load_panel() if panel is None else panel:
        field = MODEL_PROXIES.get(target['name'])
        value = None
        state = 'not_implemented'
        reason = 'No molecular field or observation model is implemented for this target'
        if field is not None:
            values = grid.get_field(field)
            value = float(values[position]) if position is not None else float(values.mean())
            if not np.isfinite(value):
                raise ValueError(f'Non-finite model field: {field}')
            state = 'modeled_proxy'
            reason = 'Legacy model field; molecular specificity and physical calibration unconfirmed'
        records.append({
            'analyte_id': target['analyte_id'], 'name': target['name'],
            'category': target['category'], 'state': state, 'value': value,
            'field': field, 'units': 'legacy_model_units' if field else None,
            'evidence_class': 'hypothesis', 'sensor_measured': False,
            'validation_status': 'unconfirmed', 'reason': reason,
            'support': {'kind': 'voxel' if position is not None else 'whole_cube_mean',
                        'position': list(position) if position is not None else None},
        })
    return records

