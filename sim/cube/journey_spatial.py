"""Conservative route model coupled to Cube's dimensioned diffusion operator.

The tissue grid is physical, but its geometry, velocity, reaction rates and HA
coupling are assumptions, not a fit to oral or arterial measurements.
"""
import math
import numpy as np
from sim.cube.physical_transport import TransportDomain, diffuse

DOMAIN = TransportDomain((1, 1, 20), (20., 20., 100.), ())
VOLUME = DOMAIN.voxel_volume_um3
STEP_SECONDS = 10.

def diffusivity(diameter_nm):
    # Stokes-Einstein at assumed 310 K and 1 mPa.s. Tissue hindrance is separate.
    return 1.380649e-23 * 310 / (3 * math.pi * .001 * diameter_nm * 1e-9) * 1e12

def branch(p, treated, substeps=20):
    c = np.zeros(DOMAIN.shape)
    oral, blood, plaque, cleared, uptake = 1., 0., 0., 0., 0.
    integrity, inflammation, cumulative = .65, .35, 0.
    d = diffusivity(p['diameter_nm']) * p['hindrance']
    frames = []
    for t in range(121):
        tissue = float(c.sum()*VOLUME)
        frames.append(dict(step=t, seconds=t*STEP_SECONDS, oral=oral, tissue=tissue,
            blood=blood, plaque=plaque, cleared=cleared, uptake=uptake,
            integrity=integrity, inflammation=inflammation, cumulative=cumulative,
            profile=(c.ravel()*VOLUME).tolist()))
        if t == 120: break
        for _ in range(substeps):
            dt = 1/substeps
            repair = p['strength'] * p['coupling'] * p['cd44'] if treated and t >= p['start'] else 0.
            permeability = .008+.06*(1-integrity)
            entry = permeability*oral*dt
            oral -= entry
            c[0,0,0] += entry/VOLUME
            # Reuse the established finite-volume, zero-flux diffusion solver.
            c, _ = diffuse(c, DOMAIN, d, dt*STEP_SECONDS)
            # Positive upwind advection. Distal outflow enters blood exactly once.
            fraction = p['velocity']*dt*STEP_SECONDS/DOMAIN.spacing_um[2]
            if fraction > 1: raise ValueError('Advection CFL bound exceeded')
            out = c*fraction
            c -= out
            c[:,:,1:] += out[:,:,:-1]
            entry_blood = float(out[0,0,-1]*VOLUME)
            capture = c*(.003*dt)
            c -= capture
            uptake += float(capture.sum()*VOLUME)
            cleared_blood = .015*p['clearance']*blood*dt
            to_plaque = .006*p['plaque_capture']*blood*dt
            from_plaque = .002*plaque*dt
            blood += entry_blood-cleared_blood-to_plaque+from_plaque
            plaque += to_plaque-from_plaque
            cleared += cleared_blood
            cumulative += entry_blood
            tissue = float(c.sum()*VOLUME)
            di=(.16*tissue*(1-inflammation)-.025*inflammation)*dt
            db=(.012*(1-integrity)-.04*inflammation*integrity+.045*repair*(1-integrity))*dt
            inflammation += di
            integrity += db
    return frames

def metrics(frames):
    def auc(key):
        return sum((a[key]+b[key])*.5*STEP_SECONDS for a,b in zip(frames,frames[1:]))
    peak=max(frames,key=lambda f:f['blood'])
    return dict(blood_auc=auc('blood'), tissue_auc=auc('tissue'), plaque_auc=auc('plaque'),
        peak_blood=peak['blood'], peak_seconds=peak['seconds'],
        cumulative_blood_entry=frames[-1]['cumulative'], cleared=frames[-1]['cleared'],
        plaque_retained=frames[-1]['plaque'], tissue_uptake=frames[-1]['uptake'],
        final_inflammation=frames[-1]['inflammation'])
