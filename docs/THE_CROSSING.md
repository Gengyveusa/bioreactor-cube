> v3 adds learning, a quantum-inspired benchmark, schooling and the information/energy/terrain/time contract. See [LEARNING_AND_SCALES.md](LEARNING_AND_SCALES.md). The v2 transport equations below are unchanged.

# The Crossing v2 — one vesicle, two worlds

Open `/journey` after starting the local Cube server. The portable HTML contains 12 paired runs and the eight sensitivity scenarios for the default run. The local server enables arbitrary validated settings and live Meta. The original numerical Cube and research lab remain at `/world` and `/`.

## What changed

- OMV-001 is a synthetic *P. gingivalis* exemplar with a chosen diameter, explicit membrane depiction, cargo classes, source links and unknowns. Its default 168.2 nm diameter is informed by an ATCC 33277 preparation mean; it is not a measurement of one actual vesicle. Cargo co-occurrence and copy numbers remain unknown. PPAD is presented as W83 comparator evidence, not assigned to this particle.
- The route ends in a **pre-existing carotid plaque**. The glowing vesicle is a narrative conditioned on successful arrival. Ensemble fractions include clearance and uptake; not every OMV arrives.
- A real carotid figure from the user's archive is reproduced unchanged, with attribution. A manually approximated panel-D contour supplies a teaching extrusion. Depth is invented, not recovered from microscopy. No SAM3D or tomography reconstruction was performed.
- Twenty tissue voxels now use the existing `physical_transport.diffuse` solver, coupled to directed transport, uptake, circulating clearance, plaque capture and release.
- The HA atlas contains 16 evidence-graded chains, upstream processes, receptor/kinase branches, downstream effectors, counterevidence, study contexts and candidate measurements. It is a selected map, not every possible pathway and not a complete kinetic simulation.
- Paired worlds share a camera and timeline. AUC, peaks, cumulative entry, retained fractions and clearance are reported alongside final states. Eight sensitivity scenarios probe the assumptions; they are not statistical confidence intervals.
- The Meta guide can inspect vesicles and pathways, navigate to plaque, control paired view/follow/source/time, change bounded parameters and challenge the model. Six prior turns are retained per session in server memory, with current numerical context recomputed independently. The browser notebook saves up to 24 entries and supports exact-run restore and JSON export.
- Optional browser dictation and narration start only on a button press. Recognition availability and processing depend on the browser/provider. No voice cloning, image upload to Meta or always-on microphone is included.

## The experiment judges should see

Press **Judge story**. The five stops are: one defined vesicle; the HA–barrier hypothesis; spatial paired worlds; a pre-existing carotid destination; the unexpected endpoint. The final button runs the sensitivity challenge.

Default v2 computation: total blood exposure is lower in the HA hypothesis, while final inflammation is slightly higher. This is an output of assumed kinetics, not evidence of an HA treatment effect. Rewind to see the timing and test null coupling before interpreting it.

Use **Follow OMV-001**, **Walk / Fly**, drag to look, WASD to move, E/Q to ascend/descend, scroll to approach, **Overview**, **Compare worlds**, and the time slider. Manual movement exits following. Click structures to inspect their evidence status. The hero is deliberately enlarged and the narrative world combines different scales.

## Mathematical contract

One unit of ensemble probability begins at the oral surface. At every sampled frame:

`oral + sum(tissue voxels) + blood + plaque + tissue uptake + cleared = 1`.

Time is 120 display steps. The physical transport clock assumes **10 seconds per step**, across a 100 µm tissue domain with twenty 5 µm cells and an assumed 20×20 µm cross-section. This is a local tissue segment, not the full mouth-to-heart distance. Systemic circulation is a receiving compartment, not a resolved artery/heart network.

Free diffusion uses `D = k_B T / (3π η d)` with **assumed** T=310 K, viscosity=1 mPa·s and a separate tissue mobility factor (default 0.25). Size influences this diffusion coefficient. The upwind velocity defaults to 1 µm/s, an assumed effective directed transport, not a measured interstitial velocity. Its magnitude and the 10-second clock do not establish physiological transit time.

Twenty operator-split updates occur per display step. Oral influx per step is `(0.008 + 0.06 × (1 − B)) × oral`. The proximal tissue face receives this influx. Closed-face diffusion is handled by the existing positivity-preserving solver, followed by upwind transport with distal efflux to blood. Tissue capture rate is 0.003 per step. Blood clearance rate is 0.015 times the clearance multiplier per step; plaque capture is 0.006 times its multiplier; plaque release is 0.002 per step. Uptake and clearance are absorbing ledgers, with no return to circulation.

Bounded indices evolve as:

```
dI/dstep = 0.16 × tissue × (1 − I) − 0.025 × I
dB/dstep = 0.012 × (1 − B) − 0.04 × I × B + 0.045 × u × (1 − B)
u = strength × coupling × CD44_available after the selected start; otherwise 0
```

These are assumed reduced equations. Cytokine concentrations, individual receptor kinetics, plaque growth, mineralization and clinical risk are **not** integrated. HA molecular weight is recorded as context and does not automatically prescribe efficacy. The CD44 switch removes the assumed repair term; it is not a complete biological knockout with all pleiotropic effects. Matrix-vesicle pathways are distinct host mechanisms and do not turn the OMV into a host vesicle.

AUC uses trapezoidal integration at 10-second sample spacing. Peak time is resolved only to this sampling interval. Sensitivity changes one parameter at a time; the eight rows do not establish joint robustness or population uncertainty. Null controls compare complete trajectories. The v1 compartment implementation remains in `journey_legacy.py` for provenance; v1 and v2 are different models and their values should not be mixed.

## Source and geometry provenance

The full source catalog and per-chain references are in `data/journey/evidence.json`. Seventeen primary publications were retrieved through PubMed; selected full texts were read for the diameter/strain and HA size/mechanism context. The date and study setting travel with the atlas. A direct causal human OMV route from this oral preparation to this carotid specimen has not been established.

Carotid image: Truong et al., *PLOS ONE* (2022), Figure 2, DOI 10.1371/journal.pone.0265598.g002, CC BY 4.0. Panels A–C are synchrotron μCT; D–F are glycophorin A histology. Original pixels and scale bars are retained. Panel-D outer/lumen contours are manually approximate; the approximate 98 px/mm scale calibration is documented, not used to infer a 3D depth or set transport coefficients. All contour points and the original image SHA-256 are in the catalog. This is not electron microscopy of OMV-001.

The existing enamel and tongue references are unchanged archived oral-article images. Their primary acquisition details remain unverified. They are evidence references, not reconstructed tissue or a source of fitted coefficients. No algorithmic segmentation quality claim is made.

## Validation and limits

Run `python3 -m unittest discover -s tests -p 'test_journey*.py' -v` and `node tests/check_journey_ui.cjs` and `node tests/check_journey_live_ui.cjs`. Checks cover conservation, positivity, no pre-intervention effect, null coupling, CD44 channel removal, HA-size metadata invariance, inverse-diameter diffusion, timestep refinement, source resolution, safe tool dispatch, session memory and HTTP boundaries. The Node test exercises controls with a DOM/canvas test double; it does not replace a visual browser inspection.

The live Meta service and microphone require the user’s local browser/server. Provider behavior and real speech capture cannot be certified by the test double. The prior build had a live Meta tool-call demonstration; this upgraded build needs to be started locally to verify the new provider actions end to end.

## Rebuild

The current standalone live HTML embeds `web/journey-v2.js` and `data/journey/evidence.json`. After modifying either, run `python3 scripts/assemble_journey.py`, then `python3 scripts/build_journey_snapshot.py`. This generates `demo/cube-journey-recorded.html`. `web/journey-base.html` retains the prior UI as a reference only; do not use it to overwrite v2. The full source image is embedded in the HTML and kept under `data/journey/` with the original attribution record.
