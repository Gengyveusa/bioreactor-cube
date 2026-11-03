# Fasolati Cube: research experiment workspace

This build connects a visual 3D diffusion experiment to a bounded Meta research
agent. The deterministic experiment works without an API key. The agent can
inspect the existing model's limits and request experiments through explicit
tools; every completed run includes numerical checks and reproducible provenance.

Cube World adds a navigable view of that same field, with spatially grounded
questions and a bounded scene-configuration tool. It is available at `/world`;
see [CUBE_WORLD.md](CUBE_WORLD.md).

## Start

### Mac quick start

For the downloadable bundle, unzip `fasolati-cube.zip`. Open Terminal, type
`bash` followed by a space, drag `start-mac.command` from the extracted folder
into Terminal, and press Return. This supplies the correct path even when the
folder is in Dropbox or its path contains spaces.

The launcher works from any current folder. It checks for Python 3.10 or newer,
creates an isolated environment, installs NumPy, and starts the app with a
hidden API-key prompt. If Python is missing or too old, it provides the official
installer link. After installation, open a new Terminal window and rerun it.
On a python.org installation, also run `Install Certificates.command` in the
installed Python folder if HTTPS certificate errors prevent package installation
or provider access. Python's macOS instructions are at
https://docs.python.org/3/using/mac.html.

Paste the key only when Terminal displays `Meta API key (hidden; retained only
for this process):`. No characters appear while entering it. Press Return,
leave Terminal open, and visit http://127.0.0.1:8765. Press Control-C in Terminal
to stop the server. For numerical experiments without a key, use the manual
launch command below without `--meta`.

When updating a running copy, stop the old server with Control-C and launch the
newly extracted copy. The key is requested again because it is not saved to disk.

### Sensor CSV intake

The sensor panel accepts one local CSV channel, its units and explicit source
declaration, and returns a time trace, file-quality checks and provenance.
`Load synthetic example` exercises the importer without implying a physical
measurement. The new server advertises `sensor_csv_v1`; if the page requests a
restart, stop and relaunch the updated server. See [SENSOR_INTAKE.md](SENSOR_INTAKE.md)
for the format and scientific limits. No real sensor export has been imported
or device connected by this build, and quantitative model comparison remains
blocked pending calibration and matched observations.

### Manual setup

From a checkout of branch `hackathon/cube-research-loop`, with Python 3.10 or newer:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install 'numpy>=1.26' 'pytest>=8.0'
python -m sim.cube.lab_server
```

On Windows, activate with `.venv\Scripts\activate` instead. Open
http://127.0.0.1:8765 in a browser. Click **Run experiment**, switch between
initial and final fields, rotate the Cube, inspect the evidence, and download the
JSON record. Inputs share one color scale so smoothing remains comparable.

For live Meta investigation, stop the server and run:

```bash
python -m sim.cube.lab_server --meta
```

Enter the existing Meta API key at the hidden terminal prompt. It remains in the
server process environment and is not written to a file or sent to the browser.
Do not paste it into chat, a research question, or Git. If `MODEL_API_KEY` is
already configured in the process environment, no prompt is needed. The default
model is `muse-spark-1.3`; `META_MODEL` can select another supported model.

The configured-key indicator is not proof of successful API access. A successful
live request, with provider output and execution receipts, is the integration
check. Local automated tests use a mocked provider. After installing the
Python 3.14 certificates on the user's Mac, a live browser run with
`muse-spark-1.3` succeeded on 2026-10-27 Pacific time; the recorded results are
described below. API calls consume credits and may incur charges according to
the account's billing settings.

Example prompt: “Run the same initial field with diffusion coefficients 200 and
1000 square micrometers per second. Compare their variance after one second and
explain what still needs physical validation.” The last complete experiment is
visualized; receipts retain the metrics of every completed tool call.

## What is new for the hackathon

The solver and the synthetic 115-target registry predate the event. The physical
transport baseline is commit `45d5415e5aa148553756c9a737f27b36643eecd4`.
This work is isolated above branch `fasolati/synthetic-panel-115` at commit
`5c697ce7d20e8027fa31473c35460a6d1b325a3d`.

New event work consists of:

- A bounded experiment runner with seeded initial fields, numerical checks,
  input/output/code hashes, units and explicit limitations.
- An interactive browser workspace with initial/final fields and evidence export.
- A Meta Responses adapter exposing `inspect_run` and `run_transport`, with
  bounded `create_world` configuration available only in the world interface.
- A navigable 200-micrometer world and server-recomputed spatial context for
  questions about the current numerical field.
- A local HTTP service and protocol, validation and numerical regression tests.

The adapter uses https://api.meta.ai/v1/responses. It preserves response output
items, encrypted reasoning and phases when returning tool outputs. It handles at
most five model requests, eight tool calls and three experiment invocations per
question. Experiments allow grids of 8–32 cells per axis, duration 0–10 seconds,
diffusivity 0–2000 square micrometers per second, and at most 2000 solver steps.
No shell, arbitrary filesystem or network tool is exposed to the model.

The service binds only to loopback and checks Host and Origin. It is a local
research workspace, not a production multi-user deployment.

## Scientific scope

The current field is a synthetic lactate-like tracer in a 200-micrometer cube.
Concentration is in arbitrary units. The coefficient is illustrative; it has
not been estimated from sensor data. This demo solves passive diffusion with
closed or periodic boundaries. Reactions, flow, binding and tissue dynamics
are excluded. Solver time does not calibrate the whole biological model.

Mass conservation, nonnegative concentrations and decreasing spatial variance
verify numerical behavior. They do not establish biological validity, clinical
utility, treatment effects or Gengyve efficacy. The registry contains seven
modeled proxies and 108 unimplemented targets, not 115 measured biomarkers.
The prior whole-engine scoreboard (8 passed, 7 failed, 1 blocked) is historical
and is not replaced by this narrower transport validation.

Synthetic experiments support software demonstrations without sensor access.
The next physical validation milestone is one timestamped sensor channel with defined
units, calibration and sample context. Compare its observed trajectory against
a specified simulation before expanding the model. 3D bioprinting is a future
experimental option where controlled geometry is useful; no print or tissue
experiment has been performed by this demo.

## Verify and record

```bash
python -m pytest -q tests/test_world_context.py tests/test_sensor_intake.py tests/test_research_lab.py tests/test_meta_agent.py tests/test_lab_server.py tests/test_physical_transport.py tests/test_fasolati.py tests/test_cube.py tests/test_fused.py
python -m sim.cube.transport_benchmark --out output/hackathon-benchmark
python -m sim.cube.research_lab --out output/cube-lab/run.json
python scripts/build_cube_snapshot.py
```

The benchmark checks an analytical diffusion reference at three resolutions.
`demo/cube-recorded.html` is a self-contained, interactive view of one recorded
numerical run. It cannot execute new experiments or call Meta; start the local
server for those operations. The snapshot can be regenerated from the same code.

`demo/cube-world-recorded.html` is the corresponding navigable recorded world.
It supports exploration and grid-cell inspection without a server. Live Meta
questions require the updated local service at `/world`.

Verification in the build environment on 2026-10-28 UTC: 111 targeted tests
passed, and all five analytical transport benchmark checks passed. The UI's
JavaScript passed a syntax check.

The sensor-intake addition expands the targeted suite to 177 passing tests,
including CSV validation, time handling, source declarations, size limits and
local HTTP routing. The new sensor UI passed JavaScript syntax and focused
DOM-mock checks, including extreme numeric ranges and old-server guards. Visual
browser verification of the sensor panel subsequently passed on the user's Mac
on 2026-10-28 Pacific time: 11 synthetic samples over 10 seconds, all four intake
checks, rendered timeline, locked synthetic source declaration and result
invalidation after metadata changes. No physical sensor was used.

Separately, a live browser integration run was verified on the user's Mac on
2026-10-27 Pacific time with Python 3.14 and NumPy 2.5.4, after installing
Python's certificates. The main browser view was visually inspected in a
screenshot. The live `muse-spark-1.3` response executed `inspect_run` and
`run_transport`: two successful tool calls and no failed tool calls.

The experiment used a 16 × 16 × 16 grid, seed 7, diffusion coefficient
480 square micrometers per second, duration 1 second and closed boundaries.
Run `7985055a844a8b85` returned relative mass error
`1.4268861138406552e-16`; all three numerical checks passed: mass conservation,
nonnegative concentrations and variance that did not increase. The captured
record is [live-meta-verification.json](../demo/live-meta-verification.json).

This confirms one live model/tool/browser cycle and its numerical checks. It
does not establish clinical, sensor or bioprinting validation, or constitute
production load testing.

The world extension expands the targeted suite to 227 passing tests. It has
automated checks for server-recomputed spatial context,
grid boundaries and resolution changes, presentation-tool validation, request
budgets and local HTTP protections. Frontend syntax and focused DOM-mock checks
cover movement, input focus, field sampling and recorded mode. Its new Meta
world tool has not yet been exercised against the live provider, and browser
visual verification awaits the updated local server; the direct HTML preview
was blocked by the browser connection's URL policy.

For the eventual submission, preserve this baseline attribution and live
verification record, capture the demo video, and prepare the required public
repository. This branch does not change repository visibility or submit an entry.
