# Cube World

Cube World turns the current numerical concentration field into a navigable
research scene. Open `/world` on the running local server, or choose **Enter
Cube World** from the experiment lab. No sensor measurements are required.

The space is the same 200 × 200 × 200 micrometer domain as the transport solver.
Displayed marks represent grid-cell values, not individual molecules, biological
cells or physical sensor observations. Glyph size and glow are visual aids.
Initial and final fields share a concentration scale in arbitrary units.

## Explore and ask

Enter the world, move with WASD or the on-screen controls, and drag to look
around. Ascend/descend controls let you explore the volume. Reset to the overview
when needed. Typing a question must not move the camera.

Initial means simulated time zero. After transport means the selected experiment
duration. Camera motion, waiting and rotation do not evolve the numerical field.

The question panel sends the loaded experiment parameters, camera position,
displayed field and optional selected grid cell to the local server. The server
validates these inputs and recomputes the experiment before supplying numerical
context to Meta. This avoids trusting client-provided concentrations or metrics.
The context identifies its run, units, grid spacing and sampling position.

The world-enabled Meta agent can use:

- `inspect_run` to inspect the existing baseline and scope.
- `run_transport` to execute a bounded numerical experiment.
- `create_world` to choose a bounded scene title, palette and annotation setting.

`create_world` configures the renderer. It cannot create arbitrary code, execute
scripts, import remote assets or turn invented biological structures into
validated evidence. A new transport experiment updates the visible field and
has its own receipts. The local context describes the experiment at the time
the question was submitted; later experiments may differ.

Example questions:

- “Build an amber research world and explain the concentration at my location.”
- “What does the concentration at this position mean, and what are its limits?”
- “Run the same initial field for two seconds and explain how its variance changes.”

Live responses use the server's existing Meta key and account credits. Keys
remain outside the browser. Each question is independent; the interface does not
claim conversational memory across requests. Sensor CSV imports are not sent
to Meta by this world interface.

## Recorded exploration

`demo/cube-world-recorded.html` contains a generated numerical experiment and
the renderer in one file. Open it directly for movement and field inspection
without Python or an API key. It explicitly disables live Meta operations and
new experiment execution. Regenerate both recorded viewers with:

```bash
python scripts/build_cube_snapshot.py
```

The current implementation is a browser scene, not a VR headset application,
multiplayer environment, biological digital twin or reconstructed tissue.
Physical calibration, sensors and bioprinting remain future validation work.
