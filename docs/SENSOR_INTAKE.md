# First sensor channel: local CSV intake

Cube Lab can import one time series and inspect its numerical consistency. This
is a file adapter, not a live device connection. No recorded sensor export was
found in the focused project-file search, so the included example is synthetic.

## Use the panel

Start the updated app, open http://127.0.0.1:8765, and find **Sensor evidence · Import one channel**.
Choose a CSV, enter the channel and unit, and declare whether its source is a
measurement or a synthetic example. Device and sample IDs are optional. Use a
non-identifying sample label. Calibration defaults to unknown; choosing
documented requires a reference, which the importer does not verify.

Click the inspection button to display the trace, summary, file-quality checks,
provenance and a downloadable JSON report. **Load synthetic example** supplies
explicitly artificial data to exercise the workflow. The report does not become
a claim of real measurement when the example is loaded.

Imported contents go only to the local Cube server and are not automatically
included in Meta requests. They are held in the page and response, not written
into the project by the server. Download the report to preserve it. Reloading
the page discards the loaded intake.

## File contract

Use UTF-8 CSV, at most 256 KiB and 2–2000 rows, with exactly one of these headers:

```csv
elapsed_s,value
0,1.2
1,1.3
```

or:

```csv
timestamp,value
2026-10-27T15:00:00-07:00,1.2
2026-10-27T15:00:01-07:00,1.3
```

These examples describe syntax only. Values are not sensor observations.

Timestamps require seconds and a timezone, with at most six fractional-second
digits. They are normalized to UTC and plotted as seconds since the first
sample. Supplied elapsed seconds are nonnegative and preserved. Times must
strictly increase; duplicates and reversed order are rejected. Every value must
be finite. Negative values are retained because the importer cannot assume a
particular channel's physical range.

Missing values, `ND`, censored values such as `<LOD`, NaN and infinity are
rejected. Do not replace these with zero to force an import. Preserve the
original export so a device-specific adapter can later model missingness and
detection limits correctly. Do not strip instrument quality flags or mix channels
to make a multicolumn device export fit this narrow contract.

## Comparison gate

The current Cube experiment uses a synthetic initial field and arbitrary
concentration units. Importing a real physical unit, or supplying a calibration
reference, does not make it quantitatively comparable. The API always returns
`comparison.status = not_comparable`; the UI shows the reason. It does not
produce a prediction error, fitted diffusion parameter or clinical interpretation.

The next adapter needs a real device export and a defined channel, unit,
calibration method, specimen/compartment, sampling time and sampling location or
volume. Raw versus corrected values, recovery and LOD/LOQ also need explicit
handling before physical model comparison. This matches the direction of the
project's proposed `transport_observation` contract; this first importer is not
the complete contract.

The project sources located during implementation were a sensor literature
inventory and proposed architecture, not recorded measurements:

- [OVN Translocation Program Master Architecture](https://www.dropbox.com/scl/fi/7wvszj309lutqydx4jhjv)
- [Graphene Biosensor sensor inventory](https://www.dropbox.com/scl/fi/b6ptl128wtf5i2etvhwtx)
- [Fasolati Product Proposal](https://www.dropbox.com/scl/fi/5sgbv8vq40fa5c6oxy1g5)

## API and evidence

`POST /api/sensor` accepts `{csv_text, metadata}`. Required metadata:
`channel_name`, `unit`, `source_kind` (`measurement` or `synthetic_demo`).
Optional metadata: `sensor_id`, `sample_id`, `calibration_status` (`unknown` or
`documented`), and `calibration_reference`.

The report includes normalized observations, summary statistics, safe validation
messages and SHA-256 hashes of the original CSV, normalized metadata, samples
and parser code. Hashes support traceability; they do not authenticate the device
or prove that source declarations are true. No uploads, database or hardware
drivers are configured.

The example files are `data/sensor/synthetic_tracer.csv` and its metadata JSON.
Only data-structure and numerical consistency checks are available. Calibration,
biological validation, sensor deployment and bioprinting remain separate work.
