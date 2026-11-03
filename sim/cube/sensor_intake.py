"""Inspect one locally supplied sensor CSV without interpreting its biology.

Declared units, source and calibration remain user assertions. This parser does
not calibrate the Cube, establish device authenticity, or transmit observations.
"""
from __future__ import annotations

import csv
from datetime import datetime, timezone
import hashlib
import io
import json
import math
from pathlib import Path
import re
import statistics
import unicodedata


MAX_CSV_BYTES = 256 * 1024
MAX_ROWS = 2000
PARSER_VERSION = "1.0"
TIMESTAMP_FORMAT = re.compile(
    r"\d{4}-\d{2}-\d{2}[Tt ]\d{2}:\d{2}:\d{2}(?:\.\d{1,6})?"
    r"(?:[Zz]|[+-](?:[01]\d|2[0-3])(?::?[0-5]\d)?)"
)
METADATA_KEYS = {
    "channel_name", "unit", "sensor_id", "sample_id", "calibration_status",
    "calibration_reference", "source_kind",
}
COMPARISON_REASON = (
    "Cube uses arbitrary concentration units and synthetic initial conditions; "
    "physical calibration and matched observation are required."
)


class SensorValidationError(ValueError):
    """A fixed, safe validation message suitable for the local HTTP response."""


def _digest(value):
    encoded = json.dumps(value, sort_keys=True, separators=(",", ":"),
                         allow_nan=False).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _text(value, name, *, default=None, limit=128):
    if not isinstance(value, str):
        raise SensorValidationError(f"{name} must be text.")
    value = value.strip()
    if not value and default is not None:
        return default
    if not value or len(value) > limit:
        raise SensorValidationError(f"{name} must contain 1–{limit} characters.")
    if any(unicodedata.category(char).startswith("C") for char in value):
        raise SensorValidationError(f"{name} cannot contain control characters.")
    return value


def _metadata(value):
    if not isinstance(value, dict) or set(value) - METADATA_KEYS:
        raise SensorValidationError("Expected only the supported sensor metadata fields.")
    for name in ("channel_name", "unit", "source_kind"):
        if name not in value:
            raise SensorValidationError(f"{name} is required.")
    result = {
        "channel_name": _text(value["channel_name"], "channel_name"),
        "unit": _text(value["unit"], "unit"),
        "sensor_id": _text(value.get("sensor_id", ""), "sensor_id", default="not supplied"),
        "sample_id": _text(value.get("sample_id", ""), "sample_id", default="not supplied"),
        "calibration_status": _text(value.get("calibration_status", "unknown"), "calibration_status"),
        "calibration_reference": _text(value.get("calibration_reference", ""),
                                       "calibration_reference", default="", limit=256),
        "source_kind": _text(value["source_kind"], "source_kind"),
    }
    if result["source_kind"] not in ("measurement", "synthetic_demo"):
        raise SensorValidationError("source_kind must be measurement or synthetic_demo.")
    if result["calibration_status"] not in ("unknown", "documented"):
        raise SensorValidationError("calibration_status must be unknown or documented.")
    if result["calibration_status"] == "documented" and not result["calibration_reference"]:
        raise SensorValidationError("Documented calibration requires a calibration_reference.")
    return result


def _number(value, name, row):
    try:
        number = float(value)
    except (ValueError, OverflowError):
        raise SensorValidationError(f"Row {row}: {name} must be a finite number.") from None
    if not math.isfinite(number):
        raise SensorValidationError(f"Row {row}: {name} must be a finite number.")
    return number


def _timestamp(value, row):
    try:
        value = value.strip()
        # datetime accepts arbitrary separators, overflowing offset minutes and
        # silently truncates fractional seconds after six digits. Reject those
        # forms before parsing rather than silently changing observation times.
        if TIMESTAMP_FORMAT.fullmatch(value) is None:
            raise ValueError
        parsed = datetime.fromisoformat(value.upper().replace("Z", "+00:00"))
        if parsed.tzinfo is None or parsed.utcoffset() is None:
            raise ValueError
        return parsed.astimezone(timezone.utc)
    except (ValueError, OverflowError):
        raise SensorValidationError(
            f"Row {row}: timestamp needs an ISO8601 date, time with seconds, timezone and at most six fractional digits."
        ) from None


def inspect_sensor(payload):
    """Validate a bounded CSV and return observations, summaries and provenance.

    CSV header must be ``elapsed_s,value`` or ``timestamp,value``. Elapsed values
    are preserved; timezone-aware timestamps are converted to seconds since the
    first row. Negative observations are valid for channels that support them.
    """
    if not isinstance(payload, dict) or set(payload) != {"csv_text", "metadata"}:
        raise SensorValidationError("Expected csv_text and metadata.")
    csv_text = payload["csv_text"]
    if not isinstance(csv_text, str):
        raise SensorValidationError("csv_text must be text.")
    try:
        csv_bytes = csv_text.encode("utf-8")
    except UnicodeEncodeError:
        raise SensorValidationError("CSV must contain valid UTF-8 text.") from None
    if not csv_bytes or len(csv_bytes) > MAX_CSV_BYTES:
        raise SensorValidationError("CSV must contain 1–262144 UTF-8 bytes.")
    metadata = _metadata(payload["metadata"])
    samples = []
    origin = None
    try:
        reader = csv.reader(io.StringIO(csv_text.removeprefix("\ufeff"), newline=""), strict=True)
        header = next(reader, None)
        if header not in (["elapsed_s", "value"], ["timestamp", "value"]):
            raise SensorValidationError("CSV header must be elapsed_s,value or timestamp,value.")
        time_basis = header[0]
        for row_number, row in enumerate(reader, start=2):
            if len(samples) >= MAX_ROWS:
                raise SensorValidationError("CSV may contain at most 2000 data rows.")
            if len(row) != 2 or any(not cell.strip() for cell in row):
                raise SensorValidationError(f"Row {row_number}: exactly two nonempty cells are required.")
            value = _number(row[1], "value", row_number)
            if time_basis == "timestamp":
                observed = _timestamp(row[0], row_number)
                if origin is None:
                    origin = observed
                seconds = (observed - origin).total_seconds()
            else:
                seconds = _number(row[0], "elapsed_s", row_number)
                if seconds < 0:
                    raise SensorValidationError(f"Row {row_number}: elapsed_s cannot be negative.")
            if samples and seconds <= samples[-1]["elapsed_s"]:
                raise SensorValidationError(f"Row {row_number}: times must be strictly increasing, without duplicates.")
            samples.append({"elapsed_s": seconds, "value": value})
    except csv.Error:
        raise SensorValidationError("CSV contains invalid quoting or an excessively long field.") from None
    if len(samples) < 2:
        raise SensorValidationError("CSV must contain at least two data rows.")

    values = [sample["value"] for sample in samples]
    provenance = {
        "csv_sha256": hashlib.sha256(csv_bytes).hexdigest(),
        "metadata_sha256": _digest(metadata),
        "sample_sha256": _digest(samples),
        "code_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "parser_version": PARSER_VERSION,
    }
    checks = [
        {"name": "Supported CSV structure", "passed": True,
         "detail": f"Two columns and {len(samples)} data rows; no missing cells."},
        {"name": "Finite observations", "passed": True,
         "detail": "Every observation is finite; negative values are retained."},
        {"name": "Strictly increasing times", "passed": True,
         "detail": "No duplicate or reversed observation times."},
        {"name": "Units and source declared", "passed": True,
         "detail": "Required metadata supplied; declarations are not independently verified."},
    ]
    limitations = [
        "Imported CSV only; no live hardware connection or sensor identity verification.",
        "Source kind, units and calibration reference are supplied declarations, not verified measurements or calibration.",
        "These checks establish file structure and numerical consistency, not device accuracy or biological validity.",
        "No conversion to Cube concentration units, prediction-error score, clinical interpretation or efficacy claim is produced.",
    ]
    if metadata["source_kind"] == "synthetic_demo":
        limitations.insert(0, "Synthetic demonstration data; these are not physical sensor measurements.")
    if metadata["calibration_status"] == "unknown":
        limitations.append("Calibration is unknown; physical accuracy has not been established.")
    else:
        limitations.append("A calibration reference was supplied; its contents and applicability have not been verified.")
    return {
        "intake_id": _digest(provenance)[:16],
        "source_kind": metadata["source_kind"],
        "metadata": metadata,
        "time_basis": time_basis,
        "time_origin": origin.isoformat().replace("+00:00", "Z") if origin else None,
        "samples": samples,
        "summary": {"count": len(samples), "min": min(values), "max": max(values),
                    "mean": statistics.mean(values),
                    "duration_s": samples[-1]["elapsed_s"] - samples[0]["elapsed_s"]},
        "checks": checks,
        "provenance": provenance,
        "comparison": {"status": "not_comparable", "reason": COMPARISON_REASON},
        "limitations": limitations,
    }
