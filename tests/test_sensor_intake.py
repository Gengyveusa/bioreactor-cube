import hashlib
import json

import pytest

from sim.cube.sensor_intake import inspect_sensor, SensorValidationError, MAX_CSV_BYTES, MAX_ROWS


def payload(csv_text="elapsed_s,value\n0,1\n1,2\n", **metadata):
    return {"csv_text": csv_text, "metadata": {
        "channel_name": "Bench pH channel", "unit": "pH", "source_kind": "measurement", **metadata}}


def test_import_preserves_observations_and_declares_comparison_limit():
    supplied = payload("elapsed_s,value\n10,-0.5\n11,0\n13,1.25\n")
    result = inspect_sensor(supplied)
    assert result["samples"] == [{"elapsed_s": 10., "value": -.5},
                                 {"elapsed_s": 11., "value": 0.},
                                 {"elapsed_s": 13., "value": 1.25}]
    assert result["summary"] == {"count": 3, "min": -.5, "max": 1.25, "mean": .25, "duration_s": 3.}
    assert result["time_basis"] == "elapsed_s" and result["time_origin"] is None
    assert result["metadata"]["calibration_status"] == "unknown"
    assert result["metadata"]["sensor_id"] == "not supplied"
    assert result["comparison"]["status"] == "not_comparable"
    assert "arbitrary concentration units" in result["comparison"]["reason"]
    assert all(check["passed"] for check in result["checks"])
    assert inspect_sensor(supplied) == result
    json.dumps(result, allow_nan=False)


def test_timezone_offsets_normalize_to_absolute_chronological_order():
    result = inspect_sensor(payload("timestamp,value\n2026-10-27T23:59:59-07:00,6.5\n2026-10-28T07:00:00+00:00,6.6\n2026-10-28T09:00:01+02:00,6.7\n"))
    assert result["time_origin"] == "2026-10-28T06:59:59Z"
    assert [row["elapsed_s"] for row in result["samples"]] == [0., 1., 2.]
    assert result["summary"]["duration_s"] == 2.


def test_original_bytes_and_semantic_metadata_have_separate_hashes():
    a = inspect_sensor(payload())
    b = inspect_sensor(payload("\ufeffelapsed_s,value\r\n0,1\r\n1,2\r\n"))
    c = inspect_sensor(payload(unit="mV"))
    assert a["samples"] == b["samples"]
    assert a["provenance"]["sample_sha256"] == b["provenance"]["sample_sha256"]
    assert a["provenance"]["csv_sha256"] != b["provenance"]["csv_sha256"]
    assert a["provenance"]["metadata_sha256"] != c["provenance"]["metadata_sha256"]
    assert a["intake_id"] != b["intake_id"] != c["intake_id"]
    assert a["provenance"]["csv_sha256"] == hashlib.sha256(payload()["csv_text"].encode()).hexdigest()


def test_synthetic_and_documented_are_declarations_not_validity_claims():
    result = inspect_sensor(payload("elapsed_s,value\n0,7\n1,7\n", source_kind="synthetic_demo",
                                    calibration_status="documented", calibration_reference="Bench notebook 17"))
    assert result["summary"]["min"] == result["summary"]["max"] == result["summary"]["mean"] == 7
    assert result["source_kind"] == "synthetic_demo"
    assert "not physical sensor measurements" in " ".join(result["limitations"])
    assert "have not been verified" in " ".join(result["limitations"])
    assert result["comparison"]["status"] == "not_comparable"


@pytest.mark.parametrize("csv_text", [
    "", "elapsed_s,value\n", "elapsed_s,value\n0,1\n", "time,value\n0,1\n1,2\n",
    "elapsed_s,value,unit\n0,1,mV\n1,2,mV\n", "value,elapsed_s\n1,0\n2,1\n",
    "elapsed_s,value\n0,1\n1,\n", "elapsed_s,value\n0,1\n,2\n",
    "elapsed_s,value\n0,1\n1\n", "elapsed_s,value\n0,1\n1,2,3\n",
    "elapsed_s,value\n0,1\n\n1,2\n", "elapsed_s,value\n0,1\n0,2\n",
    "elapsed_s,value\n1,1\n0,2\n", "elapsed_s,value\n-1,1\n0,2\n",
    "elapsed_s,value\n0,NaN\n1,2\n", "elapsed_s,value\n0,inf\n1,2\n",
    "elapsed_s,value\n0,-Infinity\n1,2\n", "elapsed_s,value\n0,1e99999\n1,2\n",
    "elapsed_s,value\n0," + "9" * 400 + "\n1,2\n",
    "elapsed_s,value\n0,1\nNaN,2\n", "elapsed_s,value\n0,1\n1e99999,2\n",
    "elapsed_s,value\n0,=SUM(1)\n1,2\n", 'elapsed_s,value\n0,1\n1,"2\n',
    "timestamp,value\n2026-10-27T00:00:00,1\n2026-10-27T00:00:01,2\n",
    "timestamp,value\n2026-10-27T00:00:00Z,1\n2026-10-26T00:00:01Z,2\n",
    "timestamp,value\n2026-10-27T00:00:00Z,1\n2026-10-27T01:00:00+01:00,2\n",
    "timestamp,value\n2026-02-30T00:00:00Z,1\n2026-03-01T00:00:00Z,2\n",
    "timestamp,value\n0001-01-01T00:00:00+01:00,1\n0001-01-01T00:00:01+01:00,2\n",
    "timestamp,value\n2026-10-27T00:00:00+01:60,1\n2026-10-27T00:00:01+01:60,2\n",
    "timestamp,value\n2026-10-27X00:00:00Z,1\n2026-10-27X00:00:01Z,2\n",
    "timestamp,value\n2026-10-27T00:00:00.1234567Z,1\n2026-10-27T00:00:01Z,2\n",
])
def test_rejects_invalid_csv_without_repairing_or_silently_sorting(csv_text):
    with pytest.raises(ValueError):
        inspect_sensor(payload(csv_text))


@pytest.mark.parametrize("key,value", [
    ("source_kind", "observed"), ("source_kind", None), ("channel_name", ""), ("unit", "   "),
    ("channel_name", "x" * 129), ("unit", "x" * 129), ("unit", "mV\x00"),
    ("sensor_id", 123), ("sample_id", [1]), ("calibration_status", "verified"),
    ("calibration_status", "documented"), ("calibration_reference", "x" * 257),
    ("extra", "unexpected"),
])
def test_rejects_invalid_metadata(key, value):
    with pytest.raises(ValueError):
        inspect_sensor(payload(**{key: value}))


@pytest.mark.parametrize("key", ["channel_name", "unit", "source_kind"])
def test_required_declarations_are_not_invented(key):
    supplied = payload()
    del supplied["metadata"][key]
    with pytest.raises(ValueError):
        inspect_sensor(supplied)


@pytest.mark.parametrize("supplied", [None, [], {}, {"csv_text": ""},
    {"csv_text": None, "metadata": {}}, {"csv_text": "x", "metadata": []},
    {"csv_text": "x", "metadata": {}, "extra": 1}])
def test_payload_shape_is_bounded(supplied):
    with pytest.raises(ValueError):
        inspect_sensor(supplied)


def test_byte_and_row_budgets():
    with pytest.raises(ValueError, match="UTF-8 bytes"):
        inspect_sensor(payload("µ" * (MAX_CSV_BYTES // 2 + 1)))
    valid = "elapsed_s,value\n" + "".join(f"{i},1\n" for i in range(MAX_ROWS))
    assert inspect_sensor(payload(valid))["summary"]["count"] == MAX_ROWS
    with pytest.raises(ValueError, match="2000"):
        inspect_sensor(payload(valid + f"{MAX_ROWS},1\n"))


def test_extreme_finite_observations_have_finite_mean():
    result = inspect_sensor(payload("elapsed_s,value\n0,1.7976931348623157e308\n1,1.7976931348623157e308\n"))
    assert result["summary"]["mean"] == result["summary"]["max"]
    json.dumps(result, allow_nan=False)


def test_optional_blank_identifiers_do_not_imply_identity():
    result = inspect_sensor(payload(sensor_id=" ", sample_id="", calibration_reference=""))
    assert result["metadata"]["sensor_id"] == result["metadata"]["sample_id"] == "not supplied"


def test_validation_error_is_safe_to_display_without_raw_observations():
    with pytest.raises(SensorValidationError) as error:
        inspect_sensor(payload("elapsed_s,value\n0,private-unparseable-observation\n1,2\n"))
    assert "Row 2" in str(error.value)
    assert "private-unparseable-observation" not in str(error.value)
