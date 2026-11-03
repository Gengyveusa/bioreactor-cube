"""Exercise the actual loopback HTTP handler without contacting a model provider."""

import http.client
from http.server import ThreadingHTTPServer
import json
import threading
import urllib.request

import pytest

from sim.cube.lab_server import Handler
from sim.cube.research_lab import BASELINE, digest


@pytest.fixture(scope="module")
def server():
    instance = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    worker = threading.Thread(
        target=instance.serve_forever, kwargs={"poll_interval": 0.05}, daemon=True
    )
    worker.start()
    try:
        yield instance
    finally:
        instance.shutdown()
        instance.server_close()
        worker.join(timeout=3)
        assert not worker.is_alive()


@pytest.fixture(autouse=True)
def offline_environment(monkeypatch):
    monkeypatch.delenv("MODEL_API_KEY", raising=False)
    monkeypatch.delenv("META_MODEL", raising=False)

    def forbidden_provider_request(*args, **kwargs):
        raise AssertionError("HTTP integration tests must not contact a model provider")

    monkeypatch.setattr(urllib.request.OpenerDirector, "open", forbidden_provider_request)


def request(server, method, path, body=None, headers=None):
    connection = http.client.HTTPConnection("127.0.0.1", server.server_port, timeout=3)
    try:
        connection.request(method, path, body=body, headers=headers or {})
        response = connection.getresponse()
        raw = response.read()
        assert len(raw) == int(response.getheader("Content-Length"))
        return response.status, dict(response.getheaders()), raw
    finally:
        connection.close()


def post(server, value, path="/api/experiment", headers=None):
    return request(
        server, "POST", path, json.dumps(value).encode(),
        {"Content-Type": "application/json", **(headers or {})},
    )


@pytest.mark.parametrize("configured", [False, True])
def test_status_reports_configuration_without_exposing_key(server, monkeypatch, configured):
    secret = "integration-test-secret-never-send-to-provider"
    if configured:
        monkeypatch.setenv("MODEL_API_KEY", secret)
        monkeypatch.setenv("META_MODEL", "test-model")
    status, headers, raw = request(server, "GET", "/api/status")
    assert status == 200
    assert json.loads(raw) == {
        "meta_configured": configured,
        "model": "test-model" if configured else "muse-spark-1.3",
        "baseline": BASELINE,
        "features": ["sensor_csv_v1", "world_v1", "journey_v1"],
    }
    assert secret not in raw.decode()
    assert secret not in str(headers)
    assert headers["Cache-Control"] == "no-store"


def test_experiment_executes_real_transport_and_returns_verifiable_fields(server):
    local_origin = f"http://localhost:{server.server_port}"
    status, _, raw = post(
        server, {"grid": 8, "seconds": 0.5},
        headers={"Origin": local_origin, "Host": f"localhost:{server.server_port}"},
    )
    assert status == 200
    result = json.loads(raw)
    assert result["mode"] == "deterministic"
    assert result["initial"]["shape"] == result["final"]["shape"] == [8, 8, 8]
    assert len(result["final"]["values"]) == 8 ** 3
    assert result["initial"]["values"] != result["final"]["values"]
    assert result["metrics"]["variance_after"] < result["metrics"]["variance_before"]
    assert all(check["passed"] for check in result["checks"])
    assert result["provenance"]["final_sha256"] == digest(result["final"])
    assert result["receipts"][1]["result"]["substeps"] > 0


@pytest.mark.parametrize("method", ["GET", "POST"])
@pytest.mark.parametrize("headers", [
    {"Host": "attacker.example"},
    {"Origin": "https://attacker.example"},
    {"Origin": "null"},
])
def test_foreign_host_or_origin_is_rejected(server, method, headers):
    if method == "GET":
        status, _, raw = request(server, method, "/api/status", headers=headers)
    else:
        status, _, raw = post(server, {}, headers=headers)
    assert status == 403
    assert json.loads(raw) == {"error": "Local origin required"}


@pytest.mark.parametrize("value", [
    [], None, {"grid": True}, {"seconds": -1}, {"unexpected": 1},
    {"seconds": 10 ** 400}, {"diffusion_um2_s": 10 ** 400},
])
def test_invalid_experiment_requests_are_client_errors(server, value):
    status, _, raw = post(server, value)
    assert status == 400
    assert json.loads(raw) == {"error": "Invalid request or experiment parameters"}


def test_malformed_json_is_rejected_and_releases_execution_lock(server):
    status, _, _ = request(
        server, "POST", "/api/experiment", b'{"grid":',
        {"Content-Type": "application/json"},
    )
    assert status == 400
    status, _, _ = post(server, {"grid": 8, "seconds": 0})
    assert status == 200


def test_non_json_media_type_is_rejected(server):
    status, _, raw = request(
        server, "POST", "/api/experiment", b"{}", {"Content-Type": "text/plain"}
    )
    assert status == 415
    assert json.loads(raw) == {"error": "JSON required"}


def test_body_above_limit_is_rejected(server):
    status, _, raw = request(
        server, "POST", "/api/experiment", b"{}" + b" " * 8191,
        {"Content-Type": "application/json"},
    )
    assert status == 413
    assert "8192" in json.loads(raw)["error"]


def test_agent_without_key_returns_safe_unavailable_response(server, monkeypatch):
    opener_attempts = []

    def unexpected_opener(*args, **kwargs):
        opener_attempts.append(True)
        raise AssertionError("Missing credentials must be rejected before creating a client")

    monkeypatch.setattr(urllib.request, "build_opener", unexpected_opener)
    status, _, raw = post(
        server, {"question": "Run a small diffusion experiment."}, path="/api/agent"
    )
    assert status == 503
    assert json.loads(raw) == {
        "error": "Set MODEL_API_KEY on the server to enable live Meta reasoning."
    }
    assert opener_attempts == []


def test_sensor_upload_stays_separate_from_simulation_and_provider(server):
    payload = {"csv_text":"timestamp,value\n2026-10-27T12:00:00Z,1.5\n2026-10-27T12:00:01Z,1.7\n",
               "metadata":{"channel_name":"lactate","unit":"mM","source_kind":"measurement"}}
    status, _, raw = post(server, payload, path="/api/sensor")
    assert status == 200
    result = json.loads(raw)
    assert result["source_kind"] == "measurement"
    assert result["samples"] == [{"elapsed_s":0.0,"value":1.5},{"elapsed_s":1.0,"value":1.7}]
    assert result["comparison"]["status"] == "not_comparable"
    assert "model" not in result and "final" not in result


def test_sensor_invalid_data_returns_safe_actionable_error(server):
    payload = {"csv_text":"elapsed_s,value\n0,1\n0,2\n",
               "metadata":{"channel_name":"test","unit":"mM","source_kind":"measurement"}}
    status, _, raw = post(server, payload, path="/api/sensor")
    assert status == 400
    assert "time" in json.loads(raw)["error"].lower()
    assert "Traceback" not in raw.decode()


def test_sensor_request_limit_does_not_expand_experiment_limit(server):
    payload = {"csv_text":"elapsed_s,value\n"+"".join(f"{i/100},0.123456789\n" for i in range(1000)),
               "metadata":{"channel_name":"demo","unit":"arbitrary units","source_kind":"synthetic_demo"}}
    assert len(json.dumps(payload)) > 8192
    status, _, raw = post(server, payload, path="/api/sensor")
    assert status == 200 and json.loads(raw)["summary"]["count"] == 1000
    status, _, _ = post(server, payload, path="/api/experiment")
    assert status == 413


def test_sensor_import_rejects_foreign_origin(server):
    status, _, _ = post(server, {}, path="/api/sensor",headers={"Origin":"https://attacker.example"})
    assert status == 403


def test_world_page_uses_local_only_document_headers(server, monkeypatch, tmp_path):
    from sim.cube import lab_server
    page = tmp_path / "world.html"
    page.write_text("<!doctype html><title>Cube World</title>")
    monkeypatch.setattr(lab_server, "WORLD_WEB", page)
    status, headers, raw = request(server, "GET", "/world")
    assert status == 200
    assert b"Cube World" in raw
    assert headers["Content-Type"] == "text/html; charset=utf-8"
    assert "frame-ancestors 'none'" in headers["Content-Security-Policy"]


def world_request():
    from sim.cube.research_lab import DEFAULTS
    return {"question": "What is here?", "context": {
        "parameters": {**DEFAULTS, "grid": 8}, "position_um": [100, 100, 100],
        "field": "final", "selection": None,
    }}


def test_world_agent_http_request_delivers_bounded_context_to_agent(server, monkeypatch):
    from sim.cube import meta_agent
    seen = []

    def fake_agent(question, *, world_context):
        seen.append((question, world_context))
        return {"answer": "Mock response, no provider contacted", "world": None}

    monkeypatch.setattr(meta_agent, "run_agent", fake_agent)
    payload = world_request()
    status, _, raw = post(server, payload, path="/api/world/agent")
    assert status == 200
    assert seen == [(payload["question"], payload["context"])]
    assert json.loads(raw)["world"] is None


@pytest.mark.parametrize("payload", [None, {}, {"question": "What?"}, {"question": "What?", "context": None}])
def test_world_http_rejects_missing_context(server, payload):
    status, _, raw = post(server, payload, path="/api/world/agent")
    assert status == 400
    assert json.loads(raw) == {"error": "Invalid request or experiment parameters"}


def test_world_http_validates_before_provider_and_retains_body_origin_guards(server):
    payload = world_request()
    payload["context"]["position_um"] = [float("nan"), 0, 0]
    status, _, raw = post(server, payload, path="/api/world/agent")
    assert status == 400
    assert "NaN" not in raw.decode()
    status, _, _ = post(server, world_request(), path="/api/world/agent",
                        headers={"Origin": "https://attacker.example"})
    assert status == 403
    status, _, _ = request(server, "POST", "/api/world/agent", b" " * 8193,
                          {"Content-Type": "application/json"})
    assert status == 413
    status, _, raw = post(server, world_request(), path="/api/world/agent")
    assert status == 503
    assert "Set MODEL_API_KEY" in json.loads(raw)["error"]
