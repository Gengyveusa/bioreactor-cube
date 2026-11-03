"""Mock-wire adapter tests only: these do not call Meta or verify account access."""

import copy
import io
import json
import urllib.error

import pytest

from sim.cube import meta_agent as agent


def function(name="inspect_run", args=None, call_id="call_1"):
    return {"type": "function_call", "name": name, "arguments": json.dumps(args or {}),
            "call_id": call_id, "status": "completed"}


def final(text="The recorded checks passed."):
    return {"type": "message", "role": "assistant",
            "content": [{"type": "output_text", "text": text}]}


def response(*output):
    return {"status": "completed", "output": list(output)}


class FakeClient:
    def __init__(self, *replies):
        self.replies = iter(replies)
        self.requests = []

    def create_response(self, payload):
        self.requests.append(copy.deepcopy(payload))
        return next(self.replies)


def test_stateless_replay_preserves_all_output_and_keeps_fields_out_of_model(monkeypatch):
    experiment = {"initial": {"shape": [8, 8, 8], "values": [0] * 512},
                  "final": {"shape": [8, 8, 8], "values": [1] * 512},
                  "metrics": {"relative_mass_error": 0}, "provenance": {"source": "synthetic"},
                  "checks": [{"name": "mass", "passed": True}]}
    seen = []
    monkeypatch.setattr(agent, "run_experiment", lambda args: seen.append(args) or experiment)
    reasoning = {"type": "reasoning", "id": "rs_1", "summary": [], "encrypted_content": "opaque"}
    commentary = {**final("Running the numerical experiment."), "phase": "commentary"}
    call = function("run_transport", {"grid": 8, "seconds": 1})
    client = FakeClient(response(reasoning, commentary, call), response(final()))

    result = agent.run_agent("Run a transport experiment", client)

    assert seen == [{"grid": 8, "seconds": 1}]
    assert result["experiment"] is experiment
    assert result["mode"] == "live_meta"
    assert result["receipts"][0]["status"] == "success"
    first, second = client.requests
    assert first["store"] is False
    assert first["include"] == ["reasoning.encrypted_content"]
    assert first["parallel_tool_calls"] is False
    assert first["tool_choice"] == "auto"
    assert second["instructions"] == first["instructions"]
    assert second["input"][1:4] == [reasoning, commentary, call]
    returned = second["input"][4]
    assert returned["type"] == "function_call_output"
    assert returned["call_id"] == "call_1"
    receipt = json.loads(returned["output"])
    assert receipt == result["receipts"][0]
    assert "initial" not in receipt["result"] and "final" not in receipt["result"]
    assert receipt["result"]["provenance"] == experiment["provenance"]


@pytest.mark.parametrize("name,args", [
    ("delete_files", {}), ("inspect_run", {"extra": 1}),
    ("run_transport", {"grid": True}), ("run_transport", {"grid": 33}),
    ("run_transport", {"seconds": float("nan")}),
    ("run_transport", {"diffusion_um2_s": 2001}),
    ("run_transport", {"boundary": "infinite"}), ("run_transport", {"seed": -1}),
    ("run_transport", {"shell": "do something"}),
])
def test_invalid_tools_have_visible_error_receipts_and_do_not_execute(monkeypatch, name, args):
    def forbidden(*_):
        pytest.fail("Invalid arguments must not execute a tool")
    monkeypatch.setattr(agent, "run_experiment", forbidden)
    monkeypatch.setattr(agent, "inspect_baseline", forbidden)
    client = FakeClient(response(function(name, args)), response(final("The request was rejected.")))
    result = agent.run_agent("Try a tool", client)
    assert result["receipts"][0]["status"] == "error"
    assert "error" in result["receipts"][0]["result"]
    assert result["experiment"] is None
    assert result["answer"].startswith("No Cube tools were executed successfully")


def test_invalid_json_arguments_are_returned_as_tool_error(monkeypatch):
    call = function("run_transport")
    call["arguments"] = "{broken json"
    client = FakeClient(response(call), response(final("Unable to run.")))
    result = agent.run_agent("Run", client)
    assert result["receipts"][0]["result"]["error"] == "Tool arguments are not valid JSON."


def test_oversized_integer_is_validation_error_without_executing(monkeypatch):
    def forbidden(*_):
        pytest.fail("Out-of-range integers must not execute an experiment")
    monkeypatch.setattr(agent, "run_experiment", forbidden)
    client = FakeClient(response(function("run_transport", {"seconds": 10**400})),
                        response(final("The requested duration exceeds the lab limit.")))
    result = agent.run_agent("Run", client)
    assert result["receipts"][0]["result"]["error"] == "seconds must be a finite number from 0 to 10."
    assert result["experiment"] is None


def test_three_experiment_cap_keeps_last_successful_experiment(monkeypatch):
    executions = []
    def run(args):
        executions.append(args)
        return {"run_id": len(executions), "initial": {}, "final": {}}
    monkeypatch.setattr(agent, "run_experiment", run)
    client = FakeClient(response(*(function("run_transport", call_id=f"c{i}") for i in range(4))),
                        response(final("Three runs completed; the fourth was rejected.")))
    result = agent.run_agent("Run four experiments", client)
    assert len(executions) == 3
    assert result["experiment"]["run_id"] == 3
    assert result["receipts"][-1]["status"] == "error"


def test_tool_cap_limits_execution_even_if_model_ignores_parallel_false(monkeypatch):
    executions = []
    monkeypatch.setattr(agent, "inspect_baseline", lambda: executions.append(1) or {"baseline": "test"})
    client = FakeClient(response(*(function(call_id=f"c{i}") for i in range(10))))
    result = agent.run_agent("Inspect", client)
    assert len(executions) == 8
    assert len(result["receipts"]) == 8
    assert "tool-call limit" in result["answer"]
    assert len(client.requests) == 1


def test_five_request_cap_preserves_receipts(monkeypatch):
    monkeypatch.setattr(agent, "inspect_baseline", lambda: {"baseline": "test"})
    client = FakeClient(*(response(function(call_id=f"c{i}")) for i in range(5)))
    result = agent.run_agent("Inspect", client)
    assert len(client.requests) == 5
    assert len(result["receipts"]) == 5
    assert "five-request" in result["answer"]


def test_duplicate_call_id_never_executes_twice(monkeypatch):
    executions = []
    monkeypatch.setattr(agent, "inspect_baseline", lambda: executions.append(1) or {})
    client = FakeClient(response(function(), function()))
    with pytest.raises(agent.MetaAPIError, match="repeated"):
        agent.run_agent("Inspect", client)
    assert len(executions) == 1


@pytest.mark.parametrize("reply", [
    {"status": "failed", "error": {"message": "untrusted provider details"}},
    {"status": "incomplete", "output": [final("Partial") ]},
    response(), response(final("")), response({**final("Only commentary"), "phase": "commentary"}),
])
def test_failed_incomplete_and_empty_results_are_explicit_errors(reply):
    with pytest.raises(agent.MetaAPIError):
        agent.run_agent("Inspect", FakeClient(reply))


def test_plain_answer_does_not_fabricate_execution_receipts():
    result = agent.run_agent("What can you do?", FakeClient(response(final("I can inspect or simulate."))))
    assert result["receipts"] == []
    assert result["experiment"] is None
    assert result["answer"].startswith("No Cube tools were executed successfully")


def test_missing_key_has_no_fallback_or_fake_live_result(monkeypatch):
    monkeypatch.delenv("MODEL_API_KEY", raising=False)
    with pytest.raises(agent.MetaAPIError, match="Set MODEL_API_KEY"):
        agent.run_agent("Run an experiment")


def test_http_wire_uses_fixed_endpoint_server_key_and_json(monkeypatch):
    monkeypatch.setenv("MODEL_API_KEY", "mock-secret-not-a-real-key")
    seen = []
    class Opener:
        def open(self, request, timeout):
            seen.append((request, timeout))
            return io.BytesIO(json.dumps(response(final())).encode())
    monkeypatch.setattr(agent.urllib.request, "build_opener", lambda *_: Opener())
    client = agent.MetaClient(timeout=7)
    payload = {"model": agent.DEFAULT_MODEL, "input": "test"}
    client.create_response(payload)
    request, timeout = seen[0]
    assert request.full_url == agent.ENDPOINT
    assert request.get_method() == "POST"
    assert request.get_header("Authorization") == "Bearer mock-secret-not-a-real-key"
    assert json.loads(request.data) == payload
    assert timeout == 7


@pytest.mark.parametrize("status", [301, 401, 403, 429, 500, 504])
def test_provider_error_body_and_key_are_not_exposed(monkeypatch, status):
    secret = "mock-secret-not-a-real-key"
    monkeypatch.setenv("MODEL_API_KEY", secret)
    class Opener:
        def open(self, *_args, **_kwargs):
            raise urllib.error.HTTPError(agent.ENDPOINT, status, secret, {}, io.BytesIO(secret.encode()))
    monkeypatch.setattr(agent.urllib.request, "build_opener", lambda *_: Opener())
    with pytest.raises(agent.MetaAPIError) as caught:
        agent.MetaClient().create_response({"input": "test"})
    assert secret not in str(caught.value)
    assert str(status) in str(caught.value)


def test_redirect_handler_refuses_to_forward_authorization():
    handler = agent._NoRedirect()
    assert handler.redirect_request(None, None, 302, "", {}, "https://other.example") is None
