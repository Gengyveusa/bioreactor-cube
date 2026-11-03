"""Evidence grounding and bounded scene configuration; no live provider calls."""
import copy
import json

import pytest

from sim.cube import meta_agent
from sim.cube.research_lab import DEFAULTS, run_experiment
from sim.cube.world_context import (
    describe_world, prepare_world_context, validate_world_config, validate_world_context,
)


def viewer(**updates):
    return {"parameters": {**DEFAULTS, "grid": 8}, "position_um": [0, 100, 200],
            "field": "final", "selection": None, **updates}


class FakeClient:
    def __init__(self, *replies):
        self.replies = iter(replies)
        self.requests = []

    def create_response(self, payload):
        self.requests.append(copy.deepcopy(payload))
        return {"status": "completed", "output": next(self.replies)}


def final():
    return [{"type": "message", "role": "assistant", "content": [
        {"type": "output_text", "text": "This is a synthetic concentration field."}]}]


def call(name, args, call_id="world_1"):
    return {"type": "function_call", "status": "completed", "call_id": call_id,
            "name": name, "arguments": json.dumps(args)}


def test_context_reconstructs_the_real_field_and_keeps_position_distinct_from_voxel_center():
    context, experiment, evidence = prepare_world_context(viewer())
    independently_run = run_experiment(context["parameters"])
    assert experiment == independently_run
    assert evidence["position_um"] == [0, 100, 200]
    assert evidence["domain_um"] == [200, 200, 200]
    assert evidence["grid_spacing_um"] == 25
    observation = evidence["observation"]
    assert observation["indices"] == [0, 4, 7]
    assert observation["cell_center_um"] == [12.5, 112.5, 187.5]
    assert observation["value"] == independently_run["final"]["values"][(0 * 8 + 4) * 8 + 7]
    assert observation["units"] == "arbitrary simulation units"
    assert evidence["field_time_s"] == 1
    assert evidence["experiment"]["run_id"] == experiment["run_id"]
    assert all(item["passed"] for item in evidence["experiment"]["checks"])
    assert "values" not in json.dumps(evidence)


def test_selected_initial_voxel_uses_zero_time_and_full_resolution_index():
    _, experiment, evidence = prepare_world_context(viewer(field="initial", selection=[7, 1, 3]))
    assert evidence["observation"]["indices"] == [7, 1, 3]
    assert evidence["observation"]["basis"] == "selected grid cell"
    assert evidence["observation"]["value"] == experiment["initial"]["values"][(7 * 8 + 1) * 8 + 3]
    assert evidence["field_time_s"] == 0


@pytest.mark.parametrize("position", [
    None, [1, 2], [1, 2, 3, 4], [True, 2, 3], [0, -1, 0], [200.01, 0, 0],
    [float("nan"), 0, 0], [float("inf"), 0, 0], [10**400, 0, 0], ["1", 0, 0],
])
def test_invalid_coordinates_are_rejected_before_computation(position):
    with pytest.raises(ValueError):
        validate_world_context(viewer(position_um=position))


@pytest.mark.parametrize("selection", [
    [], [1, 2], [1, 2, 3, 4], [True, 1, 2], [1.0, 1, 2], [8, 1, 2], [-1, 1, 2],
])
def test_invalid_selection_is_rejected(selection):
    with pytest.raises(ValueError):
        validate_world_context(viewer(selection=selection))


@pytest.mark.parametrize("value", [
    None, [], {}, {**viewer(), "metrics": {"relative_mass_error": 0}},
    viewer(parameters={"grid": 8}), viewer(field="sensor"),
    viewer(parameters={**DEFAULTS, "seconds": 10**400}),
    viewer(parameters={**DEFAULTS, "seconds": 10, "grid": 32, "diffusion_um2_s": 2000}),
])
def test_context_rejects_incomplete_forged_or_unbounded_fields(value):
    with pytest.raises(ValueError):
        validate_world_context(value)


def test_resolution_change_clears_stale_selected_indices():
    context = validate_world_context(viewer(parameters={**DEFAULTS, "grid": 16}, selection=[15, 15, 15]))
    experiment = run_experiment({**DEFAULTS, "grid": 8})
    evidence = describe_world(context, experiment)
    assert evidence["selection"] is None
    assert evidence["observation"]["indices"] == [0, 4, 7]
    assert evidence["parameters"]["grid"] == 8


@pytest.mark.parametrize("config", [
    {"html": "<script>"}, {"title": "<b>Cube</b>"}, {"title": "\nTitle"},
    {"title": ""}, {"title": "x" * 81}, {"title": 1}, {"palette": "custom"},
    {"annotations": 1}, {"annotations": "false"}, {"url": "https://example.com"},
])
def test_world_configuration_rejects_nonpresentation_or_malformed_values(config):
    with pytest.raises(ValueError):
        validate_world_config(config)


def test_world_question_gets_server_evidence_without_claiming_a_meta_build():
    client = FakeClient(final())
    result = meta_agent.run_agent("What is here?", client, world_context=viewer())
    assert result["world"] is None
    assert result["receipts"] == []
    assert result["experiment"]["run_id"] == result["world_context"]["experiment"]["run_id"]
    assert result["answer"].startswith("No Meta-requested Cube tools")
    content = client.requests[0]["input"][0]["content"]
    evidence = json.loads(content[1]["text"].split("\n", 1)[1])
    assert evidence == result["world_context"]
    assert "values" not in content[1]["text"]
    assert "create_world" in [tool["name"] for tool in client.requests[0]["tools"]]


def test_create_world_is_available_only_in_world_mode_and_preserves_simulation():
    args = {"title": "Diffusion Observatory", "palette": "violet", "annotations": False}
    world_client = FakeClient([call("create_world", args)], final())
    result = meta_agent.run_agent("Build a world", world_client, world_context=viewer())
    assert result["world"] == args
    assert result["receipts"][0]["status"] == "success"
    assert result["receipts"][0]["result"]["world"] == args
    assert result["experiment"]["parameters"] == viewer()["parameters"]
    plain_client = FakeClient([call("create_world", args)], final())
    plain = meta_agent.run_agent("Build a world", plain_client)
    assert "create_world" not in [tool["name"] for tool in plain_client.requests[0]["tools"]]
    assert plain["receipts"][0]["status"] == "error"
    assert "world" not in plain


def test_context_reconstruction_is_separate_from_three_requested_runs():
    calls = [call("run_transport", {"grid": 8, "seconds": i / 10}, f"run_{i}") for i in range(4)]
    client = FakeClient(calls, final())
    result = meta_agent.run_agent("Compare four runs", client, world_context=viewer())
    assert [receipt["status"] for receipt in result["receipts"]] == ["success"] * 3 + ["error"]
    assert result["experiment"]["parameters"]["seconds"] == 0.2
    assert result["world_context"]["field_time_s"] == 0.2
    assert result["world_context"]["experiment"]["run_id"] == result["experiment"]["run_id"]
    receipt = result["receipts"][2]["result"]
    assert receipt["world_observation"] == result["world_context"]
    assert "initial" not in receipt and "final" not in receipt
    returned_receipts = [json.loads(item["output"]) for item in client.requests[1]["input"] if item.get("type") == "function_call_output"]
    assert returned_receipts == result["receipts"]


def test_resolution_change_permanently_clears_selection_when_grid_returns():
    context = viewer(parameters={**DEFAULTS, "grid": 16}, selection=[15, 15, 15])
    client = FakeClient([
        call("run_transport", {"grid": 8}, "coarse"),
        call("run_transport", {"grid": 16}, "restore"),
    ], final())
    result = meta_agent.run_agent("Change resolution and change back", client, world_context=context)
    observations = [receipt["result"]["world_observation"] for receipt in result["receipts"]]
    assert [item["parameters"]["grid"] for item in observations] == [8, 16]
    assert all(item["selection"] is None for item in observations)
    assert observations[0]["observation"]["indices"] == [0, 4, 7]
    assert observations[1]["observation"]["indices"] == [0, 8, 15]
    assert result["world_context"]["selection"] is None
    assert result["world_context"]["observation"]["basis"] == "grid cell containing the viewer position"
    assert context["selection"] == [15, 15, 15]  # Caller-owned input is unchanged.


def test_invalid_context_is_rejected_without_contacting_meta():
    client = FakeClient()
    with pytest.raises(ValueError):
        meta_agent.run_agent("What is here?", client, world_context=viewer(position_um=[-1, 0, 0]))
    assert client.requests == []
