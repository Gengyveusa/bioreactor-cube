"""Ground a navigable scene in server-recomputed numerical experiment data.

Camera coordinates are supplied by the viewer, but field values, numerical
checks and provenance are recomputed locally. Nothing here represents a real
sensor observation or anatomical structure.
"""
from __future__ import annotations

import math
import unicodedata

from sim.cube.research_lab import DEFAULTS, run_experiment, validate_plan


def validate_world_context(value):
    """Accept only the viewer's bounded navigation and experiment parameters."""
    if not isinstance(value, dict) or set(value) != {
        "parameters", "position_um", "field", "selection"
    }:
        raise ValueError("Expected world parameters, position, field and selection.")
    parameters = value["parameters"]
    if not isinstance(parameters, dict) or set(parameters) != set(DEFAULTS):
        raise ValueError("World context requires all five experiment parameters.")
    parameters = validate_plan(parameters)
    position = value["position_um"]
    if not isinstance(position, list) or len(position) != 3 or any(
        type(c) not in (int, float) or not 0 <= c <= 200 or not math.isfinite(c)
        for c in position
    ):
        raise ValueError("World position must contain three finite coordinates from 0 to 200 micrometers.")
    if value["field"] not in ("initial", "final"):
        raise ValueError("World field must be initial or final.")
    selection = value["selection"]
    if selection is not None and (
        not isinstance(selection, list) or len(selection) != 3 or any(
            type(i) is not int or not 0 <= i < parameters["grid"] for i in selection
        )
    ):
        raise ValueError("World selection must be three grid indices or null.")
    return {
        "parameters": parameters,
        "position_um": [float(c) for c in position],
        "field": value["field"],
        "selection": list(selection) if selection is not None else None,
    }


def describe_world(context, experiment):
    """Return compact evidence for the active field, without voxel arrays.

If a model-requested run changes resolution, the previous index selection is
cleared; observation falls back to the viewer's bounded camera location.
"""
    n = experiment["parameters"]["grid"]
    spacing = 200.0 / n
    selection = context["selection"] if context["parameters"]["grid"] == n else None
    indices = selection or [min(n - 1, int(c / spacing)) for c in context["position_um"]]
    i, j, k = indices
    field_name = context["field"]
    field = experiment[field_name]
    value = field["values"][(i * n + j) * n + k]
    return {
        "source": "server-recomputed synthetic numerical experiment",
        "parameters": experiment["parameters"],
        "position_um": context["position_um"],
        "field": field_name,
        "selection": selection,
        "domain_um": [200.0, 200.0, 200.0],
        "grid_spacing_um": spacing,
        "field_time_s": 0.0 if field_name == "initial" else experiment["parameters"]["seconds"],
        "observation": {
            "basis": "selected grid cell" if selection is not None else "grid cell containing the viewer position",
            "indices": indices,
            "cell_center_um": [(index + 0.5) * spacing for index in indices],
            "value": value,
            "units": "arbitrary simulation units",
            "meaning": "Concentration at a numerical grid cell, not an individual molecule or biological cell.",
        },
        "experiment": {
            key: experiment[key]
            for key in ("run_id", "metrics", "checks", "provenance", "limitations")
        },
        "scope": "Viewer position is user-supplied; numerical evidence is recomputed on the server. No sensor data is included.",
    }


def prepare_world_context(value):
    """One bounded local reconstruction; no model request or tool-call budget."""
    context = validate_world_context(value)
    experiment = run_experiment(context["parameters"])
    return context, experiment, describe_world(context, experiment)


def validate_world_config(value):
    """Allow only presentation choices. Model text never becomes executable code."""
    if not isinstance(value, dict) or set(value) - {"title", "palette", "annotations"}:
        raise ValueError("World configuration accepts only title, palette and annotations.")
    title = value.get("title", "Cube World")
    if not isinstance(title, str) or not 1 <= len(title.strip()) <= 80 or any(
        unicodedata.category(c).startswith("C") or c in "<>" for c in title
    ):
        raise ValueError("World title must be 1–80 plain-text characters without markup or controls.")
    palette = value.get("palette", "teal")
    if palette not in ("teal", "amber", "violet"):
        raise ValueError("World palette must be teal, amber or violet.")
    annotations = value.get("annotations", True)
    if type(annotations) is not bool:
        raise ValueError("World annotations must be true or false.")
    return {"title": title.strip(), "palette": palette, "annotations": annotations}
