"""Source preparation contract, independent of candidate-program verification."""

import collections, math, hashlib, json
from pathlib import Path
import numpy as np


class SourcePreparationError(ValueError):
    pass


def need(condition, message):
    if not condition:
        raise SourcePreparationError(message)


def finite(value, label, nonnegative=False):
    need(
        isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value),
        f"{label}: not a finite number",
    )
    need(not nonnegative or value >= 0, f"{label}: negative count/time")


def road_schema(roadnet):
    roads = {r["id"]: r for r in roadnet["roads"]}
    result = {}

    def direction(link):
        points = roads[link["startRoad"]]["points"]
        a, b = points[:2]
        angle = math.atan2(b["y"] - a["y"], b["x"] - a["x"]) % (2 * math.pi)
        cardinal = (
            "E"
            if angle < math.pi / 4 or angle >= 7 * math.pi / 4
            else "N" if angle < 3 * math.pi / 4 else "W" if angle < 5 * math.pi / 4 else "S"
        )
        return cardinal + {"go_straight": "T", "turn_left": "L", "turn_right": "R"}.get(
            link.get("type"), "T"
        )

    controlled = [
        i
        for i in roadnet["intersections"]
        if not i["virtual"] and len(i["trafficLight"]["lightphases"]) > 1
    ]
    for node, inter in enumerate(controlled):
        phases = [p for p in inter["trafficLight"]["lightphases"] if p["availableRoadLinks"]]
        need(bool(phases), f"Roadnet node{node} has no physical service phase")
        result[node] = {
            phase: sorted(
                {direction(inter["roadLinks"][index]) for index in p["availableRoadLinks"]}
            )
            for phase, p in enumerate(phases)
        }
    need(bool(result), "Roadnet contains no controlled nodes")
    return result


def moments(values):
    array = np.asarray(values, dtype=float)
    need(array.size > 0, "No measurements for required statistic")
    need(bool(np.isfinite(array).all()), "Non-finite measurement pool")
    stats = dict(
        min=float(array.min()),
        max=float(array.max()),
        mean=float(array.mean()),
        std=float(array.std()),
        count=int(array.size),
    )
    stats.update(
        zip(
            ["p50", "p75", "p90", "p95", "p99"],
            map(float, np.percentile(array, [50, 75, 90, 95, 99], method="linear")),
        )
    )
    stats["zero_fraction"] = float(np.mean(array == 0))
    return stats


def check_rows(header, rows, schema, required_keys, mode, horizon, action_interval=20):
    """Validate EVERY serialized row against authoritative topology, not observed IDs."""
    need(mode in ["phase_argmax", "binary_keep_switch"], "Unknown action mode")
    expected_nodes = set(schema)
    need(header["config"]["num_step"] == horizon, "Source horizon mismatch")
    intersections = header["config"]["intersections"]
    need(isinstance(intersections, list) and bool(intersections), "Missing declared intersections")
    need(
        all(
            type(x.get("id")) is int and type(x.get("num_phases")) is int and x["num_phases"] > 0
            for x in intersections
        ),
        "Declared node and phase counts must be integers, not booleans",
    )
    declared = {x["id"]: x["num_phases"] for x in intersections}
    need(len(declared) == len(intersections), "Duplicate declared source node")
    need(set(declared) == expected_nodes, "Header node coverage differs from roadnet")
    need(
        all(declared[n] == len(schema[n]) for n in schema),
        "Header physical-phase count differs from roadnet",
    )
    need(
        bool(header.get("schema", {}).get("binary_interface")) == (mode == "binary_keep_switch"),
        "Source action interface mismatch",
    )
    need(
        type(horizon) is int
        and type(action_interval) is int
        and action_interval > 0
        and horizon > 0
        and horizon % action_interval == 0,
        "Invalid source decision horizon/interval",
    )
    need(bool(required_keys), "No required observed traffic feature")
    seen = collections.defaultdict(set)
    values = collections.defaultdict(list)
    binary_values = collections.defaultdict(list)
    count = phase_rows = 0
    for row in rows:
        node = row.get("intersection_id")
        step = row.get("timestep")
        need(type(node) is int and node in schema, "Unknown source node")
        need(
            type(step) is int and 0 <= step < horizon // action_interval, "Invalid source timestep"
        )
        need(step not in seen[node], f"Duplicate source row node{node} step{step}")
        seen[node].add(step)
        phases = row.get("phases")
        need(isinstance(phases, list) and bool(phases), "Missing source phase list")
        ids = [p.get("phase_id") for p in phases]
        need(
            all(type(pid) is int for pid in ids)
            and len(set(ids)) == len(ids)
            and set(ids) == set(schema[node]),
            f"Physical phase coverage mismatch node{node} step{step}",
        )
        current = row.get("current_phase")
        action = row.get("teacher_action")
        need(type(current) is int and current in schema[node], "Invalid current physical phase")
        need(type(action) is int and action in schema[node], "Invalid teacher physical action")
        need(
            [p["phase_id"] for p in phases if p.get("selected") is True] == [action],
            "Selected teacher label mismatch",
        )
        repeated = {}
        binary_reference = None
        for phase in phases:
            pid = phase["phase_id"]
            state = phase.get("state")
            need(isinstance(state, dict), "Missing phase state")
            need(
                type(state.get("current_phase")) is int and state["current_phase"] == current,
                "Invalid/inconsistent current phase across source bundle",
            )
            directions = state.get("controlled_directions")
            fbd = state.get("features_by_direction")
            need(
                isinstance(directions, list)
                and len(set(directions)) == len(directions)
                and set(directions) == set(schema[node][pid]),
                f"Controlled directions differ from roadnet node{node} phase{pid}",
            )
            need(
                isinstance(fbd, dict) and set(fbd) == set(directions),
                "Directional feature coverage mismatch",
            )
            for direction, features in fbd.items():
                need(
                    isinstance(features, dict) and set(required_keys) <= set(features),
                    f"Missing required direction feature node{node} phase{pid} {direction}",
                )
                for key, value in features.items():
                    finite(value, key, nonnegative=key != "pressure")
                if "waiting_time" in features and "sum_waiting_time" in features:
                    need(
                        features["waiting_time"] == features["sum_waiting_time"],
                        "Waiting-time alias disagreement",
                    )
                if direction in repeated:
                    need(
                        features == repeated[direction],
                        "Same movement differs across source phases",
                    )
                repeated[direction] = features
                for key in required_keys:
                    values[key].append(float(features[key]))
            if mode == "binary_keep_switch":
                binary = state.get("binary_features")
                need(isinstance(binary, dict), "Missing native binary state")
                for key in ["green_vehicle", "red_vehicle"]:
                    finite(binary.get(key), key, True)
                if binary_reference is None:
                    binary_reference = binary
                else:
                    need(binary == binary_reference, "Binary inputs differ across physical phases")
            for key in ["Q_value", "logit"]:
                if phase.get(key) is not None:
                    finite(phase[key], key)
            phase_rows += 1
        if mode == "binary_keep_switch":
            label = row.get("teacher_binary_action")
            need(type(label) is int and label in [0, 1], "Missing/invalid native action label")
            ids = sorted(schema[node])
            following = ids[(ids.index(current) + 1) % len(ids)]
            need(
                action == (current if label == 0 else following),
                "Native action label does not match cyclic successor",
            )
            for key in ["green_vehicle", "red_vehicle"]:
                binary_values[key].append(float(binary_reference[key]))
        count += 1
    need(set(seen) == expected_nodes, "Missing source nodes")
    target = set(range(horizon // action_interval))
    need(all(seen[node] == target for node in schema), "Missing source timesteps")
    need(count == sum(len(v) for v in seen.values()), "Source row count disagreement")
    # The aligned collector defines total_samples as global decision frames,
    # not the number of serialized node/frame rows (collect_teacher.py:83).
    if "total_samples" in header["stats"]:
        need(
            header["stats"]["total_samples"] == horizon // action_interval,
            "Header global decision-frame count mismatch",
        )
    finite(header["stats"].get("avg_travel_time"), "Teacher ATT", True)
    need(header["stats"]["avg_travel_time"] > 0, "Teacher ATT must be positive")
    return dict(
        status="passed",
        source_nodes=len(seen),
        source_bundles=count,
        physical_phase_states=phase_rows,
        decision_frames_per_node=horizon // action_interval,
        required_direction_keys=required_keys,
        direction_statistics={k: moments(v) for k, v in values.items()},
        binary_statistics={k: moments(v) for k, v in binary_values.items()},
    )


def check_statistics(actual, expected):
    need(set(actual) == set(expected), "Feature-statistic name coverage mismatch")
    for name, reference in expected.items():
        row = actual[name]
        need(set(row) == set(reference), f"Statistic field coverage mismatch:{name}")
        for key, value in reference.items():
            finite(row[key], f"{name}.{key}")
            need(
                math.isclose(row[key], value, rel_tol=1e-10, abs_tol=1e-10),
                f"Statistic mismatch:{name}.{key}: stored={row[key]}, recomputed={value}",
            )
    return dict(
        status="passed",
        feature_names=sorted(actual),
        scalar_fields=sum(len(v) for v in actual.values()),
    )


def prepare_source(
    loader, teacher_model, observation_features, config, action_mode, *, project_root=None
):
    """Check authoritative topology and every source row before generating candidates.

    Missing source files/metadata are preparation errors, never candidate failures.
    The returned measurements also independently check the existing stats builder.
    """
    try:
        meta = loader.get_meta()
        collector = loader.get_collector_config()
        need(
            meta.get("model", "").casefold() == teacher_model.casefold(),
            "Source teacher metadata mismatch",
        )
        dataset = meta.get("dataset")
        flow = meta.get("flow_idx")
        need(
            isinstance(dataset, str) and bool(dataset) and type(flow) is int and flow >= 0,
            "Missing source dataset/flow metadata",
        )
        root = (
            Path(project_root) if project_root is not None else Path(__file__).resolve().parents[3]
        )
        road = root / "data" / dataset / "roadnet.json"
        flow_file = road.parent / f"flow_{flow}.json"
        need(road.is_file() and flow_file.is_file(), "Source roadnet/flow file missing")
        for key, path in [("roadnet_sha256", road), ("flow_sha256", flow_file)]:
            if key in meta:
                need(
                    hashlib.sha256(path.read_bytes()).hexdigest() == meta[key],
                    f"Source {key} mismatch",
                )
        # Legacy phase rollouts may omit schema; aligned binary files must declare it.
        header = loader.get_header()
        mode = "binary_keep_switch" if action_mode == "binary_keep_switch" else "phase_argmax"
        required = []
        for feature in observation_features:
            for key in ["pressure", "num_waiting_vehicle", "num_vehicle", "waiting_time"]:
                if key in feature.lower():
                    if key not in required:
                        required.append(key)
                    break
        audit = check_rows(
            header,
            loader.timesteps,
            road_schema(json.loads(road.read_text())),
            required,
            mode,
            collector.get("num_step"),
            config.get("action_interval", 20),
        )
        audit.update(
            teacher_model=teacher_model,
            dataset=dataset,
            flow=flow,
            action_mode=mode,
            trajectory_sha256=hashlib.sha256(Path(loader.json_file_path).read_bytes()).hexdigest(),
        )
        return audit
    except SourcePreparationError:
        raise
    except (OSError, KeyError, TypeError, ValueError, StopIteration) as exc:
        raise SourcePreparationError(f"Source preparation failed: {exc}") from exc


def check_prepared_statistics(audit, extractor, actual):
    """Compare source-derived fields without enabling optional percentile fields."""
    need(audit.get("status") == "passed", "Source data must pass preparation first")
    kinds = {"current_phase": dict(min=0.0, max=1.0, mean=0.5, std=0.5)}
    kinds.update(
        {
            {
                "pressure": "pressure_sum",
                "num_vehicle": "num_vehicle_sum",
                "num_waiting_vehicle": "num_waiting_vehicle_sum",
                "waiting_time": "waiting_time_sum",
            }[k]: v
            for k, v in audit["direction_statistics"].items()
        }
    )
    expected = {spec.name: dict(kinds[spec.kind]) for spec in extractor.specs if spec.kind in kinds}
    if audit["action_mode"] == "binary_keep_switch":
        expected.update(audit["binary_statistics"])
    optional = {"p50", "p75", "p90", "p95", "p99", "zero_fraction"}
    need(
        set(actual) == set(expected),
        "Prepared feature-statistic names differ from source measurements",
    )
    for name, row in expected.items():
        fields = set(actual[name])
        need(set(row) - optional <= fields <= set(row), f"Missing/unknown statistic fields: {name}")
        expected[name] = {k: v for k, v in row.items() if k in fields}
    return check_statistics(actual, expected)
