#!/usr/bin/env python3
"""Single public entry point for the submission artifact."""

from __future__ import annotations

import argparse
from contextlib import contextmanager
from functools import lru_cache
import hashlib
import importlib.metadata
import json
import math
import os
from pathlib import Path
import platform
import shlex
import sys
import tempfile
from datetime import datetime, timezone
from typing import Any


from ppl_tsc.paths import PROJECT_ROOT as ROOT

SRC = ROOT / "src"
DEFAULT_API_PROFILE = ROOT / "configs/api.local.json"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))


def _read_json_value(path: Path) -> Any:
    with path.open(encoding="utf-8") as f:
        return json.load(f)


def _read_json(path: Path) -> dict[str, Any]:
    data = _read_json_value(path)
    if not isinstance(data, dict):
        raise ValueError(f"Configuration must be a JSON object: {path}")
    return data


def _section(config: dict[str, Any], name: str) -> dict[str, Any]:
    value = config.get(name)
    if not isinstance(value, dict):
        raise ValueError(f"Missing object section {name!r} in experiment config")
    return value


def _required(section: dict[str, Any], key: str, section_name: str) -> Any:
    value = section.get(key)
    if value is None or value == "":
        raise ValueError(f"Missing required setting {section_name}.{key}")
    return value


def _root_path(value: str) -> Path:
    path = Path(value).expanduser()
    return path.resolve() if path.is_absolute() else (ROOT / path).resolve()


def _portable_path(value: Any) -> Any:
    """Render paths inside the artifact relative to its root in saved metadata."""
    if value is None:
        return None
    path = Path(str(value))
    if path.is_absolute() and path.is_relative_to(ROOT):
        return path.relative_to(ROOT).as_posix()
    return value


def _load_local_env(path: Path = ROOT / ".env") -> None:
    """Load local LLM_* variables without overriding the process environment."""
    if not path.is_file():
        return
    for line_number, raw_line in enumerate(
        path.read_text(encoding="utf-8").splitlines(),
        start=1,
    ):
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        if line.startswith("export "):
            line = line[len("export ") :].strip()
        key, separator, raw_value = line.partition("=")
        key = key.strip()
        if not separator or not key.startswith("LLM_"):
            continue
        try:
            values = shlex.split(raw_value, comments=True, posix=True)
        except ValueError as exc:
            raise ValueError(f"Invalid .env value at line {line_number}: {path}") from exc
        value = values[0] if values else ""
        os.environ.setdefault(key, value)


def _apply_api_profile(
    config: dict[str, Any],
    profile_path: Path | None,
) -> None:
    if profile_path is None:
        return
    profile = _read_json(profile_path)
    allowed = {
        "provider",
        "endpoint",
        "request_path",
        "model",
        "api_key_env",
        "temperature",
        "auth_mode",
    }
    unknown = sorted(set(profile) - allowed)
    if unknown:
        raise ValueError(f"Unsupported API profile settings in {profile_path}: {unknown}")
    if any("key" in key.casefold() and key != "api_key_env" for key in profile):
        raise ValueError(
            f"API profiles must not contain credentials; put the key in {ROOT / '.env'}"
        )
    _section(config, "llm").update(profile)


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _sha256_source_tree() -> str:
    digest = hashlib.sha256()
    paths = list(SRC.rglob("*.py")) + [
        ROOT / name for name in ("run.py", "evaluate.py", "train_teacher.py", "collect_teacher.py")
    ]
    # Include the bundled simulator: local fixes affect execution provenance.
    paths += [
        path
        for path in (ROOT / "CityFlow").rglob("*")
        if path.suffix in {".cpp", ".h", ".hpp", ".py"} or path.name == "CMakeLists.txt"
    ]
    for path in sorted(
        path
        for path in paths
        if path.is_file()
        and not any(
            part in {"build", ".git", "__pycache__"} or part.endswith(".egg-info")
            for part in path.relative_to(ROOT).parts
        )
    ):
        digest.update(path.relative_to(ROOT).as_posix().encode("utf-8"))
        digest.update(path.read_bytes())
    return digest.hexdigest()


def _package_versions() -> dict[str, str]:
    versions: dict[str, str] = {}
    for package in (
        "numpy",
        "torch",
        "gymnasium",
        "scipy",
        "cma",
        "cityflow",
    ):
        try:
            versions[package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            versions[package] = "not-installed"
    return versions


def _teacher_rollout_entries(
    manifest_path: Path,
) -> tuple[str, list[dict[str, Any]]]:
    """Return the manifest mode and every rollout entry it references."""
    manifest = _read_json_value(manifest_path)
    if (
        not isinstance(manifest, list)
        or not manifest
        or any(not isinstance(item, dict) or "roadnet_configs" in item for item in manifest)
    ):
        raise ValueError(
            "Teacher manifests must be a non-empty list of per-source runs; "
            "joint road-network search is not part of this release"
        )
    return "sequential", manifest


@lru_cache(maxsize=32)
def _rollout_preview_cached(path: str, content_sha256: str) -> tuple[dict, dict | None]:
    """Keep only a header/sample; release the full trajectory after parsing."""
    from ppl_tsc.data.loader import TeacherDataLoader

    loader = TeacherDataLoader(path, verbose=False)
    return loader.get_header(), loader.timesteps[0] if loader.timesteps else None


def _rollout_preview(path: Path) -> tuple[dict, dict | None]:
    # Use content hashes: filesystem timestamps can miss rapid same-size writes.
    # Retain only a bounded header/sample cache, not complete trajectories.
    path = path.resolve()
    return _rollout_preview_cached(str(path), _sha256_file(path))


def _read_rollout_meta(path: Path) -> dict[str, Any]:
    header, _ = _rollout_preview(path)
    if not header["meta"]:
        raise ValueError(f"Teacher rollout has no meta object: {path}")
    return dict(header["meta"])


def _validate_rollout_entry(
    entry: dict[str, Any],
    *,
    label: str,
) -> tuple[Path, str, str, int]:
    path_value = entry.get("teacher_data_path") or entry.get("teacher_data")
    teacher_model = str(entry.get("teacher_model", "")).strip()
    if not path_value or not teacher_model:
        raise ValueError(f"{label} must specify teacher_data_path and teacher_model")

    rollout_path = _root_path(str(path_value))
    if not rollout_path.is_file():
        raise FileNotFoundError(f"Teacher rollout not found: {rollout_path}")

    meta = _read_rollout_meta(rollout_path)
    dataset = str(meta.get("dataset", "")).strip()
    meta_model = str(meta.get("model", "")).strip()
    flow_idx = meta.get("flow_idx")
    if meta_model and meta_model.casefold() != teacher_model.casefold():
        raise ValueError(
            f"{label} names teacher {teacher_model!r}, but rollout metadata names "
            f"{meta_model!r}: {rollout_path}"
        )
    if not dataset or not isinstance(flow_idx, int):
        raise ValueError(
            f"Teacher rollout metadata must contain dataset and integer flow_idx: {rollout_path}"
        )

    for simulator_input in (
        ROOT / "data" / dataset / "roadnet.json",
        ROOT / "data" / dataset / f"flow_{flow_idx}.json",
    ):
        if not simulator_input.is_file():
            raise FileNotFoundError(
                f"CityFlow input required by {rollout_path} was not found: {simulator_input}"
            )
    return rollout_path, teacher_model, dataset, flow_idx


def _validate_ecolight_action_interface(path: Path, action_mode: str) -> None:
    """Reject a binary EcoLight rollout paired with the wrong student head."""
    if action_mode == "auto":
        action_mode = "keep_switch"
    if action_mode not in {"keep_switch", "binary_keep_switch", "phase_argmax"}:
        raise ValueError(f"Unsupported EcoLight action mode {action_mode!r}")
    header, sample = _rollout_preview(path)
    if not isinstance(sample, dict) or not sample.get("phases"):
        raise ValueError(f"EcoLight rollout has no phase observations: {path}")
    state = sample["phases"][0].get("state", {})
    binary = state.get("binary_features") if isinstance(state, dict) else None
    aligned = (
        "teacher_binary_action" in sample
        or isinstance(header.get("meta", {}).get("binary_supervision"), dict)
        or isinstance(binary, dict)
    )
    if action_mode == "binary_keep_switch":
        if (
            "teacher_binary_action" not in sample
            or not isinstance(binary, dict)
            or not {"green_vehicle", "red_vehicle"} <= set(binary)
        ):
            raise ValueError(
                f"binary_keep_switch requires aligned EcoLight binary actions and "
                f"green/red inputs: {path}"
            )
    elif aligned:
        raise ValueError(
            f"EcoLight {action_mode} requires the original phase-level rollout; "
            f"aligned binary inputs require binary_keep_switch: {path}"
        )


def _entry_seed(
    entry: dict[str, Any],
    *,
    explicit_keys: tuple[str, ...],
    default: int,
) -> int:
    """Resolve a sequential entry seed from its override or rollout metadata."""
    for key in explicit_keys:
        if entry.get(key) is not None:
            return int(entry[key])
    path_value = entry.get("teacher_data_path") or entry.get("teacher_data")
    if not path_value:
        raise ValueError("Sequential entry has no teacher_data_path")
    rollout_path = _root_path(str(path_value))
    rollout_seed = _read_rollout_meta(rollout_path).get("seed")
    return int(rollout_seed) if rollout_seed is not None else int(default)


def _entry_simulation_seed(entry: dict[str, Any], default: int) -> int:
    return _entry_seed(
        entry,
        explicit_keys=("simulation_seed", "sim_seed"),
        default=default,
    )


def _entry_random_seed(entry: dict[str, Any], default: int) -> int:
    return _entry_seed(
        entry,
        explicit_keys=("random_seed",),
        default=default,
    )


def _result_artifacts(
    entry: dict[str, Any],
    *,
    output_dir: Path,
    population_size: int,
    generations: int,
    simulation_seed: int,
) -> tuple[Path, Path, Path]:
    rollout_path = _root_path(str(entry.get("teacher_data_path") or entry["teacher_data"]))
    meta = _read_rollout_meta(rollout_path)
    teacher = str(entry["teacher_model"])
    scenario = str(meta["dataset"])
    flow_idx = int(meta["flow_idx"])
    entry_simulation_seed = _entry_simulation_seed(entry, simulation_seed)
    run_dir = output_dir / teacher / scenario / f"flow{flow_idx}"
    suffix = f"seed{entry_simulation_seed}_pop{population_size}_gen{generations}"
    return (
        run_dir / f"results_{suffix}.json",
        run_dir / f"deployment_model_{suffix}.json",
        run_dir / f"deployment_model_{suffix}.py",
    )


def _artifacts_are_complete(paths: tuple[Path, Path, Path]) -> bool:
    if not all(path.is_file() and path.stat().st_size > 0 for path in paths):
        return False
    try:
        result = _read_json(paths[0])
        deployment = _read_json(paths[1])
        best = result["best_individual"]
        score = float(best["score"])
        metrics = best["metrics"]
        if not math.isfinite(score) or score < 0 or metrics.get("error"):
            return False
        if score != float(deployment["score"]) or deployment["metrics"] != metrics:
            return False
        for key in ("student_travel_time", "teacher_travel_time"):
            if not math.isfinite(float(metrics[key])) or float(metrics[key]) <= 0:
                return False
        if best.get("function_code") != deployment.get("function_code"):
            return False
        if result.get("run_config") != deployment.get("run_config"):
            return False
        if result.get("per_intersection_bias", {}) != deployment.get("per_intersection_bias", {}):
            return False
        code = deployment["function_code"]
        if not isinstance(code, str) or "def compute_utility" not in code:
            return False
        if code.strip() not in paths[2].read_text(encoding="utf-8"):
            return False
        compile(code, str(paths[1]), "exec")
    except (OSError, ValueError, KeyError, TypeError, SyntaxError):
        return False
    return True


@contextmanager
def _run_output_guard(config, resolved, teacher_data_paths, output_dir):
    """Bind a result directory to its inputs before any search or resume scan."""
    import fcntl

    sources = {}
    for path in teacher_data_paths:
        meta = _read_rollout_meta(path)
        sources[str(path.resolve())] = _sha256_file(path)
        root = Path(resolved.get("method_project_root") or ROOT)
        for name in ("roadnet.json", f"flow_{meta['flow_idx']}.json"):
            data_path = root / "data" / meta["dataset"] / name
            sources[str(data_path.resolve())] = _sha256_file(data_path)
    identity = {
        "schema": "release_run_identity_v1",
        "config": config,
        "source_sha256": _sha256_source_tree(),
        "traffic_sha256": _sha256_file(Path(resolved["traffic_config"])),
        "input_sha256": sources,
        "teacher_manifest_sha256": (
            _sha256_file(Path(resolved["teacher_manifest"]))
            if resolved.get("teacher_manifest")
            else None
        ),
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    with (output_dir / ".run.lock").open("a") as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            raise ValueError("Another run is writing this output directory") from exc
        protocol = output_dir / "run_identity.json"
        if protocol.exists():
            if _read_json(protocol) != identity:
                raise ValueError(
                    "Run configuration, implementation or input data changed; preserve old results and use a new output directory"
                )
        else:
            if (
                (output_dir / "run_manifest.json").exists()
                or any(output_dir.rglob("deployment_model*.json"))
                or any(output_dir.rglob("results_*.json"))
            ):
                raise ValueError(
                    "Existing results have no run identity; use a new output directory instead of silently reusing legacy results"
                )
            temporary = protocol.with_suffix(".tmp")
            try:
                temporary.write_text(json.dumps(identity, indent=2, allow_nan=False) + "\n")
                temporary.replace(protocol)
            finally:
                temporary.unlink(missing_ok=True)
        yield


def _prepare_resume_manifest(
    *,
    config: dict[str, Any],
    output_dir: Path,
    resolved: dict[str, Any],
    internal_argv: list[str],
) -> bool:
    """Filter a sequential manifest to unfinished entries.

    Returns True when the complete matrix is already present.
    """
    if resolved["run_mode"] != "sequential" or not resolved.get("teacher_manifest"):
        raise ValueError("--resume currently supports sequential teacher manifests only")
    manifest_path = Path(resolved["teacher_manifest"])
    manifest = _read_json_value(manifest_path)
    if not isinstance(manifest, list):
        raise ValueError("--resume requires a JSON-array teacher manifest")

    search = _section(config, "search")
    population_size = int(search.get("population_size", 5))
    generations = int(search.get("generations", 5))
    simulation_seed = int(resolved["simulation_seed"])
    expected_identity = {}
    if resolved.get("method_profile") == "full_100_256_v1":
        from ppl_tsc.version import implementation_identity

        expected_identity = implementation_identity()
    pending = []
    for entry in manifest:
        paths = _result_artifacts(
            entry,
            output_dir=output_dir,
            population_size=population_size,
            generations=generations,
            simulation_seed=simulation_seed,
        )
        if not _artifacts_are_complete(paths):
            pending.append(entry)
            continue
        if expected_identity:
            recorded = _read_json(paths[1]).get("run_config", {})
            mismatched = [
                key for key, value in expected_identity.items() if recorded.get(key) != value
            ]
            if mismatched:
                raise ValueError(
                    "Cannot resume current full profile from a different or "
                    "unrecorded implementation revision: "
                    + ", ".join(mismatched)
                    + ". Preserve the existing policies and use a fresh output "
                    "directory for the current implementation."
                )
    completed = len(manifest) - len(pending)
    resolved["resume_completed"] = completed
    resolved["scheduled_rollouts"] = len(pending)
    print(f"Resume scan: {completed} complete, {len(pending)} pending.")
    if not pending:
        return True

    runtime_dir = ROOT / ".cache/runtime_manifests"
    runtime_dir.mkdir(parents=True, exist_ok=True)
    # Multiple cells commonly use the basename "manifest.json". Each
    # invocation needs its own file; a shared basename silently substitutes
    # another cell's teacher, flow or seed when launches overlap.
    with tempfile.NamedTemporaryFile(
        mode="w",
        encoding="utf-8",
        dir=runtime_dir,
        prefix=f"{manifest_path.stem}.",
        suffix=".pending.json",
        delete=False,
    ) as f:
        json.dump(pending, f, ensure_ascii=False, indent=2)
        runtime_path = Path(f.name)
    try:
        argument_index = internal_argv.index("--teachers_config") + 1
    except ValueError as exc:
        raise ValueError("Internal batch arguments have no --teachers_config") from exc
    internal_argv[argument_index] = str(runtime_path)
    return False


def _build_internal_argv(
    config: dict[str, Any],
    *,
    offline: bool,
    disable_cache: bool,
    validate_only: bool,
) -> tuple[list[str], Path, dict[str, Any], list[Path]]:
    experiment = _section(config, "experiment")
    data = _section(config, "data")
    llm = _section(config, "llm")
    search = _section(config, "search")
    evaluation = _section(config, "evaluation")

    provider = str(llm.get("provider", "openai_compatible")).strip().lower()
    backend_map = {
        "openai_compatible": "1",
        "local": "1",
        "yunwu": "4",
    }
    if provider not in backend_map:
        raise ValueError(
            f"Unsupported llm.provider {provider!r}; choose from {sorted(backend_map)}"
        )

    traffic_config = _root_path(str(data.get("traffic_config", "configs/traffic.json")))
    output_dir = _root_path(str(_required(experiment, "output_dir", "experiment")))
    endpoint = str(_required(llm, "endpoint", "llm"))
    model = str(_required(llm, "model", "llm"))
    request_path = str(llm.get("request_path", "/v1/chat/completions"))
    temperature = float(llm.get("temperature", 0.0))
    random_seed = int(experiment.get("random_seed", 0))
    simulation_seed = int(evaluation.get("simulation_seed", random_seed))

    if not traffic_config.is_file():
        raise FileNotFoundError(f"Traffic config not found: {traffic_config}")

    teacher_data_value = data.get("teacher_data")
    teacher_manifest_value = data.get("teacher_manifest")
    if bool(teacher_data_value) == bool(teacher_manifest_value):
        raise ValueError(
            "Configure exactly one of data.teacher_data (single run) or "
            "data.teacher_manifest (sequential batch)."
        )

    teacher_data_paths: list[Path] = []
    teachers: set[str] = set()
    scenarios: set[str] = set()
    flow_indices: set[int] = set()
    entry_simulation_seeds: set[int] = set()
    entry_random_seeds: set[int] = set()
    if teacher_manifest_value:
        teacher_manifest = _root_path(str(teacher_manifest_value))
        if not teacher_manifest.is_file():
            raise FileNotFoundError(f"Teacher manifest not found: {teacher_manifest}")
        run_mode, rollout_entries = _teacher_rollout_entries(teacher_manifest)
        artifact_keys = set()
        for index, entry in enumerate(rollout_entries):
            rollout_path, teacher, dataset, flow_idx = _validate_rollout_entry(
                entry,
                label=f"Teacher manifest entry {index}",
            )
            if teacher.casefold() == "ecolight":
                _validate_ecolight_action_interface(
                    rollout_path, str(search.get("action_mode", "auto"))
                )
            teacher_data_paths.append(rollout_path)
            teachers.add(teacher)
            scenarios.add(dataset)
            flow_indices.add(flow_idx)
            if run_mode == "sequential":
                artifact_key = (
                    teacher,
                    dataset,
                    flow_idx,
                    _entry_simulation_seed(entry, simulation_seed),
                )
                if artifact_key in artifact_keys:
                    raise ValueError(
                        "Sequential entries would overwrite the same result path; use separate output directories for independent search seeds"
                    )
                artifact_keys.add(artifact_key)
                entry_simulation_seeds.add(_entry_simulation_seed(entry, simulation_seed))
                entry_random_seeds.add(_entry_random_seed(entry, random_seed))
        data_argv = ["--teachers_config", str(teacher_manifest)]
    else:
        teacher_model = str(_required(data, "teacher_model", "data"))
        rollout_path, teacher, dataset, flow_idx = _validate_rollout_entry(
            {
                "teacher_data": teacher_data_value,
                "teacher_model": teacher_model,
            },
            label="Experiment data",
        )
        if teacher.casefold() == "ecolight":
            _validate_ecolight_action_interface(
                rollout_path, str(search.get("action_mode", "auto"))
            )
        teacher_data_paths.append(rollout_path)
        teachers.add(teacher)
        scenarios.add(dataset)
        flow_indices.add(flow_idx)
        teacher_manifest = None
        run_mode = "single"
        entry_simulation_seeds.add(simulation_seed)
        entry_random_seeds.add(random_seed)
        data_argv = [
            "--teacher_data_path",
            str(rollout_path),
            "--teacher_model",
            teacher_model,
        ]

    from ppl_tsc.config import validate_config as validate_current_profile

    validate_current_profile(config)
    if config.get("method", {}).get("profile") == "full_100_256_v1":
        for rollout_path in teacher_data_paths:
            source_header, _ = _rollout_preview(rollout_path)
            if (
                source_header.get("meta", {}).get("direction_feature_semantics")
                != "compatible_shared_lane_mean_v1"
            ):
                raise ValueError(
                    "Current full method requires matching compatible-shared-lane source observations"
                )
    environment_semantics = json.loads(traffic_config.read_text()).get(
        "waiting_time_semantics", "per_decision"
    )
    for rollout_path in teacher_data_paths:
        header, _ = _rollout_preview(rollout_path)
        meta = header.get("meta", {})
        if (
            meta.get("model") == "TinyLight"
            and meta.get("environment_semantics", "legacy_per_read") != environment_semantics
        ):
            raise ValueError(
                "TinyLight trajectory feature semantics differ from the environment. "
                "Regenerate trajectories for corrected training; historical replay requires "
                "a separate traffic config with waiting_time_semantics=legacy_per_read."
            )

    horizon_matched = True
    simulation_steps = evaluation.get("simulation_steps")
    if simulation_steps is not None:
        if int(simulation_steps) <= 0:
            raise ValueError("evaluation.simulation_steps must be positive")
        for rollout_path in teacher_data_paths:
            header, _ = _rollout_preview(rollout_path)
            teacher_steps = header.get("config", {}).get("num_step")
            if teacher_steps is None or int(teacher_steps) != int(simulation_steps):
                horizon_matched = False
        if not horizon_matched and not evaluation.get("allow_horizon_mismatch", False):
            raise ValueError(
                "Student simulation horizon differs from the recorded teacher horizon. "
                "Use simulation_steps=null for matched evaluation; the explicit "
                "allow_horizon_mismatch setting is intended only for smoke tests."
            )

    api_key_env = str(llm.get("api_key_env", "LLM_API_KEY"))
    api_key = os.getenv(api_key_env, "").strip()
    if offline or validate_only:
        api_key = "CACHE_REPLAY_ONLY"
    elif provider == "local" and not api_key:
        api_key = "EMPTY"
    elif not api_key:
        raise ValueError(f"API key is missing. Set {api_key_env} before running this experiment.")

    method_project_root = ROOT
    # Public JSON provider takes precedence over stale research-shell settings.
    os.environ["LLM_BACKEND"] = "yunwu" if provider == "yunwu" else "openai_compatible"
    os.environ["LLM_API_KEY"] = api_key
    os.environ["LLM_REQUEST_PATH"] = request_path
    os.environ["LLM_AUTH_MODE"] = str(llm.get("auth_mode", "bearer"))
    os.environ["LLM_TEMPERATURE"] = str(temperature)

    cache_dir = _root_path(str(llm.get("cache_dir", ".cache/llm")))
    cache_mode = "off" if disable_cache else str(llm.get("cache_mode", "readwrite"))
    if offline:
        cache_mode = "readonly"
    os.environ["LLM_CACHE_DIR"] = str(cache_dir)
    os.environ["LLM_CACHE_MODE"] = cache_mode
    cache_key_mode = str(llm.get("cache_key_mode", "call_sequence"))
    if cache_key_mode not in {"call_sequence", "prompt"}:
        raise ValueError("llm.cache_key_mode must be call_sequence or prompt")
    os.environ["LLM_CACHE_KEY_MODE"] = cache_key_mode
    os.environ.setdefault("LLM_USAGE_LOG", str(output_dir / "llm_usage.jsonl"))
    os.environ.setdefault("LLM_TRACE_DIR", str(output_dir / "llm_traces"))

    argv = [
        "ppl_tsc.search.runner",
        *data_argv,
        "--config_path",
        str(traffic_config),
        "--output_dir",
        str(output_dir),
        "--llm_backend",
        backend_map[provider],
        "--llm_api_endpoint",
        endpoint,
        "--llm_model",
        model,
        "--action_mode",
        str(search.get("action_mode", "auto")),
        "--population_size",
        str(int(search.get("population_size", 5))),
        "--n_generations",
        str(int(search.get("generations", 5))),
        "--mutation_rate",
        str(float(search.get("mutation_rate", 1.0))),
        "--max_constraint_retries",
        str(int(search.get("constraint_retries", 3))),
        "--max_init_attempts",
        str(int(search.get("max_initial_attempts", 30))),
        "--bias_fit_max_iter",
        str(int(search.get("bias_fit_iterations", 20))),
        "--bias_fit_bound",
        str(float(search.get("bias_bound", 0.5))),
        "--score_mode",
        str(evaluation.get("score_mode", "action_plus_perf")),
        "--action_weight",
        str(float(evaluation.get("action_weight", 1.0))),
        "--performance_weight",
        str(float(evaluation.get("performance_weight", 1.0))),
        "--sim_seed",
        str(simulation_seed),
        "--random_seed",
        str(random_seed),
    ]

    optional_ints = (
        ("max_timesteps", "--max_timesteps"),
        ("simulation_steps", "--sim_num_step"),
    )
    for key, flag in optional_ints:
        value = evaluation.get(key)
        if value is not None:
            argv.extend([flag, str(int(value))])

    time_budget_seconds = search.get("time_budget_seconds")
    if time_budget_seconds is not None:
        value = float(time_budget_seconds)
        if value <= 0.0:
            raise ValueError("search.time_budget_seconds must be positive")
        argv.extend(["--time_budget_seconds", str(value)])

    if not bool(search.get("slot_based_parent", True)):
        argv.append("--no_slot_based_parent")
    if not bool(search.get("fit_phase_bias", True)):
        argv.append("--no_fit_bias_during_evolution")
    if not bool(search.get("constraint_validation", True)):
        argv.append("--no_constraint_validation")

    resolved = {
        "provider": provider,
        "endpoint": endpoint,
        "request_path": request_path,
        "model": model,
        "temperature": temperature,
        "action_mode": str(search.get("action_mode", "auto")),
        "performance_horizon_matched": horizon_matched,
        "cache_mode": cache_mode,
        "cache_key_mode": cache_key_mode,
        "cache_dir": str(cache_dir),
        "run_mode": run_mode,
        "teacher_manifest": str(teacher_manifest) if teacher_manifest else None,
        "teacher_rollout_count": len(teacher_data_paths),
        "teachers": sorted(teachers),
        "scenarios": sorted(scenarios),
        "flow_indices": sorted(flow_indices),
        "traffic_config": str(traffic_config),
        "output_dir": str(output_dir),
        "method_project_root": str(method_project_root),
        "random_seed": random_seed,
        "simulation_seed": simulation_seed,
        "time_budget_seconds": (
            None if time_budget_seconds is None else float(time_budget_seconds)
        ),
        "seed_mode": (
            "manifest_override_then_rollout_metadata"
            if run_mode == "sequential"
            else "experiment_defaults"
        ),
        "random_seeds": sorted(entry_random_seeds) or [random_seed],
        "simulation_seeds": sorted(entry_simulation_seeds) or [simulation_seed],
    }
    if run_mode == "single":
        resolved["teacher_data"] = str(teacher_data_paths[0])
    return argv, output_dir, resolved, teacher_data_paths


def _write_manifest(
    config_path: Path,
    config: dict[str, Any],
    resolved: dict[str, Any],
    output_dir: Path,
    started_at: str,
    teacher_data_paths: list[Path],
) -> Path:
    artifacts = []
    if output_dir.exists():
        for path in sorted(output_dir.rglob("*")):
            relative_path = path.relative_to(output_dir)
            is_transient_log = bool(relative_path.parts) and relative_path.parts[0] == "logs"
            if path.is_file() and path.name != "run_manifest.json" and not is_transient_log:
                artifacts.append(
                    {
                        "path": relative_path.as_posix(),
                        "sha256": _sha256_file(path),
                        "bytes": path.stat().st_size,
                    }
                )

    traffic_config = Path(resolved["traffic_config"])
    teacher_rollout_files = [
        {
            "path": (path.relative_to(ROOT).as_posix() if path.is_relative_to(ROOT) else str(path)),
            "sha256": _sha256_file(path),
            "bytes": path.stat().st_size,
        }
        for path in teacher_data_paths
    ]
    method_project_root = Path(resolved.get("method_project_root") or ROOT).expanduser().resolve()
    cityflow_paths: set[Path] = set()
    for path in teacher_data_paths:
        meta = _read_rollout_meta(path)
        dataset = str(meta["dataset"])
        flow_idx = int(meta["flow_idx"])
        cityflow_paths.update(
            {
                method_project_root / "data" / dataset / "roadnet.json",
                method_project_root / "data" / dataset / f"flow_{flow_idx}.json",
            }
        )
    cityflow_input_files = [
        {
            "path": (path.relative_to(ROOT).as_posix() if path.is_relative_to(ROOT) else str(path)),
            "sha256": _sha256_file(path),
            "bytes": path.stat().st_size,
        }
        for path in sorted(cityflow_paths)
    ]
    teacher_manifest = resolved.get("teacher_manifest")
    api_profile = resolved.get("api_profile")
    resolved_for_manifest = dict(resolved)
    for key in (
        "cache_dir",
        "teacher_manifest",
        "traffic_config",
        "output_dir",
        "method_project_root",
        "teacher_data",
        "api_profile",
    ):
        if key in resolved_for_manifest:
            resolved_for_manifest[key] = _portable_path(resolved_for_manifest[key])
    manifest = {
        "status": "completed",
        "started_at_utc": started_at,
        "finished_at_utc": datetime.now(timezone.utc).isoformat(),
        "experiment_config": config,
        "resolved_non_secret_settings": resolved_for_manifest,
        "reproducibility": {
            "python": sys.version,
            "platform": platform.platform(),
            "packages": _package_versions(),
            "source_sha256": _sha256_source_tree(),
            "config_sha256": _sha256_file(config_path),
            "traffic_config_sha256": _sha256_file(traffic_config),
            "teacher_manifest_sha256": (
                _sha256_file(Path(teacher_manifest)) if teacher_manifest else None
            ),
            "api_profile_sha256": (_sha256_file(Path(api_profile)) if api_profile else None),
            "teacher_rollouts": teacher_rollout_files,
            "cityflow_inputs": cityflow_input_files,
        },
        "artifacts": artifacts,
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = output_dir / "run_manifest.json"
    with manifest_path.open("w", encoding="utf-8") as f:
        json.dump(manifest, f, ensure_ascii=False, indent=2)
    return manifest_path


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Run LLM-guided symbolic policy distillation from one JSON config."
    )
    parser.add_argument(
        "--config",
        default="configs/examples/MPLight.json",
        help="Experiment JSON path, relative to the submission root by default.",
    )
    parser.add_argument(
        "--offline",
        action="store_true",
        help="Replay cached LLM responses; no network request is made.",
    )
    parser.add_argument(
        "--no-cache",
        action="store_true",
        help="Disable LLM response caching for this run.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Validate and print resolved non-secret settings without running.",
    )
    parser.add_argument(
        "--output-dir",
        help="Override experiment.output_dir without editing the JSON config.",
    )
    parser.add_argument(
        "--api-profile",
        help=(
            "Override llm settings with a non-secret JSON profile. If omitted, "
            "configs/api.local.json is loaded automatically when present."
        ),
    )
    parser.add_argument(
        "--no-api-profile",
        action="store_true",
        help="Ignore configs/api.local.json and use llm settings from the experiment.",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="For a sequential matrix, skip runs with all expected artifacts present.",
    )
    parser.add_argument(
        "--continue-on-error",
        action="store_true",
        help="Attempt the remaining matrix after an individual run fails, then fail overall.",
    )
    args = parser.parse_args()

    if args.api_profile and args.no_api_profile:
        parser.error("--api-profile and --no-api-profile cannot be used together")
    _load_local_env()

    config_path = _root_path(args.config)
    config = _read_json(config_path)
    if args.no_api_profile:
        api_profile_path = None
    elif args.api_profile:
        api_profile_path = _root_path(args.api_profile)
        if not api_profile_path.is_file():
            raise FileNotFoundError(f"API profile not found: {api_profile_path}")
    elif DEFAULT_API_PROFILE.is_file():
        api_profile_path = DEFAULT_API_PROFILE.resolve()
    else:
        api_profile_path = None
    _apply_api_profile(config, api_profile_path)
    if args.output_dir:
        _section(config, "experiment")["output_dir"] = args.output_dir
    from ppl_tsc.config import configure

    configure(config)
    internal_argv, output_dir, resolved, teacher_data_paths = _build_internal_argv(
        config,
        offline=args.offline,
        disable_cache=args.no_cache,
        validate_only=args.dry_run,
    )
    resolved["method_profile"] = config.get("method", {}).get("profile", "custom_configuration")
    if resolved["method_profile"] == "full_100_256_v1":
        from ppl_tsc.config import PROMPT_TEMPLATE
        from ppl_tsc.version import implementation_identity

        resolved["prompt_template"] = PROMPT_TEMPLATE
        resolved.update(implementation_identity())
    resolved["verifier"] = {"source_bundles": 100, "coordinates": 256}
    resolved["api_profile"] = str(api_profile_path) if api_profile_path else None
    if args.continue_on_error:
        internal_argv.append("--continue_on_error")

    print("Resolved experiment:")
    print(json.dumps(resolved, ensure_ascii=False, indent=2))
    if args.dry_run:
        print("Configuration is valid. No experiment was started.")
        return 0

    with _run_output_guard(config, resolved, teacher_data_paths, output_dir):
        matrix_complete = False
        if args.resume and not args.dry_run:
            matrix_complete = _prepare_resume_manifest(
                config=config,
                output_dir=output_dir,
                resolved=resolved,
                internal_argv=internal_argv,
            )

        started_at = datetime.now(timezone.utc).isoformat()
        if matrix_complete:
            manifest = _write_manifest(
                config_path,
                config,
                resolved,
                output_dir,
                started_at,
                teacher_data_paths,
            )
            print(f"All expected runs are already complete. Run manifest: {manifest}")
            return 0

        previous_cwd = Path.cwd()
        try:
            os.chdir(ROOT)
            from ppl_tsc.search.runner import main as method_main

            method_main(internal_argv[1:], method_profile=resolved["method_profile"])
        finally:
            os.chdir(previous_cwd)

        manifest = _write_manifest(
            config_path,
            config,
            resolved,
            output_dir,
            started_at,
            teacher_data_paths,
        )
        print(f"Run manifest: {manifest}")
        return 0


if __name__ == "__main__":
    raise SystemExit(main())
