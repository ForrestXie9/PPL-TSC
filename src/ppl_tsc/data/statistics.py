"""Pool source observations using counts and population variance (ddof=0)."""

import math


def merge_feature_statistics(groups):
    merged = {}
    for group in groups:
        for kind, stats in group.items():
            if kind == "current_phase":
                # This is a documented binary-indicator convention, not a sample.
                continue
            n = stats.get("count")
            if not isinstance(n, int) or isinstance(n, bool) or n <= 0:
                raise ValueError(f"Missing positive sample count for {kind}")
            mean, std = float(stats["mean"]), float(stats["std"])
            low, high = float(stats["min"]), float(stats["max"])
            if not all(math.isfinite(x) for x in (mean, std, low, high)) or std < 0 or low > high:
                raise ValueError(f"Invalid source statistics for {kind}")
            if kind not in merged:
                merged[kind] = dict(count=n, mean=mean, m2=n * std * std, min=low, max=high)
            else:
                old = merged[kind]
                total = old["count"] + n
                delta = mean - old["mean"]
                old["m2"] += n * std * std + delta * delta * old["count"] * n / total
                old["mean"] += delta * n / total
                old["count"] = total
                old["min"], old["max"] = min(old["min"], low), max(old["max"], high)
    for stats in merged.values():
        stats["std"] = math.sqrt(max(0.0, stats.pop("m2") / stats["count"]))
    return merged


def observed_direction_keys(loader):
    """Read the serialized student schema, not teacher observation-name aliases."""
    keys = set()
    for inter in loader.intersection_ids:
        for states, *_ in loader.get_action_consistency_data(inter, max_timesteps=None):
            for state in states.values():
                for features in (state.get("features_by_direction") or {}).values():
                    keys.update(features)
    return sorted(keys)


def source_feature_contract(extractor, measured_statistics, loader):
    """Keep measured feature kinds and the declared current-phase indicator."""
    extractor.specs = [spec for spec in extractor.specs if spec.kind in measured_statistics]
    extractor.feature_names = [spec.name for spec in extractor.specs]
    statistics = {spec.name: dict(measured_statistics[spec.kind]) for spec in extractor.specs}
    return statistics, observed_direction_keys(loader)
