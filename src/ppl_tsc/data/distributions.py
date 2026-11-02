"""Compute distribution summaries from the source observation pool."""

import numpy as np

FIELDS = ["p50", "p75", "p90", "p95", "p99", "zero_fraction"]


def extend(stats, values_by_kind):
    for kind, values in values_by_kind.items():
        if not values:
            continue
        array = np.asarray(values, dtype=float)
        if not np.isfinite(array).all():
            raise ValueError("Nonfinite source percentile input")
        row = stats[kind]
        assert int(row["count"]) == array.size
        assert row["min"] == float(array.min()) and row["max"] == float(array.max())
        result = np.percentile(array, [50, 75, 90, 95, 99], method="linear")
        row.update({key: float(value) for key, value in zip(FIELDS[:5], result)})
        row["zero_fraction"] = float(np.mean(array == 0))
    return stats
