"""Private isolated interpreter: trusted input frames, scalar JSON responses."""

import base64
import contextlib
import json
import math
import os
import pickle
import resource
import sys


def main():
    # Applied before importing NumPy or executing generated code.
    resource.setrlimit(resource.RLIMIT_AS, (2 * 1024**3, 2 * 1024**3))
    resource.setrlimit(resource.RLIMIT_CPU, (120, 120))
    resource.setrlimit(resource.RLIMIT_FSIZE, (1024**2, 1024**2))
    resource.setrlimit(resource.RLIMIT_CORE, (0, 0))
    resource.setrlimit(resource.RLIMIT_NOFILE, (64, 64))
    import numpy as np

    wire = sys.stdout
    function = None
    MAX_FRAME = 8 * 1024**2
    for line in sys.stdin:
        try:
            if len(line) > MAX_FRAME:
                raise ValueError("Policy input exceeds frame limit")
            request = json.loads(line)
            with (
                open(os.devnull, "w") as sink,
                contextlib.redirect_stdout(sink),
                contextlib.redirect_stderr(sink),
            ):
                if request["op"] == "load":
                    namespace = {"np": np, "numpy": np, "__builtins__": __builtins__}
                    exec(request["code"], namespace)
                    function = namespace.get("compute_utility")
                    if not callable(function):
                        raise ValueError("compute_utility must be callable")
                    response = {"ok": True, "loaded": True}
                elif request["op"] == "call":
                    # These bytes originate only in the trusted parent process.
                    # Responses deliberately use JSON, never worker-produced pickle.
                    state, phase, stats = pickle.loads(base64.b64decode(request["args"]))
                    # NumPy otherwise only warns and can let generated code hide
                    # invalid intermediate arithmetic behind a final finite check.
                    # Underflow to zero is allowed; invalid/divide/overflow fail the
                    # candidate and enter the normal review feedback path.
                    with np.errstate(divide="raise", invalid="raise", over="raise", under="ignore"):
                        value = function(state, phase_id=phase, feature_stats=stats)
                    if value is None:
                        raise ValueError("compute_utility returned None; expected a finite scalar")
                    if isinstance(value, (complex, np.complexfloating)):
                        raise ValueError("Complex utility")
                    value = float(value)
                    if not math.isfinite(value):
                        raise ValueError("Nonfinite utility")
                    response = {"ok": True, "value": value}
                else:
                    raise ValueError("Unknown worker operation")
        except BaseException as exc:
            response = {"ok": False, "error_type": type(exc).__name__, "error": str(exc)[:2000]}
        wire.write(json.dumps(response, allow_nan=False) + "\n")
        wire.flush()


if __name__ == "__main__":
    main()
