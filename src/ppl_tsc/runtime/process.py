"""Bounded generated-policy execution outside the research process."""

from __future__ import annotations
import atexit
import base64
import json
import os
from pathlib import Path
import pickle
import select
import shutil
import signal
import subprocess
import sys
import threading
import weakref

_LIVE = weakref.WeakSet()


class PolicyExecutionError(ValueError):
    """A worker failed, exceeded limits or returned an invalid scalar."""


class PolicyProcess:
    def __init__(self, code: str, call_timeout: float = 2.0, load_timeout: float = 15.0):
        self.call_timeout = call_timeout
        self._lock = threading.Lock()
        self.process = None
        from ppl_tsc.verification.identity import require_no_process_identity

        require_no_process_identity(code)
        worker = Path(__file__).with_name("worker.py").resolve()
        mode = os.environ.get("PPL_POLICY_ISOLATION", "sandbox")
        if mode not in {"sandbox", "process"}:
            raise ValueError("PPL_POLICY_ISOLATION must be sandbox or process")
        executable = str(Path(sys.executable).absolute())
        command = [executable, "-I", "-u", str(worker)]
        if mode == "sandbox":
            bwrap = shutil.which("bwrap")
            if not bwrap:
                raise PolicyExecutionError(
                    "Install bubblewrap for default filesystem/network isolation; "
                    "PPL_POLICY_ISOLATION=process explicitly selects process/resource limits only"
                )
            command = [
                bwrap,
                "--unshare-all",
                "--die-with-parent",
                "--new-session",
                "--cap-drop",
                "ALL",
                "--tmpfs",
                "/tmp",
            ]
            # Create private /tmp before binding runtime paths: a venv under /tmp
            # must remain visible. Only its runtime directory is rebound read-only.
            # Mount interpreter/runtime dependencies, not the research workspace.
            mounted = []
            for path in ["/usr", "/lib", "/lib64", "/bin", sys.base_prefix, sys.prefix]:
                p = Path(path).absolute()
                if p.exists() and not any(p == q or p.is_relative_to(q) for q in mounted):
                    command += ["--ro-bind", str(p), str(p)]
                    mounted.append(p)
            # A venv created from another venv can link through that older
            # environment. Recreate only the interpreter chain, without
            # exposing the older environment's packages or other files.
            interpreter = Path(executable)
            visited = set()
            while True:
                if interpreter in visited:
                    raise PolicyExecutionError("Cyclic Python interpreter symlink")
                visited.add(interpreter)
                covered = any(interpreter == p or interpreter.is_relative_to(p) for p in mounted)
                if interpreter.is_symlink():
                    target = os.readlink(interpreter)
                    if not covered:
                        command += [
                            "--dir",
                            str(interpreter.parent),
                            "--symlink",
                            target,
                            str(interpreter),
                        ]
                    interpreter = Path(os.path.abspath(interpreter.parent / target))
                else:
                    if not covered:
                        command += ["--ro-bind", str(interpreter), str(interpreter)]
                    break
            command += [
                "--ro-bind",
                str(worker),
                "/worker.py",
                "--proc",
                "/proc",
                "--dev",
                "/dev",
                "--dir",
                "/work",
                "--chdir",
                "/work",
                executable,
                "-I",
                "-u",
                "/worker.py",
            ]
        env = {
            "PATH": "/usr/bin:/bin",
            "LANG": "C.UTF-8",
            "OMP_NUM_THREADS": "1",
            "OPENBLAS_NUM_THREADS": "1",
            "MKL_NUM_THREADS": "1",
            "NUMEXPR_NUM_THREADS": "1",
        }
        self.process = subprocess.Popen(
            command,
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
            env=env,
            start_new_session=True,
            bufsize=0,
        )
        _LIVE.add(self)
        try:
            self._request({"op": "load", "code": code}, load_timeout)
        except BaseException:
            self.close()
            raise

    def _request(self, payload, timeout):
        import time

        with self._lock:
            process = self.process
            if process is None or process.poll() is not None:
                raise PolicyExecutionError("Policy worker exited or was closed")
            wire = (json.dumps(payload, allow_nan=False) + "\n").encode()
            if len(wire) > 8 * 1024**2:
                raise PolicyExecutionError("Policy input exceeds frame limit")
            deadline = time.monotonic() + timeout
            # Nonblocking writes and reads include pipe stalls in the wall limit.
            os.set_blocking(process.stdin.fileno(), False)
            os.set_blocking(process.stdout.fileno(), False)
            response = bytearray()
            sent = 0
            try:
                while b"\n" not in response:
                    remaining = deadline - time.monotonic()
                    if remaining <= 0:
                        raise TimeoutError(f"Policy exceeded {timeout:g}s wall limit")
                    readable, writable, _ = select.select(
                        [process.stdout], [process.stdin] if sent < len(wire) else [], [], remaining
                    )
                    if writable:
                        sent += os.write(process.stdin.fileno(), wire[sent : sent + 65536])
                    if readable:
                        data = os.read(process.stdout.fileno(), 65536)
                        if not data:
                            raise PolicyExecutionError("Policy worker exited without a result")
                        response.extend(data)
                        if len(response) > 65536:
                            raise PolicyExecutionError("Policy response exceeds limit")
                result = json.loads(response)
            except (OSError, TimeoutError, ValueError) as exc:
                self.close()
                raise PolicyExecutionError(str(exc)) from exc
            if not result.get("ok"):
                raise PolicyExecutionError(f"{result.get('error_type')}: {result.get('error')}")
            return result

    def __call__(self, state, phase_id=None, feature_stats=None):
        args = base64.b64encode(pickle.dumps((state, phase_id, feature_stats), protocol=4)).decode(
            "ascii"
        )
        return self._request({"op": "call", "args": args}, self.call_timeout)["value"]

    def __deepcopy__(self, memo):
        # Python function deepcopy also shares the function's module globals.
        return self

    def close(self):
        process = self.process
        self.process = None
        if process is not None:
            if process.poll() is None:
                try:
                    os.killpg(process.pid, signal.SIGKILL)
                except ProcessLookupError:
                    pass
            try:
                process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait()
            for stream in (process.stdin, process.stdout):
                if stream:
                    stream.close()

    def __del__(self):
        self.close()


@atexit.register
def _close_workers():
    for worker in list(_LIVE):
        worker.close()
