"""Atomic evaluation records: a failed retry must not look like an old success."""
from contextlib import contextmanager
import fcntl
import json
import os
from pathlib import Path
import tempfile


def atomic_json(path, value):
    path = Path(path)
    # Validate serialization before touching the destination.
    payload = json.dumps(value, indent=2, allow_nan=False) + '\n'
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, name = tempfile.mkstemp(prefix='.' + path.name + '.', dir=path.parent)
    temporary = Path(name)
    try:
        with os.fdopen(fd, 'w') as stream:
            stream.write(payload)
            stream.flush()
            os.fsync(stream.fileno())
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)


@contextmanager
def evaluation_record(path, context):
    """Serialize writers and record running/failed/complete explicitly.

    Caller must protect input/output aliases before entering this context.
    SIGKILL can leave running, but cannot leave a previous complete result.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.with_name('.' + path.name + '.lock').open('a') as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            raise ValueError('Another evaluation is writing this result path') from exc
        atomic_json(path, {**context, 'status': 'running'})
        committed = False
        def complete(result):
            nonlocal committed
            if committed:
                raise RuntimeError('Evaluation result was already committed')
            atomic_json(path, {**context, **result, 'status': 'complete'})
            committed = True
        try:
            yield complete
            if not committed:
                raise RuntimeError('Evaluation exited without a complete result')
        except BaseException as exc:
            atomic_json(path, {**context, 'status': 'failed',
                               'error_type': type(exc).__name__, 'error': str(exc)})
            raise


def new_gzip_jsonl(path, rows):
    """Publish a complete new trajectory, preserving any existing destination."""
    import gzip
    path = Path(path)
    if path.exists():
        raise ValueError('Trajectory output already exists; use a new path')
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, name = tempfile.mkstemp(prefix='.' + path.name + '.', dir=path.parent)
    temporary = Path(name)
    try:
        with os.fdopen(fd, 'wb') as raw:
            with gzip.GzipFile(filename='', mode='wb', fileobj=raw, mtime=0) as stream:
                for row in rows:
                    stream.write((json.dumps(row, allow_nan=False) + '\n').encode())
            raw.flush()
            os.fsync(raw.fileno())
        os.link(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)
