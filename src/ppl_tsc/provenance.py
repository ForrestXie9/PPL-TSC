"""Identify a run by the contents of its executable source."""

import hashlib
from pathlib import Path
from ppl_tsc.paths import PROJECT_ROOT


def source_tree_sha256(root: Path = PROJECT_ROOT) -> str:
    digest = hashlib.sha256()
    paths = list((root / "src").rglob("*.py")) + [
        root / name for name in ("run.py", "evaluate.py", "train_teacher.py", "collect_teacher.py")
    ]
    # Include the bundled simulator: local fixes affect execution provenance.
    paths += [
        path
        for path in (root / "CityFlow").rglob("*")
        if path.suffix in {".cpp", ".h", ".hpp", ".py"} or path.name == "CMakeLists.txt"
    ]
    for path in sorted(
        path
        for path in paths
        if path.is_file()
        and not any(
            part in {"build", ".git", "__pycache__"} or part.endswith(".egg-info")
            for part in path.relative_to(root).parts
        )
    ):
        digest.update(path.relative_to(root).as_posix().encode("utf-8"))
        digest.update(path.read_bytes())
    return digest.hexdigest()


def source_identity(root: Path = PROJECT_ROOT) -> dict[str, str]:
    return {"method": "PPL-TSC", "source_sha256": source_tree_sha256(root)}
