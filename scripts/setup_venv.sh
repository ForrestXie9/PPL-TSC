#!/usr/bin/env bash
set -euo pipefail
artifact_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${artifact_root}"
command -v bwrap >/dev/null || { echo 'Install bubblewrap (e.g. apt install bubblewrap) before using isolated policies.'; exit 1; }
task_python="${PPL_TSC_PYTHON:-python3.10}"
"${task_python}" -c 'import sys; assert sys.version_info[:2] == (3, 10), "Use Python 3.10 for the pinned reproduction environment"'
if ! "${task_python}" -c 'import ensurepip' >/dev/null 2>&1; then
  echo 'Python 3.10 needs venv/ensurepip. Install python3.10-venv, or set PPL_TSC_PYTHON to a complete Python 3.10 interpreter.'
  exit 1
fi
"${task_python}" -m venv .venv
# Pin a CMake generation supported by the bundled simulator. A direct official
# CPU wheel avoids resolving unrelated packages through the PyTorch index.
task_torch_wheel="$(.venv/bin/python -c 'import platform; assert platform.system()=="Linux" and platform.machine()=="x86_64", "Validated setup supports Linux x86_64"; print("https://download.pytorch.org/whl/cpu/torch-2.9.1%2Bcpu-cp310-cp310-manylinux_2_28_x86_64.whl#sha256=7210713b66943fdbfcc237b2e782871b649123ac5d29f548ce8c85be4223ab38")')"
if [[ -n "${PPL_TSC_TORCH_WHEEL:-}" ]]; then
  task_torch_wheel="$(.venv/bin/python -c 'import hashlib,sys; from pathlib import Path; p=Path(sys.argv[1]).resolve(); assert hashlib.sha256(p.read_bytes()).hexdigest()=="7210713b66943fdbfcc237b2e782871b649123ac5d29f548ce8c85be4223ab38", "CPU wheel digest mismatch"; print(p.as_uri())' "${PPL_TSC_TORCH_WHEEL}")"
fi
.venv/bin/python -m pip install --index-url https://pypi.org/simple \
  --timeout 30 --retries 3 --cache-dir outputs/setup_cache \
  "torch @ ${task_torch_wheel}" -r requirements.txt 'setuptools==79.0.1'
PATH="${artifact_root}/.venv/bin:${PATH}" .venv/bin/python -m pip install --no-build-isolation ./CityFlow
.venv/bin/python scripts/check_environment.py
echo 'Environment ready: source .venv/bin/activate'
