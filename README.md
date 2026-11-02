# PPL-TSC

Code for **Discovering Programmatic Policies from Reinforcement Learning-Based Traffic Signal Controllers** (NeurIPS 2026).

Train a DRL teacher, collect its trajectories, and distill an executable symbolic traffic controller.

## Install

Linux x86_64, Python 3.10, a C++ toolchain, and `bubblewrap` are required.

```bash
bash scripts/setup_venv.sh
source .venv/bin/activate
```

## Train and distill

Example: MPLight on Hangzhou1, flow 0. Run these commands from the repository root.

```bash
python train_teacher.py --teacher MPLight --scenario Hangzhou1 --flow 0 --output-dir outputs/teacher
python collect_teacher.py --teacher MPLight --scenario Hangzhou1 --flow 0 --checkpoint outputs/teacher/checkpoint.pth --output outputs/trajectory.jsonl.gz
python scripts/prepare_distillation.py --variant MPLight --trajectory outputs/trajectory.jsonl.gz --output-dir outputs/student
python configure_api.py
python run.py --config outputs/student/config.json
```

`configure_api.py` saves your API key locally. Distillation makes paid LLM requests.
Add `--dry-run` to `run.py` to check a configuration without calling the API.
`python scripts/check_environment.py` runs a local self-check without API calls.
Run records include source and input hashes for reproducibility.

Teachers: **CoLight, FRAP, MPLight, TinyLight, EcoLight**.
Student variants: `CoLight`, `FRAP`, `MPLight`, `TinyLight`, `EcoLight-original`,
`EcoLight-native`, `EcoLight-phase-selection`. Use the matching `--teacher` and `--variant`.

Default search: population 5, generations 5; CMA-ES population 20, maximum 20 iterations.
Configuration templates are in `configs/examples/`; all commands share `configs/traffic.json`.
The default fitness is ABF + exp(min(Delta, 0)), where Delta is relative ATT improvement.
Each source intersection and physical phase has an offset in [-0.5, 0.5].

### EcoLight

All three student variants use `--teacher EcoLight`.
**EcoLight-original** retains the earlier phase-feature interface: score the current
phase and its cyclic successor using their directional vehicle features.
**EcoLight-native** uses EcoLight's exact green/red lane vehicle totals to score
KEEP and SWITCH on the same state. Both advance only to the cyclic next phase.
**EcoLight-phase-selection** scores all candidate phases and selects the highest.
Choose the interface with `--variant`; the example below uses `EcoLight-original`.

```bash
python train_teacher.py --teacher EcoLight --scenario Hangzhou1 --flow 0 --output-dir outputs/EcoLight/teacher
python collect_teacher.py --teacher EcoLight --scenario Hangzhou1 --flow 0 --checkpoint outputs/EcoLight/teacher/checkpoint.pth --output outputs/EcoLight/trajectory.jsonl.gz
python scripts/prepare_distillation.py --variant EcoLight-original --trajectory outputs/EcoLight/trajectory.jsonl.gz --output-dir outputs/EcoLight/student
python run.py --config outputs/EcoLight/student/config.json
```

## Evaluate

Replace `POLICY.json` with the generated `deployment_model*.json` path.
Changing `--flow` evaluates the fixed policy on another flow of the same network.

```bash
python evaluate.py --deployment POLICY.json --scenario Hangzhou1 --flow 1 --output outputs/evaluation.json
```

For another network, export the shared formula and frozen feature statistics explicitly;
source intersection offsets are removed by this export.

```bash
python scripts/export_formula_transfer.py --deployment POLICY.json --output outputs/transfer.json
python evaluate.py --deployment outputs/transfer.json --scenario Atlanta --flow 0 --steps 900 --output outputs/transfer_result.json
```

`evaluate_teacher.py --help` describes the teacher evaluation protocols.
Use `--help` on each command for additional options.

## Code

- `src/ppl_tsc/`: data contracts, LLM prompts, verification, search, calibration, and policy execution.
- `src/agent/`: the five DRL teachers and their training code.
- `src/env/` and `src/utilities/`: traffic environments and shared utilities.
- `configs/`: method and environment settings.
- `data/`: seven networks and five traffic flows per network.
- `CityFlow/`: simulator source and build dependencies.
- `scripts/`: installation, preparation, and transfer export.

No pretrained weights, collected trajectories, generated policies, or API keys are included.
Third-party attribution and licensing notes are in [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md).
