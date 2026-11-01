# Discovering Programmatic Policies from Reinforcement Learning-Based Traffic Signal Controllers

**PPL-TSC · NeurIPS 2026 — research code release candidate**

Discovering Programmatic Policies from Reinforcement Learning-Based Traffic Signal Controllers.

This folder contains the current PPL-TSC implementation, five DRL teacher implementations,
from-scratch training/trajectory collection, and fixed-policy evaluation. **No pretrained
weights, teacher trajectories, searched policies, API keys or historical experiment outputs
are included.** See [中文操作说明](README.zh-CN.md).

## Installation

Validated platform: Linux x86_64, Python 3.10. Install a C++ toolchain, Python 3.10 development
headers/venv, and `bubblewrap`; user namespaces must be available for isolated policy execution.
The local CityFlow source and its dependencies are included.

```bash
bash scripts/setup_venv.sh
source .venv/bin/activate
```

The installer downloads dependencies and builds CityFlow. It uses pinned CPU PyTorch;
GPU configuration is optional and not required. See `requirements-current-lock.txt`.

## From scratch: train → collect → distill → evaluate

Run commands from this folder. The default example is MPLight, Hangzhou1, flow0.

```bash
python train_teacher.py --teacher MPLight --scenario Hangzhou1 --flow 0 \
  --episodes 128 --selection best_training_att --output-dir outputs/teachers/MPLight/Hangzhou1/flow0

python collect_teacher.py --teacher MPLight --scenario Hangzhou1 --flow 0 \
  --checkpoint outputs/teachers/MPLight/Hangzhou1/flow0/checkpoint.pth \
  --output outputs/trajectories/MPLight_Hangzhou1_flow0.jsonl.gz

python scripts/prepare_distillation.py --variant MPLight \
  --trajectory outputs/trajectories/MPLight_Hangzhou1_flow0.jsonl.gz \
  --output-dir outputs/students/MPLight_Hangzhou1_flow0

cp .env.example .env
# Edit .env locally: LLM_API_KEY=your_key
# The generated config sets provider endpoint/model; change these for your provider.
python run.py --config outputs/students/MPLight_Hangzhou1_flow0/config.json --dry-run --no-api-profile
python run.py --config outputs/students/MPLight_Hangzhou1_flow0/config.json --no-api-profile
```

Collection automatically checks `audit.json` beside the checkpoint (or `--training-audit`),
including completed training, teacher identity, checkpoint hash and source roadnet.
A legacy checkpoint without an audit requires `--allow-unaudited-checkpoint` and is
explicitly marked unaudited; this flag never bypasses a mismatching audit. Collection
can use another flow on the same roadnet; cross-roadnet teacher transfer is a separate protocol.

The last command makes paid third-party LLM requests. Prompts include source statistics,
feature descriptions, generated code, and source evaluation/error feedback. No target-domain
feedback is used. Dry-run is offline. Do not share `.env` or request/cache outputs containing
private experiment data.

Locate the resulting policy, then evaluate it (replace `DEPLOYMENT.json` with that file):

```bash
find outputs/students/MPLight_Hangzhou1_flow0 -name 'deployment_model*.json'
python evaluate.py --deployment DEPLOYMENT.json --scenario Hangzhou1 --flow 0 --output outputs/source.json
python evaluate.py --deployment DEPLOYMENT.json --scenario Hangzhou1 --flow 1 --output outputs/cross_flow.json
```

Same-roadnet tests retain source offsets and frozen feature statistics. For a deliberately
**formula-only** cross-roadnet evaluation, export an explicit separate policy:

```bash
python scripts/export_formula_transfer.py --deployment DEPLOYMENT.json --output outputs/formula_transfer.json
python evaluate.py --deployment outputs/formula_transfer.json --scenario Hangzhou2 --flow 0 \
  --steps 3600 --output outputs/cross_network.json
```

This removes source-local offsets and records them and the source hash in the export; it does
not claim to transfer all calibrated parameters. All target evaluation is held out from search.
To compare against a teacher, use the **same source checkpoint** on the target through a
compatible source-agent-0 transfer controller; a target-trained teacher is a different baseline.

## Teachers and interfaces

| Teacher | Training | Student interface |
|---|---|---|
| CoLight | `train_teacher.py --teacher CoLight` (original internal sharing) | phase scores |
| FRAP | `train_teacher.py --teacher FRAP` | phase scores |
| MPLight | `train_teacher.py --teacher MPLight` (shared network internally) | phase scores |
| TinyLight | `train_teacher.py --teacher TinyLight` | phase scores |
| EcoLight | `train_teacher.py --teacher EcoLight` | native binary keep/switch, phase-feature naive keep/switch, or phase-selection |

Use `--variant CoLight`, `FRAP`, `MPLight`, `TinyLight`, `EcoLight-native`,
`EcoLight-naive`, or `EcoLight-phase-selection` in `prepare_distillation.py`.
All three EcoLight students use trajectories from the same binary DRL teacher:

| Student variant | Action mode | Inputs and decision |
|---|---|---|
| `EcoLight-native` | `binary_keep_switch` | Global green/red vehicle counts; compare KEEP and SWITCH action scores. |
| `EcoLight-naive` (historical phase-feature interface) | `keep_switch` | Phase-level directional features; compare current phase with its cyclic successor only. |
| `EcoLight-phase-selection` | `phase_argmax` | Phase-level directional features; select the highest-scoring phase among all candidates. |

For naive and phase-selection, the preparation tool writes a separate physical-phase
projection, preserves teacher physical action labels, current phase and directional features,
removes native-only binary fields, and records the original trajectory hash. The original
trajectory is unchanged. Naive does not select arbitrary phases and is not the native
binary-feature interface. Use independent output directories for the three variants.

Example after collecting an EcoLight trajectory:

```bash
.venv/bin/python scripts/prepare_distillation.py --variant EcoLight-naive \
  --trajectory outputs/trajectories/EcoLight_Hangzhou1_flow0.jsonl.gz \
  --output-dir outputs/EcoLight-naive
.venv/bin/python run.py --config outputs/EcoLight-naive/config.json --dry-run
.venv/bin/python run.py --config outputs/EcoLight-naive/config.json
```

This restores the historical **interface**, using the current method configuration;
it does not claim to reproduce historical policies or performance.

`train_teacher.py --resume` resumes its own saved training state. Defaults: 128 episodes,
minimum training ATT checkpoint selection, CPU. The TinyLight trainer is the project's
corrected **unfrozen 128-episode variant**, not the canonical pruned 64+128 training protocol.
Fresh training is not guaranteed to reproduce historical checkpoint scores.

## Original-family teacher transfer

FRAP/EcoLight retain independent per-intersection training. No added shared-training variant
is shipped. MPLight and CoLight retain parameter sharing already present in PI-Light's code.
The original `015_baseline_transfer.py` copies **source agent 0** into every target-local agent.
Our explicit runner follows that mapping and strictly checks tensor names/shapes:

```bash
python evaluate_teacher.py --checkpoint outputs/teachers/MPLight/Hangzhou1/flow0/checkpoint.pth \
  --training-audit outputs/teachers/MPLight/Hangzhou1/flow0/audit.json \
  --scenario Hangzhou2 --flow 0 --protocol source_agent0 --output outputs/teacher_transfer.json
```

For original/same-network cross-flow evaluation, use `--protocol same_network` and the original
scenario: this preserves each source intersection's own weights. CoLight cross-network transfer
is conditional on dimensions and feature/action semantics, not universally supported. TinyLight
cross-network transfer is not established by the original baseline script. See
[teacher audit and protocol](docs/TEACHER_PROTOCOL.md).

## Current method

- Version `0.4.5-scale-reminder`; Figure A.2 prompt structure with current compact contracts.
- Population 5 × 5 generations; strict fitness improvement replaces its slot's parent.
- Fitness = ABF + CP; CP = exp(min(Delta, 0)), Delta = (teacher ATT − student ATT)/(abs(teacher ATT)+1e-6).
- Source per-intersection/phase calibration, bound ±0.5; CMA-ES population 20, maximum 20 iterations
  (native early stopping remains enabled), zero initialization, sigma = bound/3.
- Numerical/feature/traffic-contract verification and source statistics; normalization is chosen
  by the generated program. Finite tests are not a universal safety or traffic-optimality proof.
- The new active-phase scale reminder is included. Its measured benefit is mixed; neither
  universal superiority over teachers nor an independent generalization benefit of ABF is claimed.
- Source horizons: Atlanta 900 s, LosAngeles 1800 s, other scenarios 3600 s; action interval 20 s.

## Layout

`src/`: method, agents, simulator adapters, verifier and calibration.
`configs/examples/`: seven runnable template configurations.
`data/`: seven roadnets and five flows each (inputs, not learned models).
`CityFlow/`: simulator source and third-party notices.
`benchmarks/`: collector/evaluation dependencies.
`reference/pi_light/`: unmodified selected upstream files for provenance (not standalone entrypoints).
`scripts/`: setup, configuration, explicit transfer export and package checks.

## Release status and provenance

This is a source-code release candidate. Project license
and upstream teacher/data attribution remain to be confirmed by the authors; see
[THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md). No project license is invented here.
The latest execution audit is `docs/COMPREHENSIVE_REVIEW_VALIDATION.json`; earlier reports describe earlier revisions. The full offline audit uses two-episode teachers and deterministic LLM fixtures. Separate local acceptance runs completed 128-episode teachers and real LLM searches; trained artifacts and private logs are not included. `MANIFEST.sha256` records shipped files.

## Offline checks

```bash
python scripts/check_package.py
python -m unittest discover -s tests -v
```

Tests that execute policies require functional bubblewrap/user namespaces.

### Module validation update

Fresh environment installation, five teachers on single/multiple-intersection training smokes,
six full-budget offline fixture searches, 16 policy replays and 42 tests pass. LLM responses
are deterministic fixtures, not fresh paid generations or performance evidence. The private
/tmp mount ordering was corrected for temporary virtual environments; setup now checks isolated
execution. Details and remaining scope limits: `docs/MODULE_VALIDATION.json`.

### Evaluation and file integrity checks

Teacher evaluation defaults to the source training audit seed; student evaluation defaults
to the deployment simulation seed. For paired comparisons, explicitly pass the same
`--seed` to both, especially if trajectory collection used a different seed.
Trajectory projection requires a new destination and only publishes a complete file.
Unknown calibration intersection/phase indices are rejected. Evaluation output paths
must differ from input checkpoints, audits and deployments.
The additional boundary audit (48 tests and six exact source replays) is recorded in
`docs/DEEP_AUDIT_VALIDATION.json`. These checks establish tested execution paths, not
universal performance or traffic-safety guarantees.

A further three-round audit is recorded in `docs/REPEATED_AUDIT_VALIDATION.json`:
55 unit tests, all 12 test modules run independently, five audited teacher collections,
six configuration preflights, an interrupted-versus-continuous MPLight training check,
and six exact policy replays from a separately extracted directory.

### Safe retries and matrix resume

A real run records `run_identity.json` before search and holds `.run.lock` while
writing results. Reusing that directory requires identical configuration, source
implementation, teacher trajectories, topology/flow files and traffic configuration.
Keep changed experiments in a new output directory. Existing legacy result directories
without this identity are preserved and refused for automatic reuse. An unchanged
failed run can be retried; `--resume` remains limited to sequential matrices and skips
only complete, mutually consistent result/deployment artifacts. Two matrix entries
that map to the same output path are rejected, including different search seeds with
the same teacher, scenario, flow and simulation seed. Use separate directories for them.

### Failure integrity

Evaluation JSON reports `running`, `failed` or `complete`; aggregate only `complete`
records. A retry invalidates the old success before simulation starts, and result
writes are atomic. Concurrent writers to the same result file are rejected. A hard
process kill may leave `running`, which is not a success. Trajectory collection
requires a new output path and publishes only a complete gzip. Unknown feature
contracts are rejected. The JSON provider overrides stale `LLM_BACKEND` values;
legacy COMP dispatch cannot bypass offline mode. See
`docs/FAILURE_INTEGRITY_VALIDATION.json` for 71 tests and real replay/failure checks.

## Repeatable systematic acceptance audit

The latest consolidated result is `docs/COMPREHENSIVE_REVIEW_VALIDATION.json`:
157 checks, 87 unit tests (18 test files also run independently), fresh two-episode
training and collection for all five teachers on Hangzhou1 and Atlanta, seven
full-budget offline searches, 21 student evaluations, explicit teacher transfer
checks, and training/search resume checks. The frozen source stayed unchanged
throughout that audit. Earlier reports document their own earlier revisions.

```bash
python scripts/audit_release.py --output-dir ../PPL_TSC_Audit_Run
```

Use a new directory outside the source tree. Each command and assertion records its
status; a failure stops the audit and leaves later stages explicitly not run. LLM
text is supplied by deterministic test fixtures with network access forbidden.
Training, verification, calibration, simulation, selection and exports run normally.
This is not a live-provider, full-convergence, GPU, or all-input safety validation.
Previous audit reports describe their respective earlier rounds.

## Statistics validation and action diagnostics

The verifier checks proven frozen `feature_stats` accesses, including literal nested
fields, local aliases and provably unreachable None/empty fallback branches. Raw observation names and
statistical fields are distinct: `feature_stats['green_vehicle']['p95']` is valid
when present; `feature_stats['green_vehicle']['num_vehicle']` is not. A `.get`
default does not make an absent source statistics field valid. Prompt guidance
uses the same contract. This AST check is not a proof covering arbitrary dynamic
key construction or every Python control-flow pattern.

`evaluate.py` now records `policy_diagnostics` by default (`--diagnostics` remains
accepted). Per intersection, at least 20 decisions and a switch rate <= 1% or >= 99%
produce descriptive `near_always_hold` / `near_always_switch` flags. These thresholds
are diagnostic conventions, not traffic safety rules. They do not reject policies
or change fitness, calibration or search budgets. Low demand, phase availability
and other legitimate conditions can explain extreme rates; inspect demand and ATT.
Raw action comparisons use states visited by the deployed policy, not a separate
uncalibrated rollout. Existing artifacts and historical metrics remain unchanged.
