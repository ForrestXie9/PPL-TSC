# Teacher implementations and transfer protocol

## Source and version

The reference is a locally archived [firepd/PI-Light](https://github.com/firepd/PI-Light)
source tree, rather than a fresh checkout of each teacher paper's official repository.
Selected unmodified files are preserved in `reference/pi_light/`. Runtime differences
are recorded in `teacher_*.patch`, with hashes in `TEACHER_IMPLEMENTATION_AUDIT.json`.
The runtime implementations include inherited fixes and are not pristine upstream copies.

Packaging and English documentation edits do not change PPL-TSC's search, calibration,
scoring, prompt structure or teacher parameter-sharing organization. FRAP and EcoLight
retain independent per-intersection training. Results from separate shared-teacher
experiments must not be relabeled as results from these original-family teachers.

## Teacher organization

| Teacher | Parameter organization | Runtime differences from reference | Cross-network transfer |
|---|---|---|---|
| FRAP | Independent network per intersection. | Original organization, with inherited device support; see patch. | Copy source agent 0 weights; rebuild target phase/movement mappings. |
| EcoLight | Independent two-input, two-output keep/switch network per intersection. | Inherited device support and action decoding using the observed physical phase. | Copy source agent 0 weights; rebuild target green/red demand and phase mappings. |
| MPLight | Shared layers and optimizer across intersections, as in the reference. | `mplight.py` is identical to the reference; its FRAP base has device support. | Copy source agent 0 weights; retain native sharing on the target. |
| CoLight | Shared encoding, attention and output layers. | Inherited heterogeneous-phase handling, valid-action masking, observation padding/truncation and gradient clipping; see patch. | Load only when dimensions and feature/action semantics are compatible. Sharing does not establish arbitrary-topology transfer. |
| TinyLight | Independent intersection structures; input/output sizes depend on lanes/phases. | Inherited device and architecture-gradient fixes; this trainer is the unfrozen variant. | The original transfer script does not establish this protocol; this package supports same-network evaluation. |

CoLight's observation padding/truncation is inherited runtime behavior. It can discard
observations; compatible tensor shapes do not establish semantic compatibility or good
control performance. Exact reproduction of an upstream implementation requires reviewing
these differences. The transfer loader itself does not truncate or pad learned tensors
and does not ignore missing or unexpected state-dictionary keys.

## What the original transfer script does

`015_baseline_transfer.py` trains on the source network for 64 episodes, then:

1. Creates target-local agents using the target road network's phase mapping.
2. Copies the learned parameters of `env.n_agent[0].network_local` to each target agent.
3. Runs target inference without training or storing experience.

For independent FRAP/EcoLight teachers, this transfers **source intersection 0's network**.
It does not combine all source networks or introduce shared training. Source agent 0 is
fixed before target evaluation and is not selected using target ATT.

The original script uses parameter-wise zip/copy. The package instead checks every named
tensor and shape before strict loading. Target topology mappings are constructed locally.
Student and teacher must correspond to the same declared source training task. If the
student uses trajectories from all source intersections while the transferred teacher uses
only source agent 0, report this asymmetry explicitly.

## Public evaluation commands

`evaluate_teacher.py --protocol same_network` preserves each source intersection's own
weights and requires the road-network hash to match. Use it for the source flow and
same-network cross-flow evaluation.

`evaluate_teacher.py --protocol source_agent0` uses the fixed source agent 0 on another
network. The checkpoint must match its training audit hash. Added shared-training artifacts
are rejected. Incompatible transfers fail explicitly rather than silently changing the
model. CoLight transfer may be attempted subject to compatibility; TinyLight rejects this
unestablished protocol.

The default trainer uses the corrected 128-episode protocol and selects the checkpoint
with the best training ATT. This differs from the upstream 64-episode transfer script.
`--episodes 64 --selection last` matches that training length and final-checkpoint choice,
but does not remove other implementation or environment differences.

## Included implementations

No added `FRAP_Shared_Teacher_Retrain` or `EcoLight_Shared_Teacher_Retrain` implementation,
loader or collection option is included. Native MPLight/CoLight sharing is retained.

## Recorded interface validation

Same-network cross-flow checks completed for all five teacher families. FRAP, EcoLight and
MPLight completed Hangzhou1-to-Hangzhou2 and Hangzhou1-to-Atlanta interface checks. CoLight
ran on Hangzhou1-to-Hangzhou2; Hangzhou1-to-Atlanta was rejected for incompatible input
dimensions (22 versus 15) and phase outputs (6 versus 5).

These checks used temporary two-episode models and establish tested interfaces, not
performance claims. See `TEACHER_TRANSFER_VALIDATION.json` for that validation record.
