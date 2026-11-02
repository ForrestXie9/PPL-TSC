"""CMA-ES calibration of source intersection/phase offsets to teacher agreement."""

from typing import Dict, Optional
import numpy as np
import hashlib
import json
from ppl_tsc.runtime.utility import FunctionUtility
from ppl_tsc.config import CALIBRATION


def fit_biases(
    self,
    utility_function: "FunctionUtility",
    max_timesteps: Optional[int] = None,
    max_iter: int = CALIBRATION.max_iterations,
    bias_bound: float = CALIBRATION.bias_bound,
    verbose: bool = True,
) -> Dict[int, Dict[int, float]]:
    """
    Calibration stage: fit per-intersection, per-phase additive bias to maximize behavioral fidelity.

    For each intersection, optimize independent bias values for each traffic signal phase.
    The shared utility function remains unchanged; only per-phase biases are tuned.
    Final Q-value: Q_adjusted = Q_raw(from utility function) + bias[phase_id]

    Optimization objective: maximize behavioral fidelity (fraction of correct argmax matches)
    This directly optimizes which action is selected, not just probability distributions.

    Args:
        utility_function: FunctionUtility instance (shared compute_utility function)
        max_timesteps: Max timesteps for action data (None = use all)
        max_iter: Maximum CMA-ES iterations per intersection
        bias_bound: Bound for each bias value (e.g. 5.0 -> bias in [-5, 5])
        verbose: Whether to print progress

    Returns:
        per_intersection_bias: {inter_id: {phase_id: bias_float}}
            - Outer dict: one entry per intersection
            - Inner dict: one bias value per phase (typically 4 phases: 0, 1, 2, 3)
    """
    if not np.isfinite(bias_bound) or bias_bound <= 0:
        raise ValueError("bias_bound must be finite and positive")
    if isinstance(max_iter, bool) or not isinstance(max_iter, (int, np.integer)) or max_iter <= 0:
        raise ValueError("max_iter must be a positive integer")
    import cma  # type: ignore

    # Require a controller interface that can apply phase_bias during action selection.
    if not callable(getattr(utility_function, "select_action", None)):
        if verbose:
            print(
                "    fit_per_intersection_bias: controller has no select_action, returning empty bias"
            )
        return {}
    self.assert_utility_runtime(utility_function)
    feature_extractor = (
        self.feature_extractor.extract if hasattr(self.feature_extractor, "extract") else None
    )
    per_intersection_bias: Dict[int, Dict[int, float]] = {}

    intersection_ids = sorted(self.data_loader.intersection_ids)
    verbose_samples = False
    self.calibration_details = {}

    for inter_id in intersection_ids:
        action_data = self.data_loader.get_action_consistency_data(
            inter_id, max_timesteps=max_timesteps, include_timestep=True
        )
        if not action_data:
            per_intersection_bias[inter_id] = {}
            if verbose:
                print(f"    Intersection {inter_id}: no action_data available, skipping")
            continue

        # Phase IDs from first sample
        state_dict_0 = action_data[0][0]
        phase_ids = sorted(state_dict_0.keys())
        n_phases = len(phase_ids)
        if n_phases == 0:
            per_intersection_bias[inter_id] = {}
            if verbose:
                print(f"    Intersection {inter_id}: no phases in state_dict, skipping")
            continue

        if verbose:
            print(f"    Intersection {inter_id}: {len(action_data)} samples, {n_phases} phases")
            try:
                unmodified_consistency = self._compute_action_consistency_with_data(
                    utility_function,
                    {},
                    action_data,
                    feature_extractor,
                    phase_bias=None,
                    debug_counts=False,
                    raise_on_error=False,
                )
                print(f"      Unmodified consistency: {unmodified_consistency:.4f}")
            except Exception as e:
                print(f"      Unmodified consistency: failed ({type(e).__name__}: {e})")

        if (
            verbose
            and verbose_samples
            and action_data
            and hasattr(utility_function, "compute_all_phases")
            and callable(utility_function.compute_all_phases)
        ):
            state_dict, teacher_action = action_data[0][:2]
            q_values = utility_function.compute_all_phases(state_dict, feature_extractor)
            student_action = max(q_values.items(), key=lambda x: x[1])[0] if q_values else 0
            q_display = ", ".join([f"{pid}:{q:.3f}" for pid, q in sorted(q_values.items())])
            print(f"      All Q values: {{{q_display}}}")
            print(
                f"      Sample: teacher={teacher_action}, student={student_action}, match={teacher_action==student_action}"
            )

            # Show actual state structure to diagnose function issues
            print("      State structure for phase 0:")
            phase0_state = state_dict.get(0, {})
            if isinstance(phase0_state, dict):
                for key in list(phase0_state.keys())[:5]:  # First 5 keys
                    val = phase0_state[key]
                    if isinstance(val, dict):
                        print(f"        {key}: {{...}} (dict with {len(val)} items)")
                    else:
                        print(f"        {key}: {val} ({type(val).__name__})")
            print(f"      Current phase: {state_dict.get(0, {}).get('current_phase', '?')}")

        cached_bias_consistency = self._build_cached_bias_consistency(
            utility_function,
            action_data,
        )

        def _consistency_from_x(x: np.ndarray) -> float:
            phase_bias = dict(zip(phase_ids, x.tolist()))
            try:
                consistency = cached_bias_consistency(phase_bias)
                if np.isfinite(consistency):
                    return float(consistency)
            except Exception:
                pass
            return 0.0

        evaluated = []

        def objective(x: np.ndarray) -> float:
            phase_bias = dict(zip(phase_ids, x.tolist()))
            try:
                # Optimize action_consistency directly (negative because minimize)
                # This optimizes argmax matching, not just probability distribution
                consistency = cached_bias_consistency(phase_bias)
                # Return negative (scipy minimizes, we want to maximize consistency)
                obj_value = -float(consistency) if np.isfinite(consistency) else 1e10
                evaluated.append((np.asarray(x, dtype=float).copy(), obj_value))
                return obj_value
            except Exception as e:
                if verbose:
                    print(f"      objective error: {e}")
                return 1e10

        x0 = np.zeros(n_phases)
        try:
            # Evaluate objective at initial point
            initial_obj = objective(x0)
            initial_consistency = -initial_obj if initial_obj < 1e9 else 0.0
            if verbose:
                print(f"      Initial consistency: {initial_consistency:.4f}")
            popsize = CALIBRATION.population_size

            # CMA-ES optimization (more stable for non-smooth objectives)
            x0 = np.zeros(n_phases, dtype=float)
            sigma0 = max(1e-6, bias_bound * CALIBRATION.sigma_ratio)
            cma_opts = {
                "bounds": [[-bias_bound] * n_phases, [bias_bound] * n_phases],
                "popsize": popsize,
                "maxiter": max_iter,
                "verbose": -9,
                # CMA-ES otherwise seeds itself from wall-clock entropy.
                "seed": (self.random_seed + int(inter_id)) % 2147483646 + 1,
            }

            es = cma.CMAEvolutionStrategy(x0, sigma0, cma_opts)
            iter_state = {"i": 0, "last": None, "best": float("-inf")}

            while not es.stop() and iter_state["i"] < max_iter:
                solutions = es.ask()
                values = [objective(x) for x in solutions]
                es.tell(solutions, values)

                iter_state["i"] += 1
                best_idx = int(np.argmin(values))
                consistency = -float(values[best_idx]) if np.isfinite(values[best_idx]) else 0.0
                if consistency > iter_state["best"]:
                    iter_state["best"] = consistency

                # Per-iteration output disabled for consistency (was only for first_intersection)
                iter_state["last"] = consistency

            optimized_bias = es.result.xbest if es.result.xbest is not None else x0
            optimized_bias = np.clip(
                np.array(optimized_bias, dtype=float), -bias_bound, bias_bound
            ).tolist()

            from ppl_tsc.calibration.ties import choose_tied_bias

            trace_digest = hashlib.sha256(
                json.dumps([(x.tolist(), v) for x, v in evaluated], allow_nan=False).encode()
            ).hexdigest()
            optimized_bias = choose_tied_bias(
                evaluated, cached_bias_consistency.samples, phase_ids
            ).tolist()
            chosen_abf = cached_bias_consistency(dict(zip(phase_ids, optimized_bias)))
            best_observed_abf = max(-value for _, value in evaluated)
            self.calibration_details[str(inter_id)] = dict(
                population=20,
                maxiter=max_iter,
                iterations=iter_state["i"],
                cma_evaluations=int(es.result.evaluations),
                sigma0=sigma0,
                bound=bias_bound,
                trace_sha256=trace_digest,
                best_observed_abf=best_observed_abf,
                chosen_abf=chosen_abf,
                tie_rule="least_changes",
            )

            class _ResShim:
                def __init__(self, xbest, fbest, nfev, success):
                    self.x = xbest
                    self.fun = fbest
                    self.nfev = nfev
                    self.success = success

            stop_reasons = es.stop()
            success = not (
                stop_reasons and ("maxiter" in stop_reasons or "maxfevals" in stop_reasons)
            )
            res = _ResShim(
                xbest=np.array(optimized_bias, dtype=float),
                fbest=-chosen_abf,
                nfev=int(es.result.evaluations) if es.result.evaluations is not None else 0,
                success=success,
            )

            # Extract optimized bias values (ALWAYS use res.x, not x0)
            optimized_bias = res.x.tolist()

            if verbose:
                # Detailed diagnostic: show optimization progress
                final_consistency = -res.fun if res.fun < 1e9 else 0.0
                initial_consistency = -initial_obj if initial_obj < 1e9 else 0.0
                total_actions = len(action_data)
                initial_steps = (
                    int(round(initial_consistency * total_actions)) if total_actions > 0 else 0
                )
                final_steps = (
                    int(round(final_consistency * total_actions)) if total_actions > 0 else 0
                )
                bias_values = ", ".join(
                    [f"P{pid}:{val:+.3f}" for pid, val in zip(phase_ids, optimized_bias)]
                )
                nfev = getattr(res, "nfev", "?")
                print(
                    f"    Intersection {inter_id}: "
                    f"consistency {initial_consistency:.4f}({initial_steps}/{total_actions})->"
                    f"{final_consistency:.4f}({final_steps}/{total_actions}), "
                    f"bias=[{bias_values}], success={res.success}, nfev={nfev}"
                )

            # A tie provides no fidelity benefit: retain the original policy.
            final_consistency = -res.fun if res.fun < 1e9 else 0.0
            initial_consistency = -initial_obj if initial_obj < 1e9 else 0.0
            if final_consistency <= initial_consistency:
                if verbose:
                    print(
                        f"    Intersection {inter_id}: consistency did not improve "
                        f"({initial_consistency:.4f}->{final_consistency:.4f}), revert bias to zeros"
                    )
                per_intersection_bias[inter_id] = dict(zip(phase_ids, [0.0] * n_phases))
            else:
                per_intersection_bias[inter_id] = dict(zip(phase_ids, optimized_bias))
        except Exception as e:
            raise RuntimeError(f"Calibration failed at {inter_id}: {e}") from e

    return per_intersection_bias
