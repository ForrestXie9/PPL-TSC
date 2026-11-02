"""Diagnostic values with the controller's feasible actions and tie order."""


def decision_values(utility, states, stats):
    if utility.action_mode == "binary_keep_switch":
        cur, nxt, keep, switch = utility.compute_binary_action_values(
            states, feature_stats=stats, raise_on_error=True
        )
        return {cur: float(keep), nxt: float(switch)}
    if utility.action_mode == "keep_switch":
        ids = utility._sorted_phase_ids(states)
        if not ids:
            raise ValueError("Cannot diagnose an empty action state")
        current = states[ids[0]].get("current_phase", ids[0])
        if current not in states:
            current = ids[0]
        following = ids[(ids.index(current) + 1) % len(ids)]
        # KEEP wins a tie, exactly like select_action.
        return {
            p: float(
                utility.compute(states[p], phase_id=p, feature_stats=stats, raise_on_error=True)
            )
            for p in dict.fromkeys([current, following])
        }
    return utility.compute_all_phases(states, feature_stats=stats, raise_on_error=True)
