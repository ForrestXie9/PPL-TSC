"""Source episode diagnostics; no extra simulator runs or fitness terms."""
import copy

def install_observer():
    from .lane_recorder import Recorder
    from env.intersection import Intersection
    from LLM_SR_DISTILLATION_method_v2.evaluator import Evaluator
    original = Evaluator._compute_student_travel_time_simulation
    def simulate(self, utility_function, parameters, per_intersection_bias=None, raise_on_error=True):
        observer = Recorder()
        active = {'intersection': None}
        last = {}
        snapshots = {}
        original_step = Intersection.step
        original_run = self.simulator.run_episode
        original_select = utility_function.select_action
        own_run = self.simulator.__dict__.get('run_episode')
        own_select = utility_function.__dict__.get('select_action')
        def select(states, *args, **kwargs):
            result = original_select(states, *args, **kwargs)
            if active['intersection'] is not None:
                inter = active['intersection']
                last[inter] = {'states': copy.deepcopy(states), 'requested_phase': int(result)}
            return result
        def run(policy):
            def wrapped(env, inter):
                active['intersection'] = inter
                try:
                    return policy(env, inter)
                finally:
                    active['intersection'] = None
            return original_run(wrapped)
        def step(inter, action, interval):
            original_step(inter, action, interval)
            observer.tick(inter, interval)
            for lane in inter.n_in_lane_id:
                clock = observer.lanes.get((inter.inter_idx, lane))
                if clock is None:
                    continue
                if clock['without_green'] <= 0 or inter.inter_idx not in last:
                    continue
                key = (inter.inter_idx, lane)
                old = snapshots.get(key)
                if old is None or clock['without_green'] > old['continuous_wait_without_green_seconds']:
                    snapshots[key] = dict(intersection=inter.inter_idx, lane=lane,
                        continuous_wait_without_green_seconds=clock['without_green'],
                        source_tick_time=float(inter.eng.get_current_time()),
                        **last[inter.inter_idx])
                if len(snapshots) > 12:
                    del snapshots[min(snapshots, key=lambda k: snapshots[k]['continuous_wait_without_green_seconds'])]
        Intersection.step = step
        self.simulator.run_episode = run
        utility_function.select_action = select
        try:
            result = original(self, utility_function, parameters, per_intersection_bias, raise_on_error)
        finally:
            Intersection.step = original_step
            if own_run is None:
                del self.simulator.run_episode
            else:
                self.simulator.run_episode = own_run
            if own_select is None:
                del utility_function.select_action
            else:
                utility_function.select_action = own_select
        summary = observer.summarize()
        examples = []
        stats = getattr(self.feature_extractor, 'feature_stats', None)
        for row in sorted(snapshots.values(), key=lambda r: r['continuous_wait_without_green_seconds'], reverse=True)[:3]:
            states = row.pop('states')
            phase_bias = (per_intersection_bias or {}).get(row['intersection'],
                           (per_intersection_bias or {}).get(str(row['intersection']), {})) or {}
            from .decision_values import decision_values
            raw = decision_values(utility_function, states, stats)
            if utility_function.action_mode == 'binary_keep_switch':
                cur = next(iter(states.values()))['current_phase']
                nxt = next((p for p in raw if p != cur), cur)
                view = {'current_phase': cur, 'next_phase': nxt,
                        'binary_features': next(iter(states.values())).get('binary_features')}
            else:
                view = states
            adjusted = {p: q + phase_bias.get(p, phase_bias.get(str(p), 0.)) for p, q in raw.items()}
            from .joint_response import inspect_response
            try:
                row['joint_response'] = inspect_response(utility_function, states, stats, phase_bias)
            except Exception as exc:
                row['joint_response'] = {'status': 'incomplete', 'error': type(exc).__name__ + ': ' + str(exc), 'no_hard_rejection': True}
            row.update(source_decision_state=view, raw_utilities=raw, adjusted_utilities=adjusted,
                       phase_bias=phase_bias, raw_choice=max(raw, key=raw.get),
                       adjusted_choice=max(adjusted, key=adjusted.get))
            if row['adjusted_choice'] != row['requested_phase']:
                raise ValueError('Lane diagnostic recomputed action differs from recorded policy request')
            examples.append(row)
        self._source_rollout_feedback['lane_permission_diagnostic'] = dict(
            summary=summary, examples=examples, extra_episodes=0, extra_teacher_queries=0,
            source_only=True, sampling='Per engine tick; utility state from most recent decision (at most one action interval earlier).')
        return result
    Evaluator._compute_student_travel_time_simulation = simulate
