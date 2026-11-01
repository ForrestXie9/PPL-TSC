"""Read-only lane-permission diagnostics from existing source episodes."""
class Recorder:
    def __init__(self):
        self.lanes = {}
        self.intersections = {}
        self.decisions = []
        self.current_controller = None
        self.current_mode = None

    def initialize(self, inter):
        return self.intersections.setdefault(inter.inter_idx, {
            'effective_phase': 0, 'hold_seconds': 0., 'max_hold_seconds': 0.,
            'effective_switches': 0, 'logical_transition_seconds': 0.,
            'green_seconds_by_phase': {}, 'requests': 0, 'request_switches': 0,
            'bias_changed_requests': 0, 'waiting_lane_ticks': 0,
            'waiting_with_green_ticks': 0, 'waiting_without_green_ticks': 0})

    def tick(self, inter, interval):
        # Native Intersection.step does not command a yellow lamp on entering
        # logical -1: the old engine phase remains until the next set_tl_phase.
        # Track effective phase from those original command sites, not -1.
        row = self.initialize(inter)
        if inter.current_phase >= 0:
            if inter.current_phase != row['effective_phase']:
                row['effective_switches'] += 1
                row['hold_seconds'] = 0.
            row['effective_phase'] = inter.current_phase
        else:
            row['logical_transition_seconds'] += interval
        row['hold_seconds'] += interval
        row['max_hold_seconds'] = max(row['max_hold_seconds'], row['hold_seconds'])
        phase = row['effective_phase']
        row['green_seconds_by_phase'][phase] = row['green_seconds_by_phase'].get(phase, 0.) + interval
        permitted = set(inter.n_phase[phase].n_available_startlane_id)
        serviceable = set(lane for phase in inter.n_phase for lane in phase.n_available_startlane_id)
        waiting = inter.eng.get_lane_waiting_vehicle_count()
        for lane in inter.n_in_lane_id:
            if lane not in serviceable:
                continue
            key = (inter.inter_idx, lane)
            clock = self.lanes.setdefault(key, {'without_green': 0., 'max_without_green': 0.,
                                                'waiting': 0., 'max_waiting': 0.})
            has_waiting = waiting.get(lane, 0) > 0
            clock['waiting'] = clock['waiting'] + interval if has_waiting else 0.
            clock['max_waiting'] = max(clock['max_waiting'], clock['waiting'])
            denied = has_waiting and lane not in permitted
            clock['without_green'] = clock['without_green'] + interval if denied else 0.
            clock['max_without_green'] = max(clock['max_without_green'], clock['without_green'])
            row['waiting_lane_ticks'] += has_waiting
            row['waiting_with_green_ticks'] += has_waiting and lane in permitted
            row['waiting_without_green_ticks'] += denied

    def summarize(self):
        return {'per_intersection': self.intersections,
                'max_continuous_waiting_without_green_seconds': max((v['max_without_green'] for v in self.lanes.values()), default=0.),
                'max_continuous_waiting_seconds': max((v['max_waiting'] for v in self.lanes.values()), default=0.),
                'max_effective_phase_hold_seconds': max((v['max_hold_seconds'] for v in self.intersections.values()), default=0.),
                'effective_switches': sum(v['effective_switches'] for v in self.intersections.values()),
                'requests': sum(v['requests'] for v in self.intersections.values()),
                'bias_changed_requests': sum(v['bias_changed_requests'] for v in self.intersections.values()),
                'top_waiting_lanes': [dict(intersection=k[0], lane=k[1], **v) for k, v in
                                      sorted(self.lanes.items(), key=lambda x: x[1]['max_without_green'], reverse=True)[:10]],
                'limitations': ['Lane permission is not measured departure or turn-specific service.',
                                'Continuous waiting may involve different vehicles on the same lane.',
                                'Effective lamp phase is inferred from original set_tl_phase command sites; logical yellow is not a commanded yellow lamp in this runtime.']}

