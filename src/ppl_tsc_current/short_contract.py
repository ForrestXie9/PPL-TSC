"""Keep all generation constraints in one mode-specific block."""
import json

ACTIVE_PHASE_SCALE_REMINDER = 'Keeping the current phase can reduce switching losses, but its influence should match traffic-score differences; after normalization, avoid masking higher traffic demand in competing phases.'

def build(generator):
    from .concise_contract import raw_keys,statistics
    mode=generator._decision_mode()
    if mode not in {'phase_argmax','keep_switch','binary_keep_switch'}:raise ValueError(mode)
    binary=mode=='binary_keep_switch';s=statistics(generator,binary)
    lines=['EXECUTABLE DATA CONTRACT:',
      'Function: compute_utility(state, phase_id=None, feature_stats=None).']
    if binary:
        lines += [
          'Interface: evaluate the same state for actions0=KEEP and1=SWITCH to the cyclic next phase; switch iff Q_SWITCH>Q_KEEP, otherwise KEEP. The template candidate phase means binary action here.',
          'state={current_phase:int,next_phase:int,binary_features:{green_vehicle:float,red_vehicle:float}}. Traffic values are nonnegative exact green/red lane totals, not direction means; phase_id is the action ID, not the physical phase number.'
        ]
        semantics='Increasing green_vehicle must not decrease Q_KEEP-Q_SWITCH; increasing red_vehicle must not decrease Q_SWITCH-Q_KEEP, with other inputs fixed. Branch on KEEP/SWITCH, preserving these meanings under physical-phase relabelling; SWITCH does not serve every red direction.'
    else:
        keys=raw_keys(generator)
        lines += [
          'Interface: evaluate each physical candidate phase_id:int, then argmax. state={current_phase:int,controlled_directions:list[str],features_by_direction:dict[str,dict[str,float]]}. Direction traffic fields: '+','.join(keys)+'.'
        ]
        if mode=='keep_switch':
            lines[-1]=lines[-1].replace('evaluate each physical candidate phase_id:int, then argmax.', 'evaluate physical phase_id:int only for the current phase and its cyclic successor; switch iff Q(next)>Q(current), otherwise keep. Do not select other phases.')
        if 'pressure' in keys:lines.append('pressure is signed movement/lanelink pressure, not a vehicle count.')
        if any(k!='pressure' for k in keys):lines.append('Vehicle/waiting fields are nonnegative direction means of distinct compatible incoming lanes. Shared lanes overlap across directions; sums are not unique vehicle totals or observed turn counts.')
        semantics='Increasing available controlled traffic must not decrease Q, other inputs fixed; signed pressure uses NUMERIC order (more-negative is not increased demand). Zeroing vehicle/waiting values and positive pressure, retaining nonpositive pressure, must not increase Q. Making a phase active must not lower Q; do not force inactive phases to zero. Reordering the same collection of controlled directions or consistently relabelling physical phase IDs must not change raw Q; names, positions and arbitrary IDs identify inputs, not traffic priorities.'
        if 'pressure' not in keys:semantics=semantics.replace('; signed pressure uses NUMERIC order (more-negative is not increased demand)','')
        if {'waiting_time','sum_waiting_time'}<=set(keys):semantics+=' waiting_time/sum_waiting_time aliases vary together.'
    lines += [
      'feature_stats: optional frozen SOURCE dict. min/max=empirical extrema; mean/std/count, p50/p75/p90/p95/p99 and zero_fraction describe the same observation pool.',
      ('Statistics describe exact green/red totals.' if binary else 'Statistics describe raw direction measurements, not phase aggregates. phase_active is binary support (0/1,mean0.5,std0.5), not observed frequencies.')
    ]
    if s['aliases']:lines.append('Raw-measurement aliases (not normalized inputs): '+json.dumps(s['aliases'],sort_keys=True,separators=(',',':')))
    elif not binary and any(k.endswith(('_raw','_norm')) for k in s['available']):lines.append('Raw/norm names denote raw-measurement distributions, not normalized inputs.')
    lines.append('SOURCE STAT VALUES: '+json.dumps(s['values'],sort_keys=True,separators=(',',':'),allow_nan=False))
    lines += [
      ACTIVE_PHASE_SCALE_REMINDER,
      'CONSTRAINTS:',
      '- Feature use: compute utility only from declared inputs and fixed source statistics; identical inputs must produce identical outputs. Read fields/statistics with safe .get defaults; invent no missing keys, derived phase_features or normalized inputs. Statistics are feature_stats[feature][statistic], e.g. green_vehicle/p95; raw field names such as num_vehicle are not statistic fields. Safe defaults must not hide misspelled or absent source statistic keys. Keep source statistics fixed on targets. Normalization inside f is optional; guard missing/constant ranges and out-of-source values (Min-Max may be <0 or >1).'+(' Do not normalize phase_active.' if not binary else ''),
      '- Traffic semantics: '+semantics+' Traffic-dependent response is required; constants/holding-only rules are insufficient. Unused individual inputs, equality and saturation are allowed; no minimum slope, holding bonus or forced switching is required.',
      '- Numeric safety: import NumPy as np if used; initialize Q=0.0; return float(Q) if np.isfinite(Q) else 0.0. All intermediates/source bundles must be finite; the fallback does not excuse invalid arithmetic. Guard possible zero divisors, remainder, floor division, reciprocal and negative powers. Signed ratio: a/d if abs(d)>1e-8 else 0.0; magnitude scaling: a/max(abs(d),1e-8); adding1/epsilon to signed values is insufficient. sqrt argument>=0, log argument>0; negative-base fractional powers are invalid. Guard transformed inputs and keep code readable.',
      'Calibration: modify only raw f. Each candidate separately fits external offsets per SOURCE intersection/physical phase in [-0.5,0.5] to teacher action agreement, not ATT.'+(' KEEP/SWITCH uses current/next physical-phase offsets.' if mode in {'binary_keep_switch','keep_switch'} else '')+' Parent ATT/ABF are AFTER calibration; offsets may reverse rankings. Verifier checks raw utilities; f/offsets stay fixed for same-network transfer.',
      'Checks/diagnostics sample source/boundary states, not causal traffic effects or all possible states.',
      'Output: one Python code block with ### Function, complete executable function, then ### Explanation and2–3 rationale sentences.'
    ]
    return '\n'.join(lines)
