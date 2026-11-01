"""Public version of the uniformly evaluated full-method configuration."""
import json
import os
from pathlib import Path

PROFILE='full_100_256_v1'
PROMPT_TEMPLATE='paper_figure_a2_v1'
LANE_MARKER='SOURCE JOINT RESPONSE FEEDBACK v1'
SHARED_LANE_DESCRIPTION="""COMPATIBLE SHARED-LANE DEMAND CONTRACT v1
Vehicle-count and waiting measurements use compatible incoming lanes: each lane contributes once to every movement direction allowed by its roadlinks; the direction reducer is the mean of these distinct lane measurements. Shared-lane values overlap across directions. These are potential demand measurements, NOT observed turn-specific counts, and summing directions is not a unique vehicle total. A lane with multiple roadlinks to the SAME direction is counted once. Roadlink ordering does not choose which direction receives its measurements.
Only documented available raw fields may be read. Pressure retains its existing movement/lanelink construction and signed values; do not reinterpret it as a vehicle count. Native KEEP/SWITCH reads only unchanged binary_features.green_vehicle/red_vehicle; direction mapping does not alter those exact binary inputs. Only available measured source statistics are included in feature_stats using the existing aggregation/statistic rules. phase_active statistics describe declared binary support (0/1, mean 0.5, std 0.5), not measured source frequencies; unsupported teacher observation names and their statistics are omitted. Normalization remains optional inside the program. Do not invent turning ratios, target statistics, missing features or unique demand totals.
An active-phase preference may reduce unnecessary switching but its utility increment can overpower competing demand. No active term is banned or required, no coefficient or output scale is prescribed. All existing feature constraints, calibration and fitness remain unchanged."""


def control_score_description():
    return 'Control score is exp(min(Delta, 0)), with Delta = (teacher ATT - student ATT) / (abs(teacher ATT) + 1e-6). It is capped at 1 when student ATT is equal to or lower than teacher ATT; worse ATT produces a score below 1. Preserve the joint ABF-plus-control objective.'


def validate_config(config):
    profile=config.get('method',{}).get('profile')
    if profile is None:
        return
    if profile!=PROFILE:
        raise ValueError('Unknown method profile: '+str(profile))
    if config.get('method',{}).get('prompt_template',PROMPT_TEMPLATE)!=PROMPT_TEMPLATE:
        raise ValueError('Current full profile requires prompt_template='+PROMPT_TEMPLATE)
    search=config['search'];evaluation=config['evaluation']
    expected={'population_size':5,'generations':5,'constraint_retries':3,'max_initial_attempts':30,'fit_phase_bias':True,'bias_fit_iterations':20,'bias_bound':.5,'slot_based_parent':True}
    for key,value in expected.items():
        if search.get(key)!=value:
            raise ValueError(f'{PROFILE} requires search.{key}={value!r}')
    if evaluation.get('action_weight')!=1. or evaluation.get('performance_weight')!=1. or evaluation.get('score_mode')!='action_plus_perf':
        raise ValueError('Full method requires ABF plus exponential control fitness capped at 1')
    if evaluation.get('simulation_steps') is not None or evaluation.get('max_timesteps') is not None:
        raise ValueError('Full method inherits the complete source horizon and teacher observations')


def configure(config):
    validate_config(config)
    if config.get('method',{}).get('profile')!=PROFILE:
        return
    os.environ.update(PPL_METHOD_PROFILE=PROFILE,PPL_PROMPT_TEMPLATE=PROMPT_TEMPLATE,PPL_SOURCE_PERCENTILES='1',PPL_CMA_TIE_RULE='least_changes',PPL_CMA_SIGMA_RATIO=str(1/3),PPL_ADAPTIVE_MARGIN='0',PPL_SEARCH_FEEDBACK='baseline',PPL_PHASE_BIAS_FEEDBACK='baseline')


def append_trace(name,row):
    directory=os.environ.get('PPL_SOURCE_DIAGNOSTIC_DIR')
    if directory:
        path=Path(directory)/name;path.parent.mkdir(parents=True,exist_ok=True)
        with path.open('a') as stream:
            stream.write(json.dumps(row,allow_nan=False)+'\n')


def failed_parent_feedback(metrics):
    """Keep a failed candidate repairable without inventing rollout metrics."""
    error = (metrics or {}).get('error')
    if not error:
        raise ValueError('Missing source lane diagnostic after evaluation; no recorded failure explains it')
    return ('SOURCE EVALUATION FAILURE\n' + str(error) +
            '\nThis parent did not complete source evaluation. Its ABF, ATT and lane '
            'diagnostics are unavailable; do not interpret them as zero or infer traffic '
            'behavior from this failure. Repair the recorded execution issue while '
            'preserving the documented feature/action interface and traffic constraints. '
            'The offspring must pass the usual review, calibration and source evaluation. '
            'Keep the original objectives and candidate budget.')


def install():
    if os.environ.get('PPL_METHOD_PROFILE')!=PROFILE:
        return
    from LLM_SR_DISTILLATION_method_v2.evolution import Evolution
    if getattr(Evolution,'_public_current_profile_installed',False):
        return
    Evolution._public_current_profile_installed=True
    from LLM_SR_DISTILLATION.llm_generator import LLMGenerator
    for method_name in ('_build_data_contract_text','_build_binary_data_contract_text'):
        old=getattr(LLMGenerator,method_name)
        def contract(self,old=old,binary=method_name=='_build_binary_data_contract_text'):
            text=old(self)
            text+='\n- Every original source bundle is executed for finite raw utilities before calibration; the source raw values may be reused in calibration. Finite perturbation tests are not an all-state proof.'
            if not binary:
                text+='\n- Zero-positive-demand comparison zeros vehicle/waiting fields and positive pressure, preserves nonpositive pressure, phase_id and current_phase, and requires Q_zero_positive <= Q_observed. Holding only traffic fixed, making this phase active must not lower its utility. No minimum holding bonus, switching frequency, fairness or waiting threshold is imposed.'
            return text
        setattr(LLMGenerator,method_name,contract)
    from .observer import install_observer
    install_observer()
    original = Evolution._build_parent_info
    def parent(self, individual):
        text = original(self, individual)
        # Frozen uniform bias range used in the completed full-method study.
        text = text.replace('Each bias is bounded by +/-1, so a pairwise bias difference can reach 2.',
                            'Each bias is bounded by +/-0.5, so a pairwise bias difference can reach 1.')
        lane = (individual.metrics or {}).get('source_rollout_feedback', {}).get('lane_permission_diagnostic')
        appendix = None
        if lane is None:
            appendix = failed_parent_feedback(individual.metrics)
            append_trace('semantic_feedback_trace.jsonl', {
                'profile': PROFILE, 'lane_feedback_appended': False,
                'source_evaluation_failed': True, 'appendix': appendix,
                'parent_lane_summary': None})
            return text + '\n' + appendix
        compact = {k: lane['summary'][k] for k in ['max_continuous_waiting_without_green_seconds', 'max_continuous_waiting_seconds', 'max_effective_phase_hold_seconds']}
        appendix = (LANE_MARKER + '\nExisting parent SOURCE episode only: ' + json.dumps(compact) +
            '\nThese examples include only lanes with at least one available serving phase. Demand-response probes are counterfactuals bounded by source feature ranges, not a causal simulation or proof of permanent lock. A continuously waiting lane without green permission is a service-exclusion signal; permission is not departure, and a shared lane is not turn-specific demand. Long holding without unmet demand can be appropriate. No switching target, hard hold threshold, safety penalty or extra evaluation episode is imposed.\n' +
            '\n'.join('Source lane example: ' + json.dumps(e, allow_nan=False) for e in lane['examples'][:3]) +
            '\nCompare the same-state raw and adjusted utilities with parent ATT and ABF. Offsets may legitimately reverse rankings. An unchanged action alone is not a violation. Inspect source ATT, ABF and the full decision values together. Do not hard-code lane/intersection ids, times, or the examples. Multiple structural changes are allowed. Keep the original scoring/calibration objectives, feature interface and budgets.')
        text += '\n' + appendix
        append_trace('semantic_feedback_trace.jsonl', {'profile': PROFILE, 'lane_feedback_appended': appendix is not None,
                     'appendix': appendix, 'parent_lane_summary': lane['summary'] if lane else None})
        return text
    Evolution._build_parent_info = parent
    from .paper_prompts import install as install_paper_prompts
    install_paper_prompts()

    from .yield_prompts import install as install_yield_prompts
    install_yield_prompts()
