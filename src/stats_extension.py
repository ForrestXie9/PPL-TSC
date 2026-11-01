"""Optional source statistics enrichment; old fields and the observation pool stay intact."""
import os
import numpy as np

MARKER='FROZEN SOURCE FEATURE DISTRIBUTION v1'
FIELDS=['p50','p75','p90','p95','p99','zero_fraction']
DESCRIPTION='''FROZEN SOURCE FEATURE DISTRIBUTION v1
Additional feature_stats fields are optional frozen SOURCE per-direction empirical statistics: p50, p75, p90, p95 and p99 are the indicated percentiles, computed from the SAME observation pool as min/max/mean/std/count; zero_fraction is the fraction exactly zero. The original fields are unchanged. These are not summed-phase statistics, physical limits, guaranteed bounds, target measurements or teacher action labels. Phase_active remains a binary indicator and has no percentile fields.
If useful, the program may use these fields to understand common magnitudes versus source extremes, or to choose its own finite scaling. No normalization, quantile, clipping, utility scale, formula or phase-active bonus is prescribed. Percentiles can be zero or coincide; a positive gap is not guaranteed. Guard every denominator and any transformed expression. Values in a later flow can lie outside every source percentile; source statistics stay frozen and must never be refitted to the target. Preserve the same documented raw feature keys and all existing constraints, teacher-agreement calibration, bias range and two-term scoring formula.'''


def extend(stats,values_by_kind):
    flag=os.environ.get('PPL_SOURCE_PERCENTILES','0')
    if flag not in {'0','1'}:raise ValueError('Unknown source-percentile setting')
    if flag=='0':return stats
    for kind,values in values_by_kind.items():
        if not values:continue
        array=np.asarray(values,dtype=float)
        if not np.isfinite(array).all():raise ValueError('Nonfinite source percentile input')
        row=stats[kind]
        assert int(row['count'])==array.size
        assert row['min']==float(array.min()) and row['max']==float(array.max())
        result=np.percentile(array,[50,75,90,95,99],method='linear')
        row.update({key:float(value) for key,value in zip(FIELDS[:5],result)})
        row['zero_fraction']=float(np.mean(array==0))
    return stats

BINARY_DESCRIPTION='''FROZEN SOURCE FEATURE DISTRIBUTION v1
Optional p50, p75, p90, p95, p99 and zero_fraction describe the SAME frozen SOURCE observation pool as each entry's min/max/mean/std/count; all old fields and aggregation weights are unchanged.
For green_vehicle and red_vehicle these are statistics of the exact binary teacher inputs: each decision/intersection contributes one green-lane total and one red-lane total. They are NOT per-direction statistics or teacher action labels. Other metadata aliases, if present, retain their original per-direction meanings; traffic for this native head must still be read only from binary_features.green_vehicle and binary_features.red_vehicle.
The program may use these fields to understand common source magnitudes versus extremes, or choose its own finite scaling. No normalization, quantile, clipping, utility scale, formula or bias change is prescribed. Percentiles can be zero or coincide; guard every denominator and transformed expression. These empirical source statistics are not physical limits or guaranteed bounds, and must remain frozen in other flows. Preserve the two binary action IDs (0 KEEP, 1 SWITCH), all existing constraints, teacher-agreement calibration, bias range and two-term scoring formula.'''
