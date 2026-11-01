"""Observe existing decisions; both study arms use identical instrumentation."""
from collections import Counter
from source_rollout_legacy import SourceRolloutRecorder as LegacyRecorder,render_rollout
from source_phase_summary import summarize,decompose


class SourceRolloutRecorder(LegacyRecorder):
    def __init__(self,*args,**kwargs):
        super().__init__(*args,**kwargs)
        self.phase_counts={};self.phase_changes={};self.phase_longest={}
        self.phase_matches={};self.phase_legal={};self.phase_last={};self.phase_run={}

    def _observe(self,intersection,states,requested_action):
        super()._observe(intersection,states,requested_action)
        requested=int(requested_action)
        current=int(next(iter(states.values()))['current_phase'])
        assert requested in states
        if intersection not in self.phase_counts:
            self.phase_counts[intersection]=Counter();self.phase_changes[intersection]=0
            self.phase_longest[intersection]=0;self.phase_matches[intersection]=0
            self.phase_legal[intersection]=set();self.phase_run[intersection]=0
        self.phase_counts[intersection][requested]+=1
        self.phase_legal[intersection].update(int(p) for p in states)
        self.phase_matches[intersection]+=int(requested==current)
        previous=self.phase_last.get(intersection)
        self.phase_changes[intersection]+=int(previous is not None and previous!=requested)
        self.phase_run[intersection]=self.phase_run[intersection]+1 if previous==requested else 1
        self.phase_longest[intersection]=max(self.phase_longest[intersection],self.phase_run[intersection])
        self.phase_last[intersection]=requested

    def finish(self,evaluator,utility,bias):
        result=super().finish(evaluator,utility,bias)
        result['phase_request_summary']=summarize(self.phase_counts,self.phase_changes,self.phase_longest,
                                                  self.phase_matches,self.phase_legal)
        assert result['phase_request_summary']['decisions']==result['decisions']
        for example in result['examples']:
            example['phase_bias_decomposition']=decompose(example)
        return result
