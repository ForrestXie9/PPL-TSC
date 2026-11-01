#!/usr/bin/env python3
"""Export an explicit formula-only cross-network variant without changing its source."""
import argparse,hashlib,json
from pathlib import Path

def main():
 p=argparse.ArgumentParser(description=__doc__)
 p.add_argument('--deployment',type=Path,required=True)
 p.add_argument('--output',type=Path,required=True)
 a=p.parse_args()
 if a.output.exists() or a.output.resolve()==a.deployment.resolve():p.error('Output must be a new file')
 d=json.loads(a.deployment.read_text())
 if not isinstance(d.get('feature_stats'),dict) or d.get('action_mode') not in ['phase_argmax','binary_keep_switch','keep_switch']:p.error('Invalid deployment contract')
 removed=d.get('per_intersection_bias',{})
 d['per_intersection_bias']={}
 d['transfer_export']={'protocol':'formula_only_source_statistics_v1','source_deployment_sha256':hashlib.sha256(a.deployment.read_bytes()).hexdigest(),'removed_source_bias':removed,'description':'Fixed formula and source feature statistics; source-local biases explicitly removed. No target search, calibration or teacher rollout. This is not transfer of the full calibrated policy.'}
 a.output.parent.mkdir(parents=True,exist_ok=True);a.output.write_text(json.dumps(d,indent=2,allow_nan=False)+'\n');print(a.output)
if __name__=='__main__':main()
