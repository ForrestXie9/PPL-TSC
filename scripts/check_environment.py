#!/usr/bin/env python3
"""Offline import and default-isolated policy execution check."""
import json,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'src'))
def main():
 import cityflow,numpy,torch,cma,gymnasium,sklearn,skopt,deap
 from LLM_SR_DISTILLATION.policy_process import PolicyProcess
 worker=PolicyProcess('def compute_utility(state, phase_id=None, feature_stats=None):\n    return float(state["x"] + 1)\n')
 try:
  assert worker({'x':2})==3
 finally:worker.close()
 print(json.dumps({'status':'passed','python':sys.version.split()[0],'numpy':numpy.__version__,'torch':torch.__version__,'isolated_policy':'passed','API_calls':0}))
if __name__=='__main__':main()
