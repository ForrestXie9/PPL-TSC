"""Lexicographic tie choice among already evaluated CMA candidates; no objective penalty."""
import numpy as np

def actions(samples,bias):
 result=[]
 for mode,_teacher,values in samples:
  if mode=='phase_argmax':
   result.append(max(((p,q+float(bias.get(p,0.))) for p,q in values),key=lambda x:x[1])[0])
  else:
   current,following,keep,switch=values
   result.append(following if switch+float(bias.get(following,0.)) > keep+float(bias.get(current,0.)) else current)
 return tuple(result)

def choose_tied_bias(evaluated,samples,phase_ids):
 if not samples or not evaluated:raise ValueError('Nonempty source samples and evaluated candidates required')
 original=actions(samples,{})
 def rank(pair):
  vector,value=pair
  if not np.all(np.isfinite(vector)) or not np.isfinite(value):raise ValueError('Nonfinite candidate')
  chosen=actions(samples,dict(zip(phase_ids,vector)))
  correct=sum(action==sample[1] for action,sample in zip(chosen,samples))
  if abs(-value-correct/len(samples))>1e-12:raise ValueError('Recorded objective does not match actions')
  changes=sum(a!=b for a,b in zip(chosen,original))
  return (-correct,changes,float(np.dot(vector,vector)))
 return min(evaluated,key=rank)[0].copy()
