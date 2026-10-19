"""Summaries from completed measured results; no pooled-row significance."""
import runtime_profile
import json,pathlib
import numpy as np
ROOT=pathlib.Path(__file__).resolve().parents[1]
def group(name):
 if name.startswith('actuators_'):return 'validation' if name.endswith(('_11','_12')) else 'held-out walks'
 if name.startswith('loads_hatch_mix_fast'):return 'fast waves'
 if name.startswith('loads_hatch_mix_slow'):return 'slow waves'
 return name
def main():
 allrows={}
 for directory in ['clock_rpm_controls','strong_transition_controls']:
  result=json.loads((ROOT/'outputs'/directory/'results.json').read_text())
  for r in result['rows']:
   g=group(r['recording'])
   for arm,metrics in r['arms'].items():
    k=(g,arm);allrows.setdefault(k,[]).append({'recording':r['recording'],**metrics})
 rows=[]
 for (g,arm),rs in allrows.items():rows.append({'group':g,'arm':arm,'recordings':len(rs),'one_step_normalized_MSE':float(np.mean([r['one_step_normalized_MSE'] for r in rs])),'rollout16_normalized_MSE':float(np.mean([r['rollout16_normalized_MSE'] for r in rs])),'per_recording':rs})
 out=ROOT/'outputs/real_controls_summary.json';out.write_text(json.dumps({'scope':'Equal recording-weighted descriptive means; no independent row/seed confidence intervals. Ridge roots include boundary cases while HGB excludes roots crossing regime flags; do not interpret that tiny protocol difference as a model effect.','rows':rows},indent=2),encoding='utf-8');print(json.dumps({'rows':len(rows),'saved':str(out.name)}))
if __name__=='__main__':main()
