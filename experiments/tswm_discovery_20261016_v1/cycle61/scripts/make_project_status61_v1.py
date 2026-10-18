"""Add a fresh project-level status without rewriting historical reports."""
from common61 import *

def main():
 c=read(OUT/'CLOSURE61_V1.json');s=read(OUT/'SUMMARY61_V1.json');old=read(PROJECT/'cycle60/outputs/SUMMARY60_V1.json')
 assert c['status']=='C61_LEARNED_EXISTING_DURATION_CONTROL_SCREEN_CLOSED'
 assert read(OUT/'post_state61_v1.json')['status']=='C61_CLOSED_RESULTS_AND_PUBLICATION_SNAPSHOT_READY'
 text=f'''# Current research status after C61

Overall goal: ACTIVE / UNACHIEVED. Accepted final research directions: **0**.

The goal remains a distinct, useful and experimentally supported time-series world-model research problem, not a single-dataset optimization or a count of reproductions. No new framework is claimed.

## What the latest experiments changed

C60 completed twelve fixed-budget ordinary neural state-space/GRU fits on two measured systems. Native-component NSSM three-seed mean MSE / fit variance was {old['groups']['N87']['native_component_nssm']['mean_primary_all_expected']:.8g} on N87 and {old['groups']['pick_place']['native_component_nssm']['mean_primary_all_expected']:.8g} on pick-place, below all tested fixed memory banks. This removes present evidence for a new mixed-memory necessity in these cases. N87 reused a historically viewed exploratory split; pick-place was one chronological device record. These are separately fitted controls, not shared-checkpoint transfer.

C61 executed existing state-age likelihood/filter/survival code and twelve fixed-budget unlabeled fits in two controlled finite-duration families. All selected models were sealed before test scoring; independent density/posterior/survival replay and score arithmetic were retained. Its prespecified outcome is **{s['decision']}**.

'''
 for family,g in s['groups'].items():
  h=g['hsmm'];text+=f"- {family}: HSMM mean NLL above known-law reference {h['conditional_nll_above_oracle_mean']:.8g} nats/step; remaining-life probability MSE {h['survival_probability_mse_vs_oracle_mean']:.8g}; all-seed sufficiency pass {h['all_seeds_pass_prespecified_sufficiency_screen']}.\n"
 text+='''
C61 is a prior-art exclusion, with zero real-data domains, known model order/support and fresh episode starts. It does not show that all regime, adaptation or stochastic modeling problems are solved. Known-law references are privileged numerical ceilings; fitted methods learn from observations only. The failed first support-mask adapter, corrected native extension and Python metadata/runtime boundary remain disclosed.

## What is not yet established

No unexplained cross-system capability deficit and distinct method improvement have survived the necessary existing controls. Generic stochastic latent, explicit duration, mixed memory, environment latent, assimilation-aware representation or conformal-calibration wrappers are insufficient novelty descriptions by themselves. Existing ingredients do not preclude a useful new combination; the missing evidence is an observable residual or efficiency advantage beyond competent nearest methods on independently specified data.

The next decision must follow failure attribution and primary-source comparison. Do not retune the opened C60/C61 test sets or promote a known-method repair to a new framework. Keep negative results and independent raw evidence.

## Evidence

- [C60 actual measured-system audit](cycle60/README.md)
- [C60 outcome-dependent interpretation](cycle60/INTERPRETATION60_V1.md)
- [C61 existing duration-control audit](cycle61/README.md)
- [C61 all-seed evidence](cycle61/outputs/EVALUATION61_V1.json)
- [C61 research disposition](cycle61/final_research_direction.md)
- [Additional primary-literature boundaries](cycle61/NEXT_RESEARCH_BOUNDARIES61_V1.md)

Raw data, checkpoints, latent audit truth and predictions remain on the authorized server G disk. Only code and compact evidence are published to Gifoe/wm4health; no local result-file synchronization.
'''
 p=PROJECT/'CURRENT_RESEARCH_STATUS_V61.md';b=text.encode('utf-8')
 with p.open('xb') as f:f.write(b)
 obj={'utc':utc(),'status':'PROJECT_STATUS_V61_ADDED_WITHOUT_REWRITING_HISTORY','path':str(p),'bytes':len(b),'sha256':sha(p),'git_blob_sha1':hashlib.sha1(b'blob '+str(len(b)).encode()+b'\0'+b).hexdigest(),'content':text}
 dump(OUT/'PROJECT_STATUS61_V1.json',{k:v for k,v in obj.items() if k!='content'});print(json.dumps(obj),flush=True)

if __name__=='__main__':main()
