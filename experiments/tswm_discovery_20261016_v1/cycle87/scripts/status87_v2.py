from common87_v1 import *
def main():
 v=read(O/'VERIFICATION87_V4.json');assert v['status']=='SOURCE_FIELDS_SELECTIONS_METRICS_AFFINE_REPLAYED_ROLLOUT_QUALIFICATION_INCOMPLETE';d=read(O/'STUDY_DECISION87_V2.json');q=read(O/'NUMERICAL_QUALIFICATION87_V1.json')
 text='''# Current research status - C87

Overall objective: ACTIVE / UNACHIEVED. Final accepted directions:0.

The current sampler-method candidate is deprioritized as a main research route. C87 completed9 unchanged full-size FIM-ODE adaptations (3 data arms x3 optimizer seeds) and a14-model,2-family,3-cohort held-out comparison. Resampling effects are not consistent across seeds. A same-supervision ordinary empirical-prior control is stronger in all54 reported matched native comparisons under the declared RK4/TF32 protocol. It is an existing Bayesian construction, not a new method.

Important numerical limit: independent learned-rollout qualification did NOT pass its1% criterion on the first frozen-model cohort (2.17%, then1.55% with original batch shape). Adapted-model supplemental checks were not reached. The failed checks remain failed, and the simulation comparisons are not certified continuous-time errors. Initial weights, all63 source-VAL checkpoint selections/RNG pairing, direct native field arrays, physical references, empirical posterior conditioning, metric arithmetic and exact affine flow were replayed separately. Do not call this a fully verified14-model continuous-time benchmark.

Close prior art further restricts novelty: Arruda et al. arXiv:2604.18319 already studies selection-aware amortized inference. This does not solve every task-prior/deployment shift, but excludes claiming that selection-aware simulation itself is new. No causal defect of the released checkpoint was established.

Read cycle87/experimental_results.md, cycle87/POSTSTUDY_ANALYSIS87_V2.md, cycle87/NUMERICAL_OUTCOME87_V1.md, cycle87/NOVELTY_ADDENDUM87_V1.md and cycle87/final_research_direction.md. This is a negative/limited study, not completion of the research objective. Future candidates must clear closest-prior-art and strong prior-aware ordinary controls before another large learner campaign. C71's original-budget campaign continues separately; its current process must be checked live before reporting progress.

Raw data/models/predictions/logs remain on the remote15TB disk. GitHub is the sole publication destination; no local result/model/data synchronization.
'''
 write(ROOT/'CURRENT_RESEARCH_STATUS_V87_METADATA_V1.json',dict(utc=utc(),verification_sha256=sha(O/'VERIFICATION87_V4.json'),study_decision_sha256=sha(O/'STUDY_DECISION87_V2.json'),numerical_qualification_sha256=sha(O/'NUMERICAL_QUALIFICATION87_V1.json'),accepted_directions=0,goal='ACTIVE_UNACHIEVED'))
 with (ROOT/'CURRENT_RESEARCH_STATUS_V87.md').open('x',encoding='utf-8') as f:f.write(text)
if __name__=='__main__':main()
