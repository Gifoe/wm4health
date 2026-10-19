"""Write an exploratory primary-source opportunity map, not a novelty certificate."""
import csv,datetime,json,pathlib
ROOT=pathlib.Path(__file__).resolve().parents[1]
def main():
 dst=ROOT/'literature-search-20261016-dynamics-boundaries';dst.mkdir(exist_ok=True,parents=True)
 papers=[
 ['EpiCF-Bench',2026,'KDD Datasets and Benchmarks / arXiv2606.05692','https://arxiv.org/html/2606.05692v1','pure benchmark',3,3,'N/A benchmark','A','Task information and data version audited; 13 new execution configurations','Real calibrated ABM; target-scenario state inputs and duplicate county blocks constrain interpretation'],
 ['Time-series forecasting through the lens of dynamics',2026,'ICML / arXiv2507.15774v2','https://arxiv.org/html/2507.15774v2','pure method',3,4,3,'B','Learnable direct prediction map is explicitly a relaxed dynamics notion','Read definitions and implementation; no new native numeric reproduction'],
 ['LeNEPA: No-Augmentation Next-Latent Prediction for Time-Series Representation Learning',2026,'arXiv2607.00958 / MiLeTS submission code','https://arxiv.org/abs/2607.00958','pure method',3,3,3,'A','Exact input-target and temporal regularization configuration inspected; native component null tested','Do not equate representation probes with full simulation; local dependency prevents assuming turnkey reproduction'],
 ['Semigroup-JEPA: Latent Dynamics Consistency for Zero-Shot Physics Generalization',2026,'arXiv2609.10464','https://arxiv.org/html/2609.10464v1','method + benchmark',3,4,4,'A','Direct neighbor for law conditioning and multi-step encoder training','Scalar gravity provided; linear theory explicitly includes closure and residual terms; not replicated here'],
 ['One-Step Next-Latent Prediction Is Not a World Model',2026,'arXiv2609.36227','https://arxiv.org/html/2609.36227v1','theory/proof',2,3,2,'B','Explicit prior on conditional-mean composition and missing state','Elementary examples and covariance surrogate; does not execute full LeNEPA or justify every broad slogan'],
 ['Temporally Centered SIGReg Improves Multi-Task LeWorldModel Learning',2026,'arXiv2607.26924','https://arxiv.org/html/2607.26924v1','pure method',3,3,3,'A','Residual regularization and multi-task cluster contraction are existing mechanisms','Downstream is frozen-encoder behavior cloning, not direct latent planning; published cross-pipeline baselines contextual'],
 ['MotionJEPA: Preventing Temporal Feature Collapse by Capturing Visual Changes in Latent Space',2026,'arXiv2609.23881','https://arxiv.org/html/2609.23881v1','method + benchmark',4,4,4,'A','Direct neighbor for slow-feature suppression and time/flat regularizer controls','Three controlled environments and planning; source report read, numeric results not independently reproduced'],
 ['Learning Deployable Causal Action Geometry under Temporal Non-Stationarity',2026,'arXiv2610.16644','https://arxiv.org/html/2610.16644v1','pure method',3,4,3,'B','Action ranking and response geometry under drift have an explicit recent target','One-step exchangeability and geometry persistence required; private production evidence not independently verified'],
 ]
 columns=['title','year','venue_or_source','link','paper_type','insight_score','completeness_score','numeric_evidence_score','overall_label','relevance_note','quality_note']
 with (dst/'papers.csv').open('w',encoding='utf-8',newline='') as f:
  writer=csv.writer(f);writer.writerow(columns);writer.writerows(papers)
 table='| Title | Type | Insight | Completeness | Numeric evidence | Priority |\n|---|---|---:|---:|---:|---|\n'
 for p in papers:table+=f'| [{p[0]}]({p[3]}) | {p[4]} | {p[5]} | {p[6]} | {p[7]} | {p[8]} |\n'
 (dst/'papers.md').write_text('''# Exploratory literature search: world-model dynamics boundaries

Date: 2026-10-16 Asia/Shanghai. Purpose: prevent generic forecasting, information
leakage, slow-feature and multi-step claims from being relabeled as new research.
Source-quality policy: primary papers, official code, author project materials;
MDPI and untraceable sources excluded. These are provisional reading judgments,
not venue acceptance probabilities or claims of independent numeric verification.
No exhaustive or systematic-review coverage is claimed.

'''+table+'''
## Closest-work clusters and opportunity map

| Cluster | Status | What is covered | Evidence still needed |
|---|---|---|---|
| EpiCF conditional vs prospective task | benchmark gap in this release | New audit exposes target-scenario state information; standard g-method and valid history tasks already exist | A broader research contribution needs a distinct unresolved setting after fixing task/input definitions |
| Mean composition, state and memory | covered central claim | One-step critique and SG-JEPA already cover residuals, closure, history and multi-step training | A concrete failure not fixed by ordinary stochastic/history models, not another Jensen-gap example |
| Regularizer axes and slow features | crowded but open | LeNEPA, MotionJEPA, TC-LeWM and author ChronoJEPA study time axes, residuals, feature suppression and position/order | Must test native admissible constructions; the arbitrary position table does not defeat native input-layer targets |
| Causal response under drift | crowded but open | Anchored response geometry, causal forecasting and excitation theory are direct prior work | Distinct sequential setting and controlled counterexamples to strong existing methods |

## Benchmark candidates

EpiCF is small and usable for its released oracle-state conditional task. Its
current CSV does not supply branches sharing an arbitrary factual prefix, so a
new prospective branch benchmark cannot be fabricated by pairing incompatible
full-horizon paths. DoTime already supplies explicit information masks and causal
structures, but its suites and noise-pairing versions must be pinned. MotionJEPA
offers controlled feature groups; no experiment from it was executed here.

## Scope cautions

PRO-DYN's learnable mapping definition is not an assertion of physical system
identification. LeNEPA's default input-level target excludes absolute PE; generic
tensor counterexamples can be inadmissible. SG-JEPA does not silently assume zero
transition residual in its linear theory. A method's appendix limitation is not
by itself evidence that a new method is necessary.

Unscored pending sources: author ChronoJEPA repository (public primary code/report,
peer-review status unverified), Control Theory of Predictability2607.10362 (initial
screen only), Gong et al. UAI2017 (primary abstract checked as foundational prior),
Shared Gaussianization2610.10299 (full text not yet inspected). Do not infer their
complete mechanisms or experimental validity from search snippets.
''',encoding='utf-8')
 (dst/'search-notes.md').write_text('''# Search notes

Exploratory scope, recent window primarily 2026-04-16 through 2026-10-16; older
anchors retained where they directly constrain novelty. Only public concepts,
paper titles and repository names were used as external queries.

Representative safe queries: time-series world model benchmark 2026; latent
dynamics semigroup 2026; SIGReg temporal correlated samples; VICReg temporal
correlation batch; LeNEPA position; temporal SIGReg collapse position; world model
non-normal dynamics; causal temporal aggregation.

Checked primary sources: arXiv HTML and fixed PDF versions, official EpiCF and
LeNEPA repositories, Zenodo v1.2 API/files, PRO-DYN repository, MotionJEPA author
project, author ChronoJEPA code/report, PMLR UAI2017 record. Source snippets,
automatic paper summaries, social-media mirrors and AI reviews were discovery
leads only and are not used to validate technical or empirical claims.

Unknowns: original EpiCF paper-table normalization/configuration correspondence;
exact public CSV export aggregation; full native run of LeNEPA; full-source
availability for some newest methods; complete recent-work coverage. Missing
code is a reproducibility limitation, not evidence of scientific falsity.

Handoff: do not draft a novelty or acceptance claim from this map. Read exact
model assumptions and evaluate a falsifiable distinct failure before selection.
''',encoding='utf-8')
 (dst/'idea-grounding.md').write_text('''# Idea grounding

No candidate accepted. One executed benchmark-information finding and two
rejected strong hypotheses are more defensible than a manufactured new method.

Rescue routes for covered central claims: an operational information contract
could become a useful broader benchmark contribution only after demonstrating
failures across appropriate existing tasks and avoiding tasks already solved by
DoTime or history-conditioned causal methods. Regularization-axis analysis needs
a failure admissible in the native architecture and not already repaired by
MotionJEPA, TC-LeWM, dual-axis or standard reconstruction/context baselines.
Neither rescue route currently has the evidence needed for recommendation.
''',encoding='utf-8')
 (ROOT/'outputs'/'literature_scout_status.json').write_text(json.dumps({'written_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'mode':'exploratory','scored_primary_sources':len(papers),'provisional_scores':True,'accepted_directions':0},indent=2),encoding='utf-8')
 print('Wrote literature scout',len(papers),'scored primary sources')
if __name__=='__main__':main()
