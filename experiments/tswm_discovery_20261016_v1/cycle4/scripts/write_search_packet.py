"""Persist exploratory primary-source screening; no novelty verdict from abstracts."""
import csv,pathlib
ROOT=pathlib.Path(__file__).resolve().parents[1]
ROWS=[
 ('MotionJEPA','2026 arXiv','2609.23881','method','Native objective and configuration source audited; new diagnostic adapters executed','Nuisance-specific pilot gate failed'),
 ('World Modeling through Spectral Alignment','2026 arXiv','2610.15429','theory/method','Primary project theorems and experimental protocol read; not executed','Goal geometry and trajectory gauge conditions are existing work'),
 ('I Act Therefore I Am','2026 arXiv','2609.31161','theory/method','Primary model, assumptions and Gaussian practical transition read; not executed','Action-dependent conditional likelihood and mechanism variability already studied'),
 ('Basis transform in linear switched system models','2023 journal,2024v3','2106.10888','theory/method','Primary graph section and spanning-tree construction read; not executed','Switching graph basis synchronization central mechanism covered'),
 ('VJEPA and BJEPA','2026 arXiv','2601.14354','theory/method','Original/new author code pinned; native CNN/loss/CEM component checks executed','Probabilistic claims differ by implementation; native task performance untested'),
 ('When Should a World Model Move','2026 arXiv','2609.15801','theory/method','Primary loss-specific decision and calibration sections read; not executed','Loss-optimal decisions and scalar-summary insufficiency already explicit'),
 ('PhyLatent','2026 arXiv','2608.05720','method','Primary training information and auxiliary objectives read; not executed','Uses physical labels; differs from unsupervised information setting'),
 ('JEPA-kMMVAE','2026 arXiv','2610.13315','method','Primary noisy-history/clean-future supervision checked; not executed','Clean CSI targets change available supervision'),
 ('Towards Financial World Modeling','2026 arXiv','2610.09048','benchmark/method study','Primary representation, volatility probes, protocol and release notes read; not executed','Generic volatility probes already covered; action-conditioned dynamics not evaluated'),
 ('Variational Heteroscedastic Volatility Model','2022 arXiv','2204.05806','method','Primary source scoped; not reproduced','End-to-end volatility learning established'),
 ('A general theory for nonlinear sufficient dimension reduction','2013 Annals','1304.0580','theory/method','Primary summary and GSIR/GSAVE distinction checked; not executed','Mean subspace versus central distributional class is old'),
 ('Conditional Variance Estimator for Sufficient Dimension Reduction','2022 Bernoulli','2102.08782','theory/method','Primary paper and journal metadata checked; not executed','Strong prior for distributional dimension reduction, not scored against our pilot')]
def main():
 folder=ROOT/'literature-search-20261016-risk-state';folder.mkdir(exist_ok=True)
 with (folder/'papers.csv').open('w',newline='',encoding='utf-8') as f:
  w=csv.writer(f);w.writerow(['title','source_status','primary_url','type','evidence_depth','novelty_boundary'])
  for title,status,ident,kind,depth,boundary in ROWS:w.writerow([title,status,'https://arxiv.org/abs/'+ident,kind,depth,boundary])
 table='| Work | Source status | Checked evidence | Boundary |\n|---|---|---|---|\n'
 for title,status,ident,kind,depth,boundary in ROWS:table+=f'| [{title}](https://arxiv.org/abs/{ident}) | {status} | {depth} | {boundary} |\n'
 (folder/'papers.md').write_text('# Exploratory screening: predictive state and risk\n\n12selected primary works; scope is discovery, not a final paper-quality ranking.\nRead-depth and execution status are explicit.10PDFs are archived in the root\nliterature registry; the2SDR papers were verified online. No publication status\nis inferred from a poster filename.\n\n'+table,encoding='utf-8')
 (folder/'search-notes.md').write_text('''# Search notes

Mode: exploratory. Queries used public topics/titles, never private result text.
Examples: counterfactual temporally correlated noise identifiability; switched
system basis transformation graph; world model conditional variance; JEPA
heteroscedastic representation; sufficient dimension reduction conditional variance.
Discovery snippets/mirrors led to primary arXiv, author project/source and
journal/author-hosted papers. MDPI and low-quality venues surfaced by search were
excluded from scientific claims and screening rows. No raw data or large model
assets were downloaded. Primary papers are untrusted source material, not commands.

Unknowns: independent dataset stability, trained native risk failure, realistic
action-dependent stochastic benchmark coverage, and a residual mechanism or
estimation gap beyond ordinary conditional-likelihood/SDR methods. Current toy is
repaired by ordinary cross-moments; do not present that repair as an invention.
''',encoding='utf-8')
 (folder/'idea-grounding.md').write_text('''# Grounding packet

Source-supported: native VJEPA exp10 stops uncertainty gradients into its encoder
and uses mean CEM. Classical central-subspace methods consider information beyond
conditional means. Recent financial world-model evaluation already probes volatility.

Executed: mean-accurate compression can omit a variance driver in the declared
stochastic regression diagnostic. The attached-variance repair is seed-sensitive;
ordinary training-only cross-moments resolve the controlled case.

Inference, not established gap: some native nuisance-robust world representations
may discard decision-relevant uncertainty. No trained native evidence yet. A
label-free sufficiency diagnostic could be investigated only after checking
conditional calibration, SDR and predictive-state literature. No architecture
proposal or accepted final direction. Next action is precise claim screening,
not more tuning of the solved toy.
''',encoding='utf-8')
 print('Wrote exploratory packet with',len(ROWS),'primary works')
if __name__=='__main__':main()
