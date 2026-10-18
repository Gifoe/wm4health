import sys,pathlib,subprocess
sys.path.insert(0,str(pathlib.Path(__file__).resolve().parent))
from common70_v1 import *
q=read(OUT/'QUALIFICATION70_V1.json');r=read(OUT/'INDEPENDENT_REPLAY70_V1.json');assert r['status']=='PASS'
for stem in ['qualification70_v1','replay70_v1']:
 ex=read(OUT/(stem+'_exit.json'));assert ex['exit_code']==0 and ex['log_sha256']==sha(OUT/(stem+'_stdout.log'))
assert r['qualification_sha256']==sha(OUT/'QUALIFICATION70_V1.json') and len(q['cells'])==24
for row in q['cells']:assert sha(row['array_path'])==row['array_sha256']
def write(name,body):
 with (ROOT/name).open('x',encoding='utf-8') as f:f.write(body+'\n')
write('README.md','''# C70: controlled Gaussian component qualification

Executed unchanged RP-GSSM author components with explicit ordinary interface/flag adapters:24 known-system numerical fixtures,96 sequences, prefix-only inference and two future action plans. Independent Kalman/RTS replay and a fresh native cell passed. Three implementation/interface findings are reproducible, but none is a new research mechanism.

No neural model was trained here. This is not an original-paper result reproduction or an unknown-system benchmark. Previous native RP-GSSM training in C2 and controlled-source auditing in C46 already existed; the historical registry is corrected, not counted as new work. Raw numeric archives remain on server G. Overall discovery goal ACTIVE / UNACHIEVED; accepted final directions0.
''')
write('literature_review.md',(ROOT/'LITERATURE_DELTA70_V1.md').read_text(encoding='utf-8'))
write('baseline_registry.md',f'''# Execution and information registry

| Item | What actually ran | Boundary |
|---|---|---|
| RP-GSSM source {q['source_revision']} | Native ACLGParam, continuous Flax ActionNetwork, discrete mapper, ConstrainedIVFreeEnergy.get_posterior, Gaussian chain moments |15 exact source blobs verified; not a proxy neural model. |
| Ordinary interface/flag adapters | Discrete initialization signature; retain initial-state flags; constant/no-bias output masks | Authored outside upstream source. All affected original outputs/errors retained. |
| Independent dense reference |64-dimensional joint Gaussian conditioning | Known true dynamics/sensor likelihood, privileged parameters. |
| Independent replay | Joseph-form Kalman filter plus RTS smoother | Separate algorithm; full and16-observation prefix per sequence. |
| Meta-iLaD | Primary-paper scope review only | No released implementation at pinned revision; zero native fits. |
| Compositional continual benchmark | Primary-paper/project review only | Robotics neighbor; no claimed suite reproduction. |

Historical correction: RP-GSSM had actual native training in C2 and action-prior source inspection in C46. C70 is a narrow previously unexecuted branch qualification, not a first reproduction. The same C2 JAX environment is reused unchanged. The transition mapper networks are randomly initialized and frozen; known dynamics, Gaussian likelihood and fixed covariance are privileged. No trained representation or system identification claim.
''')
lines=['| Cell | posterior mean error | covariance error after explicit padding | raw cross-covariance error | future mean error |','|---|---:|---:|---:|---:|']
for row in q['cells']:
 e=row['errors'];lines.append('|'+row['id']+'|'+ '|'.join(f'{e[k]:.8g}' for k in ['posterior_mean','posterior_cov_minus_padding','posterior_cross_raw','forecast_mean'])+'|')
write('experimental_results.md','''# Numerical qualification, not learned-model performance

The predeclared factorial grid is3 seeds × continuous/discrete actions × additive/multiplicative transitions × invariant/noninvariant initial state =24 cells. Each cell has4 sequences of32 two-dimensional states. Likelihood precision is either.02 or1. The32-step full smoother is a component check; only the first16 observations are supplied for the separate future-plan test. Both future plans share the same inferred state and frozen parameters. No future observation enters that inference.

The original author's smoother explicitly adds1e-5 I to marginal covariances before computing adjacent cross-covariances. Raw outputs are preserved; report marginal comparison after removing that declared pad and raw cross-covariance discrepancy separately. Thresholds were1e-7 for mean/marginal/moment/forecast checks and1e-4 for raw cross-covariance. This permits no claim of exact equality of the returned padded joint posterior.

'''+ '\n'.join(lines)+f'''

Independent replay used96 full smoothers and96 prefix smoothers, verified15 source blobs and25 numeric archives, and freshly repeated one native cell. Kalman/RTS versus dense posterior maximum mean error is {r['max_errors']['rts_dense_mean']:.12g}; covariance {r['max_errors']['rts_dense_cov']:.12g}. Native mean differs from RTS by at most {r['max_errors']['rts_native_mean']:.12g}; raw adjacent cross-covariance error reaches {r['max_errors']['rts_native_cross_raw']:.12g}. These are numerical checks on known parameters, not96 independent learning experiments. Qualification runtime {q['runtime_seconds']:.3f}s; replay {r['runtime_seconds']:.3f}s, CPU only.
''')
flag=q['constant_flag_control']['native']
write('IMPLEMENTATION_FINDINGS70_V1.md',f'''# Reproducible implementation findings

Pinned upstream: https://github.com/gatsby-sahani/rp-gssm/tree/{q['source_revision']}

1. `ACLGParam.init` in rp_ssm/dists.py:422 calls `action_mapper.init(key, actions)`, while `DiscreteChoiceActionMapper.init` in actions.py:56 requires dimension and transition-mode arguments too. Direct initialization raises the retained TypeError. `DiscreteSignatureBridge` forwards those explicit arguments to the unchanged native mapper. This is interface binding, not a learned replacement.
2. `ACLGParam.update` at dists.py:434 does not carry the original `start_from_invariant` flag (or explicit opt_params). All12 noninvariant cells reset their returned initial mean/covariance to the default; reported errors retain the original output. `preserving_update` supplies the original flags to the same constructor. The12 default-invariant cells retain their flag. Only these exercised settings are qualified; no blanket claim that every optimizer option is fixed.
3. With the continuous mapper, `transition_matrix='constant'` / `transition_bias='none'` still lets the returned C/c affect the chain. A fixed zero-versus-one action control changes mean by {flag['mean_max_abs']:.12g} and covariance by {flag['cov_max_abs']:.12g}. A separate mapper output mask enforcing those disabled terms produces exactly equal arrays. The full original and masked arrays remain in the hashed control archive.

No upstream file, old experiment or shared environment is modified. These repairs do not establish failure of representation learning, need for a nonlinear prior, or a new world-model architecture.
''')
write('negative_evidence.md','''# Negative evidence

- The supported action-conditioned Gaussian propagation/inference components agree with two independent calculations after the ordinary interface/flag adapters. They do not support a missing-action-inference capability claim.
- Native covariance padding explains why returning a numerically regularized covariance is not exact joint-Gaussian equality; the tolerance and raw arrays are explicit. This is an implementation property, not a novel scientific gap.
- No learning occurred here. Random frozen mapper networks and known parameters cannot establish generalization, identifiability in finite samples, representation sufficiency or useful performance on unknown systems.
- C2 already trained native RP-GSSM, and C46 already inspected actions and normalization. A stale registry would have caused a duplicate reproduction. Do not reopen earlier auxiliary-bound or decoder-necessity claims.
- Meta-iLaD already covers few-shot latent environment learning under stated assumptions and discusses local-stationarity context mismatches. Lack of public code does not create novelty.
- A recent compositional benchmark already studies encoder drift, modular reuse and privileged full-task encoder exposure. Plain alignment, replay, expert addition or switching routers are insufficient novelty claims.
''')
write('hypothesis_space.md','''# Competing hypotheses and stopping decisions

| Claim | Discriminating evidence | Decision |
|---|---|---|
| Native controlled Gaussian components have no usable action-conditioned inference |24 source-based fixtures, dense conditioning, independentRTS, future-plan checks | Rejected for this known-system contract after ordinary adapters. |
| Noninvariant state can be passed through update unchanged | Original and flag-preserving constructors,12 cells | Original implementation loses flag; ordinary repair sufficient. |
| Disabled constant/no-bias continuous settings ignore action | Zero-versus-one fixed control | Original changes; masking disabled terms fixes exercised case. |
| These implementation errors establish a new research problem | No learned-model causal comparison or new method | Rejected. |
| Few-shot environment/state/dynamics inference is novel by itself | Meta-iLaD primary theory/method | Rejected. |
| Continual mechanism recombination could reveal a useful remaining capability | Only adjacent primary-paper evidence so far | Unestablished; source and prior-project audit first. |

Do not promote an explicitly author-identified limitation to our discovery or mistake a known-information oracle check for a fair learned baseline.
''')
write('research_ideas.md','''# Search decisions

No final direction accepted. Controlled Gaussian inference, ordinary initialization adapters and generic few-shot environment inference do not supply a novel problem.

One provisional search branch: mechanism reuse when a time-series world model learns new combinations sequentially, with capacity, observed information and representation learning controlled. This is not yet an idea or recommendation. The September2026 compositional-continual benchmark already exposes this issue in robotics, and previous project alignment/covariance work may cover ordinary repairs. The relevant question is whether a distinct time-series capability remains after competent continual-learning and modular controls, not whether a router can be added.

Next inspect the actual released PWM/model and continual-baseline code, pin revisions, and check historical project coverage. Do not launch a large robotics reproduction or invent a time-series benchmark before that audit. A method would require a specific residual mechanism, independent held-out combinations, a same-information capacity/compute-matched ordinary baseline and non-test-selected validation.
''')
write('final_research_direction.md','''# No accepted final direction

**Research Question:** A novel general time-series world-model problem is still unestablished. This cycle only tests whether the actual structured action-conditioned Gaussian components can support a fair baseline.

**Evidence:**24 numerical fixtures/96 sequences, two independent posterior algorithms and one fresh native rerun pass. Direct original-interface tests identify three routine implementation issues; explicit adapters suffice for the exercised cases. There are zero new trained models and no empirical claim about unknown-system prediction.

**Existing Methods:** RP-GSSM already supports controlled priors and Gaussian recognition-based inference; Meta-iLaD already learns context-conditioned latent environments/dynamics. Prior native RP-GSSM training in C2 and source auditing in C46 must not be repeated or erased. Existing continual/modular world-model work also constrains any reuse proposal.

**Proposed Idea:** None accepted. Do not rename a Gaussian observer, few-shot context encoder, ordinary replay or adapter as innovation.

**Falsification:** Any claimed new need that disappears with the explicit implementation repair or a competent same-information existing baseline is defeated. A known-parameter oracle cannot establish a learned-model failure or solution; a published limitation is not direct experimental evidence from this project.

**Feasibility:** Native JAX/Flax/Dynamax CPU components execute in the unchanged C2 environment. A singleRTX5090 remains available for a justified learned mechanism experiment. Raw numeric files remain in the authorized serverG project subtree; source/reports/compact metadata go toGitHub only.

**Next Experiment:** First audit released PWM and continual-learning implementations against the September2026 compositional benchmark and prior project history. Seek a precise untested time-series reuse contract with independent mechanism combinations and fair information/capacity controls. If ordinary alignment/replay already covers it or source access prevents an honest baseline, reject/switch rather than run a weak proxy. This is a concrete selection action, not a promised positive direction.

Overall goal ACTIVE / UNACHIEVED; accepted final directions0.
''')
write('REPRODUCE70_V1.md','''# Reproduction scope

Use the pinned source revision and exact15 blob hashes in SOURCE_SELECTION70_V1.json; do not alter upstream code. Keep an isolated environment matching QUALIFICATION70_V1.json. CPU JAXx64,4-thread process affinity. Configure sibling projectSTORAGE_LOCATION_V1.json to place artifact_root on authorizedGstorage. Only authored source/control files are staged from local; no data/model/result files are synchronized locally.

For a clean run use fresh output and artifact directories, as writes are exclusive. Run qualify70_v1.py then replay70_v1.py via their capture/hiddenPowerShell launchers. Replay requires a closed qualification exit0 and every raw archive hash. No network or dataset download is needed for the scientific numerical worker. Qualification/replay do not train RP-GSSM's empirical-RPM objective; they exercise native controlled Gaussian chains and a known Gaussian likelihood. Read the named adapter/privilege boundaries before treating any result as a baseline.
''')
status='CLOSED_ACTION_GAUSSIAN_COMPONENT_QUALIFICATION_NOT_NEW_METHOD'
with (PROJECT/'CURRENT_RESEARCH_STATUS_V70.md').open('x',encoding='utf-8') as f:f.write('''# Current status:C70

RP-GSSM controlled Gaussian components qualified on24 cells/96 sequences with independentRTS replay. Three ordinary interface/flag repairs suffice. No neural training, no new unknown-system benchmark, no accepted final direction. Historical registry corrected:C2already trained nativeRP-GSSM;C46already inspected actions. Meta-iLaDprimary theory/method rules out generic few-shot latent environment inference as novelty.

GoalACTIVE/UNACHIEVED. Next source-selection action: inspect actualPWM/continual-learning implementations and old project coverage for a precise time-series mechanism-recombination capability. Existing modularity/replay/alignment is not a new contribution. No positive claim from the adjacent robotics paper alone. Rawarrays remain serverG; code/reports/compact evidenceGitHub only.
''')
reports={p.name:sha(p) for p in ROOT.glob('*.md')};names=['QUALIFICATION70_V1.json','INDEPENDENT_REPLAY70_V1.json','qualification70_v1_exit.json','replay70_v1_exit.json'];artifacts={n:sha(OUT/n) for n in names};sources={p.relative_to(ROOT).as_posix():sha(p) for p in list((ROOT/'scripts').glob('*.py'))+list(ROOT.glob('*.ps1'))+list(ROOT.glob('*.json'))}
dump(OUT/'CLOSURE70_V1.json',dict(utc=utc(),status=status,reports=reports,artifacts=artifacts,sources=sources,environment=q['environment'],cells=24,sequences=96,raw_archives=25,neural_fits=0,known_system_and_likelihood_privileged=True,independent_replay='PASS',accepted_final_directions=0,overall_goal='ACTIVE_UNACHIEVED',local_result_files_written=0,upstream_files_modified=False))
print(json.dumps(dict(status=status,reports=len(reports),closure_sha256=sha(OUT/'CLOSURE70_V1.json'),accepted_final_directions=0)),flush=True)
