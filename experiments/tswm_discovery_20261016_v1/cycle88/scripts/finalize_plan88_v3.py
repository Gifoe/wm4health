from common88_v1 import *
import re,shutil,numpy as np

def write_md(path,text):
    with Path(path).open('x',encoding='utf-8',newline='\n') as f:f.write(text.strip()+'\n')

def public_text(value):
    if isinstance(value,dict):return {k:public_text(v) for k,v in value.items()}
    if isinstance(value,list):return [public_text(v) for v in value]
    if not isinstance(value,str):return value
    aliases={str(ROOT):'${RESEARCH_ROOT}',**{str(Path(v)):'${'+k.upper()+'}' for k,v in STORAGE.items() if isinstance(v,str) and Path(v).is_absolute()}}
    for key,label in sorted(aliases.items(),key=lambda x:-len(x[0])):
        for form in {key,key.replace('\\','/'),key.replace('/','\\')}:value=value.replace(form,label)
    return value

def main():
    result=read(O/'PAIRED_PLAN_RESULTS88_V1.json');ridge=read(O/'RIDGE_PLAN_RESULTS88_V1.json');assert read(O/'paired_plan_eval_v1/EXIT.json')['returncode']==0
    assert read(O/'VERIFICATION88_V1.json')['status']=='DATA_REGENERATION_AND_REPORTED_PAIRED_METRICS_REPLAYED'
    for row in [result,ridge]:assert sha(row['payload_path'])==row['payload_sha256']
    fits=result['fits'];rows=result['rows'];fix=all(x['ordinary_fix_passes_original_rejection_gate'] for x in result['ordinary_controls'])
    stock=[r for r in rows if r['arm']=='stock'];lagged=[r for r in rows if r['arm']=='lagged'];causal=[r for r in rows if r['arm']=='acrkn']
    extra_competent=all(r['train_mean_ratio']<.8 and r['untrained_ratio']<.8 for r in fits[1]['factual'])
    source_fits=[read(O/f'PLAN_FIT88_V1_{a}.json') for a in ['stock','lagged','acrkn']]
    pair_map={(r['family'],r['cut'],r['sign']):r for r in stock}
    response_comparable=all(r['post_response_error_rms']<=1.05*pair_map[r['family'],r['cut'],r['sign']]['post_response_error_rms'] for r in lagged)
    route='DEPRIORITIZE_NEW_METHOD_ORDINARY_FIX_SUFFICIENT_ON_DEVELOPMENTAL_POOL' if fix and extra_competent and response_comparable else 'UNRESOLVED_DEVELOPMENTAL_COMPARISON_NO_NOVELTY_ESTABLISHED'
    table=['| Model | Parameters | Duffing MSE | Reservoir MSE |','|---|---:|---:|---:|']
    for fit in fits:table.append(f"| {fit['arm']} | {fit['parameters']} | {fit['factual'][0]['mse']:.8g} | {fit['factual'][1]['mse']:.8g} |")
    table.append(f"| ordinary ARX4 | {ridge['parameters']} | {ridge['factual'][0]['mse']:.8g} | {ridge['factual'][1]['mse']:.8g} |")
    table='\n'.join(table)
    maxima={arm:max(r['early_max'] for r in rows if r['arm']==arm) for arm in ['stock','lagged','acrkn']}
    rms={arm:max(r['early_rms'] for r in rows if r['arm']==arm) for arm in ['stock','lagged','acrkn']}
    write_md(C/'experimental_results.md',f'''# C88 developmental results

{table}

One optimizer seed,512 TRAIN trajectories and64 developmental VAL trajectories from new parameter systems. These are controlled cross-task experiments using pinned native source, not paper reproductions. Checkpoints selected by factual VAL MSE only; edited plans never participate in selection. Each neural arm receives1000 updates/Adam. Observation prefix65 and future32. Sensor coordinates standardized using TRAIN only. No independent EVAL pool is opened.

Maximum early-prefix change across two families, suffix starts80/88 and signs+/-: {json.dumps(maxima)}. Maximum early RMS: {json.dumps(rms)}. Values are standardized sensor units. A zero prefix change alone does not establish correct future dynamics; full paired response errors and edited forecast MSE are retained in outputs/PAIRED_PLAN_RESULTS88_V1.json. ARX4 results are outputs/RIDGE_PLAN_RESULTS88_V1.json.

Original ordinary-fix factual/causality rejection gate: {fix}. Additional selected-vs-mean/untrained competence check: {extra_competent}. Lagged response-error<=1.05 stock in every paired cell: {response_comparable}. These are developmental observations, not independent-test significance claims.

Decision: **{route}**. Accepted novel research directions remain0; full user goal remains ACTIVE/UNACHIEVED.
''')
    write_md(C/'baseline_registry.md',f'''# C88 baseline registry

All neural cores originate from ALRhub/MTS3 commit ab971c0ec43df7664c4b727ca69d5b8dd0cab5da. Native files remain unchanged. Stock/lagged MTS3 initial state SHA256 is identical: {fits[0]['initial_sha256']}. Only the manager macro-action slice differs via an in-memory AST transform; the final-window skip is preserved. This is an ordinary causal comparator, not a proposed novel architecture. Native acRKN is an existing causal recurrent Kalman network with official latent dimensions/widths and its reported parameter count. ARX4 is a source-trained ordinary linear baseline; its six ridge values are selected by factual VAL only. It is not compute matched to neural training.

{table}

H changed prospectively from the source mobileRobot75 to32 to match the declared transferred task. Adam lr=.001/batch16/forecast-only NLL is our shared controlled protocol. These are not the original paper's benchmark settings or convergence claims.
''')
    write_md(C/'negative_evidence.md',f'''# C88 negative evidence and unresolved explanations

- Untrained stock MTS3 already beats persistence on both families. The original persistence threshold alone is inadequate evidence of learning. Selected-vs-untrained and TRAIN-mean comparisons are reported explicitly, not retroactively hidden.
- Ordinary causal indexing and native acRKN show early-prefix max changes {maxima['lagged']:.8g} and {maxima['acrkn']:.8g}; stock max is {maxima['stock']:.8g}. Functional invariance and factual/response accuracy are separate properties.
- Source whole-window conditioning is described in MTS3's paper; this is not just an accidental software typo. Evaluation uses a declared exogenous primitive-intervention contract, not the observational behavior-policy conditioning under which future actions may convey legitimate information.
- No new method is accepted. A one-line reindexing is not innovation. One model/source pool cannot support a field-wide failure claim; our earlier C79 tested native TSWM models without finding anticipation.
- The current comparison has one optimizer seed, selected developmental VAL, hand-designed discrete simulator families, and no real public action-conditioned benchmark. It does not establish generalization to all time-series world models or a stable optimization tradeoff.
- The original and stricter ordinary-control checks are reported without changing thresholds: original={fix}, added competence={extra_competent}, paired response comparability={response_comparable}.

Current scoped decision: {route}.
''')
    write_md(C/'hypothesis_space.md',f'''# C88 competing hypotheses

1. **Whole-window action abstraction causes anticipation after learning.** Measured on selected, trained checkpoints with suffix-only edits. Stock maximum early response={maxima['stock']:.8g}; per-cell values are retained. A structural dependency at random initialization was not sufficient evidence.
2. **The effect comes from masked future observations.** Identical base histories/zeroed masked observations and a separate masked-observation perturbation test distinguish it. Replay receipts are in outputs/PAIRED_PLAN_RESULTS88_V1.json.
3. **Ordinary causal alignment solves it without a novel model.** Same initialization/capacity/batches/loss/updates for stock and lagged. Original registered rejection gate={fix}; response comparability={response_comparable}. If those suffice, retire a method claim.
4. **An observed accuracy difference is training or representation failure.** Compare source competence, untrained/mean/persistence/ARX4 and native acRKN. One optimizer seed cannot distinguish a robust tradeoff from optimization noise. Do not infer irreducible failure from this pilot.
5. **All action-conditioned world models share the defect.** Not supported: C79's native TSWM tests found no anticipation, and causal recurrent models impose the relevant prefix structure. Do not generalize one hierarchy to the field.
''')
    write_md(C/'research_ideas.md',f'''# C88 research decision

Candidate: whether temporal action abstractions preserve the timing of primitive action effects. Proposed contribution type before qualification: possible empirical finding, not a novel method. Current decision: {route}.

An ordinary reindexing is insufficient novelty regardless of its performance. A viable remaining question would require a replicated and practically important limitation after strong causal alternatives, plus a mechanism beyond conventional causal masking/filtering. This experiment does not establish that residual. Do not expand the architecture or tune edited-plan test cases to maintain this route.
''')
    write_md(C/'final_research_direction.md',f'''# No final research direction established by C88

**Research Question:** Does slow action aggregation preserve primitive intervention timing, and is any failure left after ordinary causal alternatives?

**Evidence:** Three1000-update native-core fits on two nonlinear partially observed controlled families; identical stock/lagged initialization and batches; factual checkpoint selection; paired simulator suffix interventions; existing acRKN and source-trained ARX4 controls. Full metrics and negative evidence are retained. Maximum early-prefix change: {json.dumps(maxima)}.

**Existing Methods:** Recurrent causal filtering and ordinary time alignment already enforce nonanticipation. MTS3, TSWM, intervention audits, hierarchical endpoint planning and counterfactual-consistency training are direct prior art, with different action/query semantics. See LITERATURE_BOUNDARY88_V1.md.

**Proposed Idea:** No new architecture is promoted. The implemented lagged-index control is an ordinary comparator. Current route decision is {route}.

**Falsification:** A competent, accurate ordinary causal control eliminates the claimed need for a new method; lack of competent learning or a robust effect also fails to support a gap. Ordinary-control registered factual/causality gate={fix}, additional competence={extra_competent}, response comparability={response_comparable}.

**Feasibility:** All current fits run on the single RTX5090 with native source and small task-local data; raw trajectories, weights and prediction arrays stay on server data storage. Scripts/configurations/hashes/results/logs are published to the authorized GitHub repository.

**Next Experiment:** If the ordinary control suffices, stop this method route. If not, resolve the concrete remaining optimization/response discrepancy before any positive novelty claim. The separate original-budget modular-meta-learning campaign remains independent and must not be restarted or shortened. No independent EVAL pool has been used in C88, and no final top-conference direction is accepted.
''')
    write_md(C/'literature_review.md',(C/'LITERATURE_BOUNDARY88_V1.md').read_text(encoding='utf-8'))
    write_md(C/'README.md',f'''# C88: trained temporal-action-abstraction falsification

Decision: **{route}**. This is a developmental, source-pinned controlled comparison, not a reproduction of reported paper scores. It does not complete the research objective or establish a novel method.

Read PLAN_CAUSALITY_PROTOCOL88_V1.md for the prospective contract; experimental_results.md for all factual scores; outputs/PAIRED_PLAN_RESULTS88_V1.json for paired response/causality measures; negative_evidence.md and final_research_direction.md for the decision boundary. Original source files were not modified. Three native-core fits completed1000 updates each; no held-out EVAL was used.
''')
    write_md(C/'RUN_BOOK88_V1.md','''# C88 executed run book

Runtime: Python3.10/Torch2.8.0+cu128/NumPy1.26.4/SciPy1.15.3/PyYAML6.0.3 on RTX5090. Highest float32 matmul precision, TF32 disabled in this experiment, two CPU threads and disabled wandb. Actual original benchmark hyperparameters are not claimed.

Pinned source: ALRhub/MTS3 ab971c0ec43df7664c4b727ca69d5b8dd0cab5da. Clone/check out with line ending conversion disabled, then verify the52 file hashes in outputs/SOURCE_MANIFEST88_V1.json. The archive contains runtime-path placeholders; use a fresh research workspace containing cycle88/scripts and protocol documents and initialize the manifest source_root to the local pinned checkout. Supply a workspace-root STORAGE_LOCATION_V1.json with data_root, artifact_root, checkpoint_root and cache_root pointing to separate writable directories. Do not execute into archived completed outputs.

Executed order: plan_data88_v1.py; three-update stock runtime benchmark; capture_plan88_v1.py (stock/lagged/acRKN1000 updates each); ridge_plan88_v1.py; queue_evaluate88_v1.py/evaluate_plan88_v1.py after successful terminal campaign state; verify_plan88_v2.py; finalize_plan88_v3.py. The queue waits on the actual Windows process handle, not just a status filename. Training logs and checkpoint/data hashes are in the archive. selected88_v1.pt is selected by factual VAL MSE, final88_v1.pt also stores optimizer and NumPy/Torch/CUDA RNG states. Data/code seeds and equations are in the source/protocol.

Raw arrays/checkpoints are server-only and deliberately excluded from GitHub; deterministic generator and exact hashes support regeneration. Published paths use ${RESEARCH_ROOT}/${DATA_ROOT}/${ARTIFACT_ROOT}/${CHECKPOINT_ROOT}/${CACHE_ROOT} placeholders. No payload is synchronized to the local desktop. No checkpoint selection or training uses simulator hidden states, parameters, future observations, edited-plan metrics or an independent EVAL pool.
''')
    decision=dict(utc=utc(),route=route,ordinary_fix_original_gate=fix,additional_competence=extra_competent,paired_response_comparable=response_comparable,anticipation_maxima=maxima,anticipation_rms_maxima=rms,scientific_neural_fits=3,updates_per_fit=1000,optimizer_seeds=1,independent_eval_accessed=False,accepted_research_directions=0,goal_status='ACTIVE_UNACHIEVED',previous_goal_turn='NO_PROGRESS',current_goal_turn='PROGRESS_ACTUAL_FITS_AND_PAIRED_FALSIFICATION',data_and_weights_remain_remote=True,local_payload_sync=False)
    write(O/'RESEARCH_DECISION88_V1.json',decision)
    status=f'''# Current time-series world-model research status — C88

Goal ACTIVE/UNACHIEVED; accepted novel directions0. C87 remains a limited negative study with incomplete numerical rollout qualification. C88 now adds three actual1000-update learned-core fits, an ordinary ARX4, and paired primitive-action suffix tests on two nonlinear partially observed families. This is a controlled transfer experiment, not original-paper score reproduction.

Current C88 decision: {route}. Original ordinary-fix gate={fix}; additional competence={extra_competent}; paired response comparability={response_comparable}. A simple causal reindexing is not a novel method. Full results/negative evidence are in cycle88/. No positive generalization claim is supported by one developmental pool/one seed. Raw data, weights and predictions remain remote; GitHub receives source, compact results and logs only. The independent C71 full-budget campaign is preserved.
'''
    write_md(ROOT/'CURRENT_RESEARCH_STATUS_V88.md',status)
    pub=O/'publication_v1';pub.mkdir(parents=True,exist_ok=True);manifest=[]
    candidates=[p for p in C.iterdir() if p.suffix=='.md']+list((C/'scripts').glob('*.py'))
    names=['SOURCE_MANIFEST88_V1.json','NATIVE_QUALIFICATION88_V1.json','PLAN_DATA88_V1.json','PLAN_FIT88_V1_stock.json','PLAN_FIT88_V1_lagged.json','PLAN_FIT88_V1_acrkn.json','RIDGE_PLAN_RESULTS88_V1.json','PAIRED_PLAN_RESULTS88_V1.json','RESEARCH_DECISION88_V1.json','VERIFICATION88_V1.json']
    candidates += [O/n for n in names]
    candidates += list((O/'plan_campaign_v1').glob('*.log'))
    candidates += [O/'paired_plan_eval_v1/evaluation.log']
    for src in sorted(candidates):
        rel=src.relative_to(C).as_posix();dest=pub/rel;dest.parent.mkdir(parents=True,exist_ok=True)
        if src.suffix=='.json':raw=json.dumps(public_text(read(src)),indent=2,ensure_ascii=False,allow_nan=False).encode('utf-8')
        else:raw=public_text(src.read_text(encoding='utf-8')).encode('utf-8')
        with dest.open('xb') as f:f.write(raw)
        manifest.append(dict(path=rel,server_snapshot_file=str(dest),sha256=hashlib.sha256(raw).hexdigest(),git_blob_sha1=hashlib.sha1(b'blob '+str(len(raw)).encode()+b'\0'+raw).hexdigest(),bytes=len(raw),encoding='utf-8'))
    write(pub/'manifest.json',dict(utc=utc(),project='tswm_discovery_20261016_v1',cycle=88,files=manifest,raw_data_weights_or_predictions_included=False,decision=route))
    write(O/'CONTINUATION_HANDOFF88_V1.json',dict(**decision,publication_manifest_path=str(pub/'manifest.json'),publication_manifest_sha256=sha(pub/'manifest.json'),published=False,remaining='Publish verified snapshot; continue full research goal without promoting ordinary reindexing as innovation'))
    log(dict(stage='plan_results_finalized',route=route,publication_manifest_sha256=sha(pub/'manifest.json'),files=len(manifest)))

if __name__=='__main__':main()
