"""Archive native qualification evidence, candidate lineage and next decisive test."""
import pathlib,json,hashlib,datetime
R=pathlib.Path(__file__).resolve().parents[1];O=R/'outputs';load=lambda p:json.loads(p.read_text(encoding='utf-8'));sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
q=load(O/'native_v3/qualification.json');v=load(O/'verification_v1.json');assert q['status']=='PASS_WITH_NARROW_METADATA_REPAIR' and v['status']=='PASS'
assert v['qualification_sha256']==sha(O/'native_v3/qualification.json')
summary=f'''本轮PROGRESS；整体ACTIVE_UNACHIEVED，接受最终方向0。新增近邻SPF固定commit a7624ae25b47b474d5a31ca22b3ac6e6907795dd，29个source files/Gitblob+SHA确认。执行原生core/collator/scheduler与原函数体的context adapter，两次50updates preflight共100步（均不作效果）；V2原假设失败保留，V3记录而不强求跨尺度共享。独立复核9000个调度值max{v['max_schedule_abs']:.3g}，checkpoint replay max{v['checkpoint_replay_max_abs']:.3g}，同尺度续接差{v['independent_within_level_resume_max_abs']:.3g}。此前真实TEST与本轮comparative effects均未评分。'''
lit='''# C43 Literature and novelty boundary

[SPF official code](https://github.com/stanfordmlgroup/spf) identifies the work as CVPR2026. The [arXiv v1 Methods §2.2](https://arxiv.org/html/2512.02268v1) describes direct clean outputs at different temporal resolutions and temporal funneling. That capability already exists; adding a coarse-timescale head is not a new research gap. The final CVF PDF fetch was denied, so this reading is explicitly the arXiv version plus pinned current author code, not a claim of checking every camera-ready detail.

The C41 sources mr-Diff, WaveletDiff and generative MLMC remain relevant, but SPF is a closer complete sampler for cheap temporal-resolution queries. MLMC requires its coarse independent term and paired correction to have the same coarse marginal; it does not require that coarse trajectory to equal a fine aggregation. A separate native coarse/fine call with common randomness is therefore a valid ordinary contender. This is a mathematical constraint on the proposed mechanism, not an experimental claim about its quality.

NCDSSM+modular dynamics, LIFTS and Compositional Interfaces remain the closest constraints for the main observation/dynamics route. Their general model class should not be used as a blanket rejection of new trainability/transfer interactions. C42 showed a known-structure signal, not an accepted deep framework.
'''
registry=f'''# C43 Baseline registry

{summary}

Baseline: author SPF numerical backbone, native temporal scheduler, native multiscale collate and AST-extracted training/sampling methods. Small setting: 8×8 output grid/onechannel, temporal lengths1/1/16, inner64/depth1/twoheads, sinusoidal positions, native SDPA,6Euler calls per query. Actual model parameter count{q['parameters']} includes inherited parameters; do not infer count from the tiny active width alone. Current environmentPython3.10/torch2.8cu128 differs from declared3.12/2.5.1, disclosed and numerically qualified. Source numerical files are unchanged; one unused text-encoder package re-export omitted in a runtime copy. Five wheels project-local, global environment unchanged.

Only optional partial-stage returned metadata initialized toNone after preserving its native runtime error. All formerly valid full outputs remain exactly unchanged. Full Lightning/climate-pipeline/headline reproduction has not run. The qualification checkpoint is not eligible for method-quality evidence. Full-condition projection gradients areNone in the native code path; that follows its no_grad context copy. This does not show the model cannot use conditioning. A scored pilot must declare this and allow an ordinary trainable-condition control when attributing failure.
'''
results=f'''# C43 Actual execution results

{summary}

V3 GPU50step qualification runtime{q['runtime_seconds']:.4f}s, peak allocated{q['peak_allocated_GiB']:.4f}GiB; excludes import/default parent construction and is not a runtime benchmark. Loss first{q['first_loss']:.6f}/last{q['last_loss']:.6f} on different random paths/timepoints; no convergence claim. Synthetic evolving-field fixture supplies only initial field toconditioning. Native labelsdecadal/yearly/monthly are API names with no climate interpretation.

|native target|shape|same RNG replay|unchanged original full API|
|---|---|---:|---:|
'''
for x in q['checks']:results+=f"|{x['timescale']}|{q['shapes'][x['timescale']]}|{x['replay_max_abs']}|{x['original_valid_modular_max_abs']}|\n"
results+=f'''
Original standard predict and seeded modular full outputs also agree exactly for each level. Optional stop_stage=1 failed because returnedindices was not assigned; metadata-onlyrepair preserves formerly validoutputs. Within-level split/resume afterrepair agrees exactly, verified in a new process from checkpoint.

Cross-resolution first-stage difference{q['shared_first_stage_max_abs']:.9g} at these discarded preflight weights. Independent scheduler endpoints by resolution: {v['stage0_training_timestep_end_by_level']}. Global time embedding depends on the whole path normalization, so identical cross-level prefix is not a valid native assumption. The difference is small and no quality consequence was tested. V2's mistakenzero assertion remains in outputs/native_v2/qualification.json; it did not invalidate native fullsampling.

All observations are API/gradient/schedule qualification. No MC/MLMC query error, model bias, realworld effect or proposedmethod benefit has yet been measured.
'''
negative=f'''# C43 Negative evidence

{summary}

- Direct cheap coarse generation is already supplied by SPF; no novelty claim from that capability.
- Stage resume does not imply an identical prefix across target resolutions. Original V2 assertion failed; an independent NumPy scheduler reconstruction explains the source of different time coordinates. No relaxedquality threshold or score manipulation occurred.
- The optional metadata defect has an ordinary repair. It does not justify a new model, and native standard coarse sampling already worked.
- The conditioning stem hasNone gradients through native detached context packing; random fixed features may still be used by later layers. No causal performance explanation is claimed.
-50step loss is not convergence evidence. No effect table may use the qualification model. Initial unused-import failure and source adaptation are disclosed.
- Internal cross-resolution differences do not imply worse true conditional simulation; C39's negative lesson remains governing.
'''
hyp='''# C43 Competing hypotheses and discriminating tests

H1: native SPF plus ordinary shared-random-number MLMC already supplies useful multi-resolution path queries. Test a genuinely fitted native conditional law, correcting with separately generated coarse draws of the same marginal. If it wins, no special architecture is needed for that utility.

H2: an exactly shared clean coarse marginal improves paired correction variance/cost enough to matter beyond H1. It must improve actual total error per end-to-endtime, not only reduce a coordinate-dependent residual. Strong ordinary coupling, an AR/SSM generator and the overhead of pilot allocation must be included. No need to force a newarchitecture before this evidence.

H3: apparent query gain is merely cheap coarse readout or a worse fine law. Freeze nontrivial function families; compare a high-sample fitted-law reference separately from known-process truth. Report bias and stochastic error independently. Include high-frequency/threshold functions expected to fail.

H4: observation/dynamics composition has a useful scalable learning interaction beyond C42 ordinary HMM. Main fallback: likelihood/evidence uncertainty × reusable nonlinear dynamics with same input/anchors/budget. Closest complete NCDSSM+modular implementation must be present. A generic SSM graph can express many models; novelty must be assessed at the actual learning recipe and evidence, without relabeling a known implementation.
'''
ideas='''# C43 Idea evolution and boundaries

Target: generic ICML/ICLR/NeurIPS/KDD family, specific venue unknown. Exploratory mechanism discovery, not numeric idea scoring or manuscript drafting. Inputs: userframeworkscope, C41/C42actualcontrols, new pinned SPF code. Data and complete method choices remain contingent on decisive evidence.

|internal child|parent/operator|mechanism/claim|nearest overlap and minimum falsifier|status|
|---|---|---|---|---|
|A0 ordinary factorization|C42/refine|joint observation/dynamics sharing yields finite-data combination gain|exact11 is an ordinary HMM; already known model|positive structural signal; no new-method claim|
|A1 uncertainty-bearing observation interface|C41/combine|uncertain observation evidence supports nonlinear dynamic-module reuse|NCDSSM+modular dynamics; same-data strongest complete combination absorbs gain|needs mechanism/evidence; main fallback|
|B0 resolution-aware path queries|C41B/refine|native coarse/fine sampling with query-dependent MLMC improves useful error/cost|SPF+classical common-random-number coupling, cheapSSM/AR dominates|new close baseline qualified; next decisive ordinary-control pilot|
|A2 coordinate-free mechanism transplant|A1/instrument|distinguish learned-coordinate alignment from transferable observable transition behavior|PSR/PSIM and dynamical alignment; may simply rename old objectives|needs-search; not selected, no new loss proposed|

One overlap challenge materially narrowed B0: SPF already has clean multiscale outputs and resumable APIs. One evidence challenge: intermediate-state mismatch is not task harm; ordinary separate-call coupling needs no exact aggregation equality. Revised B0 tests useful query variance/cost first. This is an admissible evidence route for a potential framework recipe, not a guarantee of novelty. A1 remains a different fallback; do not arbitrarily force the entire userproject into rareevents or climate data.

Risks: known-component equivalence(requiresnewresult), too fewrealworlddomains(requiresnewresult), inadequatebaselineoptimization(design/evidencefixable), unclearpracticalqueryuse(needsmechanism/evidence), sourceadapterlimits(reproducibilitydisclosure). No score or acceptanceprobability is assigned.
'''
final=f'''# C43 Current research direction

**Research Question**：可复用TS world model能否以与查询所需信息相适应的成本提供有用的随机未来推演，而不限于固定horizon逐点误差？观测/动态组合复用仍是并列开发目标。

**Evidence**：{summary} 新证据证明已有SPF确实有多尺度/续接能力；它没有证明任何新方法优于它。

**Existing Methods**：SPF直接覆盖粗尺度生成，ordinaryMLMC允许合法的分开耦合调用；cheap AR/SSM必须保留。主路线的NCDSSM/LIFTS/modular近邻不能遗漏。

**Proposed Idea**：当前只提出“查询所需尺度×采样分配”的可复用框架候选；exact shared clean marginal的必要性未知，不预先添加loss或架构。新贡献可能来自必要交互与有效训练/推断配方，不能来自组件更名。

**Falsification**：SPF+ordinarycoupling或cheap AR/SSM支配质量成本；只有平凡均值函数获益；模型bias主导；需要逐数据集换结构；以内部残差代替真实效用。出现则否决对应主张。

**Feasibility**：固定source与nativecore已在RTX5090执行并checkpointreplay。无需下载庞大ClimateSuite；后续先fittedknownprocess训练，确认价值后才扩到不同真实领域。

**Next Experiment**：新协议冻结两seed真正trained conditional SPF、同一fittedlaw highsample reference、独立process truth、nativecommonrandomcouplingMLMC与MC/cheapARSSM、非平凡函数与总成本。qualificationweights全部排除。只有ordinarycontrol有无法覆盖的具体问题时再设计shared-marginal架构。

候选A1为不同fallback，最终接受方向0；整体ACTIVE_UNACHIEVED。
'''
reports={'literature_review.md':lit,'baseline_registry.md':registry,'experimental_results.md':results,'negative_evidence.md':negative,'hypothesis_space.md':hyp,'research_ideas.md':ideas,'final_research_direction.md':final,'CURRENT_STATE.md':'# C43 — native qualification completed, goal active\n\n'+summary+'\n\nSee SCIENTIFIC_DECISION.md. No scored method pilot or realdataTEST was run. Do not call the overall research objective complete.\n'}
for name,body in reports.items():(R/name).write_text(body,encoding='utf-8')
names=list(reports)+['SCIENTIFIC_DECISION.md','NATIVE_QUALIFICATION_PLAN.md','IMPORT_ADAPTATION.md']
manifest={'reports':{name:sha(R/name) for name in names},'qualification_sha256':sha(O/'native_v3/qualification.json'),'verification_sha256':sha(O/'verification_v1.json'),'source_receipts_sha256':sha(R/'source_archive/spf/CACHE_RECEIPTS.json'),'runtime_adapter_sha256':sha(R/'runtime_source/spf/ADAPTER_MANIFEST.json'),'formal_comparative_fits':0,'discarded_preflight_updates':100,'accepted_final_directions':0,'goal':'ACTIVE_UNACHIEVED'}
(R/'REPORT_MANIFEST.json').write_text(json.dumps(manifest,indent=2),encoding='utf-8');print(json.dumps({'reports':len(names),'discarded_preflight_updates':100,'formal_comparative_fits':0,'goal':'ACTIVE_UNACHIEVED'}))
