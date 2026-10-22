"""Verify the Validation-only branch and preserve negative evidence without a method claim."""
import pathlib,json,hashlib,csv,datetime
import numpy as np
ROOT=pathlib.Path(__file__).resolve().parents[1];OUT=ROOT/'outputs'
def read(p):return json.loads(pathlib.Path(p).read_text(encoding='utf-8-sig'))
def sha(p):return hashlib.sha256(pathlib.Path(p).read_bytes()).hexdigest()
assert read(OUT/'decoded_validation_v1_exit.json')['exit_code']==0
j=read(OUT/'decoded_validation_v1/RESULTS_V1.json');assert len(j['table'])==18
assert j['protocol_sha256']==sha(ROOT/'OFFICIAL_DECODED_CONTROL_PROTOCOL_V1.md')
fixed=read(OUT/'matched_cap_v1/RESULTS_V1.json');new=read(OUT/'decoded_control_v1/RESULTS_V1.json')
assert len(new['rows'])==6 and all(x['exit_code']==0 for x in new['rows'])
checks=[]
for row in j['table']:
 assert sha(row['arrays']['path'])==row['arrays']['sha256'] and row['epoch_reselected'] is False
 with np.load(row['arrays']['path'],allow_pickle=False) as z:a=z['origin_mse'];b=z['origin_mae']
 assert len(a)==row['validation_origins'] and np.isfinite(a).all() and np.isfinite(b).all()
 for v,key in [(a,'validation_mse'),(b,'validation_mae')]:
  e=abs(float(np.sum(v,dtype=np.float64)/len(v))-row[key]);assert e<2e-11;checks.append(e)
 reg=next(x for x in (new['rows'] if row['mode']=='decoded' else fixed['rows']) if x['seed']==row['seed'] and x['horizon']==row['horizon'] and (row['mode']=='decoded' or x['mode']==row['mode']))
 p=pathlib.Path(reg['metrics_path']).parent;m=read(p/'metrics.json')
 assert sha(p/'metrics.json')==reg['metrics_sha256']==row['metrics_sha256'] and sha(m['checkpoint_path'])==row['checkpoint_sha256']
 h=list(csv.DictReader((p/'history.csv').open(encoding='utf-8-sig')));assert len(h)==100 and min(h,key=lambda x:float(x['valid_loss']))['epoch']==str(row['selection_epoch'])
 if row['mode']=='decoded':
  cfg=read(p/'config.json');args=cfg['arguments'];assert args['mse_weight']==0 and args['cosine_weight']==0 and args['perceptual_weight']==1 and args['reconstruction_weight']==0
  assert cfg['flags']==['use_latent','freeze_encoder','freeze_decoder'] and read(p/'SOURCE_UNCHANGED_AFTER_V1.json')['upstream_unchanged']
  init=read(p/'INITIALIZATION_V1.json');assert init['steps_per_epoch']=={336:10,720:9}[row['horizon']]
summary=[]
for H in (336,720):
 q=dict(horizon=H)
 for mode in ('latent','raw','decoded'):
  rows=[x for x in j['table'] if x['mode']==mode and x['horizon']==H]
  for split in ('validation','test'):q[mode+'_'+split+'_mean']=float(np.mean([x[split+'_mse'] for x in rows]))
 q['decoded_to_raw_validation_ratio']=q['decoded_validation_mean']/q['raw_validation_mean'];q['decoded_test_excess_over_raw']=q['decoded_test_mean']/q['raw_test_mean']-1
 q['decoded_test_improvement_over_latent']=1-q['decoded_test_mean']/q['latent_test_mean'];q['within_band']=q['decoded_to_raw_validation_ratio']<=1.05;summary.append(q)
decision='STOP_CURRENT_ARCHITECTURE_MOTIVATION_OBJECTIVE_CONTROL_SUFFICIENT' if all(x['within_band'] for x in summary) else 'RUN_REGISTERED_ITRANSFORMER_COMPATIBILITY_CONTROLS'
assert decision==j['decision'];dense=read(OUT/'fixed100_dense_verification_v1/verification.json')
assert dense['status']=='TWELVE_FIXED100_DENSE_MODELS_VERIFIED'
if decision.startswith('STOP_'):assert not (OUT/'strong_itransformer_v1_launch.json').exists() and not (OUT/'strong_itransformer_v1').exists()
dest=OUT/'decoded_evidence_v1';dest.mkdir(exist_ok=False)
obj=dict(status='REGISTERED_ATTRIBUTION_DECISION_VERIFIED',utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),decision=decision,summary=summary,validation_origin_scalar_checks=len(checks),max_scalar_error=max(checks),source_validation_result_sha256=sha(OUT/'decoded_validation_v1/RESULTS_V1.json'),dense_verification_sha256=sha(OUT/'fixed100_dense_verification_v1/verification.json'),strong_itransformer_planned_cells=18,strong_itransformer_executed_cells=0 if decision.startswith('STOP_') else None,completed_new_fits_this_campaign=30,completed_dense_frozen_model_audits=24,full_causal_attribution=False,independent_generalization=False,new_method=False,accepted_final_directions=0,full_research_goal_achieved=False)
(dest/'verification.json').write_text(json.dumps(obj,indent=2),encoding='utf-8')
lines=['# 官方已有损失解释了验证集差距；停止当前扩模动机','','更新：2026-10-23。本轮六个decoded-loss对照均训练完100epoch。使用既有AE、既有预测器和源码已有的观测MSE选项，不是新方法。测试集早已在历史研究中查看；结果属于诊断。','','## 预先规定的决策','','按18个已选定checkpoint的确定性、逐元素加权完整Validation MSE判断，不重新选择任何epoch。若decoded的三种子平均不超过raw的1.05倍，两个horizon都满足，则停止本案新架构动机，不启动十八组iTransformer兼容性训练。该5%是描述性优先级阈值，不是统计等效检验。','','| Horizon | raw Val MSE | latent Val MSE | decoded Val MSE | decoded/raw |','| --- | ---: | ---: | ---: | ---: |']
for q in summary:lines.append(f"| {q['horizon']} | {q['raw_validation_mean']:.9f} | {q['latent_validation_mean']:.9f} | {q['decoded_validation_mean']:.9f} | {q['decoded_to_raw_validation_ratio']:.4f} |")
lines+=['','决策：`'+decision+'`。两个horizon都进入预设范围。十八组强预测器分支代码已准备，但条件未触发，实际执行数为0；不将其记为失败、复现或效果证据。','','## 已有Test诊断仍有残差','','| Horizon | raw Test MSE | latent Test MSE | decoded Test MSE | decoded相对raw仍高 |','| --- | ---: | ---: | ---: | ---: |']
for q in summary:lines.append(f"| {q['horizon']} | {q['raw_test_mean']:.9f} | {q['latent_test_mean']:.9f} | {q['decoded_test_mean']:.9f} | {q['decoded_test_excess_over_raw']*100:.2f}% |")
lines+=['','已有decoded损失使两种horizon的Test平均误差下降，但未关闭对raw的Test差距。不能声称整个失败已唯一归因为latent loss，也不能因Test残差推翻Val规则继续搜权重或扩模。验证与测试时期的差异、函数类、AE来源及原生选模噪声仍属于未分离因素。','','## 逐种子结果','','| Horizon | Mode | Seed | 确定性Val MSE | Test MSE | native选择统计量 | 选择epoch |','| --- | --- | ---: | ---: | ---: | ---: | ---: |']
for x in j['table']:lines.append(f"| {x['horizon']} | {x['mode']} | {x['seed']} | {x['validation_mse']:.9f} | {x['test_mse']:.9f} | {x['native_selection_statistic']:.9f} | {x['selection_epoch']} |")
lines+=['','native选择统计量是原生随机shuffle、batch MSE不加权平均的最小值；当前Val列是同一已选权重的确定性完整集合测量。两者不能混写，当前测量没有事后重选checkpoint。','','## 固定100模型的逐小时闭合','','上一轮报告明确没有完成的十二模型逐小时评估现已完成并独立核验。','', '| Horizon | latent逐小时MSE | raw逐小时MSE | raw相对降低 |','| --- | ---: | ---: | ---: |']
for q in dense['summary']:lines.append(f"| {q['horizon']} | {q['latent_dense_mean']:.9f} | {q['raw_dense_mean']:.9f} | {q['raw_relative_reduction']*100:.2f}% |")
lines+=['',f"十二个模型中{sum(x['dense_better_than_native'] for x in dense['table'])}个逐小时误差低于原稀疏起点，另外两个较高。原始六个latent模型全部改善的结论属于初轮权重，不能替换为当前模型的结果。整体仍不支持预期的系统性密集查询退化。重叠窗口不当作独立样本。",'', '## 研究判断','','“重构好的状态未必适合动力学”这个宽泛命题已被相邻文献直接研究；本次结果不能作为新颖性依据。已有观测损失能解释当前验证集表现落差，进一步削弱本案新架构的必要性。保留Test残差，但暂不推荐由此立项。','','本研究目标仍未完成，已接受最终新颖方向0。下一步先闭合C71既有冻结评估；若继续考察三篇主线，需要选择一个明确未被旧控制或近期文献覆盖的能力失败，并有独立系统证据。不能再用新名称包装当前损失变化。','','## 复核与产物','','本campaign累积30个新训练单元、24个冻结模型逐小时审计；其中本轮新增6个训练与12个逐小时审计。另测量18个既有已选权重的Val。当前脚本独立重算36个Val均值、核查配置/曲线/数据与权重哈希，并从Val重新计算分支；六个新Test指标已按独立分块算术核验。原始数据、预测和权重留服务器。','','[冻结协议](../../OFFICIAL_DECODED_CONTROL_PROTOCOL_V1.md) · [新颖性核查](../../NOVELTY_SCREEN_20261023_V2.md) · [机器结论](verification.json) · [逐小时复核](../fixed100_dense_verification_v1/verification.json) · [全部Val测量](../decoded_validation_v1/RESULTS_V1.json)','']
(dest/'experimental_results.md').write_text('\n'.join(lines),encoding='utf-8')
reports={
'literature_review.md':(ROOT/'NOVELTY_SCREEN_20261023_V2.md').read_text(encoding='utf-8-sig'),
'baseline_registry.md':(ROOT/'baseline_registry.md').read_text(encoding='utf-8-sig')+'\n\nCurrent closure adds six official decoded-loss DLinear controls, twelve fixed100 dense audits and eighteen frozen selected-weight Val measurements. Campaign totals:30 new fits and24 dense model audits. The conditional iTransformer18-cell branch was not triggered and has zero actual fits. It is prepared source, not reproduced performance.\n',
'negative_evidence.md':'# Negative evidence\n\nGeneric representation/transition compatibility has direct recent prior art. The registered ordinary decoded loss reaches the raw Validation band at both horizons, so the new-architecture motivation is stopped. Test residuals remain and are not claimed solved. Fixed100 dense queries improve10/12 models; this does not support a systematic dense-query breakdown. No iTransformer compatibility fit was run because its Validation condition was false. A rejected launcher prerequisite is an execution mistake, not a model failure.\n',
'hypothesis_space.md':'# Attribution state\n\nDifferent early stopping alone does not explain the original six-cell raw advantage. Existing decoded supervision closes the deterministic Validation gap under the registered band, so a new architecture is unnecessary for this case decision. It improves but does not close Test error. No unique causal explanation is assigned: temporal shift, decoder/predictor function class, AE lineage and native randomized validation selection remain distinct possibilities. The original broad novelty claim is rejected by primary literature; further work needs a distinct observable residual.\n',
'research_ideas.md':'# Candidate disposition\n\nRejected as a novel direction: generic good-reconstruction/bad-dynamics or simple decoded-loss repair. Current experiments give an ordinary explanation on Validation, not a new method. Test generalization discrepancy is an observed residual, but is not automatically a new world-model research gap; it still lacks mechanism isolation and independent-system evidence. Do not initiate Test-driven loss searches or the untriggered capacity branch.\n',
'final_research_direction.md':'# Current recommendation\n\nAccepted final novel directions:0. Stop the present architecture motivation under the registered Validation-only rule. Six new official decoded-loss fits and twelve fixed100 dense audits are complete;18 selected models have deterministic Val measurements. Raw/decoded Test residuals remain disclosed. Recent direct prior art rules out the generic compatibility phrase as novelty. The remaining historical C71 assessment must continue unchanged from its confirmed live process. The broader world-model research objective remains active; no replacement idea is accepted without a distinct, independently supported failure.\n'}
for name,tx in reports.items():(dest/name).write_text(tx,encoding='utf-8')
(dest/'manifest.json').write_text(json.dumps(dict(status=obj['status'],decision=decision,files={p.name:sha(p) for p in dest.iterdir() if p.is_file()},full_research_goal_achieved=False),indent=2),encoding='utf-8')
print(json.dumps(dict(status=obj['status'],decision=decision,summary=summary,strong_itransformer_executed_cells=obj['strong_itransformer_executed_cells'])),flush=True)
