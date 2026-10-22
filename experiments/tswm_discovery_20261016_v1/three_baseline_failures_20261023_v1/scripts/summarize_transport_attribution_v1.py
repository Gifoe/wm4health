"""Interpret verified frozen compositions; stop the current architecture story without hiding residuals."""
import pathlib,json,hashlib,datetime
import numpy as np
ROOT=pathlib.Path(__file__).resolve().parents[1];OUT=ROOT/'outputs';DEST=OUT/'transport_evidence_v1'
def read(p):return json.loads(pathlib.Path(p).read_text(encoding='utf-8-sig'))
def sha(p):return hashlib.sha256(pathlib.Path(p).read_bytes()).hexdigest()
v=read(DEST/'verification.json');raw=read(OUT/'transport_audit_v1/RESULTS_V1.json')
assert v['status']=='FROZEN_COMPOSITION_SCORES_AND_ERROR_IDENTITIES_VERIFIED' and v['source_result_sha256']==sha(OUT/'transport_audit_v1/RESULTS_V1.json')
assert not (OUT/'strong_itransformer_v1_launch.json').exists()
facts=[]
for H in (336,720):
 val=next(x for x in v['summary'] if x['horizon']==H and x['split']=='val');test=next(x for x in v['summary'] if x['horizon']==H and x['split']=='test');q=dict(horizon=H)
 for split in ('val','test'):
  r=[x for x in raw['records'] if x['horizon']==H and x['split']==split]
  q[split+'_history_recon']=float(np.mean([x['diagnostics']['history_reconstruction_mse'] for x in r]));q[split+'_future_truth_recon']=float(np.mean([x['diagnostics']['true_future_reconstruction_mse'] for x in r]))
 q['history_recon_test_to_val_ratio']=q['test_history_recon']/q['val_history_recon'];q['truth_recon_test_to_val_ratio']=q['test_future_truth_recon']/q['val_future_truth_recon']
 for arm in ['raw_after_input_recon','output_recon_after_raw','both_recon_transport','latent_decoded']:
  q[arm+'_test_excess']=test['forecasts'][arm]['relative_mse_excess_over_raw']
 for split,row in [('val',val),('test',test)]:
  q[split+'_decoded_shift']=row['forecasts']['latent_decoded']['shift_mse'];q[split+'_decoded_cross']=row['forecasts']['latent_decoded']['error_cross_term']
 facts.append(q)
lines=['# 冻结组合后的归因判断','','本轮0个optimizer update，使用原来的18个已选预测器权重和同一发布AE；12个split×horizon×seed审计单元全部完成，324个独立数值/误差恒等式检查通过。没有重选checkpoint、修改loss、启动被Val规则排除的强模型分支或根据Test选择组合。','','## 可直接观察的差异','','| Horizon | 历史重构Val MSE | 历史重构Test MSE | Test/Val |','| --- | ---: | ---: | ---: |']
for q in facts:lines.append(f"| {q['horizon']} | {q['val_history_recon']:.9f} | {q['test_history_recon']:.9f} | {q['history_recon_test_to_val_ratio']:.3f}× |")
lines+=['','| Horizon | F(R(x))相对raw Test差距 | R(F(x))差距 | 双侧组合差距 | decoded预测器差距 |','| --- | ---: | ---: | ---: | ---: |']
for q in facts:lines.append(f"| {q['horizon']} | {q['raw_after_input_recon_test_excess']*100:.2f}% | {q['output_recon_after_raw_test_excess']*100:.2f}% | {q['both_recon_transport_test_excess']*100:.2f}% | {q['latent_decoded_test_excess']*100:.2f}% |")
lines+=['','历史重构对原raw预测器的影响并非可以忽略；输出重构也会产生可测的额外误差。相同AE在两个时期的重构质量不同，因而不能把所有Test残差直接归到潜态转移学习。','双侧冻结组合在H336比decoded差，在H720比decoded好；它不是统一胜出的新方法。F(R(x))也不是另一个训练后的模型，不能把其增量直接加到任意潜态模型误差上。','','## 误差抵消不能省略','','| Horizon | Val decoded偏移平方 | Val有符号交叉项 | Test偏移平方 | Test有符号交叉项 |','| --- | ---: | ---: | ---: | ---: |']
for q in facts:lines.append(f"| {q['horizon']} | {q['val_decoded_shift']:.9f} | {q['val_decoded_cross']:.9f} | {q['test_decoded_shift']:.9f} | {q['test_decoded_cross']:.9f} |")
lines+=['','decoded与raw预测差异的平方并不等于它们的风险差；交叉项在两个时期明显不同。该代数事实不独自证明validation overfitting、shortcut learning或某个因果机制。对一个更好预测器而言，负交叉项本来也可以是正常的有效修正。','','## 当前排除与保留','','- “仅仅早停轮数不同”不足以解释原先全部差距；固定100对照已经排除这种完整解释。','- 不能主张“历史AE重构对这个raw预测器几乎没有影响”：本轮观测到3.35%–5.01%的Test影响。','- 不能把E的不可逆信息丢失当成已证实原因：R还包含解码器，即使E保留信息，D也可能引入重构误差。','- 不能把R(y)或R(F(x))的误差称为潜态预测下界，也不能把双侧组合看作容量匹配或独立训练方法。','- AE预训练来源、跨时期误差、模型函数类、优化和native随机Val选模仍没有全部分离；完整Test残差尚未唯一归因。','','## 研究处置','','停止由此开发新loss、新Loop或新架构。这个案例提供的是配方级复现差距及普通控制证据，尚不能确立通用world-model研究缺口。宽泛的重构/动力学不兼容已有直接先行工作。保留负面结果和Test残差，原Val分支规则继续有效。','已经核对C79旧记录：它既测过无anticipation，也测过lagged impulse支持；现成causal decay/conv/attention选项已在原文，不能再把“点态Gate没有滞后作用”当作新发现重新实验。','当前最终接受方向0。C71继续同一冻结评估，实际终态到达前不据部分seed下结论。下一项研究问题必须来自一个不同的、尚未被原有控制或直接先行工作覆盖的能力残差，并在独立系统确认。','','[六路径全表和公式](experimental_results.md) · [复核机器表](verification.json) · [既有新颖性核查](../../NOVELTY_SCREEN_20261023_V2.md) · [C79已做的行动时间审计](../../../cycle79/PLAN_CONTRACT_FINDING79_V2.md)','']
outputs={'INTERPRETATION_V1.md':'\n'.join(lines),
'literature_review.md':'# Literature boundary\n\nSee [the primary-source novelty screen](../../NOVELTY_SCREEN_20261023_V2.md). No new broad state/transition-compatibility novelty is established. The present frozen compositions are diagnostic controls, not another primary-paper reproduction or new method.\n',
'baseline_registry.md':(OUT/'decoded_evidence_v1/baseline_registry.md').read_text(encoding='utf-8-sig')+'\n\nThis follow-up adds zero fits, twelve split/seed/horizon audit cells and six fixed compositions per cell. All three learned forecast paths qualify against existing predictions/Val scores. The existing C79 impulse audit is reused rather than repeated.\n',
'negative_evidence.md':'# Negative evidence\n\nThe input roundtrip is not negligible for the existing raw predictor; output roundtrip also changes forecast risk. Both-side transport is not a uniform improvement over decoded training. These observations contradict simplistic attribution of all residuals to the latent transition, but do not uniquely prove an encoder/decoder cause. Existing ordinary decoded training already reaches the predeclared Val band, so the conditional18-fit branch remains untriggered.\n',
'hypothesis_space.md':'# Remaining explanations\n\nConfirmed: AE roundtrip quality differs across the two periods and has a measurable effect on the unchanged raw forecaster. Not identified uniquely: encoder information loss, decoder bias, pretraining lineage, predictor function class, optimization and stochastic native selection. Error cross terms matter; a negative cross term is not itself evidence of shortcut learning. No new architecture hypothesis is accepted.\n',
'research_ideas.md':'# Candidate status\n\nReject generic good-reconstruction/bad-dynamics as novelty. Current residual is a source-recipe and transfer-attribution case, with several ordinary explanations still confounded. Do not turn the untrained transported composition into a new world-model method. Reusing the prior C79 lagged/anticipation audit prevents duplicate discovery. A future candidate needs a distinct capability failure and independent-system evidence.\n',
'final_research_direction.md':'# Current recommendation\n\nAccepted final novel directions:0. Stop method development for this case; retain the measured residual and all ordinary controls. The frozen audit completes12 cells with zero updates and validates error identities; it does not provide unique causal attribution or independent generalization. Do not reopen the untriggered capacity branch, generic decoder-alignment idea, or already audited pointwise-Gate lag issue. Continue the existing C71 assessment unchanged and require a distinct, externally unclosed failure for the next primary research question. The overarching goal remains active.\n'}
for name,tx in outputs.items():
 with (DEST/name).open('x',encoding='utf-8') as f:f.write(tx)
with (DEST/'interpretation_facts_v1.json').open('x') as f:json.dump(dict(facts=facts,optimizer_updates=0,full_causal_attribution=False,novel_method=False),f,indent=2)
with (DEST/'manifest_addendum_v1.json').open('x') as f:json.dump(dict(source_verification_sha256=sha(DEST/'verification.json'),files={name:sha(DEST/name) for name in list(outputs)+['interpretation_facts_v1.json']},accepted_final_directions=0),f,indent=2)
print(json.dumps(dict(status='FROZEN_ATTRIBUTION_INTERPRETATION_RECORDED',facts=facts,accepted_final_directions=0)),flush=True)
