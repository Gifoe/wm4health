"""Recompute every frozen composition score and its signed error decomposition."""
import pathlib,json,hashlib,datetime
import numpy as np
ROOT=pathlib.Path(__file__).resolve().parents[1];OUT=ROOT/'outputs'
def read(p):return json.loads(pathlib.Path(p).read_text(encoding='utf-8-sig'))
def sha(p):return hashlib.sha256(pathlib.Path(p).read_bytes()).hexdigest()
assert read(OUT/'transport_audit_v1_exit.json')['exit_code']==0
source=OUT/'transport_audit_v1/RESULTS_V1.json';j=read(source);assert j['status']=='TWELVE_FROZEN_COMPOSITION_CELLS_COMPLETE'
assert [(x['horizon'],x['seed'],x['split']) for x in j['records']]==[(h,s,split) for h in (336,720) for s in (42,43,44) for split in ('val','test')]
assert j['protocol_sha256']==sha(ROOT/'FROZEN_TRANSPORT_PROTOCOL_V1.md')
for p,h in j['source_sha256'].items():assert sha(p)==h
checks=[];table=[];paths=[]
for row in j['records']:
 assert row['optimizer_updates']==0 and row['epoch_reselected'] is False and row['future_targets_only_for_scoring']
 p=row['arrays']['path'];assert sha(p)==row['arrays']['sha256']
 with np.load(p,allow_pickle=False) as z:
  mse=z['origin_mse'];mae=z['origin_mae'];shift=z['shift_mse'];cross=z['error_cross_term'];rx=z['history_reconstruction_mse'];ry=z['true_future_reconstruction_mse'];ids=z['relative_origin']
 assert mse.shape==mae.shape==shift.shape==cross.shape==(row['origins'],6) and np.array_equal(ids,np.arange(len(ids))*60)
 assert all(np.isfinite(x).all() for x in [mse,mae,shift,cross,rx,ry])
 identity=float(np.max(abs((mse-mse[:,0,None])-shift-cross)));assert identity<2e-10
 checks.append(identity)
 for i,name in enumerate(row['arms']):
  for a,key in [(mse,'mse'),(mae,'mae'),(shift,'shift_mse'),(cross,'error_cross_term')]:
   err=abs(float(a[:,i].sum(dtype=np.float64)/len(a))-row['forecasts'][name][key]);assert err<2e-11;checks.append(err)
  table.append(dict(horizon=row['horizon'],seed=row['seed'],split=row['split'],arm=name,**row['forecasts'][name]))
 for a,key in [(rx,'history_reconstruction_mse'),(ry,'true_future_reconstruction_mse')]:
  err=abs(float(a.sum(dtype=np.float64)/len(a))-row['diagnostics'][key]);assert err<2e-11;checks.append(err)
 assert row['diagnostics']['true_future_reconstruction_is_not_forecast']
 for arm,x in row['checkpoint_receipts'].items():assert sha(x['path'])==x['sha256']
 if row['split']=='test':
  assert set(row['baseline_qualification'])=={'raw','latent','decoded'}
  for q in row['baseline_qualification'].values():assert q['prediction_max_abs_error']<=q['bound'] and q['truth_max_abs_error']==0
 paths.append(dict(horizon=row['horizon'],seed=row['seed'],split=row['split'],**row['arrays']))
summary=[]
for h in (336,720):
 for split in ('val','test'):
  rr=[x for x in table if x['horizon']==h and x['split']==split];q=dict(horizon=h,split=split,forecasts={})
  for arm in j['records'][0]['arms']:
   r=[x for x in rr if x['arm']==arm];q['forecasts'][arm]={key:float(np.mean([x[key] for x in r])) for key in ['mse','mae','shift_mse','error_cross_term']}
  base=q['forecasts']['raw']['mse']
  for arm,x in q['forecasts'].items():x['relative_mse_excess_over_raw']=x['mse']/base-1
  summary.append(q)
dest=OUT/'transport_evidence_v1';dest.mkdir(exist_ok=False)
obj=dict(status='FROZEN_COMPOSITION_SCORES_AND_ERROR_IDENTITIES_VERIFIED',utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),summary=summary,table=table,scalar_checks=len(checks),max_scalar_error=max(checks),source_result_sha256=sha(source),arrays=paths,optimizer_updates=0,new_method=False,unique_causal_attribution=False,independent_confirmation=False,accepted_final_directions=0,full_research_goal_achieved=False)
(dest/'verification.json').write_text(json.dumps(obj,indent=2),encoding='utf-8')
lines=['# 冻结组合审计：AE存在本身是否解释剩余差距？','','本轮没有训练、微调、校准或重选checkpoint。复用已选raw/native-latent/decoded预测器和发布AE，固定两个horizon、三颗seed、Val/Test两种split，共12个冻结审计单元。每个单元计算六条路径；这些路径不是六个新训练模型或独立系统。','','记F为冻结普通DLinear，R=D∘E为冻结AE重构。真实未来只在全部预测完成后用于评分；R(y)只作重构诊断，未列入预测排名。','','## 完整平均结果','','| Horizon | Split | F(x) | F(R(x)) | R(F(x)) | R(F(R(x))) | native latent | decoded loss |','| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |']
for q in summary:
 vals=[q['forecasts'][a]['mse'] for a in j['records'][0]['arms']];lines.append('| '+str(q['horizon'])+' | '+q['split']+' | '+' | '.join(f'{v:.9f}' for v in vals)+' |')
lines+=['','每个数是三颗初始化的平均MSE；并未根据该表选择方法、权重或新的训练超参数。原先基于Val停止架构扩展的决定保持不变。','','## 误差变化及抵消','','以d=P−F(x)、r=F(x)−y计算，MSE(P)−MSE(F)=E[d²]+2E[rd]。后项可正可负，不能直接把重构误差当成预测误差增加量。','','| Horizon | Split | Path | ΔMSE | E[d²] | 2E[rd] |','| --- | --- | --- | ---: | ---: | ---: |']
for q in summary:
 base=q['forecasts']['raw']['mse']
 for arm in ['raw_after_input_recon','output_recon_after_raw','both_recon_transport']:
  x=q['forecasts'][arm];lines.append(f"| {q['horizon']} | {q['split']} | {arm} | {x['mse']-base:.9f} | {x['shift_mse']:.9f} | {x['error_cross_term']:.9f} |")
lines+=['','## 可支持的归因范围','','F(R(x))测量历史重构对这个已训练F的影响；R(F(x))测量一个特定输出重构对同一F的影响。R(F(R(x)))是已有预测器和AE的确定性组合，不是新学习的latent transition，也没有匹配模型深度或训练。','这些路径的差异不能唯一分离函数类、优化、数据时期变化和AE预训练来源。R(F(x))不保证找到最优latent preimage，因此其误差不是任何潜态模型的下界。R(y)诊断同样不是预测误差下界。','如果冻结组合优于某个潜态预测器，可反驳“AE存在本身必然造成同样差距”的强说法，但不能据此声称新的可复用世界模型已经学成。宽泛的表示—动力学不兼容已有直接先行工作，本轮不构成新颖性证据。','','## 复核','',f'独立读取12个per-origin数组并检查{len(checks)}个标量/恒等式，最大误差{max(checks):.3g}；三类学习模型均重放到原Test预测和原确定性Val分数，真实目标与旧Test完全一致。权重、AE、原始数据、源码哈希保持不变。','完整逐seed/逐split/逐path结果在机器表中。数据、预测和权重未复制本地；新诊断数组留服务器。','','## 当前决定','','不撤销上轮Val规则，不启动十八组未触发扩模，不把本轮组合当成新方法。保留实际残差和被否定的解释。通用world-model研究目标仍未完成，接受最终新颖方向0。','','[冻结协议](../../FROZEN_TRANSPORT_PROTOCOL_V1.md) · [全部机器结果](verification.json) · [上一轮普通损失结论](../decoded_evidence_v1/experimental_results.md)','']
(dest/'experimental_results.md').write_text('\n'.join(lines),encoding='utf-8')
(dest/'manifest.json').write_text(json.dumps(dict(status=obj['status'],files={p.name:sha(p) for p in dest.iterdir() if p.is_file()},full_research_goal_achieved=False),indent=2),encoding='utf-8')
print(json.dumps(dict(status=obj['status'],summary=summary,max_scalar_error=max(checks))),flush=True)
