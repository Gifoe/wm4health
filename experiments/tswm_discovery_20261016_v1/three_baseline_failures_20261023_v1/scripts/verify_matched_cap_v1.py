"""Independently check the registered fixed-cap attribution study and report its limits."""
import os
os.environ['OMP_NUM_THREADS']='2';os.environ['OPENBLAS_NUM_THREADS']='2';os.environ['MKL_NUM_THREADS']='2'
import pathlib,json,hashlib,csv,datetime
import numpy as np
ROOT=pathlib.Path(__file__).resolve().parents[1];OUT=ROOT/'outputs'
def read(p):return json.loads(pathlib.Path(p).read_text(encoding='utf-8-sig'))
def sha(p):
 h=hashlib.sha256()
 with pathlib.Path(p).open('rb') as f:
  while b:=f.read(1048576):h.update(b)
 return h.hexdigest()
assert read(OUT/'matched_cap_v1_exit.json')['exit_code']==0
regpath=OUT/'matched_cap_v1/RESULTS_V1.json';reg=read(regpath)
assert reg['status']=='TWELVE_FIXED100_CONTROLS_CLOSED'
assert reg['protocol_sha256']==sha(ROOT/'MATCHED_CAP_PROTOCOL_V1.md')
assert [(x['horizon'],x['seed'],x['mode']) for x in reg['rows']]==[(h,s,m) for h in (336,720) for s in (42,43,44) for m in ('latent','raw')]
old=read(OUT/'primary_evidence_v1/verification.json');rows=[];checks=[];receipts=[]
for row in reg['rows']:
 folder=pathlib.Path(row['metrics_path']).parent;m=read(folder/'metrics.json');c=read(folder/'config.json');args=c['arguments']
 assert row['exit_code']==0 and m['epochs']==100 and m['early_stop_break_ignored']
 assert sha(folder/'metrics.json')==row['metrics_sha256'] and sha(m['checkpoint_path'])==m['checkpoint_sha256'] and sha(folder/'predictions.npz')==m['predictions_sha256']
 assert args['seq_len']==720 and args['pred_len']==row['horizon'] and args['seed']==row['seed'] and args['step']==60 and args['batch_size']==32 and args['train_epochs']==100 and args['learning_rate']==.001
 assert args['label_len']==0 and args['accum_steps']==1 and args['lradj']=='cosine'
 assert m['source_commit']=='7c8ae947ee1220bf4e788ace6bc2f0f122cb26c2' and m['data_sha256']=='7e45845d54c5219bad0ae6bc1b5316cf8ff9cead5d33fa998a5a51c2e4a497ad'
 assert m['num_parameters']=={336:484512,720:1038240}[row['horizon']]
 h=list(csv.DictReader((folder/'history.csv').open(encoding='utf-8-sig',newline='')))
 assert [int(x['epoch']) for x in h]==list(range(1,101))
 best=min(h,key=lambda x:float(x['valid_loss']));assert int(best['epoch'])==m['best_epoch']
 source=read(folder/'SOURCE_UNCHANGED_AFTER_V1.json');assert source.get('upstream_unchanged',source.get('native_source_unchanged',False)) or source.get('source_native_unchanged',False)
 before=next(x for x in old['table'] if x['mode']==row['mode'] and x['seed']==row['seed'] and x['horizon']==row['horizon'])
 with np.load(folder/'predictions.npz',allow_pickle=False) as z:
  pred=z['pred'];true=z['true'];ss=sa=0.;n=0
  for j in range(0,len(pred),13):
   error=pred[j:j+13].astype(np.float64)-true[j:j+13].astype(np.float64);assert np.isfinite(error).all()
   ss+=np.einsum('ijk,ijk->',error,error);sa+=np.abs(error).sum(dtype=np.float64);n+=error.size
 assert pred.shape==true.shape and pred.shape[1:]==(row['horizon'],321)
 for key,value in [('reproduced_mse',ss/n),('reproduced_mae',sa/n)]:
  err=abs(value-m[key]);assert err<2e-11;checks.append(err)
 steps={336:10,720:9}[row['horizon']]
 if row['mode']=='raw':assert m['actual_training_updates']==100*steps and m['steps_per_epoch']==steps
 rows.append(dict(mode=row['mode'],horizon=row['horizon'],seed=row['seed'],native_mse=m['reproduced_mse'],native_mae=m['reproduced_mae'],best_epoch=m['best_epoch'],actual_epochs=100,optimizer_updates=steps*100,previous_earlystop_mse=before['native_mse'],previous_epochs=before['epochs'],relative_mse_change=(m['reproduced_mse']/before['native_mse']-1)))
 receipts.append(dict(mode=row['mode'],horizon=row['horizon'],seed=row['seed'],metrics_path=row['metrics_path'],metrics_sha256=row['metrics_sha256'],history_sha256=sha(folder/'history.csv'),config_sha256=sha(folder/'config.json'),checkpoint_sha256=m['checkpoint_sha256'],predictions_sha256=m['predictions_sha256']))
summary=[]
for horizon in (336,720):
 q=dict(horizon=horizon)
 for mode in ('latent','raw'):
  vals=np.array([x['native_mse'] for x in rows if x['horizon']==horizon and x['mode']==mode]);q[mode+'_mean']=float(vals.mean());q[mode+'_seed_sd']=float(vals.std(ddof=1))
 q['raw_relative_reduction']=(q['latent_mean']-q['raw_mean'])/q['latent_mean'];q['raw_lower_all_three_seeds']=all(next(x['native_mse'] for x in rows if x['horizon']==horizon and x['seed']==seed and x['mode']=='raw')<next(x['native_mse'] for x in rows if x['horizon']==horizon and x['seed']==seed and x['mode']=='latent') for seed in (42,43,44));summary.append(q)
survives=all(x['raw_lower_all_three_seeds'] for x in summary)
verdict='RAW_ADVANTAGE_SURVIVES_FIXED100_ON_BOTH_HORIZONS' if survives else 'NO_UNIFORM_FIXED100_LATENT_DEFICIT'
obj=dict(utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),status='TWELVE_FIXED100_FITS_NUMERICALLY_AND_SELECTION_VERIFIED',verdict=verdict,table=rows,summary=summary,max_abs_metric_error=max(checks),validation_minimum_checked=True,actual_100_epochs_checked=True,update_count_by_horizon={336:1000,720:900},source_registry_sha256=sha(regpath),protocol_sha256=reg['protocol_sha256'],receipts=receipts,new_method=False,accepted_final_directions=0,independent_confirmation=False,full_research_goal_achieved=False)
dest=OUT/'matched_evidence_v1';dest.mkdir(exist_ok=False)
(dest/'verification.json').write_text(json.dumps(obj,indent=2),encoding='utf-8')
lines=['# 固定训练预算后的失败归因','','三篇主线仍为WorldTS、LatentTSF、TSWM。本报告只更新其中LatentTSF的首个可重复失败，不把三篇不同信息协议合并排名。','','本轮累积完成24个训练单元：6个原生潜态重复、6个官方raw模式对照、12个固定100epoch归因对照；另有前12个冻结模型的逐小时推断审计。这不等于24次完整论文复现。','','## 已冻结的对照','','两种模式都使用作者原上限100epoch，原生损失、batch32、step60、学习率/余弦调度和验证集最小损失选模保持不变，只忽略early-stop的break。这是有声明的归因实验，不是未经修改的原论文复现。每个horizon的双方分别执行1000/900个小批更新。参数量相同不代表函数类相同；实际初始权重、AE额外计算和每步FLOPs没有匹配。','','| Horizon | Mode | Seed | 原早停MSE | 固定100 MSE | MAE | 验证选择epoch |','| --- | --- | ---: | ---: | ---: | ---: | ---: |']
for x in rows:lines.append(f"| {x['horizon']} | {x['mode']} | {x['seed']} | {x['previous_earlystop_mse']:.9f} | {x['native_mse']:.9f} | {x['native_mae']:.9f} | {x['best_epoch']} |")
lines+=['','| Horizon | 潜态MSE（三种子均值±SD） | raw MSE（三种子均值±SD） | raw相对降低 |','| --- | ---: | ---: | ---: |']
for q in summary:lines.append(f"| {q['horizon']} | {q['latent_mean']:.6f} ± {q['latent_seed_sd']:.6f} | {q['raw_mean']:.6f} ± {q['raw_seed_sd']:.6f} | {q['raw_relative_reduction']*100:.2f}% |")
finding=('固定100epoch后，raw在六个配对单元中仍全部更好。早停确实影响绝对结果，但“仅因为raw训练更多轮”不足以解释已观察到的完整差距。残差属于发布配方级差距，尚未唯一归因于AE、latent loss或可学习函数类。' if survives else '固定预算后没有保留六组一致的潜态劣势。撤回一致机制缺陷的提法，先按完整配对结果判断预算/选模解释。')
lines+=['','## 当前判断','',finding,'','原始逐小时查询让前六个潜态模型都略有改善，不能声称真实查询暴露了新的时序失稳；该负面证据保留。当前100epoch模型没有重新做逐小时审计，不能将原逐小时结果归到新模型。','','## 研究问题与新颖性边界','','当前可检验的问题：**重构良好的冻结状态，何时仍与学习动态的函数类和训练目标不兼容？** 它比“换loss提升Electricity”更接近world-model问题，但单一数据集的分数差不能证明通用性或新颖性。','','已有源码本身支持decoded/perceptual loss、不同AE及解冻选项；增加观测损失或扩大预测器不能直接算创新。还需分别排除：(1)发布AE预训练来源/重构误差；(2)非线性坐标变换让同样大小的DLinear具有不同表达能力；(3)latent MSE/cos与观测目标不一致。只有残差体现独立的动态可复用性、控制响应或模拟缺陷，才值得方法设计。','','下一项最小判别：固定现有AE、输入划分和训练预算，在未用作Test选择的开发数据上，用官方已有decoded-loss和有表达力的预测器做普通归因控制；先决定能否用既有做法关闭差距。若关闭，放弃把当前现象包装成新架构；若不能，再在独立系统上检验同一机制。不得继续按Test提升100→200轮或搜索loss权重。','','## 复核','',f'独立重新读取12个checkpoint与预测哈希，以不同13行累积重算24个MSE/MAE标量，最大绝对差{max(checks):.3g}；逐一确认100行epoch记录和native验证最小值对应选择epoch。三种子SD只描述初始化变动，未做跨领域显著性声明。','原Test已在历史研究中查看；本轮是诊断，不是新方法独立确认。完整数据、权重、预测、日志保留服务器，GitHub仅发布紧凑证据。','','[完整机器结果](verification.json) · [初轮与逐小时结果](../primary_evidence_v1/experimental_results.md) · [预注册预算协议](../../MATCHED_CAP_PROTOCOL_V1.md)','']
(dest/'experimental_results.md').write_text('\n'.join(lines),encoding='utf-8')
extras={
'literature_review.md':(OUT/'primary_evidence_v1/literature_review.md').read_text(encoding='utf-8-sig')+'\n\nFixed-cap follow-up: the original LatentTSF paper already studies perceptual/observation-space objectives and backbone-dependent sensitivity; the official code has decoded-loss and unfreezing options. Such repairs are existing-method controls, not a new method. No specific residual mechanism has passed an independent novelty check.\n',
'baseline_registry.md':(OUT/'primary_evidence_v1/baseline_registry.md').read_text(encoding='utf-8-sig')+'\n\nTwelve registered fixed100 controls are now complete and independently checked; this intentionally disables the early-stop break and is an attribution study, not unchanged paper reproduction. Existing 12-mode dense audits refer to the earlier weights only.\n',
'negative_evidence.md':'# Negative evidence\n\nDense hourly inference improved the six original latent models. Runtime incidents are not model deficiencies. Fixed100 finding: '+finding+'\n\nNo independent new-method benefit or general world-model gap has been established. Ordinary decoded losses and expressive predictors remain mandatory explanations/controls.\n',
'hypothesis_space.md':'# Competing explanations\n\nSeed42 numeric/execution drift is contradicted by byte-identical replay. Sparse-hour degradation is falsified. Fixed100 finding: '+finding+'\n\nDistinct remaining hypotheses: released AE lineage/reconstruction bias; changed function class under nonlinear representation; latent surrogate objective mismatch. Separate controls and independent mechanisms are required; no unique representation-information-loss attribution is accepted.\n',
'research_ideas.md':'# Failure-derived candidate\n\nQuestion: when is a reconstruction-trained frozen state compatible with learning a reusable transition, rather than just reconstructing observations? The current experimental handle is a recipe-level latent/raw gap; the fixed-cap outcome is '+verdict+'.\n\nNovelty is unestablished. Standard decoded training, unfreezing, nonlinear prediction or normalization cannot be renamed as a contribution. Prioritize one attribution case, then seek the same failure in independent systems; do not optimize this Test set.\n',
'final_research_direction.md':'# Current decision\n\nAccepted final novel directions:0. Strongest current next question is state/transition compatibility, motivated by an actual released-recipe failure rather than a new architecture proposal. Fixed100 verdict: '+verdict+'.\n\nEvidence is confined to a DLinear/Electricity attribution study, supported by independent replay and explicit negative evidence. It does not yet establish a general world-model mechanism. Existing-method controls precede novelty claims and independent-system validation. The overarching research goal remains active.\n'}
for name,tx in extras.items():(dest/name).write_text(tx,encoding='utf-8')
files={x.name:sha(x) for x in dest.iterdir() if x.is_file()}
(dest/'manifest.json').write_text(json.dumps(dict(utc=obj['utc'],status=obj['status'],verdict=verdict,files=files,full_research_goal_achieved=False),indent=2),encoding='utf-8')
print(json.dumps(dict(status=obj['status'],verdict=verdict,summary=summary,max_abs_metric_error=max(checks))),flush=True)
