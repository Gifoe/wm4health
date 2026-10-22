"""Independent numeric replay and evidence reports; no training or method selection."""
import pathlib,json,hashlib,datetime,os
os.environ['OMP_NUM_THREADS']='2';os.environ['OPENBLAS_NUM_THREADS']='2';os.environ['MKL_NUM_THREADS']='2'
import numpy as np
ROOT=pathlib.Path(__file__).resolve().parents[1];OUT=ROOT/'outputs'
def read(p):return json.loads(pathlib.Path(p).read_text(encoding='utf-8-sig'))
def sha(p):
    h=hashlib.sha256()
    with pathlib.Path(p).open('rb') as f:
        while b:=f.read(1048576):h.update(b)
    return h.hexdigest()
lp=OUT/'native_replays_v4/RESULTS_V3.json';rp=OUT/'raw_control_v1/RESULTS_V1.json'
dp={'latent':OUT/'dense_hourly_v2/RESULTS_V1.json','raw':OUT/'raw_dense_v1/RESULTS_V1.json'}
assert read(OUT/'native_replays_v4_exit.json')['exit_code']==read(OUT/'raw_controls_v1_exit.json')['exit_code']==0
assert read(OUT/'dense_hourly_v2_exit.json')['exit_code']==read(OUT/'raw_dense_v1_exit.json')['exit_code']==0
native={'latent':read(lp),'raw':read(rp)};dense={k:read(p) for k,p in dp.items()}
assert native['latent']['status']=='ALL_SIX_NATIVE_REPLAY_ARTIFACTS_VERIFIED' and native['raw']['status']=='ALL_SIX_RAW_MODE_CONTROLS_CLOSED'
assert all(x['status']=='SIX_FROZEN_MODELS_DENSE_HOURLY_AUDIT_COMPLETE' for x in dense.values())
checks=[];repeated=[];table=[];dense_arrays=[]
def check(a,b,label):
    error=float(abs(a-b));assert error<2e-11,(label,error);checks.append(dict(label=label,max_abs=error))
for mode in ('latent','raw'):
    assert [(x['horizon'],x['seed']) for x in native[mode]['rows']]==[(h,s) for h in (336,720) for s in (42,43,44)]
    for row in native[mode]['rows']:
        metric=read(row['metrics_path']);folder=pathlib.Path(row['metrics_path']).parent
        assert sha(row['metrics_path'])==row['metrics_sha256'] and sha(metric['checkpoint_path'])==metric['checkpoint_sha256'] and sha(folder/'predictions.npz')==metric['predictions_sha256']
        with np.load(folder/'predictions.npz',allow_pickle=False) as z:
            pred=z['pred'];true=z['true'];ss=sa=0.;n=0
            for j in range(0,len(pred),11):
                error=pred[j:j+11].astype(np.float64)-true[j:j+11].astype(np.float64);ss+=np.einsum('ijk,ijk->',error,error);sa+=np.abs(error).sum(dtype=np.float64);n+=error.size
        check(ss/n,metric['reproduced_mse'],f'{mode}/{row["horizon"]}/{row["seed"]}/native_mse')
        check(sa/n,metric['reproduced_mae'],f'{mode}/{row["horizon"]}/{row["seed"]}/native_mae')
        d=next(x for x in dense[mode]['records'] if x['horizon']==row['horizon'] and x['seed']==row['seed']);assert d['source_checkpoint_sha256']==metric['checkpoint_sha256'] and sha(d['arrays']['path'])==d['arrays']['sha256']
        with np.load(d['arrays']['path'],allow_pickle=False) as z:
            a=z['origin_mse'];b=z['origin_mae'];hours=z['forecast_hour'];ids=z['raw_target_origin']
        assert len(a)==d['dense_origins'] and np.isfinite(a).all() and np.isfinite(b).all() and np.all(np.diff(ids)==1)
        check(a.mean(),d['dense_mse'],f'{mode}/{row["horizon"]}/{row["seed"]}/dense_mse');check(b.mean(),d['dense_mae'],f'{mode}/{row["horizon"]}/{row["seed"]}/dense_mae')
        for phase in d['phase_scores']:
            subset=np.arange(len(a))%60==phase['phase'];assert int(subset.sum())==phase['n']
            check(a[subset].mean(),phase['mse'],'phase_mse');check(b[subset].mean(),phase['mae'],'phase_mae')
        for hour in d['hour_scores']:
            subset=hours==hour['hour'];assert int(subset.sum())==hour['n']
            check(a[subset].mean(),hour['mse'],'hour_mse');check(b[subset].mean(),hour['mae'],'hour_mae')
        dense_arrays.append(dict(mode=mode,horizon=row['horizon'],seed=row['seed'],path=d['arrays']['path'],sha256=d['arrays']['sha256']))
        if mode=='latent' and row['seed']==42:
            old=read(pathlib.Path('D:/baseline_reproduction_seed2021_v1/results/latenttsf')/f'Electricity_DLinear_H{row["horizon"]}_seed42/metrics.json')
            assert metric['checkpoint_sha256']==old['checkpoint_sha256'] and metric['predictions_sha256']==old['predictions_sha256']
            repeated.append(dict(horizon=row['horizon'],checkpoint_and_predictions_byte_identical=True))
        table.append(dict(mode=mode,horizon=row['horizon'],seed=row['seed'],native_mse=metric['reproduced_mse'],native_mae=metric['reproduced_mae'],dense_mse=d['dense_mse'],dense_mae=d['dense_mae'],best_epoch=metric['best_epoch'],epochs=metric['epochs'],parameters=metric['num_parameters'],native_origins=d['native_sparse_origins'],dense_origins=d['dense_origins']))
summary=[]
for h in (336,720):
    s=dict(horizon=h,paper_latent_mse={336:.185,720:.218}[h])
    for mode in ('latent','raw'):
        rows=[x for x in table if x['mode']==mode and x['horizon']==h]
        for metric in ('native_mse','dense_mse','native_mae','dense_mae'):
            vals=np.asarray([x[metric] for x in rows]);s[mode+'_'+metric+'_mean']=float(vals.mean());s[mode+'_'+metric+'_seed_sd']=float(vals.std(ddof=1))
        s[mode+'_epochs']=[x['epochs'] for x in rows]
    s['raw_lower_native_mse_all_three_seeds']=all(next(x['native_mse'] for x in table if x['horizon']==h and x['seed']==seed and x['mode']=='raw')<next(x['native_mse'] for x in table if x['horizon']==h and x['seed']==seed and x['mode']=='latent') for seed in (42,43,44))
    s['raw_dense_mse_relative_reduction']=(s['latent_dense_mse_mean']-s['raw_dense_mse_mean'])/s['latent_dense_mse_mean'];summary.append(s)
dest=OUT/'primary_evidence_v1';dest.mkdir(exist_ok=False)
obj=dict(utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),status='PRIMARY_TWELVE_FITS_AND_TWELVE_DENSE_AUDITS_NUMERICALLY_VERIFIED',scalar_checks=len(checks),max_abs_error=max(x['max_abs'] for x in checks),seed42_replay=repeated,table=table,summary=summary,dense_array_receipts=dense_arrays,source_native_registry_sha256=sha(lp),source_raw_registry_sha256=sha(rp),dense_registry_sha256={k:sha(v) for k,v in dp.items()},native_raw_more_earlystopping_epochs=True,independent_generalization=False,new_method=False,accepted_final_directions=0)
(dest/'verification.json').write_text(json.dumps(obj,indent=2),encoding='utf-8')
lines=['# 已完成的失败归因：三篇主线中的首个案例','','主线：WorldTS / LatentTSF / TSWM。复用的20个WorldTS/LatentTSF checkpoint和40个TSWM子集checkpoint已重新核对；TSWM的125个源码文件也与固定版本一致。这里不把原论文子集和跨任务审计写成完整排行榜复现。','','本轮在服务器完成6组原生LatentTSF重复、6组官方观测空间DLinear对照，以及12个冻结模型的逐小时评估。所有配置事先固定，原始Test已在历史研究中查看，因此本轮是诊断证据，不是独立新方法确认。','','## 完整结果','','| Horizon | Mode | Seed | 原生step60 MSE | 每小时MSE | 每小时MAE | 最佳/实际epoch |','| --- | --- | ---: | ---: | ---: | ---: | --- |']
for x in table:lines.append(f"| {x['horizon']} | {x['mode']} | {x['seed']} | {x['native_mse']:.9f} | {x['dense_mse']:.9f} | {x['dense_mae']:.9f} | {x['best_epoch']}/{x['epochs']} |")
lines.extend(['','三种子平均（标准差只描述这三个初始化）：','','| Horizon | 潜态原生MSE | 原生raw MSE | 潜态逐小时MSE | raw逐小时MSE | raw相对降低 |','| --- | ---: | ---: | ---: | ---: | ---: |'])
for s in summary:lines.append(f"| {s['horizon']} | {s['latent_native_mse_mean']:.6f} ± {s['latent_native_mse_seed_sd']:.6f} | {s['raw_native_mse_mean']:.6f} ± {s['raw_native_mse_seed_sd']:.6f} | {s['latent_dense_mse_mean']:.6f} | {s['raw_dense_mse_mean']:.6f} | {100*s['raw_dense_mse_relative_reduction']:.2f}% |")
lines.extend(['','## 直接支持的事实','','1. 两个seed42重新训练的checkpoint和预测NPZ均与历史记录逐字节相同，排除了本次适配器改变已有结果的解释。H720的三颗种子仍高于论文记录0.218；这不是原论文作者具体run的唯一原因诊断。','2. 原生step60只覆盖CSV时钟标签的10点/22点。逐小时评估覆盖全天24小时：latent每个horizon有4925/4541个起点；新工作负载没有让六个latent模型退化，反而均略改善。不能靠该实验声称时序泛化失败。','3. 原生raw DLinear在六组里均优于latent，且两者forecast backbone参数量分别同为484512/1038240；latent另有1708865个冻结AE参数。存在真实的发布配方级表现差距。','4. 但raw实际训练25–47epoch，latent11–22epoch。初始张量、实际更新数和FLOPs没有匹配，不能仅据此归因为潜态状态信息丢失。','','## 下一项判别','','[固定100epoch对照](../../MATCHED_CAP_PROTOCOL_V1.md)在看到其结果之前冻结：保留双方原损失、数据、batch、学习率与native验证选模，去除提前break，只运行作者原上限100epoch。它区分早停/训练预算解释与更深的表示/目标差异，不做100→200的逐次救援。当前结果须与其状态分开。','','如果相同预算关闭差距，应撤回新的表示机制动机；若差距仍在，再检查目标函数、冻结AE来源与函数类兼容性，并在未用于本轮选择的独立系统验证。单一Electricity结果不足以确定通用world-model方向。','','## 复核与边界','',f'独立重算{len(checks)}个标量，最大差{obj["max_abs_error"]:.3g}；12个dense forward逐个核对原生稀疏起点和raw CSV标准化目标，数据/权重哈希保持不变。完整60个phase和24个hour分组保留在机器结果中。','重叠起点不是独立样本；未声明显著性、跨领域稳健性或新方法收益。运行与共享GPU存在重叠，不把所记wall time作为公平效率排名。','', '[注册协议](../../PROTOCOL.md) · [普通控制协议](../../RAW_MODE_CONTROL_PROTOCOL_V1.md) · [校验与完整机器表](verification.json)',''])
report='\n'.join(lines)
(dest/'experimental_results.md').write_text(report,encoding='utf-8')
reports={
'literature_review.md':(ROOT/'literature_review.md').read_text(encoding='utf-8-sig')+'\n\nCurrent experiment establishes a recipe-level failure and a training-budget confound; no novel mechanism has been proposed or verified.\n',
'baseline_registry.md':(ROOT/'baseline_registry.md').read_text(encoding='utf-8-sig')+'\n\nThis turn adds six source-recipe replays and six official raw-mode controls. Twenty prior WorldTS/LatentTSF and forty TSWM checkpoint identities were currently revalidated; this is not sixty new fits.\n',
'negative_evidence.md':'# Negative evidence\n\nDense hourly queries improved all six latent runs instead of amplifying a failure. Sparse-phase coverage is real but does not explain away the H720 discrepancy. Raw-mode success demonstrates that these data can be forecast usefully with an existing ordinary method; it does not isolate an AE defect while actual training durations differ. One first worker was recovered without a capture-exit receipt; complete scientific artifacts and source/hash replay are retained. All startup/controller/encoding failures remain execution incidents.\n',
'hypothesis_space.md':'# Competing hypotheses\n\nExecution/numeric drift is contradicted by byte-identical seed42 weights/predictions and independent metrics. Initialization explains part of H336 variation but does not alone close the three tested H720 cases. The proposed sparse-to-hourly degradation is falsified. Different early stopping/update counts remain a concrete alternative to representation/loss incompatibility; the fixed100 matrix tests it. AE pretraining lineage and objective/model-class differences remain separate explanations. No single cause is accepted before that comparison.\n',
'research_ideas.md':'# Failure-derived question\n\nWhy does this released latent forecasting recipe underperform the same official raw backbone: is the deficit an artifact of selection/budget, or a systematic compatibility problem between frozen states, latent objectives and learnable transitions? The question is experimentally concrete. Novelty and cross-domain importance remain unestablished. Do not name a new architecture or call simple early-stop/decoded-loss changes an innovation.\n',
'final_research_direction.md':'# Research objective remains active\n\nAccepted final directions:0. Primary papers: WorldTS, LatentTSF and TSWM. Evidence: twelve new native-mode/control fits, twelve dense inference audits, byte-identical seed42 reproduction and independent numeric replay. The observed latent-recipe deficit is real on the tested benchmark but confounded by early stopping; a fixed100-epoch attribution study is registered. Existing raw prediction already supplies a strong control. Next decisive evidence is the matched-cap result, followed by independent mechanisms if a meaningful residual survives. No general world-model gap or novel method is established by these numbers alone.\n'}
for name,tx in reports.items():(dest/name).write_text(tx,encoding='utf-8')
manifest={name:sha(dest/name) for name in ['verification.json','experimental_results.md']+list(reports)}
(dest/'manifest.json').write_text(json.dumps(dict(utc=obj['utc'],status=obj['status'],files=manifest,full_research_goal_achieved=False),indent=2),encoding='utf-8')
print(json.dumps(dict(status=obj['status'],scalar_checks=len(checks),max_abs_error=obj['max_abs_error'],summary=summary,scientific_reports=7)),flush=True)
