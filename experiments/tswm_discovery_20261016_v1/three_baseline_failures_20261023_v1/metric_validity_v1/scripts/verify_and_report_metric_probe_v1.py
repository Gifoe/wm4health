"""Independent CPU Torch cosine calculation; no model execution, fitting or selection."""
import os
os.environ['OMP_NUM_THREADS']='2';os.environ['MKL_NUM_THREADS']='2'
import pathlib,json,hashlib,csv,datetime
import numpy as np
import torch
import torch.nn.functional as F
torch.set_num_threads(2)
ROOT=pathlib.Path(__file__).resolve().parents[1];OUT=ROOT/'outputs';CAMPAIGN=ROOT.parent
def read(p):return json.loads(pathlib.Path(p).read_text(encoding='utf-8-sig'))
def sha(p):return hashlib.sha256(pathlib.Path(p).read_bytes()).hexdigest()
def dump(n,j):
 with (OUT/n).open('x',encoding='utf-8') as f:json.dump(j,f,indent=2,allow_nan=False)
def values(a):
 t=torch.from_numpy(np.array(a,dtype=np.float64,copy=True));c=t-t.mean(dim=(0,1),keepdim=True)
 return (F.cosine_similarity(t[:,:-1],t[:,1:],dim=-1,eps=1e-8).mean().item(),F.cosine_similarity(c[:,:-1],c[:,1:],dim=-1,eps=1e-8).mean().item(),int(((t[:,:-1].norm(dim=-1)==0)|(t[:,1:].norm(dim=-1)==0)).sum()),int(((c[:,:-1].norm(dim=-1)==0)|(c[:,1:].norm(dim=-1)==0)).sum()))
j=read(OUT/'RESULTS_V1.json');assert j['status']=='TWO_NATIVE_CHECKPOINTS_METRIC_VALIDITY_PROBE_COMPLETE'
terminal=read(CAMPAIGN/'outputs/metric_validity_probe_v2_exit.json');assert terminal['exit_code']==0
assert j['protocol_sha256']==sha(ROOT/'PROTOCOL.md') and j['optimizer_updates']==0
for p,h in j['source_sha256'].items():assert sha(p)==h
checks=[];table=[];errors=[]
def eq(a,b,label):
 e=abs(float(a)-float(b));assert e<2e-12,(label,e);errors.append(e)
for rec in j['records']:
 p=pathlib.Path(rec['state_banks']['path']);assert sha(p)==rec['state_banks']['sha256']
 q=rec['qualification'];assert q['test_truth_max_abs_error']==0 and q['translated_encoder_input_recovery_max_abs_error']==0 and q['translated_forecast_max_abs_error']<=q['bound']
 with np.load(p,allow_pickle=False) as z:
  assert all(np.isfinite(z[k]).all() for k in ['raw_observations','observed_encoding','predicted_latent'])
  raw=values(z['raw_observations'])
  for k,val in zip(['ttc','centered_ttc','zero_norm_pairs','centered_zero_norm_pairs'],raw):eq(val,rec['raw_observation_descriptor'][k],rec['dataset']+':raw:'+k)
  for name in ['observed_encoding','predicted_latent']:
   bank=z[name].astype(np.float64);perms=z[name+'_permutations'];rng=np.random.default_rng(20261023)
   expected=np.asarray([[rng.permutation(bank.shape[1]) for _ in range(bank.shape[0])] for _ in range(32)],dtype=np.int32);assert np.array_equal(perms,expected)
   rr=[x for x in rec['rows'] if x['bank']==name];assert len(rr)==4
   centered=[]
   for row in rr:
    a=bank+row['offset'];v=values(a);centered.append(v[1]);assert row['pairs']==bank.shape[0]*(bank.shape[1]-1)
    for k,val in zip(['ttc','centered_ttc','zero_norm_pairs','centered_zero_norm_pairs'],v):eq(val,row[k],rec['dataset']+':'+name+':'+k)
    null=[];cnull=[]
    for k,perm in enumerate(perms):
     n=values(np.take_along_axis(a,perm[:,:,None],axis=1));null.append(n[0]);cnull.append(n[1]);eq(n[0],row['shuffle_scores'][k],'shuffle');eq(n[1],row['shuffle_centered_scores'][k],'centered_shuffle')
    eq(np.mean(null),row['shuffle_ttc_mean'],'shuffle_mean');eq(np.std(null,ddof=1),row['shuffle_ttc_sd'],'shuffle_sd');eq(np.mean(cnull),row['shuffle_centered_ttc_mean'],'centered_shuffle_mean')
    reversed_v=values(a[:,:,::-1]);eq(reversed_v[0],v[0],'orthogonal_coordinate_reversal')
    table.append(dict(dataset=rec['dataset'],bank=name,offset=row['offset'],pairs=row['pairs'],ttc=v[0],centered_ttc=v[1],shuffle_ttc_mean=float(np.mean(null)),shuffle_ttc_min=float(np.min(null)),shuffle_ttc_max=float(np.max(null)),shuffle_centered_ttc_mean=float(np.mean(cnull)),ttc_minus_shuffle_mean=float(v[0]-np.mean(null)),centered_ttc_minus_shuffle_mean=float(v[1]-np.mean(cnull)),zero_norm_pairs=v[2],centered_zero_norm_pairs=v[3]))
   assert max(centered)-min(centered)<2e-12
   checks.append(dict(dataset=rec['dataset'],bank=name,shape=list(bank.shape),pairs=bank.shape[0]*(bank.shape[1]-1),all_32_permutations_regenerated=True,centered_offset_range=max(centered)-min(centered)))
with (OUT/'METRICS_V1.csv').open('x',encoding='utf-8',newline='') as f:
 w=csv.DictWriter(f,fieldnames=list(table[0]));w.writeheader();w.writerows(table)
verification=dict(status='TTC_FORMULA_AND_ALL_NULLS_INDEPENDENTLY_VERIFIED',utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),independent_implementation='torch.float64 F.cosine_similarity CPU, worker used numpy',checks=checks,scalar_checks=len(errors),max_scalar_error=max(errors),results_sha256=sha(OUT/'RESULTS_V1.json'),protocol_sha256=sha(ROOT/'PROTOCOL.md'),terminal_receipt=terminal,optimizer_updates=0,learned_weights_unchanged=True,raw_payload_local_sync=False,accepted_final_directions=0)
dump('VERIFICATION_V1.json',verification)
lines=['# TTC指标假设诊断：分数可以改变，物理预测保持不变','','这是独立的评估假设检查，不延续已停止的Electricity分数优化，也不是新模型或新指标。两份原生DLinear/H96/seed42 checkpoint均重新通过原Test前32个origin的预测回放；指标只在Validation上计算。训练、选模和预测超参数更新均为0。','','## 结果','','| 数据 | 状态库 | 偏移 | TTC | 打乱时间TTC均值 | 全局中心化TTC | 中心化后打乱均值 |','| --- | --- | ---: | ---: | ---: | ---: | ---: |']
for x in table:lines.append(f"| {x['dataset']} | {x['bank']} | {x['offset']:g} | {x['ttc']:.9f} | {x['shuffle_ttc_mean']:.9f} | {x['centered_ttc']:.9f} | {x['shuffle_centered_ttc_mean']:.9f} |")
lines+=['','每一组保留全部32次预注册的序内时间置换。observed_encoding是各唯一Validation时点的逐点AE表示；predicted_latent是32个固定Validation起点、每条96步的潜态预测，未跨窗口拼接相邻对。两类库分开解释，前者不是预测。窗口存在重叠，没有把所有pair当作独立统计样本。','','## 实际预测资格','','| 数据 | 原Test回放最大差 | 平移包装物理预测最大差 | 恢复encoder输入最大差 |','| --- | ---: | ---: | ---: |']
for r in j['records']:
 q=r['qualification'];lines.append(f"| {r['dataset']} | {q['test_prediction_max_abs_error']:.3g} | {q['translated_forecast_max_abs_error']:.3g} | {q['translated_encoder_input_recovery_max_abs_error']:.3g} |")
lines+=['','平移尺度由TRAIN的512个固定点RMS预先决定，所有坐标使用同一偏移；没有用Val/Test选有利偏移。E_c=E+c、F_c(z)=F(z−c)+c、D_c(w)=D(w−c)采用相同已学习权重，double加减后恢复原float32输入。这里确实执行了包装后的前向，没有只写数学断言。它是函数等价包装，不宣称原训练目标或原参数族对这种变换不变。','','## 能支持和不能支持的判断','','不改变任何物理预测，仅移动潜在坐标原点，就可将TTC推近1；时间置换后的TTC也能非常高。因此，TTC绝对值高不能单独证明状态转移正确或模型学到了时间机制。平移后的模型没有得到更真实的动力学。中心化TTC在本实验平移下稳定，但这只是已知诊断，不是可独立确认动力学的指标。','','这不反证LatentTSF的预测结果或表示有用性，也没有证明其论文数值算错。论文TTC公式已核对，但当前pin中未找到其专用geometry评估入口；本实验明确给定epsilon、采样和拼接规则，只是公式重实现，不能声称复现Table3/11。CKA和有效秩未在本轮重现或审计，不能从TTC结果推断它们也失败。','','## 已知边界和反证','','坐标原点与cosine的关系是基本性质；已有表示相似性研究讨论了不变性选择的限制。不能将这个诊断重新命名为world-model创新，也不能要求对任意可逆变换不变。后续若声称新方法改善动力学，必须在同样信息、任务和预算下改善可观测行为，并超过普通中心化、尺度控制及原生预测器；单独提高TTC不构成证据。当前没有该方法实证，接受最终方向0。','','## 独立验证','','NumPy工作脚本与Torch CPU float64实现独立核对全部指标、所有32次null和坐标反转负控；'+str(len(errors))+'个scalar检查最大差'+f'{max(errors):.3g}'+'。状态库只保存在服务器数据盘，GitHub仅发布小型指标、协议、源码和SHA。','', '[原始紧凑结果](RESULTS_V1.json) · [完整CSV](METRICS_V1.csv) · [独立核验](VERIFICATION_V1.json) · [冻结协议](../PROTOCOL.md)','','来源：[LatentTSF v2](https://arxiv.org/html/2602.00297v2)；[Kornblith等，ICML2019](https://proceedings.mlr.press/v97/kornblith19a.html)。','']
reports={'experimental_results.md':'\n'.join(lines),
'literature_review.md':'# Metric and prior-art boundary\n\n[LatentTSF v2, Sec5.6/AppG](https://arxiv.org/html/2602.00297v2) defines TTC using consecutive-state cosine. This probe reimplements that formula with an explicit epsilon and sampling contract; it does not reproduce the paper geometry tables. The pinned source search did not locate the dedicated geometry entry point. [Kornblith et al., ICML2019](https://proceedings.mlr.press/v97/kornblith19a.html) discusses representation-invariance choices and limitations of arbitrary invertible-linear invariance. Translation sensitivity and centering are established mathematical issues, not an accepted novel method here.\n',
'baseline_registry.md':'# Scope\n\nTwo original native LatentTSF DLinear/H96/seed42 checkpoints: ETTh1 and Electricity. Neither comes from the later fixed100/decoded-control campaign. Released AE, raw-data and model/source hashes are requalified. No new training, selection or Test tuning. Validation-only metric probes use two separate state banks and all registered offsets/nulls. This is not an additional SOTA reproduction.\n',
'negative_evidence.md':'# Negative evidence\n\nA large TTC can be induced by a latent-origin shift while unchanged learned weights and algebraic wrappers yield exactly unchanged decoded predictions in both native checkpoints. Time-permuted latent states also reach near-one TTC at the largest fixed offset. These facts reject TTC alone as a certificate of learned temporal behaviour. They do not reject the predictive usefulness of the two models, nor provide a new world-model mechanism. The metric issue is known mathematics and does not satisfy novelty.\n',
'hypothesis_space.md':'# Competing interpretations\n\nH1: higher absolute temporal cosine certifies more faithful temporal dynamics. Unsupported as a standalone interpretation: physical predictions remain identical across a wide range of scores, and shuffled states can score highly. H2: cosine also measures common latent offsets. Supported by the frozen offset experiment. H3: an origin-invariant scalar alone measures behavioural faithfulness. Untested and not assumed; centering is just an existing control. H4: the published geometry numbers are erroneous. Not tested or claimed.\n',
'research_ideas.md':'# Decision\n\nAccept no new direction on this evidence. A possible evaluation research question is whether representation descriptors predict independently measured transition or action-response fidelity after ordinary nuisance controls. Current evidence only establishes one descriptor limitation, not a novel benchmark, an unsolved prediction task or an effective repair. Require a distinct observable failure, adversarial prior-art check, competent controls and held-out method validation before promotion.\n',
'final_research_direction.md':'# No final novel direction established\n\nThe metric-validity audit is closed and independently verified. Do not propose a centered-TTC world model or rename coordinate invariance as an algorithm. The three-paper process has produced reproducible negative evidence and prevented a false direction; the overall research-discovery objective remains ACTIVE/UNACHIEVED. Accepted final directions0. The next gate is an independently observable capability failure that survives native recipes, ordinary controls and prior art, not another scalar descriptor improvement.\n'}
for name,tx in reports.items():
 with (OUT/name).open('x',encoding='utf-8') as f:f.write(tx)
dump('MANIFEST_V1.json',dict(status='METRIC_VALIDITY_DIAGNOSTIC_CLOSED_NO_NEW_METHOD',utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),files={p.name:sha(p) for p in OUT.iterdir() if p.is_file()},verifier_source_sha256=sha(__file__),full_research_goal_achieved=False,accepted_final_directions=0))
print(json.dumps(dict(status=verification['status'],scalar_checks=len(errors),max_scalar_error=max(errors),rows=table,accepted_final_directions=0)),flush=True)
