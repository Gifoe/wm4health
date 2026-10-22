"""Independent saved-array metric checks and bounded research decisions."""
import os
os.environ['OMP_NUM_THREADS']='2';os.environ['MKL_NUM_THREADS']='2'
import pathlib,json,hashlib,csv,datetime
import numpy as np,torch
ROOT=pathlib.Path(__file__).resolve().parents[1];OUT=ROOT/'outputs_recovery_v1';torch.set_num_threads(2)
def read(p):return json.loads(pathlib.Path(p).read_text(encoding='utf-8-sig'))
def sha(p):return hashlib.sha256(pathlib.Path(p).read_bytes()).hexdigest()
def dump(n,j):
 with (OUT/n).open('x',encoding='utf-8') as f:json.dump(j,f,indent=2,allow_nan=False)
j=read(OUT/'RESULTS_V1.json');assert j['status']=='FROZEN_ACTION_ORDER_DIAGNOSTIC_COMPLETE' and j['optimizer_updates']==0
assert j['protocol_sha256']==sha(ROOT/'PROTOCOL.md') and sha(j['payload_path'])==j['payload_sha256']
terminal=read(ROOT/'execution_recovery_v1/exit.json');assert terminal['exit_code']==0
old=ROOT.parent/'cycle88';manifest=read(old/'outputs/PLAN_DATA88_V1.json');data=next(x for x in manifest['files'] if x['split']=='val');assert sha(data['path'])==j['validation_data_sha256']
assert sha(old/'outputs/SOURCE_MANIFEST88_V1.json')==j['old_source_manifest_sha256']
for x in j['fits']:
 fit=read(old/'outputs'/('GRU_FIT88_V1.json' if x['arm']=='gru' else 'PLAN_FIT88_V1_'+x['arm']+'.json'));assert sha(fit['selected_path'])==x['checkpoint_sha256']
errors=[];n=0
def check(a,b,tol=1e-10):
 global n
 if a is None or b is None:assert a is None and b is None;return
 error=abs(a-b);assert error<=tol,(a,b,error,tol);errors.append(error);n+=1
with np.load(j['payload_path'],allow_pickle=False) as z:
 assert all(np.isfinite(z[k]).all() for k in z.files)
 family=z['family'];truth=torch.from_numpy(z['factual_truth']).double()
 assert truth.shape==(64,32,16)
 for row in j['rows']:
  arm,name,f=row['arm'],row['permutation'],row['family'];ids=np.flatnonzero(family==f);p=torch.from_numpy(z[arm+'_'+name][ids]).double();b=torch.from_numpy(z[arm+'_base'][ids]).double();q=torch.from_numpy(z[name+'_truth'][ids]).double();t=truth[ids];response=p-b;effect=q-t;err=response-effect
  den=effect.square().mean().item();num=err.square().mean().item();norm=(response.norm()*effect.norm()).item()
  check(response.square().mean().sqrt().item(),row['model_response_rms']);check(den**.5,row['true_response_rms']);check(num**.5,row['response_error_rms']);check((num/den)**.5 if den>0 else None,row['response_error_over_zero_reference']);check(torch.sum(response*effect).item()/norm if norm>0 else None,row['response_cosine']);check(err[:,-1].square().mean().sqrt().item(),row['endpoint_response_error_rms'])
  check((p-q).square().mean().item(),row['edited_mse'],1e-6);check((b-t).square().mean().item(),row['factual_mse'],1e-6)
  first=32 if name=='identity' else 8 if name=='swap_middle' else 0
  check(response[:,:first].abs().max().item() if first else None,row['prefix_before_first_edit_max'])
  if name=='identity':assert torch.equal(p,b) and torch.equal(q,t)
with (OUT/'METRICS_V1.csv').open('x',encoding='utf-8',newline='') as f:
 w=csv.DictWriter(f,fieldnames=list(j['rows'][0]));w.writeheader();w.writerows(j['rows'])
verification=dict(status='ACTION_ORDER_SAVED_PREDICTIONS_INDEPENDENTLY_VERIFIED',utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),implementation='Torch CPU float64, worker used NumPy',scalar_checks=n,max_scalar_error=max(errors),native_factual_replay=[x for x in j['qualification'] if x['name']!='simulator_identity'],results_sha256=sha(OUT/'RESULTS_V1.json'),protocol_sha256=sha(ROOT/'PROTOCOL.md'),payload_sha256=j['payload_sha256'],terminal_receipt=terminal,reserved_eval_opened=False,optimizer_updates=0,accepted_final_directions=0)
dump('VERIFICATION_V1.json',verification)
lines=['# 行动顺序诊断：朴素的顺序丢失解释被否定','','四个既有模型、64条既有开发轨迹、两类动力系统、三种固定顺序编辑和identity负控，训练更新0。基础预测与各原保存数组最大差均0；模拟器identity重放最大差'+f"{j['qualification'][0]['max_abs_error']:.3g}"+'，通过原1e-5门槛。原始失败的预检和独立恢复回执均保留。','','## 抽象行动确实包含顺序信息','','| 编辑 | 仅行动重排、时间戳槽位固定：最大变化 | 行动与时间戳一起重排：最大变化 |','| --- | ---: | ---: |']
for x in j['abstraction']:lines.append(f"| {x['permutation']} | {x['fixed_timestamp_max_abs_change']:.9f} | {x['joint_action_timestamp_permutation_max_abs_change']:.3g} |")
lines+=['','原生MTS3把窗口内时间戳与行动拼接后编码；其Gaussian aggregation对带时间戳的二元组作集合聚合。重排二元组当然可以不变，但将行动移到不同时间槽会改变表示。完整模型还在fast path接收逐步行动。因此不能把这一架构说成天然只看行动直方图，也不能据此直接发明所谓顺序恢复模块。','','## 行动响应误差','','数字为 response-error RMS / true-response RMS；1对应忽略编辑、预测无行动响应，越低越好。每格是同一family32条轨迹的整体开发指标，不是独立Test均值或跨seed置信区间。','','| 模型 | 系统 | 首尾块交换 | 中间块交换 | 块次序反转 |','| --- | --- | ---: | ---: | ---: |']
for arm in ['stock','lagged','acrkn','gru']:
 for f in [0,1]:
  r=[next(x for x in j['rows'] if x['arm']==arm and x['family']==f and x['permutation']==p)['response_error_over_zero_reference'] for p in ['swap_outer','swap_middle','reverse_blocks']]
  lines.append(f"| {arm} | {'Duffing' if f==0 else 'reservoir'} | {r[0]:.6f} | {r[1]:.6f} | {r[2]:.6f} |")
lines+=['','全部24个非identity格均优于无响应参照，存在明显非零且方向相关的预测响应。stock仍有已知提前响应；在中间块交换时，lagged/acRKN/GRU的未编辑前缀变化均0，stock约0.059/0.040。这复核旧时序结果，不把同一问题重新计为发现。','','剩余误差并不小，但不能由一颗seed、原1000更新和已用于事实选模的开发bank，判定为架构信息瓶颈。模型间两系统表现有差异，不能挑单个有利格宣布新方法必要性。没有按编辑结果重选epoch、修改动作或者打开reserved EVAL。','','## 研究决定','','否定“原生时间聚合必然抹去行动顺序”的朴素解释。保留响应误差，但不继续以该解释追加新架构或同组调参。原C88 anticipation修复路线保持其既有关闭结论。已有Log-NCDE/rough-path压缩、HWM learned action编码是强近邻；简单添加timestamp、GRU或signature本身不是本轮新贡献。','','独立Torch CPU复核'+str(n)+'个scalar，最大差'+f'{max(errors):.3g}'+'；factual/edited MSE允许保存float32减法次序的1e-6差，响应量按1e-10核对。原源码、数据、checkpoint、协议与预测payload哈希复核。','','当前接受新颖方向0，整体研究目标ACTIVE/UNACHIEVED。下一条值得资格检查的候选来自既有DALI线索：同一latent context编辑，能否在不同起点和不同后续行动下由一个固定的真实动力学参数变化解释？这是待实证的机制编辑问题，不是本轮已验证缺口。原生源码与checkpoint可行性先检查，禁止用未训练的toy模型冒充论文失败。','','[完整指标CSV](METRICS_V1.csv) · [独立复核](VERIFICATION_V1.json) · [冻结协议](../PROTOCOL.md) · [模拟重放修复](../SIMULATOR_REPLAY_REPAIR.md) · [文献与候选边界](../literature_search/papers.md)','']
reports={'experimental_results.md':'\n'.join(lines),
'baseline_registry.md':'# Fixed model scope\n\nReuse four C88 source-based transfer checkpoints, selected at step1000: native MTS3, its ordinary causal-index control, native acRKN and a conventional128864-parameter GRU. No new training. The64 trajectories were the original factual validation bank, not independent confirmation. Three fixed action-block permutations plus identity are diagnostic queries. The primary three-paper registry stays unchanged.\n',
'literature_review.md':'# Closest-work boundary\n\nSee [screened sources](../literature_search/papers.md). MTS3 already includes timestamp-conditioned abstract action encoding. HWM uses learned macro-action encoding; Log-NCDE supplies existing order-aware path compression. Identifiability and intervention-support uncertainty have close2026 prior art. Neither action ordering nor generic uncertainty is an accepted novelty claim.\n',
'negative_evidence.md':'# Rejected mechanism explanation\n\nExact order blindness is false in this native timestamped abstraction and the selected learned forecasts. Every nonidentity arm/family query has response-error below a zero-response reference. The old anticipation defect persists in stock and remains removed by ordinary chronological controls; that is not a new finding. The preserved simulator-prefix preflight incident was a serialization/replay error before model loading, not model failure.\n',
'hypothesis_space.md':'# Hypotheses\n\nH1: the macro action encoder necessarily ignores action order. Rejected by source and actual selected-weight execution. H2: native forecasts are insensitive to order. Rejected on the fixed developmental queries. H3: a faithful primitive response is guaranteed by order sensitivity/nonanticipation. Not established: material response error remains. H4: the remaining error requires a novel temporal abstraction. Unsupported without stronger competence, ordinary order-aware controls and independent-system evidence.\n',
'research_ideas.md':'# Next candidate, not a final recommendation\n\nTest global mechanism meaning of a fixed latent-context edit across different roots and action queries. This is the older, previously unexecuted DALI lead, now with source/release feasibility rechecked. It differs from local physical-state editing and from making imagined trajectories merely distinguishable. Native checkpoint/model competence, simulator-parameter existence, ordinary context-manifold projection and true parameter-conditioned controls must precede any new-method claim. No native DALI execution is claimed.\n',
'final_research_direction.md':'# Decision\n\nAccepted final directions0. Close the naive action-order-loss explanation; preserve observed response residuals without attributing them uniquely. Overall research goal remains active/unachieved. Retain the distinct DALI cross-query mechanism-edit candidate only for a native-feasibility and closest-work gate. No new method or publishability claim follows from this diagnostic.\n'}
for name,tx in reports.items():
 with (OUT/name).open('x',encoding='utf-8') as f:f.write(tx)
dump('MANIFEST_V1.json',dict(status='ORDER_BLINDNESS_HYPOTHESIS_REJECTED_BOUNDED_DIAGNOSTIC_CLOSED',utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),files={p.name:sha(p) for p in OUT.iterdir() if p.is_file()},accepted_final_directions=0,full_research_goal_achieved=False))
print(json.dumps(dict(status=verification['status'],scalar_checks=n,max_scalar_error=max(errors),all_nonidentity_better_than_zero_response=all(x['response_error_over_zero_reference']<1 for x in j['rows'] if x['permutation']!='identity'),accepted_final_directions=0)),flush=True)
