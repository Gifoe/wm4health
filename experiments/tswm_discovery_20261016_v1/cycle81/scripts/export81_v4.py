"""Build server-only reports and immutable publication snapshot after verification."""
import sys,pathlib,csv,hashlib,shutil,os
sys.path.insert(0,str(pathlib.Path(__file__).resolve().parent))
from common81_v1 import *
import numpy as np
os.environ['MPLBACKEND']='Agg'
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

def write(name,body):
 with (ROOT/name).open('x',encoding='utf-8') as f:f.write(body)
def main():
 r=read(OUT/'RESULTS81_V1.json');v=read(OUT/'VERIFICATION81_V1.json');cohort=read(OUT/'COHORT81_V1.json');assert v['status'].startswith('PASS_') and read(OUT/'run81_v1_exit.json')['exit_code']==0
 systems=cohort['development'];kinds=['native','fim_readout_residual','fim_readout_zero','rff_residual','affine_residual','ordinary_weak_dictionary'];lookup={(z['system'],z['initial_condition_index'],z['kind']):z for z in r['rows']};summary=[];improved=0;beats=0;retains=True;ratios=[]
 def value(sid,k,kind):
  q=lookup[sid,k,kind];return q['nrmse'] if q['solver']['success'] and q['finite_complete'] else float('inf')
 for sid in systems:
  learned=value(sid,1,'fim_readout_residual');native=value(sid,1,'native');ordinary=min(value(sid,1,z) for z in ['rff_residual','affine_residual','ordinary_weak_dictionary']);improved+=learned<native;beats+=learned<ordinary
  if np.isfinite(native) and not np.isfinite(learned):retains=False
  ratio=learned/ordinary if ordinary>0 and np.isfinite(ordinary) else float('inf');ratios.append(ratio);summary.append(dict(system=sid,learned_nrmse=learned if np.isfinite(learned) else None,native_nrmse=native if np.isfinite(native) else None,ordinary_lower_envelope_nrmse=ordinary if np.isfinite(ordinary) else None,ratio_to_ordinary_lower_envelope=ratio if np.isfinite(ratio) else None))
 median=float(np.median(ratios));passed=bool(improved>=4 and beats>=4 and retains and median<=.9);decision='ADVANCE_TO_NEW_FROZEN_EVALUATION_PROTOCOL' if passed else 'RETIRE_THIS_READOUT_ONLY_DEVELOPMENT_ROUTE'
 dump(OUT/'DECISION81_V1.json',dict(utc=utc(),decision=decision,development_only=True,reserved_evaluation_opened=False,improves_native_new_ic=int(improved),beats_ordinary_lower_envelope=int(beats),retains_native_successes=bool(retains),median_paired_ratio=median if np.isfinite(median) else None,systems=summary,ordinary_lower_envelope_is_diagnostic_not_deployable_selector=True,qualified_final_research_directions=0,requires_stronger_prior_art_controls_even_if_passed=True))
 def fmt(z):return f'{z:.6f}' if np.isfinite(z) else 'FAIL'
 table='| System | Native | Learned residual | Learned zero | Random residual | Affine residual | Weak dictionary |\n|---|---:|---:|---:|---:|---:|---:|\n'
 for sid in systems:table+='| '+str(sid)+' | '+' | '.join(fmt(value(sid,1,k)) for k in kinds)+' |\n'
 reconstruction='| System | Native | Learned residual | Learned zero | Random residual | Affine residual | Weak dictionary |\n|---|---:|---:|---:|---:|---:|---:|\n'
 for sid in systems:reconstruction+='| '+str(sid)+' | '+' | '.join(fmt(value(sid,0,k)) for k in kinds)+' |\n'
 msg=(f'预定门槛通过，但只允许另行冻结独立评估；不接受创新或最终方向。' if passed else f'预定门槛未通过，停止这条读出层适配路线，不在相同开发系统上继续调参。')
 report=f'''# C81：预训练动力学特征能否提供有效适配方向？

{msg} 这是通用动力学能力的开发筛查，不是某个真实数据集调优，也不是所有时间序列世界模型的结论。

## 固定的实验与选择

固定 OpenFIM commit `cee2bb53be1e8e92b73f98dfdbe3605efc9119cc`、HF模型revision `00875c253d025dd33fd70ae751089b1be3a7c174` 和 ODEBench revision `91909865498f5d4bd69ea575527fee50ad329ef4`。基础checkpoint有12,968,068参数，保持全部冻结。来源和资产沿用C80哈希，不重复下载。

先依据公开维度、成功标记和512点结构选择系统，排除C80三个教程；开发ID {systems}，保留评估ID {cohort['reserved_evaluation']}。保留系统未进行预测评分。本JSON在早期审计解析过，不能声称完全未接触或保证不在作者预训练分布中。该切分是在本轮特征提取和新系统评分之前冻结。

每个系统只有第一条含噪轨迹的256点，3%乘性高斯噪声、seed2026810。所有状态可见，无行动或潜在传感器转移。第一条干净路径为重建，第二条是同方程参数的新初值模拟；双方只获其初值和时间，不获未来。模型的2D物理输出采用原生内部3D补齐和归一化。

原生最后线性层有256个隐藏特征和偏置，共257项。FIM残差、从零拟合、随机残差均有514个可适配系数；仿射残差10个，普通库20或28个。随机特征控制与FIM同特征数，且两者都保留同一个FIM基础场。它们不是全模型随机初始化控制，也未匹配有效秩或离线预训练成本。

每个族都用前192点构建编码、归一化及弱积分拟合，在可见上下文最后65点验证轨迹MSE；再用全256点重建并重拟合。没有把验证点先喂给编码器。四个修正族各3个ridge候选，普通库12候选，每系统24，共144；另36个全上下文模型含6个原生模型。没有新的神经训练。相同RK45容差、20,000 RHS调用/状态绝对值1e6失败保护，无截断裁剪。弱拟合是普通工作回归，不声称精确贝叶斯轨迹后验或校准不确定性。

## 独立新初值的全部NRMSE

NRMSE = RMSE / 完整干净参考全部坐标的合并标准差，只用于评分。

{table}

预训练残差相对原生改善 **{improved}/6**；相对随机残差/仿射残差/普通库的逐系统最小误差改善 **{beats}/6**。此逐系统下包络仅是诊断门槛，依赖评分结果，不是可部署的无标签选择器。中位配对NRMSE比为 **{fmt(median)}**；保留所有原生成功轨迹：**{retains}**。未只汇总成功结果；全部失败/部分积分原数组和日志保留。

## 观察路径重建，单独报告

{reconstruction}

低重建误差本身不足以证明转移规律正确。具体差异可见下图；本轮没有把某个重建/泛化差异自动解释为新的机制。

![Development comparison](figures/development_new_ic81_v1.png)

## 资格化、复核与工程记录

首个系统的17个上下文查询，缓存向量场与直接原生前向最大差0；隐藏层仿射重建物理场误差2.35e-7，满足事先声明的float32容差。独立脚本再核对全部72结果、288个上下文/时间/参考身份数组；全部指标重算最大差 {v['metric_max_abs']:.3g}。24个全上下文修正模型使用直接逐窗口梯形积分和augmented least squares，区别于原累计和/正规方程，复核全部系数。验证不构成另一个独立噪声seed或新实验。源70文件与权重结束后再次核验。

首次SSH会话内Start-Process返回PID后未留下日志或活进程，未进入模型；改用隐藏CIM进程启动，保留相同协议/数据切分，只有一次实际有日志的数值执行。不将启动错误计作模型失败。代码/协议可本地编写；结果、数据、模型和图不落地本地，同步GitHub仅经内存。

## 对研究问题的影响

决策：`{decision}`。开发样本只有6个系统、一个噪声seed，不能建立显著性、跨领域优越性、隐状态识别或因果干预能力。尤其不能把“冻结特征+线性更新+弱积分”改名当作创新：JFR、ALPaCA、函数编码NODE、WENDy、FORM和贝叶斯混合效应ODE是直接先例，见文献边界。

即使通过也需JFR/轨迹损失适配、WENDy误差变量处理、有效秩控制、全参数适配成本与独立基准。未通过则说明当前简单读出路线没有足够证据继续，不说明所有预训练特征、一般世界模型适配或未知观测转移都已解决。

最终推荐研究方向仍为0；研究总目标ACTIVE/UNACHIEVED。不以本轮实验数量或一两个正面单元充当成功。
'''
 write('FEATURE_ADAPTATION_SCREEN81_V1.md',report)
 write('literature_review.md','# Literature review\n\nSee [verified C81 boundaries](LITERATURE_BOUNDARIES81_V1.md) and [candidate evolution](CANDIDATE_CARDS81_V1.md). The practical feature-transfer question is tested; the component algorithms are established. All sources are primary papers or author repositories. Literature methods not executed here must not be marked reproduced.\n')
 entries={
 'baseline_registry.md':'Executed: released FIM-ODE, external weak last-layer residual fit, same features from zero, matched-count random Fourier residual, affine residual, polynomial/trigonometric weak ridge/sparse library. 144 context candidates, 72 outer trajectories, six development systems, one noise seed. JFR, WENDy, full reconstruction adaptation and coherent field posterior baselines are literature-qualified but NOT executed in this cycle.',
 'experimental_results.md':f'Complete new-IC table:\n\n{table}\n\nComplete reconstruction and every candidate/solver/parameter record are in FEATURE_ADAPTATION_SCREEN81_V1.md and outputs/RESULTS81_V1.json. Decision: {decision}.',
 'negative_evidence.md':f'The learned residual improves native on {improved}/6 systems and the ordinary diagnostic lower envelope on {beats}/6, with median paired ratio {fmt(median)}. Do not retain only favorable systems. This screen cannot rule out noise-handling, basis conditioning, trajectory-loss or rank explanations, and does not test hidden-state transfer. Final innovation not established.',
 'hypothesis_space.md':'H1 implementation mismatch: context-only native/affine qualification and raw identity checks pass. H2 transferred learned features add value: evaluated against matched-count random residual, same-feature zero, affine and weak dictionary under the frozen development gate. H3 low reconstruction implies new-IC accuracy: report both, no implication assumed. H4 a new adaptation mechanism is needed: not established; stronger existing algorithms have not been run.',
 'research_ideas.md':f'Route 1: frozen pretrained dynamics-feature adaptation, development decision {decision}. Route 2: context-support contracts, generic form already occupied and mechanism/evidence unresolved. Route 3: reusable dynamics under unknown observation changes, assumptions and concrete mechanism unresolved. No route is qualified as a final recommendation. See CANDIDATE_CARDS81_V1.md.',
 'final_research_direction.md':f'Research Question: do existing foundation dynamics features provide useful trajectory-only adaptation directions? Evidence: six new development systems, 144 context candidates, 72 complete-or-failed outcomes, independently checked identities/metrics/linear fits. Existing Methods: JFR, ALPaCA, FE+NODE, WENDy, FORM and Bayesian field/collocation inference cover the basic mechanisms. Proposed Idea: no qualified new mechanism; test the most economical readout retrofit first. Falsification: matched ordinary features explain or exceed its independent new-IC behavior. Feasibility: pinned 13M released model and public ODEBench, CPU two-thread execution, all payloads server-side. Next Experiment: respect frozen decision {decision}; do not retune opened systems. Broader latent/observation transfer needs an explicit identifiable contract before implementation. Qualified final directions0; overall ACTIVE/UNACHIEVED.'}
 for name,body in entries.items():write(name,'# '+name[:-3].replace('_',' ')+'\n\n'+body+'\n')
 csvrows=[{key:row.get(key) for key in ['system','initial_condition_index','role','kind','finite_complete','mse','rmse','mae','nrmse','parameters','selected_ridge','adaptation_seconds']}|dict(solver_success=row['solver']['success'],rollout_seconds=row['solver']['wall_seconds']) for row in r['rows']]
 with (OUT/'all_metrics81_v1.csv').open('x',encoding='utf-8',newline='') as f:w=csv.DictWriter(f,fieldnames=list(csvrows[0]));w.writeheader();w.writerows(csvrows)
 fig,ax=plt.subplots(figsize=(10,5),layout='constrained');colors=['#333333','#0072B2','#56B4E9','#D55E00','#009E73','#CC79A7'];labels=['Native FIM','Learned residual','Learned zero','Random residual','Affine residual','Weak dictionary']
 for j,(kind,color,label) in enumerate(zip(kinds,colors,labels)):
  yy=np.asarray([value(sid,1,kind) for sid in systems]);xx=np.arange(6)+(j-2.5)*.12;valid=np.isfinite(yy);ax.scatter(xx[valid],np.maximum(yy[valid],1e-9),s=35,label=label,color=color,marker='o');ax.scatter(xx[~valid],np.full((~valid).sum(),10.),marker='x',color=color,s=60)
 ax.set_yscale('log');ax.set_xticks(range(6),[str(z) for z in systems]);ax.set_xlabel('ODEBench system (fixed development cohort)');ax.set_ylabel('New-IC NRMSE (log scale)');ax.set_title('Frozen-feature adaptation: all six development systems');ax.grid(axis='y',alpha=.25);ax.legend(ncols=3,fontsize=8);figure=storage('artifact')/'development_new_ic81_v1.png';fig.savefig(figure,dpi=170);plt.close(fig)
 closure=dict(utc=utc(),decision=decision,results_sha256=sha(OUT/'RESULTS81_V1.json'),verification_sha256=sha(OUT/'VERIFICATION81_V1.json'),decision_sha256=sha(OUT/'DECISION81_V1.json'),figure_path=str(figure),figure_sha256=sha(figure),raw_trajectories=72,development_systems=6,context_candidates=144,new_neural_training_fits=0,evaluation_opened=False,accepted_final_directions=0,overall_goal='ACTIVE_UNACHIEVED',local_payload_files_written=0);dump(OUT/'CLOSURE81_V1.json',closure)
 with (PROJECT/'CURRENT_RESEARCH_STATUS_V81.md').open('x',encoding='utf-8') as f:f.write(f'# C81 current status\n\n{msg}\n\n'+table+f'\nFrozen development gate: {decision}. Reserved evaluation remains unscored. Full reports, code and compact results are under cycle81. No new mechanism or final research direction has been accepted; overall ACTIVE/UNACHIEVED. All data/weights/prediction arrays remain on the server G drive. GitHub publication only; no local payload synchronization.\n')
 dest=OUT/'publication_v1';dest.mkdir(exist_ok=False);files=['README81_V1.md','PROTOCOL81_V1.md','CANDIDATE_CARDS81_V1.md','LITERATURE_BOUNDARIES81_V1.md','FEATURE_ADAPTATION_SCREEN81_V1.md','literature_review.md',*entries];files+=['scripts/'+p.name for p in sorted((ROOT/'scripts').glob('*.py')) if p.name not in ['export81_v1.py','export81_v2.py','export81_v3.py']];files+=['outputs/'+name for name in ['METADATA_CONTRACT81_V1.json','LIVE_C71_SNAPSHOT81_V1.json','COHORT81_V1.json','QUALIFICATION81_V1.json','OPENING81_V1.json','RESULTS81_V1.json','VERIFICATION81_V1.json','DECISION81_V1.json','CLOSURE81_V1.json','all_metrics81_v1.csv','run81_v1_launch.json','run81_v1_exit.json','run81_v1_stdout.log']];pub=[]
 for rel in sorted(files):
  b=(ROOT/rel).read_text(encoding='utf-8-sig').encode();assert all(token.encode() not in b for token in json.loads(os.environ.get('C81_PRIVATE_EXPORT_PATTERNS_JSON','[]'))),(rel,'private export pattern');p=dest/'files'/rel;p.parent.mkdir(parents=True,exist_ok=True);p.write_bytes(b);pub.append(dict(path=rel,bytes=len(b),sha256=hashlib.sha256(b).hexdigest(),git_blob_sha1=hashlib.sha1(b'blob '+str(len(b)).encode()+b'\0'+b).hexdigest(),server_snapshot_file=str(p),encoding='utf-8'))
 b=figure.read_bytes();rel='figures/development_new_ic81_v1.png';p=dest/'files'/rel;p.parent.mkdir(parents=True,exist_ok=True);p.write_bytes(b);pub.append(dict(path=rel,bytes=len(b),sha256=hashlib.sha256(b).hexdigest(),git_blob_sha1=hashlib.sha1(b'blob '+str(len(b)).encode()+b'\0'+b).hexdigest(),server_snapshot_file=str(p),encoding='binary'))
 dump(dest/'manifest.json',dict(utc=utc(),files=pub,file_count=len(pub),bytes=sum(z['bytes'] for z in pub),large_data_weights_prediction_arrays_included=False,accepted_final_directions=0,overall_goal='ACTIVE_UNACHIEVED',local_payload_files_written=0));print(json.dumps(dict(status='SNAPSHOT_READY',files=len(pub),manifest_sha256=sha(dest/'manifest.json'),decision=decision)),flush=True)
if __name__=='__main__':main()
