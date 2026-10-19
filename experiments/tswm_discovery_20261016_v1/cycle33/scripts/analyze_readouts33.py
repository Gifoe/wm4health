"""Predeclared descriptive attribution, only after independent verification."""
from native_common33 import *
import datetime,math
EVAL=ROOT/'outputs/test_evaluation_v1'
OUT=ROOT/'outputs/analysis_v1'

def main():
    assert not OUT.exists()
    proof=json.loads((EVAL/'verification.json').read_text())
    assert proof['status']=='passed' and proof['evaluation_sha256']==sha(EVAL/'result.json')
    er=json.loads((EVAL/'result.json').read_text())
    tc=json.loads((FITS/'training_complete.json').read_text())
    OUT.mkdir(parents=True)
    primary=[r for r in er['records'] if r['selection']=='forecast']
    reference=[r for r in er['records'] if r['selection'] in ['original','fixed']]
    def record(ds,arm,s):
        return next(r for r in er['records'] if r['dataset']==ds and r['method']==arm and
                    (r['seed']==s and r['selection']=='forecast' if arm in ARMS else r['selection'] in ['original','fixed']))
    summaries=[]
    for ds in ['NP','PJM']:
        for arm in ARMS:
            rows=[record(ds,arm,s) for s in SEEDS]
            fits=[r for r in tc['fits'] if r['dataset']==ds and r['arm']==arm]
            summaries.append({'dataset':ds,'arm':arm,'seeds':SEEDS,
                'mae_mean':float(np.mean([r['mae'] for r in rows])),
                'mae_seed_std':float(np.std([r['mae'] for r in rows],ddof=1)),
                'mse_mean':float(np.mean([r['mse'] for r in rows])),
                'oracle_target_code_reconstruction_mae_mean':float(np.mean([r['oracle_target_code_reconstruction_mae'] for r in rows])),
                'selected_epochs':[r['selected_epoch'] for r in rows],
                'parameters':fits[0]['parameters'],
                'mean_seconds':float(np.mean([r['seconds'] for r in fits])),
                'cap_still_improving_count':sum(r['cap_still_improving'] for r in fits),
                'peak_memory_bytes_max':max(r['peak_allocated_including_shared_cache_bytes'] for r in fits)})
    contrasts=[]
    pairs=[('native_pred','native_mix'),('wide_local_pred','native_pred'),('global_pred','native_pred'),
           ('global_pred','wide_local_pred'),('native_mix','original_stage2_reference'),
           ('native_pred','original_stage2_reference'),('global_pred','seasonal_24')]
    for ds in ['NP','PJM']:
        for candidate,baseline in pairs:
            errors=[];baseerrors=[];gains=[]
            for s in SEEDS:
                a=np.load(EVAL/record(ds,candidate,s)['path'])
                b=np.load(EVAL/record(ds,baseline,s)['path'])
                assert np.array_equal(a['origins'],b['origins']) and np.array_equal(a['true'],b['true'])
                ea=np.abs(a['pred'].astype(np.float64)-a['true']).mean((1,2))
                eb=np.abs(b['pred'].astype(np.float64)-b['true']).mean((1,2))
                errors.append(eb-ea);baseerrors.append(eb)
                gains.append(float(100*(eb.mean()-ea.mean())/eb.mean()))
            diff=np.stack(errors);be=np.stack(baseerrors);n=diff.shape[1]
            # Resample whole contiguous origin blocks and decoder seeds. This is
            # conditional sensitivity, not independent-system significance.
            bounds=list(range(0,n,168));counts=np.array([min(168,n-t) for t in bounds])
            totals=np.stack([diff[:,t:t+168].sum(1) for t in bounds],1)
            bt=np.stack([be[:,t:t+168].sum(1) for t in bounds],1)
            rng=np.random.default_rng(202633033);draws=[]
            for _ in range(2000):
                si=rng.integers(0,3,3);bi=rng.integers(0,len(bounds),len(bounds))
                numerator=totals[np.ix_(si,bi)].sum();denominator=bt[np.ix_(si,bi)].sum()
                draws.append(100*numerator/denominator)
            contrasts.append({'dataset':ds,'candidate':candidate,'baseline':baseline,
                'gain_percent_each_seed':gains,'all_three_seeds_improve':all(g>0 for g in gains),
                'mean_absolute_error_reduction':float(diff.mean()),
                'pooled_gain_percent':float(100*diff.mean()/be.mean()),
                'descriptive_seed_and_168origin_block_percentile95':np.percentile(draws,[2.5,97.5]).tolist(),
                'blocks':len(bounds),'last_block_origins':int(counts[-1]),'bootstrap_draws':2000,
                'not_population_confidence_or_fresh_confirmation':True})
    selection=[]
    for r in primary:
        other=next(x for x in er['records'] if x['dataset']==r['dataset'] and x['method']==r['method'] and
                   x['seed']==r['seed'] and x['selection']=='objective')
        selection.append({'dataset':r['dataset'],'arm':r['method'],'seed':r['seed'],
             'forecast_epoch':r['selected_epoch'],'objective_epoch':other['selected_epoch'],
             'forecast_mae':r['mae'],'objective_mae':other['mae'],
             'objective_minus_forecast_mae':other['mae']-r['mae']})
    result={'status':'complete_verified_descriptive_attribution','evaluation_sha256':sha(EVAL/'result.json'),
        'verification_sha256':sha(EVAL/'verification.json'),'analysis_script_sha256':sha(__file__),
        'summaries':summaries,'contrasts':contrasts,'selection_control':selection,'references':reference,
        'accepted_final_research_directions':0,'overall_goal':'ACTIVE_UNACHIEVED',
        'limitations':['one frozen stage1 seed per dataset','two related previously studied electricity datasets',
           'same updates are not equal compute','fixed 20 epoch budget may remain optimization limited',
           'readout interventions are standard controls, not novel world model mechanisms',
           'block sensitivity conditions on old split and trained checkpoints; no population inference']}
    dump(OUT/'analysis.json',result)
    stamp=datetime.datetime.now(datetime.timezone.utc).isoformat()
    common=f'UTC {stamp}。整体通用时间序列 world-model 发现任务 ACTIVE_UNACHIEVED；接受最终方向0。\n\n'
    table='| 数据 | 普通读出 | MAE均值±seed标准差 | MSE均值 | 参数 | 选中epoch | 仍改善/cap |\n|---|---|---:|---:|---:|---|---:|\n'
    for r in summaries:
        table+=f"| {r['dataset']} | {r['arm']} | {r['mae_mean']:.6f} ± {r['mae_seed_std']:.6f} | {r['mse_mean']:.6f} | {r['parameters']} | {r['selected_epochs']} | {r['cap_still_improving_count']}/3 |\n"
    ct='| 数据 | 候选 / 基准 | MAE下降%逐seed | 汇总下降% | 168-origin区块+seed敏感性95%范围 |\n|---|---|---|---:|---|\n'
    for r in contrasts:
        ct+=f"| {r['dataset']} | {r['candidate']} / {r['baseline']} | {', '.join(f'{g:.2f}' for g in r['gain_percent_each_seed'])} | {r['pooled_gain_percent']:.2f} | {r['descriptive_seed_and_168origin_block_percentile95']} |\n"
    rt='| 数据 | 固定reference | MAE | MSE |\n|---|---|---:|---:|\n'
    for r in reference:rt+=f"| {r['dataset']} | {r['method']} | {r['mae']:.6f} | {r['mse']:.6f} |\n"
    reports={
      'experimental_results.md':common+'# 原生冻结读出归因结果\n\n'+table+'\n'+rt+'\n'+ct+
        '\n正数表示候选MAE下降。范围是旧数据、固定checkpoint条件下的描述敏感性，不是跨系统显著性。oracle true-target-code reconstruction单列于机器结果，含未来真实编码，不能作为forecast或信息充足性的证明。所有24 fits先完成选模，随后统一打开TEST。\n',
      'baseline_registry.md':common+'# 原生与诊断协议\n\n原始完整stage2仅作reference；新24 fits冻结原stage1，4种普通decoder×3 decoder seeds×2数据。native_mix与native_pred逐seed同初始化；20完整epochs，batch32，每fit22820更新，共547680更新。两臂之外的wide/global参数和计算不同。所有选模用validation forecast MAE；另完整报告训练objective选模控制。详见[NATIVE_READOUT_PLAN.md](NATIVE_READOUT_PLAN.md)。\n\n'+table+
        '\n原模型是source-native旧复现；新decoder矩阵是自主机制诊断，不是新增模型排行榜或完整官方训练复现。共享float32冻结特征cache成本另见native_cache_v1/complete.json，优化时间与显存见每fit result.json。\n',
      'negative_evidence.md':common+'# 本轮排除与未排除\n\n本轮不把fixed可逆编码的人工反例当作learned模型失败。普通prediction-only、局部容量扩大和global MLP均属现有修复；其任何收益都首先削弱“必须新框架”的动机。\n\n'+ct+
        '\n没有通过这些读出器就直接证明潜态丢失信息的结论。20epoch后仍下降的fit只能给预算内结果。没有新独立数据，没有多个stage1 seeds，没有same-input observation predictor上界；这些缺项阻止接受最终gap。此前C29–C32负证据继续保留。\n',
      'hypothesis_space.md':common+'# 竞争解释\n\n1. 辅助真编码重构产生负迁移：由同初始化native_mix/pred对照检验。若pred未稳定改善，不支持其作为共同失败机制。\n2. 局部容量或局部读出限制：由wide_local/global普通控制检验；提升不能归为潜态信息丢失。\n3. 优化预算/选模：20epoch固定cap、完整曲线、两个预列selection控制可识别部分；cap未收敛不支持principle limit。\n4. 真实表示/预测信息不足：目前尚未排除同输入普通监督与优化控制，不能接受。\n\n'+ct,
      'literature_review.md':common+'# 文献边界沿用与实证增量\n\n[WorldTS](https://arxiv.org/html/2609.31162v1#S4.SS5)原本已含预测潜态decode、真编码重构、prediction-only选项和历史RevIN信息，故不能把这些重新命名为方法创新。本轮严格调用已锁定官方核心源码269426ea6217d7152a244da0e209d922acd46d4f。\n\n[LatentTSF](https://arxiv.org/html/2602.00297v2)等latent目标协议不同；本轮不做跨协议结论。C32的surrogate/link、PSR/PSIM、JEPA相邻先例及三个独立视角核查原样保留于[cycle32](../cycle32/literature_review.md)。本轮增量是实际native checkpoint的受控读出实验，不新增“尚无先例”的主张。\n',
      'research_ideas.md':common+'# 候选判断\n\n本轮没有新增可接受研究方向。通用world-model目标不等于把某个电力数据集的读出器换大。若普通控制解释观察，应终止该机制的创新主张。若仍有残差，先做同允许信息的observation predictor及收敛控制，再决定是否值得换独立领域；不能直接设计复杂架构。\n\n'+ct,
      'final_research_direction.md':common+'# 最终方向尚未确定\n\n**Research Question：** 可复用world model的learned预测表示是否存在普通读出器/训练控制后仍不可修复的任务信息缺口？当前仍是待证伪问题。\n\n**Evidence：** 原生旧checkpoint被冻结，24个普通decoder重训、56组预列TEST输出和独立源码/原数据/缓存/初始化/更新/选模/重放/指标校验完成。详见[结果](experimental_results.md)。\n\n**Existing Methods：** 原生prediction-only监督、扩大普通decoder、global readout、统一forecast选模均为已知控制，不能作为新framework。\n\n**Proposed Idea：** 尚未提出达到创新门槛的新机制；不将本轮结果包装为新模型。\n\n**Falsification：** 若普通控制消除失败，相关动机撤销；若训练cap尚未收敛或输入不公平，不能归因于表示。\n\n**Feasibility：** 单RTX5090完成原生float32特征缓存和24个decoder fits。数据/源码/旧checkpoint SHA、样本顺序、选模和完整预测均保存。\n\n**Next Experiment：** 根据完整曲线和普通控制结果决定是否关闭target/readout路线；只有经same-input observation/优化envelope后保留的失败才值得转向新的独立领域验证。当前结果不足以确定最终顶会研究方向。\n'
    }
    for name,body in reports.items():(ROOT/name).write_text(body,encoding='utf-8')
    (ROOT/'CURRENT_STATE.md').write_text(common+'C33完成真实原生模型冻结读出归因及独立校验。该轮是机制控制，不是新方法验证。\n\n'+table,encoding='utf-8')
    print(json.dumps({'status':result['status'],'accepted_final_research_directions':0,'summaries':summaries,'contrasts':contrasts}),flush=True)

if __name__=='__main__':main()
