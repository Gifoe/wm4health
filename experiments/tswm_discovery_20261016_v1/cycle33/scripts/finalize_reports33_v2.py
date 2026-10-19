"""Evidence-based interpretation after immutable evaluation and two audits."""
import pathlib,json,hashlib,datetime,shutil,statistics
ROOT=pathlib.Path(__file__).resolve().parents[1];BASE=ROOT.parent
def sha(p):return hashlib.sha256(pathlib.Path(p).read_bytes()).hexdigest()
def main():
    out=ROOT/'outputs/interpretation_v1';assert not (out/'decision.json').exists()
    a=json.loads((ROOT/'outputs/analysis_v1/analysis.json').read_text())
    er=json.loads((ROOT/'outputs/test_evaluation_v1/result.json').read_text())
    v=json.loads((ROOT/'outputs/test_evaluation_v1/verification.json').read_text())
    p=json.loads((ROOT/'outputs/test_evaluation_v1/provenance_closure.json').read_text())
    pipeline=json.loads((ROOT/'outputs/postprocess_status.json').read_text())
    train=json.loads((ROOT/'outputs/readout_fits_v1/training_complete.json').read_text())
    cache=json.loads((ROOT/'outputs/native_cache_v1/complete.json').read_text())
    assert pipeline['status']=='completed' and v['status']==p['status']=='passed'
    assert v['evaluation_sha256']==p['evaluation_sha256']==a['evaluation_sha256']==sha(ROOT/'outputs/test_evaluation_v1/result.json')
    out.mkdir(parents=True,exist_ok=True);history=ROOT/'outputs/report_history_v1';assert history.is_dir()
    names=['literature_review.md','baseline_registry.md','experimental_results.md','negative_evidence.md',
           'hypothesis_space.md','research_ideas.md','final_research_direction.md','CURRENT_STATE.md']
    # V1 preserved the original report history before a string/Path utility
    # error. Resume presentation from those originals, without any re-fit,
    # re-evaluation, or duplicated report prefixes.
    for name in names:assert (history/name).is_file()
    for name in ['CURRENT_RESEARCH_STATUS.md','CONTINUATION.md']:assert (history/('root_'+name)).is_file()
    stamp=datetime.datetime.now(datetime.timezone.utc).isoformat()
    def summ(ds,arm):return next(r for r in a['summaries'] if r['dataset']==ds and r['arm']==arm)
    def contrast(ds,candidate,baseline):return next(r for r in a['contrasts'] if r['dataset']==ds and r['candidate']==candidate and r['baseline']==baseline)
    objective={}
    for ds in ['NP','PJM']:
        mix=[r['objective_mae'] for r in a['selection_control'] if r['dataset']==ds and r['arm']=='native_mix']
        pred=[r['objective_mae'] for r in a['selection_control'] if r['dataset']==ds and r['arm']=='native_pred']
        objective[ds]={'mix_objective_selected_mae_mean':statistics.mean(mix),
            'pred_objective_selected_mae_mean':statistics.mean(pred),
            'pred_gain_percent':100*(statistics.mean(mix)-statistics.mean(pred))/statistics.mean(mix)}
    tradeoff=100*(summ('PJM','global_pred')['mse_mean']/summ('PJM','native_pred')['mse_mean']-1)
    caps=sum(r['cap_still_improving_count'] for r in a['summaries'])
    finding=(f'UTC {stamp}。C33全部24 fits、56组TEST输出、独立数值复核与来源闭合已完成；整体目标ACTIVE_UNACHIEVED，接受最终方向0。\n\n'
        '本轮结论：**关闭当前target/readout线索作为新world-model框架的动机；不继续在NP/PJM上调这条路线。** 固定相同stage1后，普通global MLP已降低部分预测误差，因此此前误差不能直接证明latent信息不足。此为普通机制归因，不是新模型。\n\n')
    core=(f"prediction-only相对mix的primary MAE下降：NP {contrast('NP','native_pred','native_mix')['pooled_gain_percent']:.3f}%，PJM {contrast('PJM','native_pred','native_mix')['pooled_gain_percent']:.3f}%；PJM三个配对seeds并非全正。改用预列原生训练objective选模后，下降只有NP {objective['NP']['pred_gain_percent']:.3f}%、PJM {objective['PJM']['pred_gain_percent']:.3f}%，不能把primary差异全归为辅助重构的系统性负迁移。\n\n"
        f"global相对native prediction-only的MAE下降：NP {contrast('NP','global_pred','native_pred')['pooled_gain_percent']:.3f}%，PJM {contrast('PJM','global_pred','native_pred')['pooled_gain_percent']:.3f}%，各自三个decoder seeds同向。PJM的168-origin区块敏感性范围跨0，而且MSE上升{tradeoff:.3f}%；不声称跨系统稳定或全面改善。\n\n"
        '加宽local的MAE变化很小且seed不一致。global同时改变非局部聚合、参数量及对已有RevIN统计的显式条件使用，因此不能单独归因为某一种新动力学机制。它没有增加观测，但与local的函数类/参数和计算不同。\n\n')
    budget=(f"全部训练{train['seconds']:.3f}秒，float32特征缓存{cache['seconds']:.3f}秒；20epochs、每epoch1141updates、每fit22820updates，24fits共547680updates。{caps}/24 fits触发末5epoch仍改善>0.5%的cap标记，不能称充分收敛。\n\n")
    proof=(f"独立核验112个MAE/MSE、120条epoch顺序计划、24个初始化/参数/选模及48个选中checkpoint关联；指标复算最大差{v['maximum_metric_abs_error']:.1f}。TEST特征/选中head独立重放最大绝对差{v['maximum_cached_native_fixture_abs_error']:.3g}/{v['maximum_selected_head_fixture_abs_error']:.3g}。TRAIN/VAL额外{p['trainval_feature_fixtures_checked']}项特征fixture和全量target、data/config/旧权重/来源SHA一致。\n\n")
    decisions={
      'experimental_results.md':finding+core+budget+proof+'以下为原始完整自动汇总，数值未改。\n\n',
      'baseline_registry.md':finding+'本轮四个head只构成普通诊断envelope，原stage2是预算/初始化不同的reference。global参数为NP924696、PJM1678360，分别约为原local的8.02倍、4.77倍；相同updates不等于相同compute。\n\n'+budget+proof,
      'negative_evidence.md':finding+core+'撤销“真实模型残差已证明target信息不足”的主张；不据此宣布所有latent目标均充分。没有证明mixed supervision普遍有害，也没有证明global单一机制解决了world modeling。oracle真编码重构使用未来真实目标，禁止列为forecast胜利。\n\n',
      'hypothesis_space.md':finding+core+f'预算解释仍未排除：{caps}/24个fit有cap标记。表示信息不足解释尚无直接证据；本轮普通读出已取回部分可用信息。若未来重新进入此路线，需要新的独立任务与同输入observation predictor/优化控制，不能在这两旧TEST上再做loss搜索。\n\n',
      'research_ideas.md':finding+'A（future-target/report compatibility）作为当前最终方向关闭：数学问题有意义，实证不足且普通修复提供反证。B（mixed decoder监督）归因已完成，不能把alpha=0取名新方法。C（多尺度/随机target预算）仍仅是已有拥挤设计空间，尚无具体新交互，不推荐为替代标题。\n\n'+core,
      'literature_review.md':finding+'文献判据未改：WorldTS已含prediction-only route，扩大decoder或把future tokens送入普通MLP不构成新先例。本轮实证为这些普通控制提供数值，不能反推所有JEPA/PSR或概率模型已解决全部任务。没有新增未核实的创新性声明。\n\n',
      'final_research_direction.md':finding+'**Research Question：** 原生learned latent的误差是否在普通读出/监督控制后仍要求新的表示机制？本轮未发现足够支持这一新框架动机的残差。\n\n**Evidence：** '+core+
        '**Existing Methods：** native prediction-only和普通global MLP已解释部分差异；还存在选模与固定训练cap解释。\n\n**Proposed Idea：** 当前不接受新方法，不将这组普通改动包装成world-model框架。\n\n**Falsification：** 本轮实现了原路线预列的ordinary-readout反证门槛；广义target兼容性数学问题未被否定，但当前实证动机不足。\n\n**Feasibility：** '+budget+proof+
        '**Next Experiment：** 关闭NP/PJM target/readout追加调参。回到不同已发布可复用模型的明确能力问题，先冻结新独立任务和强控制；下一项具体新机制尚未选定，不能宣称已完成用户要求的Research Gap发现。\n\n以下保存自动生成过程稿，仅作历史说明；上述完成后判断为最新。\n\n',
      'CURRENT_STATE.md':finding+core+budget+proof
    }
    for name,prefix in decisions.items():(ROOT/name).write_text(prefix+(history/name).read_text(encoding='utf-8'),encoding='utf-8')
    decision={'status':'closed_as_final_framework_motivation','overall_goal':'ACTIVE_UNACHIEVED','accepted_directions':0,
        'evaluation_sha256':sha(ROOT/'outputs/test_evaluation_v1/result.json'),'verification_sha256':sha(ROOT/'outputs/test_evaluation_v1/verification.json'),
        'closure_sha256':sha(ROOT/'outputs/test_evaluation_v1/provenance_closure.json'),'script_sha256':sha(__file__),
        'native_objective_selected_control':objective,'PJM_global_MSE_increase_percent':tradeoff,'cap_flagged_fits':caps,
        'not_a_claim_all_latent_models_solved':True,'no_additional_NP_PJM_tuning_on_this_route':True,
        'report_files':[{'path':name,'sha256':sha(ROOT/name)} for name in names]}
    (out/'decision.json').write_text(json.dumps(decision,indent=2,ensure_ascii=False),encoding='utf-8')
    header=('<!-- cycle33-complete-start -->\n# C33已闭合：普通读出反证，当前不推荐新框架\n\n'+finding+core+budget+proof+
        '完整材料见[原生归因结果](cycle33/experimental_results.md)、[最终判断](cycle33/final_research_direction.md)、[冻结方案](cycle33/NATIVE_READOUT_PLAN.md)。C33训练及全部postprocess均已退出；以下旧live状态/旧“当前”仅为历史。\n<!-- cycle33-complete-end -->\n\n')
    for name in names[:-1]:(BASE/name).write_text(header+(BASE/name).read_text(encoding='utf-8'),encoding='utf-8')
    for name in ['CURRENT_RESEARCH_STATUS.md','CONTINUATION.md']:(BASE/name).write_text(header+(BASE/name).read_text(encoding='utf-8'),encoding='utf-8')
    print(json.dumps(decision,ensure_ascii=False))
if __name__=='__main__':main()
