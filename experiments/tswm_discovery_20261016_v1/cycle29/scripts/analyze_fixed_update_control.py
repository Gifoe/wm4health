"""Predefined attribution summaries applied to the independently verified update control."""
import runtime_profile
import csv, json, pathlib
import numpy as np
from world_data import ROOT, sha, dump
from analyze_world_attribution import crossed, SEEDS

OUT=ROOT/'outputs/fixed_update_control_v1'

def read(p):return json.loads(p.read_text(encoding='utf-8'))

def grids(path):
    with path.open(encoding='utf-8',newline='') as f: rows=list(csv.DictReader(f))
    result={}
    for family in ['lti','wh']:
        for method in ['compiled64','conditional128']:
            for horizon in [32,64,128]:
                for metric in ['factual','effect']:
                    grid=np.empty((3,128),dtype='float64')
                    for j,seed in enumerate(SEEDS):
                        selected=[r for r in rows if r['family']==family and r['method']==method and r['seed']==seed and int(r['horizon'])==horizon
                                  and ((r['plan']=='factual') if metric=='factual' else (r['plan']!='factual'))]
                        count=1 if metric=='factual' else 3
                        assert len(selected)==128*count
                        for i in range(128):
                            values=[float(r['clean_rmse' if metric=='factual' else 'effect_rmse']) for r in selected if int(r['system'])==i]
                            assert len(values)==count
                            grid[j,i]=sum(values)/count
                        assert abs(grid[j].mean()-np.mean([float(r['clean_rmse' if metric=='factual' else 'effect_rmse']) for r in selected]))<1e-12
                    assert np.isfinite(grid).all()
                    result[family,method,horizon,metric]=grid
    return result

def main():
    proof=read(OUT/'verification.json');original=read(ROOT/'outputs/world_verification/model_comparison_v2.json')
    assert proof['status']==original['status']=='passed'
    assert proof['results_sha256']==sha(OUT/'evaluation/results.json')
    assert original['results_sha256']==sha(ROOT/'outputs/world_evaluation_v2/results.json')
    fixed=grids(OUT/'evaluation/per_system_metrics.csv')
    old=grids(ROOT/'outputs/world_evaluation_v2/per_system_metrics.csv')
    rng=np.random.default_rng(202629401);contrasts=[];summaries=[]
    for family in ['lti','wh']:
        for h in [32,64,128]:
            for metric in ['factual','effect']:
                a=fixed[family,'compiled64',h,metric];b=fixed[family,'conditional128',h,metric]
                contrasts.append(dict(family=family,horizon=h,metric=metric,contrast='compiled64_minus_conditional128',**crossed(a,b,rng)))
                for method in ['compiled64','conditional128']:
                    a=fixed[family,method,h,metric];b=old[family,method,h,metric]
                    contrasts.append(dict(family=family,horizon=h,metric=metric,contrast=method+'_continued_minus_original',**crossed(a,b,rng)))
                    summaries.append(dict(family=family,horizon=h,metric=metric,method=method,mean=float(a.mean()),seed_means=a.mean(1).tolist(),seed_std=float(a.mean(1).std(ddof=1)),world_median=float(np.median(a.mean(0)))))
    training=read(OUT/'training_summary.json');curves=[]
    for row in training['rows']:
        details=read(OUT/'training'/f"{row['kind']}_{row['seed']}"/'result.json');history=details['history']
        origin=read(ROOT/'outputs/world_training_v2'/f"{row['kind']}_{row['seed']}"/'result.json')
        assert len(history)==64
        curves.append(dict(kind=row['kind'],seed=row['seed'],extra_updates=row['extra_updates'],original_best=origin['best_validation_mse'],continued_best=row['best_validation_mse'],best_step=row['best_step'],last_val=history[-1]['validation_mse'],val_2048_updates_earlier=history[-5]['validation_mse'],original_cap_flag_rule=bool(history[-1]['validation_mse']<history[-5]['validation_mse']*.95)))
    result=dict(status='completed',scope='same previously tested worlds; descriptive 3-seed/world attribution control, not independent confirmation or novelty',
                verification_sha256=sha(OUT/'verification.json'),attribution_plan_sha256=sha(ROOT/'RESULT_ATTRIBUTION_PLAN.md'),fixed_plan_sha256=sha(ROOT/'FIXED_UPDATE_CONTROL_PLAN.md'),
                source_sha256=sha(pathlib.Path(__file__)),crossed_implementation_sha256=sha(ROOT/'scripts/analyze_world_attribution.py'),
                summaries=summaries,contrasts=contrasts,curves=curves,accepted_final_directions=0)
    dump(OUT/'attribution.json',result)
    lines=['# 等更新次数归因控制','',
           '两组各3种子均32768更新；相同原batch顺序与验证最小选择。额外更新63488；原checkpoint和日志保留。不匹配精确训练FLOPs/耗时，不是新测试cohort。主张上限为当前有限预算已知组织方式比较。','',
           '| family | horizon | metric | compiled−wide | ratio | 描述性95%区间 | 3 seed差 |','|---|---:|---|---:|---:|---|---|']
    for r in contrasts:
        if r['contrast']!='compiled64_minus_conditional128':continue
        lines.append(f"| {r['family']} | {r['horizon']} | {r['metric']} | {r['mean_difference']:+.6f} | {r['ratio_of_means']:.4f} | [{r['difference_p025']:+.6f}, {r['difference_p975']:+.6f}] | "+', '.join(f'{v:+.4f}' for v in r['seed_differences'])+' |')
    lines+=['','全部36个对比（含各方法续训相对原始）及24个汇总见outputs/fixed_update_control_v1/attribution.json。Bootstrap只有3个训练随机种子，是描述性敏感度，不是广泛稳定性证明。','',
            '| model | seed | extra updates | 原validation min | 续训validation min | best step | cap局部下降标记 |','|---|---:|---:|---:|---:|---:|---|']
    for r in curves:lines.append(f"| {r['kind']} | {r['seed']} | {r['extra_updates']} | {r['original_best']:.6f} | {r['continued_best']:.6f} | {r['best_step']} | {r['original_cap_flag_rule']} |")
    lines+=['','cap标记复用原始规则（最后相对2048更新前下降>5%），不是收敛认证。即使为false，续训显著改善仍说明早停和验证噪声会限制原始排名解释。此轮未改变优化器/学习率，也未排除更强普通训练修复。',
            '', '独立验证覆盖18432条float64指标重算、6个checkpoint最小值/完整64次验证/原历史前缀/全batch计划及36世界神经重放。新的聚合使用上述已复核分数与原已复核bootstrap实现，不将其写成新的独立实验。']
    (ROOT/'FIXED_UPDATE_CONTROL_REPORT.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
    print(json.dumps(dict(status='completed',n_summaries=len(summaries),n_contrasts=len(contrasts),primary_h128=[r for r in contrasts if r['horizon']==128 and r['metric']=='effect' and r['contrast']=='compiled64_minus_conditional128'])),flush=True)

if __name__=='__main__':main()
