"""Report all measured-transfer evidence without turning a pilot into a new-method claim."""
import runtime_profile
import csv,json,pathlib
import numpy as np
from world_data import ROOT,sha,dump
OUT=ROOT/'outputs/real_transfer'
def read(p):return json.loads(p.read_text(encoding='utf-8'))
def main():
    proof=read(OUT/'verification.json');assert proof['status']=='passed' and proof['results_sha256']==sha(OUT/'results.json');meta=read(ROOT/'outputs/real_splits/manifest.json')
    with (OUT/'per_window_metrics.csv').open(encoding='utf-8',newline='') as f:rows=list(csv.DictReader(f))
    methods=list(dict.fromkeys(r['method'] for r in rows));families=list(dict.fromkeys(r['family'] for r in rows));summary=[]
    table=['| 物理系统 | 模型 | H128 train-std NRMSE | seed std | 训练seed数 |','|---|---|---:|---:|---:|']
    for family in families:
        records=list(dict.fromkeys(r['record'] for r in rows if r['family']==family))
        for method in methods:
            seeds=list(dict.fromkeys(r['seed'] for r in rows if r['method']==method));vals=[]
            for seed in seeds:
                per_record=[]
                for record in records:
                    selected=[float(r['nrmse_train_std']) for r in rows if r['family']==family and r['record']==record and r['method']==method and r['seed']==seed and r['horizon']=='128'];assert selected;per_record.append(np.mean(selected))
                vals.append(float(np.mean(per_record)))
            mean=float(np.mean(vals));std=float(np.std(vals,ddof=1)) if len(vals)>1 else None;summary.append(dict(family=family,method=method,nrmse128=mean,seed_std=std,n_seeds=len(vals),record_weighting='equal mean inside plant; repeated contexts are not independent plants'))
            table.append(f"| {family} | {method} | {mean:.6f} | {std if std is not None else '无'} | {len(vals)} |")
    text='''# 真实系统迁移：固定模型、固定协议

本轮只有两个独立物理系统、三个非重叠记录角色；71个窗口不是71个独立装置。这里是 recorded-input rolling-origin simulation，允许每个origin之前128步真实观测，再无输出反馈模拟128步。它不同于官方50步初始化后整段open-loop评测。未来u是历史实际执行输入，不提供未执行计划真值；不据此宣称实际在线控制收益或因果忠实性。

checkpoint只按独立合成validation选取，无真实记录上的神经训练或测试调参。单位变换只来自官方train段；外部ICL/PCFM预训练预算不同，PCFM采用原生中位数、其他神经模型采用均值点预测，不虚称完全匹配训练预算。Silverbox的重叠no-extrapolation子集未重复汇总。

'''+ '\n'.join(table)+f'''

所有4260行分数、原始预测、1136个ARX拟合与抽样神经重放已独立复核。实际校验数量以verification.json为准：{proof['metric_rows_rescored']}行，ARX {proof['arx_refits']}次，神经 {len(proof['sampled_neural_replays'])}组。表中seed std只表示3个训练随机种子的差异，不是物理装置总体置信区间。

这是一组模型组织方式的迁移证据，不自动确认新Research Gap。时标、系统先验覆盖、普通非线性识别与训练预算仍需根据具体失败排查。当前接受最终方向0，总体目标ACTIVE、UNACHIEVED。不得将该试验单独包装为一个通用新模型的充分验证。

源协议 REAL_TRANSFER_PLAN.md；官方分割及原始文件SHA见 outputs/real_splits/manifest.json；全部窗口分数见 outputs/real_transfer/per_window_metrics.csv。
'''
    (ROOT/'REAL_TRANSFER_REPORT.md').write_text(text,encoding='utf-8');dump(OUT/'summary.json',summary)
    print(json.dumps(dict(status='reported',physical_systems=meta['physical_systems'],windows=meta['n_forecast_windows'],metric_rows=proof['metric_rows_rescored'],accepted_final_directions=0)),flush=True)
if __name__=='__main__':main()
