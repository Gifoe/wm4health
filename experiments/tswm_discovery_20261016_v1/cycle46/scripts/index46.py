"""Chronological project index; preserves prior bytes with compare-before-write."""
from common46 import *
import datetime
def main():
 assert read(R/'outputs/eval_v1/verification.json')['status']=='PASS_FOR_FROZEN_CONTROLLED_SCREEN';assert read(R/'outputs/train_val_diagnostics_v1/verification.json')['status']=='PASS_TRAIN_VAL_DIAGNOSTICS_ONLY';out=R/'outputs/root_history_v1';assert not out.exists();out.mkdir(parents=True);records=[];utc=datetime.datetime.now(datetime.timezone.utc).isoformat();tag='<!-- cycle46-progress-index-start -->'
 for name in ['CURRENT_RESEARCH_STATUS.md','literature_review.md','baseline_registry.md','experimental_results.md','negative_evidence.md','hypothesis_space.md','research_ideas.md','final_research_direction.md']:
  path=R.parent/name;old=path.read_bytes();assert tag.encode() not in old;(out/name).write_bytes(old);header=f'''{tag}
# C46：普通组合对照完整执行，简单神经分离方案未获支持

UTC {utc}。研究单位仍是可复用时间序列 world-model 框架，以 WorldTS 的贡献范围为参照。整体 ACTIVE_UNACHIEVED，接受最终方向0。

本轮完成8个 fresh fits / 27250 updates，完整 held-out joint-law 评分192组 / 15360 origins，独立 NumPy/SciPy 复核。未见时间组合 ES：最佳神经普通对照约0.5491，廉价 aligned PCA/LGSSM 为0.4409；双侧分离的额外收益在两初始化之间变号。TRAIN/VAL-only known-state probes也显示这批神经状态的线性读出明显较弱。它们不支持新框架主张，也不构成所有神经世界模型的普遍失败证据。见[完整结果](cycle46/experimental_results.md)、[诊断](cycle46/DIAGNOSTIC_INTERPRETATION.md)和[优先级裁决](cycle46/PRIORITY_DECISION.md)。

普通 joint-path adapter 已完成线性解析协方差和默认常数扩散非线性步长资格核查；原作者核心未改。所有数据、选中/末次权重、日志、样本和 SHA 保留。O1 同时改变 recognition/decoder 条件，且不同模型目标和参数量不同，不能把本表解释为孤立架构因果对比。

不继续包装简单 factorization 或为了让普通方法失败而加难度。下一步先核查观测采集策略变化与物理动态分离这一新线索的强近邻及信息条件；HyperNSDE/TESAR-CDE 已覆盖许多直接组件，尚无新方法效果。见[近邻](cycle46/literature_review.md)和[未选定研究线索](cycle46/RESEARCH_BRANCH_NOTES.md)。C44 原查询协议仍明确 INCOMPLETE，不重启。

以下为历史。
<!-- cycle46-progress-index-end -->

''';assert path.read_bytes()==old;path.write_bytes(header.encode('utf-8')+old);records.append({'path':name,'before':hashlib.sha256(old).hexdigest(),'after':sha(path),'bytes':path.stat().st_size})
 dump(R/'ROOT_INDEX_RECEIPTS.json',{'utc':utc,'files':records,'goal_complete':False});print(json.dumps({'root_files_updated':len(records),'prior_bytes_preserved':True}),flush=True)
if __name__=='__main__':main()
