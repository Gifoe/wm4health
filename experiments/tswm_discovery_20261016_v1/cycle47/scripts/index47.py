"""Chronological project index; preserve prior root bytes and compare before edit."""
from common47 import *
def main():
 assert read(R/'outputs/eval_v1/verification.json')['status']=='PASS_FROZEN_FINITE_GRID_SCREEN';assert read(R/'outputs/figures_v1/visual_qa.json')['status']=='VISUALLY_INSPECTED_PASS';out=R/'outputs/root_history_v1';assert not out.exists();out.mkdir();records=[];utc=now();tag='<!-- cycle47-progress-index-start -->'
 for name in ['CURRENT_RESEARCH_STATUS.md','literature_review.md','baseline_registry.md','experimental_results.md','negative_evidence.md','hypothesis_space.md','research_ideas.md','final_research_direction.md']:
  p=R.parent/name;old=p.read_bytes();assert tag.encode() not in old;(out/name).write_bytes(old);header=f'''{tag}
# C47：状态—观测时钟新目标完成反证，当前方案降优先级

UTC {utc}。研究目标仍是可复用的时间序列 world-model 框架，以 WorldTS 的贡献范围为参照。数据用于检验能力；不把具体数据集涨点当作最终贡献。整体 ACTIVE_UNACHIEVED，接受最终方向0。

本轮完成16个 fresh fits、21200正式updates；两种有限网格动力学、两个初始化、四种同参数学习目标。封闭评估54组raw paths /13824模型—起点—策略分数，对应512条独立物理轨迹及1536个配对观测历史。NumPy/SciPy独立重算全部评分、log-domain似然与状态—计数协方差通过，ES最大差4.09e-7。见[完整结果](cycle47/experimental_results.md)与[计数说明](cycle47/COUNT_GUIDE.md)。

完整随机时钟似然已恢复受限均值时钟丢失的效用。新Laplace/hybrid目标在OU无可靠未来质量收益，在双稳态两个种子中均更差：跨策略平均ES差分别+0.009927/+0.006875。降低当前固定feature-bank方案优先级；单步更便宜尚不能证明匹配总成本或高维扩展优势。见[科学裁决](cycle47/SCIENTIFIC_DECISION.md)。

真实执行的是固定1D grid、已知emission/初始先验下的受控筛查，非HyperNSDE完整复现或真实域框架验证。HyperNSDE已有unsplit随机时钟分支；PCF-GAN/SigTPP已有全局路径分布匹配。新颖性不能来自重命名普通组件。原始数据、权重、优化器、日志、图及SHA保留，未调EVAL。C44原查询协议仍明确INCOMPLETE，不重启。

以下为历史。
<!-- cycle47-progress-index-end -->

''';assert p.read_bytes()==old;p.write_bytes(header.encode('utf-8')+old);records.append({'path':name,'before':hashlib.sha256(old).hexdigest(),'after':sha(p),'bytes':p.stat().st_size})
 dump(R/'ROOT_INDEX_RECEIPTS.json',{'utc':utc,'files':records,'goal_complete':False});print(json.dumps({'root_files_updated':len(records),'prior_bytes_preserved':True}),flush=True)
if __name__=='__main__':main()
