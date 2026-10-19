# C37 冻结学习压力筛查

2026-10-17，首次训练前冻结。不是新方法实验，也不是GEFCom原比赛排行榜复现。目标是发现可复用world-model机制是否有残差；当前接受最终方向0。

## 数据与信息

原publisher附件SHA `d68d957270edd93b26a37d0f9b5e901f942abdf34c75eacbe14e417beb16e154`，wind/solar固定zone1。使用`data_v3`，按原组织者daily-midnight+24h合同，将每origin D00对应D01..D+1 00的24个标签/forecast。原时间0..23不移动，不发明UTC；这是声明清楚的benchmark迁移，不是逐发行日志回放。太阳predictors POWER显式排除，标签仅来自train15。

history=168小时，时间origin-168h..origin-1h，目标origin+1h..+24h。历史target和历史NWP、未来NWP与6个calendar sin/cos(hour,dow,year)对全部模型一致。无真实未来weather标签，因此不训练/评分physical covariate error模型。

先按完整source origin日序列固定前56% FIT、随后14% CAL、随后15% VAL、末15% TEST；整数边界floor(.56n,.70n,.85n)。每前一分区最后一日作embargo不用，保障已用标签与下一分区origin间隔。之后统一排除history/future中标签/输入缺失或历史不足的完整窗口，记录分区counts；不插值。标准化只用FIT。TEST本轮不评分。

## 模型与已知对照

两个数据×seeds2026371/2026372×3种neural fits=12fits。共同batch64、Adam lr0.001/weight_decay0.0001、4000updates、clipgrad1、固定batch索引计划、无额外HPO；VAL step0及每500步，按raw sample CRPS选checkpoint，保留全部曲线与selected权重。4000是筛查cap，不自动称充分收敛；末1000步仍改善>1%或step0最佳时标记，不能据此宣称新机制必要。

- `quantile_mlp`：相同全部输入flatten，Linear96/GELU/Linear96/GELU/Linear(24×19)，q=.05,.10,..,.95的pinball；预测quantiles排序。线性外推至p0/1只供采样，不加物理clipping。解析逆CDF用相同均匀概率格生成128 paths。
- `kite_l1`：原作者commit3140ee824cbd80c5ec7fdf2b54666210519d0b39的KITEModel/flow_loss/layers；flowdim64/depth2/head4/mlpratio2/rank8/rate2/10Eulersteps，其余structure_max.9/min_sigma.15/aux.1/omega1/p_uncond.1/prior sample保持明确。缩小模型和自建daily训练器是component transfer，不冒充原论文复现。
- `kite_l2`：相同初始化、schedule与随机种子，仅将唯一`F.l1_loss(v_pred,v_target)`改为标准`F.mse_loss`。保存原SHA/精确替换与新SHA；不修改旧源码。是已知目标对照，不称新方法。

Native KITE采样接口平均num_samples。路径adapter只将输入repeat_interleave并调用原inference(num_samples=1)，reshape得paths；预检其平均与原num_samples=N API一致。原loss/source噪声过程不另改。不会仅因L1与平方损失不同就断言实际失败。

## 普通联合分布控制与校准

quantile_mlp同一fit的IID、CAL-PIT历史rank shuffle、32个nearest-CAL-analogue rank shuffle共享每lead完全相同的128个样本值。copula来自CAL的PIT秩，仅改变跨lead排列；主张可逐元素核验sorted samples一致，因此marginal CRPS相同。历史rank重排是Schaake/ECC类已知控制，不取新名当创新。analogue只按当时可见的forecast各列均值/标准差、history最后值/均值/标准差选邻居，无未来error gating。

所有模型另报告使用相同CAL标签的逐lead逐quantile加性校准：offset=CAL residual的对应q分位数，排序后按原path秩重排。raw与calibrated均报告，不能只给新方法校准。此是普通解释，不是额外fits。

加一个无随机神经训练的HGB point control：相同全flatten输入加lead onehot，max_iter200/max_leaf_nodes15/learning_rate.05/min_samples_leaf20/l2=1/max_bins63，无HPO；CAL残差path bootstrap和32-nearest analogue residual path。它用CAL时，与native calibrated组分开标明。内存或实际耗时需记录，不宣称参数量/计算严格匹配。

## 评分、数值与推断预算

VAL采样选模64 paths、固定seed恢复训练RNG。最终128paths；primary raw-unclipped，clipped[0,1]仅共同敏感性。每origin保存target、sample和score。主分数为marginal fair-CRPS与24h平均功率fair-CRPS；energy score/variogram(p=.5、inverse lag weight)/最大绝对ramp CRPS及MAE辅助。按origin日给配对7日block bootstrap，报告两seed，不把24小时或128paths当独立实验。

fair-CRPS=mean|X-y|-sum(i≠j)|Xi-Xj|/(2M(M-1))；energy score同样排除pair diagonal并除sqrt24。路径重排必须保持所有marginal scores相同。原始采样与50Eulersteps只在已选权重上作固定一次敏感性，不反选checkpoint。路径采样数增加仅降低MC误差，不自动解释为C36的外部forcing积分压力。

## 判断

若raw缺陷被同信息rank coupling、普通CAL、标准L2或更准确积分解释，不提出新的state机制。若几类健康ordinary方法仍有稳定、重要且可重复的跨任务残差，再形成竞争机制和新方法协议。单seed/VAL探索、容量/预算差异、训练cap和完整case选择均保留局限。不要求所有经典方法无限预算失败，也不把当前候选变成数据集调参。

原理源：Flow Matching for Generative Modeling(2210.02747)的平方向量场回归；Clark etal2004 Schaake shuffle原论文/USGS记录；原KITE代码/论文。主来源核实但本protocol不是创新声明。旧成果与失败读取器全部保留，禁止触碰cycle6 secret launcher。
