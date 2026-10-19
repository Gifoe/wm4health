# C31 主流视角：持续观测与动态适配

2026-10-17；检索重点 2026-04-17—10-17。身份：一手论文、公开源码和文件元数据核查；**没有下载 checkpoint、执行整模、训练或复现**。问题沿用 RESEARCH_BRIEF 的 RQ1–3。

## 证据与边界

| 来源 | 实际支持什么；不能据此声称什么 |
|---|---|
| [HiP-RSSM，ICLR 2022](https://openreview.net/pdf?id=ds8yZOUsea)，§2.2、附录 D Algorithm 1；[原库](https://github.com/ALRhub/HiP-RSSM) | 前 N 次交互推断任务后验，供下一 N 步局部固定动态使用；原生对象就是 changing dynamics，不是仅 state filtering。窗口与局部一致性是假设，没有未知事件类型辨别证据。 |
| [MTS3，NeurIPS 2023](https://proceedings.neurips.cc/paper_files/paper/2023/file/54d8aab579b5a9ed3395764c7341ebec-Paper-Conference.pdf)，§3，式(3)(5)–(7) | 已有慢参数转移与快状态过滤：$l_k\sim\mathcal N(Xl_{k-1}+Y\alpha_k,S)$，$z_t\sim\mathcal N(Az_{t-1}+Ba_{t-1}+Cl_k,Q)$。不是“固定机制+状态修正”。H 是预选时钟，不是事件边界推断。 |
| [CLAW，2026-09-10](https://arxiv.org/html/2609.12278v1)，§4–5 | 已有历史→hypernetwork→LoRA；训练从已知 family 采样环境。它反驳“少数据修改机制只能昂贵 SGD”，但不能替代流中 state kick/noise/parameter switch 证据。C29 已核查，此轮不重启低秩生成提案。 |
| [WorldTS，2026-09-25](https://arxiv.org/html/2609.31162v1)，§4.5；[Pythia，2026-10-04](https://arxiv.org/html/2610.05240v1)，§3、附录 A/E | 前者两阶段潜态预测/冻结后解码；后者冻结数值编码器、上下文残差校正及训练期 decoder adaptation。新上下文改变预测不等于未知动态的在线参数后验更新。Pythia-Tiny 原报告每阶段 8 卡×128 batch×10k updates，不能冒充单卡廉价从头复现。 |
| [COMET，2024](https://arxiv.org/html/2404.15109v1)，§3、4.4；[Mechanistic World Models，2026-07](https://arxiv.org/html/2607.12474v1)，§4–5 | COMET 已竞争学习并重组独立机制；有限数据效率允许是价值，不能用“已有模块”直接否决。其对象分解、二元交互与新机制适配有限制；7 月论文给机制中心范式，不提供已验证的通用 streaming 数值算法。 |

MTS3 的公开数值实现确实存在；[mobileRobot 配置](https://github.com/ALRhub/MTS3/blob/master/experiments/mobileRobot/conf/model/default_mts3.yaml#L12)为 latent observation 30、H=75、MLP 宽度 120/240。显存规模适合单 5090 是合理工程推断；旧 torch 1.13 环境不等于 Blackwell 可直接安装。state/task 交叉协方差缺失已在 C15 做过精确普通修复，本轮撤销该重复筛查，不包装成新环节。

## 新核查的实际强修复：AdaJEPA

[Wang et al., AdaJEPA，2026-06-30](https://arxiv.org/html/2606.32026v1)，式(4)：

\[
L_t=|B_t|^{-1}\sum_{(o,a,o')\in B_t}\|f_\theta(E_\phi(o),E^a(a))-\operatorname{sg}E_\phi(o')\|^2,
\qquad \Omega\leftarrow\Omega-\eta\nabla_\Omega L_t.
\]

它每次真实反馈后更新，权重在 episode 内保留、episode 间恢复；是参数适配，不能归为只更新 state。默认只更新 predictor 最后一块及 encoder projection head，一步 Adam，lr 分别 5e-4/1e-5。PointMaze 是 RGB+proprio/力输入，mass×0.2、damping×20 在环境初始化设定；论文未报告流中未知 switch、velocity kick 与 observation noise 的类型区分。Table 1 的 frozen 在两个 dynamics OOD 已有约 76–82% 成功率，适配增量通常较小。

**物理状态可用性不能含混**：[point_maze_dset.py:42](https://github.com/agentic-learning-ai-lab/adajepa/blob/51d8665b7978824bd218decab9e05ddb6eb1f47b/datasets/point_maze_dset.py#L42)将 states 克隆成 proprio；[maze_model.py:266–269](https://github.com/agentic-learning-ai-lab/adajepa/blob/51d8665b7978824bd218decab9e05ddb6eb1f47b/env/pointmaze/maze_model.py#L266)暴露完整 qpos+qvel。checkpoint 是否实际采用这些分支仍须读其归档 hydra.yaml；本轮未下载，不能宣称已核实。神经 latent 是学习所得，物理参数/事件标签不属于所列观测；该实现不提供纯数值、部分可观测合同的现成证明。

[官方仓库](https://github.com/agentic-learning-ai-lab/adajepa)，核查 commit `51d8665b7978824bd218decab9e05ddb6eb1f47b`：

- [adajepa.py:58–108](https://github.com/agentic-learning-ai-lab/adajepa/blob/51d8665b7978824bd218decab9e05ddb6eb1f47b/planning/adajepa.py#L58)是真实 loss/backward/step；:122 和 :147 是参数选择。
- [adajepa_mpc.py:126–185](https://github.com/agentic-learning-ai-lab/adajepa/blob/51d8665b7978824bd218decab9e05ddb6eb1f47b/planning/adajepa_mpc.py#L126)：`recent5` 存五段 executed segments，**不是五条单步 transition**；默认每段执行五个模型动作。未知变化适配延迟必须按真实物理步计。
- [mpc.py:122–129](https://github.com/agentic-learning-ai-lab/adajepa/blob/51d8665b7978824bd218decab9e05ddb6eb1f47b/planning/mpc.py#L122)每次从初始状态 replay 已执行动作前缀，再切新反馈；自主 switch 实验必须固定事件 raw-step 和噪声序列，否则各轮历史不一致。
- [公开 checkpoint 目录](https://drive.google.com/drive/folders/1FtEhkXN381Vc75hpbgSAMTHodraiQreH)：读取到 `mediummaze_dynamics_shift.zip` 384.4 MB、`pusht_visual_shift.zip` 295.8 MB；这是压缩归档大小，不是参数量或显存实测。[README](https://github.com/agentic-learning-ai-lab/adajepa#evaluation)有调用与数据入口；单样本顺序适配，未发现多 GPU 必需项。

最小官方入口为 `python plan.py --config-name adajepa_plan_gd_maze.yaml ckpt_base_path=/ABS/checkpoints/mediummaze_dynamics_shift eval_data_path=/ABS/data/point_maze_medium n_evals=1 planner.max_iter=2 +wandb_logging=false`。这是待执行命令。原环境 Linux/torch2.3/cu12.1 pin 需迁移到支持 Blackwell 的构建；[PyTorch 2.7/cu128](https://pytorch.org/blog/pytorch-2-7/)是已发布可选版本。另 [plan.py:534–544](https://github.com/agentic-learning-ai-lab/adajepa/blob/51d8665b7978824bd218decab9e05ddb6eb1f47b/plan.py#L534)会初始化 DINO 并反序列化整模块；现代 torch 默认 `weights_only=True` 的[兼容变化](https://docs.pytorch.org/docs/stable/notes/serialization#torch-load-with-weights-only-true)需处理。因此“单 5090 内存可行”可信，“原命令已在 5090 跑通”不成立。

## 唯一最小筛选与撤销条件

**结论 NONE：当前没有已测余差支撑新框架。** 唯一值得增加的自主压力测试，是把已发布 AdaJEPA 参数适配当强普通修复，测试其是否会把暂态当机制；这不是原生复现，也不证明通用数值迁移。

复用官方 PointMaze checkpoint，不增加离线训练；先检查其 hydra.yaml，完整保持实际输入支路。同一已记录动作前缀给所有方法，在随机且不对齐 chunk 的 τ 分别施加：(a) 一次速度冲击；(b) 持续 damping 改变；(c) 持续图像噪声；另设仅动作幅度分布改变的控制。初始创新幅度在验证集匹配；τ、事件类别和真参数仅评分。模型见同一 RGB、原协议 proprio 和已执行力；若原协议包含 qpos/qvel，所有基线同样获得，不改称 hidden state。各事件固定 raw-step，prefix replay 保持一致。

比较 frozen/history-only、原生 recent5 一步适配、论文已有 predictor-only/encoder-frozen 适配，并给普通 Huber loss+滑窗长度验证选择相同总预算。用固定 pretrained decoder 的 agent centroid 对真实位置评分，避免用可变 encoder 自身 latent loss 自证；统计 τ 后 1/5/25 步误差、冲击后错误持久化、恢复时间、峰值显存与实测更新耗时。先做 1 个样本 smoke；完整筛选不超过单卡两小时，超限则缩样本并披露。

若普通适配在动态变化有效且对冲击/噪声没有持久损害，停止该动机；若有损害，先加同信息 IMM/BOCPD、噪声模型和训练混合事件控制。只有其仍有同预算余差才讨论新环节。失败若来自像素解码或无激励，应判支持/可识别性不足，不命名为“机制更新失败”。去掉 RGB、遮蔽速度和跨数值系统验证都属于尚未执行的迁移。
