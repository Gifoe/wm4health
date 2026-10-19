# C39 原生自条件查询诊断：执行前合同

2026-10-17 V2在任何真实模型采样前冻结。V1在inputs预检因solar只有1个有效672点窗口而按原合同退出；没有加载模型、没有真实预测结果。没有新方法、训练或真实预测分数主张。复用C34已验证Sundial128M revision3212e42564493f520593e5414af4367fc4b49226及FEV固定solar/Jena两公开series；这些载体先前已用于探索，不能当新独立TEST。

## 精确对象

对每个原history H，先从原720点joint flow head抽一条未来(U,V)，再把U追加到完整H重查询V。比较 E[w(U)g(V_direct)] 与 E[w(U) E[g(V_requery)|H,U]]。不是把latent Markov CK与observation re-encode混为同一合同。若该模型只声称forecast，这个额外测试是拟研究的能力，不指控作者违背承诺。

原生公开head对任意<=720请求都是同joint720的prefix；不能把request-prefix截断当模型缺陷。主split=96+96。原brief的360+360敏感性保留；因源码PatchEmbedding会在非16倍数长度左pad，执行前增加352+352 aligned敏感性来分离patch相位。所有原history672为16倍数；full-growing H+U，不裁掉旧history。

## 固定输入、采样及控制

- 两task各原series中固定end indices为C34既有已验证首次有限672点history的end+[0,336,672,1008]，history最近672；按原timestamps核cadence。缺失整窗口跳过，不插值、不从误差挑window；至少2个有效contexts/task否则停止。加载原parquet后仅提取history片段，不取未来truth作评分或选择。
- Nouter=256，Ninner=8，两个固定独立reps seeds2026391/2026392。direct720样本定义同一prefix和direct suffix；fresh inner noise必须独立于生成prefix的noise，不保留原flow noise后半（joint flow耦合全部维度）。
- native re-query：按extended H+U重新计算mean/std，调用原模型。coordinate control：保持原H mean/std，用原model.model+原flow sampler。这是普通解释控制，不新方法；同inner noise用于两坐标arms以减小其差异估计噪声。
- 采样前原API versus controlled sampler在原672、aligned768及unaligned1032长度做same-noise资格；包括原API revin=False后手动原reference还原与控制API。所有weights before/after SHA相同，无optimizer，无grad。
- 原source、weights、C34资格manifest核SHA，不下载任何新weights/信号，不用trust_remote_code，不改原环境；runtime_profile先于NumPy/Torch，vendor隔离沿用C34。

## 预列测量及MC不确定性

各history用其原mean/std把U/V变成z单位。w=[1,tanh(prefix block mean),tanh(prefix final value)]；g=[suffix block mean,suffix final value,suffix mean of squared values]，形成9个moment probes。先逐inner取g再平均，不能平方inner均值替代E[g]。无权矩只是必要条件，weighted probes检验部分条件关联；9矩通过也不证明整个joint law一致。

对每arm/context/split/rep，保存256个direct和256个inner-mean weighted probes及所有raw predictions。差值direct−requery的均值、outer-group MC standard error、近似95%MC CI、z值全量报告；这是采样积分误差，不是跨数据/真实泛化置信区间，也不将nested draws算独立样本。全比较不做逐行显著性胜出选择；重复rep只用于看MC稳定性，无多重比较确认性结论。

数学正控：冻结coefficient AR1 U~N0,1、V=rho U+sqrt(1-rho²)epsilon，rho=.8，正确重询与joint应吻合。负控：重询独立N0,1，边际矩可匹配而weighted关联错误。固定32768outer×8inner使测量灵敏度资格可判。Gauss-Hermite64点核E[U tanh(U)]，不依赖native模型。

## 裁决与限制

若MC、normalization、patch相位、窗口合同或正确conditional API解释现象，关闭相应新机制动机。若fixed-coordinate仍有残差，仅建立内部re-query不兼容；不能声称真实质量有损，也不能单凭它提出新架构。必须再与FlowNP/NDP、AR/SSM、ordinary bridge/conditioning及已有coherence训练比较并证明实际proper-score/decision与成本影响，才可升为研究方向。C23–25 parameter-posterior配置、C34first-hit conditional-channel MLE均不复活。


## V1→V2数据修正记录

原C34已记录solar首次有限history从index864开始，原V1误将两任务都从0起算，故原四个窗口前三个包含开头缺失，按minimum2条件退出。不是模型失败。V2仅将四个history平移到该**事先存在且按finite规则选择**的anchor；solar ends1536/1872/2208/2544，Jena仍672/1008/1344/1680。其他采样/统计/控制不改，V1源码、失败日志和statistic fixture完整保留。不得将两attempt计为模型训练或声称V1出现科学失败。原C34metadata SHA256 `f804c7fe754bd0e8a8d33b37f333e1a57e32e57efad2ddaf1c0b066797ee0c35`。
