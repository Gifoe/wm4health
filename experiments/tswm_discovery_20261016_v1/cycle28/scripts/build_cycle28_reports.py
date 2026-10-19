"""Generate Chinese living reports from verified native and real-system results."""
import runtime_profile
import csv,hashlib,json,pathlib
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
ROOT=pathlib.Path(__file__).resolve().parents[1];NATIVE=ROOT/'outputs/pcfm_native';CED=ROOT/'outputs/ced_screen';OUT=ROOT/'outputs/analysis'
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def load(p):return json.loads(p.read_text(encoding='utf-8'))
def main():
 native=load(NATIVE/'results.json');nv=load(NATIVE/'verification.json');replay=load(NATIVE/'neural_replay.json');ced=load(CED/'results.json');cv=load(CED/'verification.json')
 assert nv['status']==replay['status']==cv['status']=='passed'
 assert nv['results_sha256']==replay['results_sha256']==sha(NATIVE/'results.json') and cv['results_sha256']==sha(CED/'results.json')
 with (CED/'rows.csv').open(newline='',encoding='utf-8') as f:rows=list(csv.DictReader(f))
 by={(r['record'],r['method']):r for r in rows};OUT.mkdir(exist_ok=True)
 methods=['pcfm_CD_glue1','pcfm_CD_glue0','pcfm_known_direction_glue1','pcfm_known_direction_glue0','pcfm_CP_glue1','pcfm_CP_glue0','icl_native_padded30','icl_direct10','arx_train_selected','persistence','training_mean']
 table=['| 预列方法 | 低幅记录 RMSE | 高幅记录 RMSE | 低幅 10–90% 覆盖 | 高幅 10–90% 覆盖 |','|---|---:|---:|---:|---:|']
 for method in methods:
  rs=[by[(label,method)] for label in ['low_amplitude','high_amplitude']]
  table.append('| '+method+' | '+' | '.join([f"{float(rs[0]['rmse']):.6f}",f"{float(rs[1]['rmse']):.6f}"]+[f"{100*float(r['coverage10_90']):.2f}%" if r.get('coverage10_90') else '无概率输出' for r in rs])+' |')
 table='\n'.join(table)
 plt.rcParams.update({'font.family':'DejaVu Sans','font.size':10,'axes.spines.top':False,'axes.spines.right':False,'svg.fonttype':'none','pdf.fonttype':42})
 fig,axes=plt.subplots(1,2,figsize=(11,3.9))
 colors={'icl_native_padded30':'#256B89','pcfm_CD_glue1':'#B64939','pcfm_CD_glue0':'#29906F','arx_train_selected':'#9B8A51'}
 labels={'icl_native_padded30':'Native ICL','pcfm_CD_glue1':'PCFM default glue','pcfm_CD_glue0':'PCFM glue off','arx_train_selected':'Train-selected ARX'}
 for ax,label in zip(axes,['low_amplitude','high_amplitude']):
  data=np.load(CED/f'input_{label}.npz');ax.plot(np.arange(-10,90),data['y'][400:],'k',lw=1.7,label='Measured output')
  for method in colors:
   d=np.load(CED/f'{label}_{method}.npz');ax.plot(np.arange(90),d['mean'],lw=1.3,color=colors[method],label=labels[method])
  ax.axvline(-.5,color='#888888',ls=':',lw=1);ax.set_title(label.replace('_',' '));ax.set_xlabel('Test step after the 10 allowed initial outputs');ax.grid(alpha=.16)
 axes[0].set_ylabel('Output in original dataset units');axes[0].legend(frameon=False,fontsize=8)
 fig.suptitle('Real Coupled Electric Drives: identical permitted history and future input',fontsize=12)
 fig.text(.5,.015,'Two measured records, not two populations of plants. All 90 remaining test outputs forecast without output refresh.',ha='center',fontsize=8)
 fig.tight_layout(rect=(0,.06,1,.93))
 for ext in ['png','svg','pdf']:fig.savefig(OUT/f'ced_forecasts.{ext}',dpi=180)
 plt.close(fig)
 calls=load(NATIVE/'calls.json');screen=native['screen']
 fig,axes=plt.subplots(1,2,figsize=(10.5,3.8))
 def physical(u,y0):
  v=y0;out=[]
  for t in range(len(u)):
   control=u[t-48] if t>=48 else -1;v=.95*v+.05*(20+30*(control+1));out.append(v)
  return np.array(out)
 for glue,color in [(True,'#B64939'),(False,'#256B89')]:
  selected={e['plan']:np.load(NATIVE/calls[e['call_index']]['path']) for e in screen if e['glue']==glue}
  effect=selected['ON']['output'][1,4096:,4]-selected['OFF']['output'][1,4096:,4]
  truth=(physical(selected['ON']['input'][0],selected['ON']['input'][1,0])-physical(selected['OFF']['input'][0],selected['OFF']['input'][1,0]))[4096:]
  axes[0].plot(range(128),effect,color=color,label='Glue on' if glue else 'Glue off');axes[1].plot(range(48),effect[:48],color=color)
 axes[0].plot(range(128),truth,'k--',lw=1.5,label='Known native physics');axes[1].axhline(0,color='k',ls='--',lw=1.3)
 for ax in axes:ax.axvline(48,color='#888888',lw=1,ls=':');ax.grid(alpha=.16);ax.set_xlabel('Forecast step')
 axes[0].set_ylabel('ON minus OFF temperature');axes[1].set_ylabel('Response before transport delay');axes[0].set_title('Accurate native effect at full scale');axes[1].set_title('Small anticipatory response remains');axes[0].legend(frameon=False,fontsize=8)
 fig.suptitle('One authored example is a contract screen, not evidence of a new framework gap',fontsize=11)
 fig.tight_layout(rect=(0,0,1,.93))
 for ext in ['png','svg','pdf']:fig.savefig(OUT/f'native_effect.{ext}',dpi=180)
 plt.close(fig)
 evidence=dict(status='verified',classification='PROGRESS',goal_status='ACTIVE_UNACHIEVED',accepted_final_directions=0,
  decision='NO_NEW_FRAMEWORK_JUSTIFIED_BY_NATIVE_OR_CED_SCREEN',native_results_sha256=sha(NATIVE/'results.json'),ced_results_sha256=sha(CED/'results.json'),
  native_verification=nv,native_neural_replay=replay,ced_verification=cv,native_calls=28,ced_score_rows=22,
  headline={label:{method:float(by[(label,method)]['rmse']) for method in ['icl_native_padded30','pcfm_CD_glue1','pcfm_CD_glue0','arx_train_selected']} for label in ['low_amplitude','high_amplitude']})
 (OUT/'results.json').write_text(json.dumps(evidence,indent=2),encoding='utf-8')
 reports={
 'literature_review.md':'''# 文献与创新边界：cycle28

研究对象仍是可复用的数值时间序列 world model。文献窗口冻结为2026-04-16至2026-10-16；实验在北京时间10月17日完成。三个独立视角分别保存在 research/numeric_frontier.md、capability_critic.md、adjacent_efficiency.md，完整 primary 链接和版本在笔记中。旧组件先例并不自动否定新组合；有限数据/计算的收益可以构成贡献，但必须先测出具体残差。

| 能力组 | 直接先例 | 本轮应如何解释 |
|---|---|---|
| 任意变量、图和未来条件预测 | [PCFM 官方源码](https://github.com/software-competence-center-hagenberg/PCFM)、[出版社记录](https://doi.org/10.1007/978-3-032-34896-8_15) | 完整发布新底座；9个边际分位数、masked reconstruction，不等于 joint path sampler |
| 系统级上下文识别与概率预测 | [概率 ICL-SysID](https://arxiv.org/abs/2610.13411)、[已缓存源码](https://github.com/PuraFilippo/sysid_benchmarks) | cycle16 已 materialize，本轮首次在本项目完成真实 CED 质量比较；不是新发现该论文 |
| TSFM 输入响应与适配保留 | [Attenuated in-context identification](https://arxiv.org/abs/2610.08118) | 已有合成适配、频率与遗忘诊断；普通 LoRA/分支等修复尚需排除，不能泛提 adapter 新意 |
| 可更新的低维模型动态 | [AURA](https://arxiv.org/abs/2609.31559)、[KAFFEE](https://arxiv.org/abs/2605.31547) | latent Bayes tracking 和过滤目标已有；任务与假设不同，不能声称已解决所有数值模拟 |
| 物理时间概率生成 | [VSF](https://arxiv.org/abs/2610.00976) | 原文 stochastic 数据明确是每轨迹参数不确定性；不能误读为过程噪声。当前官方树只有README/LICENSE/gitignore，未执行模型 |

与只用论文关键词相比，源码明确了 PCFM 默认 glue 是加性平移、双向时间注意力、8192窗口、CP仅历史估图；这些是契约边界，不能自身当失败证据。本轮已做实际发布推理与实测数据比较。其默认设置的问题有公开普通开关可修复；较小的原生 ICL 也表现较强，尚无支持新框架的残差。

KUMA/PRO-DYN/RP-GSSM、DynaBase/FIM 与旧 SysID 文献只作历史刷新，未重复下载或训练。缺代码/缺checkpoint只影响执行成本，不是科学Gap。没有把预测误差直接当物理忠实性，也没有把独立分位数采样制造的坏路径当作者失败。
''',
 'baseline_registry.md':'''# 基线、数据和复现状态

| 项目 | 固定来源 | 本轮执行 |
|---|---|---|
| PCFM 52,852,000参数 | Git226ddc37786ab0060ab1dbe7b0b0ebcb58b24630；HFce49c06f05efeb50bd307940392af0d8c89a3796 | 211,418,848-byte safetensors，SHA746ff3eb69f9fcaa0e934f13b5afda17be7591a987eb64c671fa05b1d87b6755；严格加载；全部5官方示例；另12个预列screen条件 |
| 原生概率 ICL 5,548,418参数 | PuraFilippo/sysid_benchmarks4c20d94ccab51abf4ebf68aa8810f4596381dd41；cycle16已缓存 | 90,193,824-byte checkpoint；仅允许argparse.Namespace的受限weights_only加载；原生noncached模型；两种预列初始化表达 |
| 训练内选参 ARX | 6个阶数×4个ridge值 | 各记录24个验证拟合，先300训练、100步free-run验证，再400refit；没有测试选参 |
| 持久性/训练均值 | 相同信息 | 两条记录全部保留 |
| CED 官方实测 | [Uppsala原档](https://uu.diva-portal.org/smash/get/diva2:1165531/FULLTEXT01.zip)，MAT SHA102c8d8012d6e850d50c13227b8c2280fe69f8bc4a845177f8bc3d761d152c89 | 两条500点记录，400校准+10初始化+90全测试；速度传感器不保留符号。两条记录不等于独立装置总体 |

PCFM 的6个图/glue设置不是6种独立模型；ICL两个初始化约定不是独立训练seed。所有模型可读相同410历史u/y和90未来u，未来y均置零占位。CED方向先验u→y也与普通SISO识别接口一致。PCFM图CP只从允许历史估计；显式lag0不使用隐藏真实延迟。

代码、计划和所有raw数组保存在服务器独立cycle28；safetensors和原始NPZ不被当作本地紧凑镜像。PCFM权重直接从服务器HF镜像正常TLS取得、计算完整SHA；没有改动旧环境或重下已缓存ICL权重。nonlinear_benchmarks1.0.1的官方划分源码经PyPI wheel hash固定，信号从原始MAT读取，不需要全局安装其依赖。
''',
 'experimental_results.md':f'''# 实验结果：新发布框架与真实部分观测系统

总体目标ACTIVE、UNACHIEVED；cycle28=PROGRESS。已执行新框架，而非只给文字Idea；没有新神经训练或已接受方法方向。

## 官方发布示例

不修改 PCFM example.py 的 seed7、常数或5组main调用顺序。记录15个原生API调用，并额外执行固定单window的12个计划/glue条件和1个masked-future占位检查，共28次调用。全部原始图、输入输出、stdout/stderr保留。原生valve三计划MAE：ON0.356464°C、OFF0.231219°C、PULSE320.661514°C；ON-OFF效应MAE0.262196°C。源码原生图/历史是特权已知物理的合成例子，不是实测因果验证。

独立first-order物理递推/lfilter、seeded原始窗口及打印分数重建通过。28文件hash、1,440,768个输出finite检查、分位数顺序检查通过；28个神经调用新进程原生重放最大差6.104e-5，满足预列float32相对/绝对容差。它是same-source numerical replay，不是独立重写网络。

原生what-if为masked reconstruction。延迟前ON响应相对OFF的RMS约0.270°C（glue开）、0.220°C（关）；其他延迟方案完整保留在contract_rows.csv，最大的提前差约1.392°C。该单window显示近似提前响应，不能证明跨系统缺陷或新架构需求。masked future target换成123456后输出差严格0；没有目标占位泄漏。9边际分位数不被误当成joint路径。

原生运行{native['elapsed_s']:.3f}s，GPUpeak {native['peak_cuda_allocated_bytes']/2**20:.2f}MiB；不是完整原论文表格复现。

## 实测CED的完整官方测试段

同一记录连续500点，前400校准；官方测试前10点只作初始化，随后90点一次free forecast；没有回灌真实输出。PCFM读取410连续历史而非拼接无关episode。ICL读取400context和10query初态，native padded30仅增加20个标准化零占位，不增加观测；direct10单列。标准化只用前400点。模型全部固定，图/glue/初始化模式预先列出，不能测试后选择最佳模式部署。

{table}

22行包括全部预列设置；原始单位RMSE，非百分比。ICL native padded30表现较强；PCFM公开的glue=False已大幅减轻默认退化。由此不能将默认低分当作新框架必要性。ICL更专门的系统识别预训练与PCFM一般预训练不同；参数量/前向次数不能作为公平的离线计算量比较。本轮没有分离其训练数据、任务先验和架构效应。

ARX48个训练内验证配置与两次最终refit均保留；验证是100步free-run。全部142分数独立重算、两条原始split/2000模型input值检查通过；增广最小二乘与AR状态协方差递推另验，参数/预测/分位数差小于9e-15。12次PCFM+4次ICL完整神经重放差0。最大分数重算差3.974e-8，来自float32 reduction。native数据与CED结果哈希分别{sha(NATIVE/'results.json')} / {sha(CED/'results.json')}。

CED运行{ced['runtime_s']:.3f}s，含加载{ced['load_s']:.3f}s；GPUpeak{ced['peak_cuda_allocated_bytes']/2**20:.2f}MiB。记录warm-forward时间，但PCFM封装、ICL裸模型与ARX验证/拟合的计时边界不同，不作速度排行。没有神经训练。两条短实测记录不构成跨装置统计总体，没有对90时间点作IID显著性声明；边际覆盖也不能证实完整概率动力学。CED真实记录没有未执行plan真值，factual误差不是干预正确性证明。
''',
 'negative_evidence.md':'''# 负面证据和被排除的推断

1. PCFM默认glue把首预测强行锚在最后观测、整体平移未来。CED低幅RMSE0.3364→0.1885，高幅0.4348→0.1549，只需公开glue=False。它是普通后处理修复，不是新机制。
2. 已有5.55M原生ICL在相同允许信息下达到RMSE0.06727/0.07603，优于本轮PCFM和线性ARX。不能据53M模型低分声称现有可复用识别模型都失效；也不能归因成纯参数量优势。
3. ICL direct10比native padded30差，说明固定训练接口/初始化表达影响结果；不能把本轮较差接口作为公平baseline来制造方法收益。
4. PCFM作者单window有小幅提前响应，但48步前物理效应严格0只是一项native合成契约诊断。没有排除普通因果mask/状态估计修复，也没有多系统确认，不提升为新方向。
5. VSF原文的随机benchmark是轨迹参数不确定性；不是过程噪声。其代码尚未发布不构成Gap。ICL概率论文已在cycle16审计，本轮不能算新发现。
6. 2610.08118已覆盖TSFM输入衰减、合成适配和保留代价；AURA/KAFFEE占据相关有限预算更新原语。但这些先例不是逻辑上否决全部组合；当前缺的是一个强修复后仍存活、可独立证实的残差。

不对PCFM继续调测试图/lag/glue或重新训练以追这个CED成绩。所有预列条件、失败解释和原生正结果保留；没有推荐最终Idea。
''',
 'hypothesis_space.md':'''# 竞争假设与当前判断

| 假设 | 区分实验 | 判断 |
|---|---|---|
| 新公开数值foundation framework无法实际执行 | 固定权重/源码，跑所有官方示例 | 反证：PCFM已实际运行，质量与原生示例核验相符 |
| PCFM的CED退化需要新状态架构 | 固定权重，公开glue开关 | 不支持；普通开关解释大量退化 |
| 成熟可复用识别器仅在toy上有效 | 同信息、完整官方CED测试 | 反证：原生概率ICL在真实部分观测传感器上表现较强 |
| 直接10query初态一定优于原生padding | 两种预列约定不选最好 | 反证：本例原生padding更强；接口分布影响不可忽略 |
| 已给正确graph lag就保证严格无提前响应 | 原生固定window计划配对 | 不满足严格保证，但差异较小且常规修复未排除，非已成立方法Gap |
| 这已建立通用world model新问题/方法 | 独立机制、强修复、有限预算优势与先例核查 | 尚未；最终接受方向0 |

下一轮不重复调这个CED，也不将单条数据默认值弱点包装成框架。有限数据/计算的实证优势是允许的创新标准；无需证明经典无限计算永远无法替代，但必须把具体新机制与最近能力先例区分。
''',
 'research_ideas.md':'''# 候选方向与筛选

当前最终接受方向0。本轮新增了真实可运行PCFM及跨框架实测证据，没有强行命名新模型。

未保留：通用graph adapter、强制首点连续、增加latent Bayes更新、为边际分位数附独立采样、加future-input filterbank。它们分别已有能力先例或普通修复，且本轮没有独立剩余优势证据。

本轮最有用的正证据是：对部分观测的真实动力系统，已经缓存的原生概率ICL在400点校准、10点初态下能作较好的90步模拟。后续研究必须保留这种强先验模型，不能只同线性ARX或默认glue的PCFM比较。

本轮不推荐新训练。要继续选题，需要从“模型数值能否跑”切到一个明确的系统级能力：在什么允许信息与有限预算下，哪种已有框架及其成熟修复仍有可复现残差。这个能力和新机制尚未冻结；不能将待定义的研究计划充当已经证实的Idea。
''',
 'final_research_direction.md':'''# 当前最终判断：尚无可推荐的研究方向

**Research Question：** 通用时间序列world model在何种可复用状态或演化能力上，仍存在超出实现、接口、普通识别修复的有限数据/计算残差？这是总问题，尚未收敛为论文选题。

**Evidence：** 本轮跑通新发布PCFM全部五组示例并记录28次推断；在两条真实CED记录上完成22组同信息预测和全量复核。PCFM关闭默认glue后大幅改善；已有原生概率ICL进一步达到0.0673/0.0760 RMSE。该证据排除“当前失败需要新通用架构”的简单论据；两条记录不是跨系统充分确认。

**Existing Methods：** PCFM已提供any-variate图/未来条件接口；概率ICL已提供上下文系统识别；AURA/KAFFEE/近期强制系统适配已有更新/先验桥接。其假设不同，没有任何单篇被宣称解决全部世界建模，但当前尚未找出可支持本项目新机制的独立残差。

**Proposed Idea：** 0个接受方向；不将glue=False、因果mask、padding修正或普通识别重命名为创新。

**Falsification：** 候选失败若经公开开关、原生正确接口或成熟识别模型消除，停止从它推导架构必要性。新方法若要靠有限预算优势成立，须在相同允许信息及第二个既有机制上对强修复形成稳定优势，不要求无限计算的不可替代性。

**Feasibility：** 两个原生模型已在单5090运行；所有源码、权重hash、数据、预测和独立复算保存在D:/tswm_discovery_20261016_v1/cycle28。无需再次下载权重或重复官方示例。

**Next Experiment：** 不继续针对这两条CED调模型。下一项方法训练前，先明确一个在本轮强ICL/PCFM和普通修复下仍失败的系统级能力，并冻结已有公开的独立任务与预算。这一步目前未完成，不能称已确定方向或已开始新训练。总体目标ACTIVE、UNACHIEVED，继续自主发现。
'''}
 for name,text in reports.items():(ROOT/name).write_text(text,encoding='utf-8')
 (ROOT/'CYCLE_STATUS.json').write_text(json.dumps({k:v for k,v in evidence.items() if k not in ['native_verification','native_neural_replay','ced_verification']},indent=2),encoding='utf-8')
 print(json.dumps(dict(reports=len(reports),native_calls=28,ced_rows=22,headline=evidence['headline'],accepted_final_directions=0)),flush=True)
if __name__=='__main__':main()
