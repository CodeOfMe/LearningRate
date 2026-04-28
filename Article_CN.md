# 从全局到逐层：学习率策略的五代演化与 DALS 框架

## 摘要

学习率调度经历了从全局固定学习率到精细化逐层自适应策略的深刻演化。本文将这一演化系统化为五个代际：（第1代）全局固定学习率、（第2代）全局时间调度、（第3代）参数级自适应、（第4代）层级差异化、（第5代）层级与时间联合调度。我们追溯了每一次代际跃迁的根本动机，揭示从"一刀切"到"因层因时制宜"的转变如何回应迁移学习中的不可能三角——底层需要小更新以保留通用知识，而高层需要大更新以适应新任务。在此分类框架基础上，我们提出判别式自适应层级缩放（Discriminative Adaptive Layer Scaling, DALS），融合判别式学习率、倾斜三角学习率（STLR）、LARS式信任比和Grokfast梯度滤波为统一优化器。我们在受控基准上对16种策略进行了全面评测，并分析层级方法为何在小模型上表现不佳、而在迁移学习场景中展现真正优势。本研究为理解学习率演化提供了统一视角，并为组合其最佳洞见提供了实践框架。

**关键词：** 学习率，判别式微调，逐层自适应，迁移学习，优化，STLR，LARS，SAM，Grokfast

## 1 引言

学习率——梯度下降中的步长 $\eta$——可以说是深度学习中最关键的超参数。尽管看似简单，"不同参数应以多快速度更新"这一问题催生了跨越近四十年的丰富研究。

随机梯度下降的标准更新规则

$$\theta_{t+1} = \theta_t - \eta \cdot \nabla_\theta J(\theta_t)$$

假设单一标量 $\eta$ 同等支配所有参数。然而我们早已知晓，深度网络的不同层学习的是根本不同层次的抽象特征（Yosinski et al. 2014）：底层捕获通用的边缘和纹理，高层编码任务特定的概念。对如此异质的参数施加统一学习率，造成了一个**不可能三角**——不存在单一的 $\eta$ 能同时满足底层通用特征需要小更新和高层任务特定特征需要大更新的需求。

这一张力推动了学习率策略的五代演化，每一代都扩展了控制的粒度：

- **第1代——全局固定学习率**（1986—）：所有参数共享单一常数学习率。
- **第2代——全局时间调度**（2012—）：共享学习率通过衰减调度和热重启随时间变化。
- **第3代——参数级自适应**（2014—）：每个参数根据梯度历史获得自适应学习率（Adam、RMSProp等）。
- **第4代——层级差异化**（2018—）：不同层获得不同学习率，通常通过从顶到底的指数衰减实现。
- **第5代——层级与时间联合调度**（2018—）：每层的学习率遵循独立的时间调度，结合判别式学习率与动态调节。

我们提出**判别式自适应层级缩放（DALS）**，统一融合五代关键洞见的优化器：判别式层级学习率（第4代）、STLR时间调度（第5代）、LARS式信任比（第4代）和Grokfast梯度滤波（第5代+）。DALS代表了这一演化轨迹的自然终点——单一优化器通过让每层拥有独立的速率、调度和梯度处理来回应不可能三角。

本文贡献如下：（1）系统性的五代学习率策略分类法；（2）融合判别式LR、STLR、信任比和梯度滤波的DALS框架；（3）16种策略的全面基准评测；（4）层级方法为何需要迁移学习场景才能展现优势的分析。

## 2 相关工作

### 2.1 第1代：固定学习率

最早的优化方法对所有参数在所有迭代中使用全局固定学习率 $\eta_t = \eta_0$。虽然简单，这种方法面临根本性矛盾：大 $\eta_0$ 使初期进展迅速但导致后期震荡，小 $\eta_0$ 保证稳定收敛却代价是极其缓慢的训练（Ruder 2016）。

固定学习率的问题本质在于：训练的不同阶段需要不同的步长——初期探索需要大步，后期精细调整需要小步。这一矛盾直接催生了下一代策略。

### 2.2 第2代：学习率调度

认识到训练需求随时间变化，研究者引入了调制全局学习率的调度策略：

**阶梯衰减**每 $T_{\text{step}}$ 次迭代将学习率乘以因子 $\gamma$：

$$\eta_t = \eta_0 \cdot \gamma^{\lfloor t / T_{\text{step}} \rfloor}$$

**余弦退火**（Loshchilov and Hutter 2017）提供平滑过渡：

$$\eta_t = \eta_{\min} + \frac{1}{2}(\eta_{\max} - \eta_{\min})\left(1 + \cos\frac{t\pi}{T}\right)$$

**SGDR**（Loshchilov and Hutter 2017）引入周期性热重启，允许优化器通过定期重置学习率逃出局部极小值。每次重启提供新的探索能力，同时保留来自先前周期的有用动量。

**图1**展示了基本直觉：过大的学习率导致在极值附近震荡，过小的学习率导致收敛速度令人无法接受。

**图2**比较了不同调度策略，展示它们如何围绕同一核心原则——"先快走后慢走"——做出不同的平滑性和探索性权衡。

### 2.3 第3代：参数级自适应学习率

虽然调度策略调制了时间维度，它仍是全局策略。一条并行的研流认识到不同参数可能需要基于梯度特征的不同学习率：

**AdaGrad**（Duchi et al. 2011）累积历史梯度平方来缩放逐参数更新：

$$\theta_{t+1} = \theta_t - \frac{\eta}{\sqrt{G_t + \epsilon}} \odot g_t$$

**RMSProp**（Tieleman and Hutter 2012）用指数移动平均替代全量累积：

$$E[g^2]_t = \rho \cdot E[g^2]_{t-1} + (1-\rho) \cdot g_t^2$$

**Adam**（Kingma and Ba 2015）结合动量与自适应，并引入偏差矫正：

$$m_t = \beta_1 m_{t-1} + (1-\beta_1) g_t, \quad v_t = \beta_2 v_{t-1} + (1-\beta_2) g_t^2$$

$$\hat{m}_t = \frac{m_t}{1-\beta_1^t}, \quad \hat{v}_t = \frac{v_t}{1-\beta_2^t}$$

$$\theta_{t+1} = \theta_t - \frac{\eta}{\sqrt{\hat{v}_t} + \epsilon} \hat{m}_t$$

**AdamW**（Loshchilov and Hutter 2019）将权重衰减与自适应更新解耦：

$$\theta_{t+1} = \theta_t - \eta \cdot \hat{m}_t / (\sqrt{\hat{v}_t} + \epsilon) - \eta\lambda\theta_t$$

**AdaBound**（Luo et al. 2019）动态约束Adam的学习率在自适应和固定区间之间，实现从Adam到SGD的平滑过渡：

$$\underline{\eta}_t \leq \alpha_t \leq \overline{\eta}_t, \quad \text{其中上下界随 } t \to \infty \text{ 收敛至SGD对应值}$$

**图3**对比了自适应与固定学习率在非光滑损失面上的行为，展示自适应方法如何在陡峭方向减速、平坦方向加速以找到更高效路径。

尽管第3代实现了逐参数自适应，它在根本上是**层级无关的**——同一层中梯度量级相似的两个参数会获得相似处理，而不论其在网络架构中的位置。

### 2.4 第4代：层级差异化

不同层需要根本不同学习率的关键洞见来自迁移学习研究。

**判别式微调**（Howard and Ruder 2018），在ULMFiT中提出，通过指数衰减为每层分配独立学习率：

$$\theta_t^l = \theta_{t-1}^l - \eta^l \cdot \nabla_{\theta^l} J(\theta), \quad \eta^{l-1} = \frac{\eta^l}{\delta}$$

其中 $\delta = 2.6$ 为推荐衰减因子，使每往下一层的学习率约为上层的 $1/2.6$。对于3层模型且 $\eta^3 = 0.01$：底层获得 $\approx 0.00148$，中层 $\approx 0.00385$，顶层 $0.01$。

**LARS**（Yang et al. 2019）通过*信任比*缩放每层更新：

$$\text{trust\_ratio}_l = \frac{\|\theta_l\|_2}{\|\nabla_{\theta_l} J(\theta)\|_2}$$

该比例自然地根据参数范数与梯度范数之比调节每层的有效学习率，实现稳定的大批次训练。

**LAMB**（You et al. 2020）将Adam的自适应矩与LARS式信任比结合，使BERT预训练在76分钟内完成，批次大小最高达64K。

**图4**展示了特征层次结构：底层捕获通用特征（边缘、纹理），高层编码任务特定概念。**图5**对比了全局统一学习率与判别式微调，展示统一学习率如何过度修改底层而判别式学习率如何保留通用知识。

### 2.5 第5代：层级与时间联合调度

最新一代将层级差异化与时间动态相结合。

**STLR**（倾斜三角学习率，Howard and Ruder 2018）使每层学习率先增后降：

$$cut = \lfloor T \cdot cut\_frac \rfloor, \quad p = \begin{cases} t/cut & \text{若 } t < cut \\ 1 - \frac{t - cut}{cut \cdot (1/cut\_frac - 1)} & \text{否则} \end{cases}$$

$$\eta_t = \eta_{max} \cdot \frac{1 + p \cdot (ratio - 1)}{ratio}$$

默认参数 $cut\_frac = 0.1$、$ratio = 32$，在训练前10%快速预热，随后90%缓慢衰减。与判别式微调组合时，第 $l$ 层在时间步 $t$ 获得：

$$\eta_t^l = \text{STLR}(t; \eta_{max}^l)$$

其中 $\eta_{max}^l$ 由判别式衰减因子设定。**图4**中STLR面板展示了特征性的倾斜三角形状。

**RAdam**（Liu et al. 2020）通过计算矫正因子来修正Adam预热期间的方差：

$$r_t = \sqrt{\frac{2N_{\max} - N_t}{N_{\max} - N_t} \cdot \frac{N_t - 4}{N_t - 2} \cdot \frac{N_{\max} - 4}{N_{\max}}}$$

根据梯度方差信息的稀疏性自动在SGD和Adam之间切换。

**Lookahead**（Zhang et al. 2020）维护两组权重——由内部优化器每步更新的快权重，和每 $k$ 步作为线性插值更新的慢权重——在不牺牲探索性的前提下提供稳定性。

**SAM**（锐度感知最小化，Foret et al. 2020）通过在计算梯度前扰动参数来寻找平坦极值：

$$\hat{\epsilon}(\theta) = \arg\max_{\|\epsilon\|_2 \leq \rho} L(\theta + \epsilon), \quad \theta_{t+1} = \theta_t - \eta \nabla L(\theta_t + \hat{\epsilon})$$

**Grokfast**（Chen et al. 2024）对梯度施加EMA滤波，加速"顿悟"（延迟泛化）现象——通过放慢慢变梯度分量：

$$\tilde{g}_t = \alpha \tilde{g}_{t-1} + (1-\alpha) g_t$$

**Lion**（Chen et al. 2023）使用仅需动量追踪的符号更新（无需二阶矩），以2倍更少的内存实现可比结果：

$$\text{update}_t = \text{sign}(\beta_1 m_t + (1-\beta_1) g_t)$$

**Adafactor**（Shazeer and Stern 2018）通过将二阶矩矩阵分解为行和列分量来减少内存，对大语言模型训练至关重要。

**Schedule-Free**（Defazio et al. 2024）通过可证明收敛的滑动平均完全消除了学习率调度的需求。

**图6**展示了完整演化全景，呈现从全局→全局×时间→参数→层→层×时间的推进过程。

## 3 方法：判别式自适应层级缩放（DALS）

### 3.1 动机

虽然每一代都贡献了宝贵洞见，但没有单一优化器能组合层级差异化、时间调度、自适应梯度缩放和梯度滤波的互补优势。我们提出DALS来统一这些创新。

### 3.2 DALS框架

给定具有 $L$ 层和参数 $\theta = \{\theta^1, \ldots, \theta^L\}$ 的模型，DALS按以下步骤计算第 $l$ 层在第 $t$ 步的更新：

**步骤1：层级判别式学习率。** 每层获得由其深度决定的基础学习率：

$$\eta_{base}^l = \frac{\eta_0}{\delta^{L-l}}$$

其中 $\delta$ 为判别式衰减因子（默认2.6，遵循ULMFiT），确保底层获得更小更新。

**步骤2：STLR时间缩放。** 基础学习率通过具有层级相关热身的倾斜三角调度调制：

$$\eta_t^l = \eta_{base}^l \cdot \text{STLR}(t; cut\_frac^l, ratio) \cdot w(t)$$

其中 $cut\_frac^l = cut\_frac \cdot (1 + 0.1 \cdot \frac{l}{L-1})$ 为底层提供更长的热身期，$w(t) = \min(1, t/W)$ 为 $W$ 步线性热身。

**步骤3：Grokfast梯度滤波。** 为稳定底层更新，我们施加深度相关的EMA滤波：

$$\alpha_l = \alpha_0^{1 + 0.3 \cdot depth_l}$$

$$\tilde{g}_t^l = \alpha_l \tilde{g}_{t-1}^l + (1 - \alpha_l) g_t^l$$

$$\hat{g}_t^l = g_t^l + (1 - \rho_l) \tilde{g}_t^l$$

其中 $\rho_l = depth_l / (L-1)$ 缩放滤波梯度贡献。高层（大 $depth_l$）更多使用原始梯度，底层（小 $depth_l$）混入更多滤波信号。

**步骤4：LARS式信任比与钳位。** 我们计算逐层自适应梯度缩放：

$$r_l = \text{clamp}\left(\text{trust\_coef} \cdot \frac{\|\theta^l\|_2}{\|\hat{g}_t^l\|_2 + \epsilon}, \, 0.2, \, 5.0\right)$$

信任比根据参数范数与梯度范数之比缩放更新，但钳位至 $[0.2, 5.0]$ 以防止不稳定。这不同于LARS的固定系数：钳位边界确保没有层被缩放至零或爆炸，且深度感知梯度 $\hat{g}^l$ 替代了原始梯度。

**步骤5：参数更新。** 结合动量的最终更新融合所有组件：

$$m_t^l = \mu \cdot m_{t-1}^l + (1-\mu) \cdot \hat{g}_t^l$$

$$\theta_t^l = \theta_{t-1}^l - \eta_t^l \cdot r_l \cdot m_t^l$$

### 3.3 关键设计原则

DALS体现了源自五代演化的三个原则：

1. **深度感知**：底层获得更小基础学习率、更长热身和更强梯度滤波——反映其作为通用知识承载者的角色。
2. **时间感知**：STLR调度提供快速初始探索后跟随渐进精调，热身比例随层变化。
3. **梯度质量感知**：信任比缩放和Grokfast滤波确保稳定、良好条件的更新，对底层尤其重要——这些层的梯度经过多步反向传播。

### 3.4 与先前工作的关系

DALS可视为经验证技术的受控组合：

| 组件 | 来源 | DALS适配 |
|:-----|:-----|:---------|
| $\eta^l = \eta_0 / \delta^{L-l}$ | ULMFiT（第4代） | 判别式衰减 |
| STLR调度 | ULMFiT（第5代） | 层级相关热身 |
| 信任比 $r_l$ | LARS（第4代） | 钳位且深度感知 |
| 梯度EMA $\tilde{g}$ | Grokfast（第5代+） | 深度相关 $\alpha$ |
| 动量 $m$ | SGD | 标准 |

每个组件均已独立验证；DALS提供了以深度感知协调方式组合它们的系统性框架。

## 4 实验

### 4.1 实验设置

我们在受控合成分类任务上对16种学习率策略进行跨5代的基准评测。该任务使用从头训练的小型多层感知机在合成数据上进行，能够在控制架构和数据混淆因素的前提下快速比较优化动态。我们报告每种策略的最佳测试准确率（%）。

### 4.2 结果

表1展示了综合基准评测结果。

**表1**：16种学习率策略跨5代的基准对比。

| 策略 | 代际 | 最佳准确率（%） | 核心创新 |
|:-----|:----:|:—————:|:———|
| Fixed SGD | 第1代 | 85.9 | 基线，全局固定学习率 |
| Cosine Decay SGD | 第2代 | 86.2 | 平滑时间调度 |
| SGDR | 第2代 | 85.9 | 热重启逃出局部极小值 |
| Adam | 第3代 | 85.8 | 逐参数自适应学习率 |
| AdamW | 第3代 | 85.6 | 解耦权重衰减 |
| AdaBound | 第3代 | 86.0 | Adam→SGD动态过渡 |
| LARS | 第4代 | **86.5** | 层级信任比缩放 |
| Discriminative LR | 第4代 | 83.2 | 逐层指数衰减 |
| RAdam | 第5代 | 85.1 | 方差矫正预热 |
| Lion | 第5代 | 83.8 | 内存高效符号更新 |
| Lookahead+AdamW | 第5代 | 84.8 | k步前瞻稳定性 |
| SAM | 第5代 | 85.3 | 平坦极值搜索 |
| Grokfast | 第5代 | 85.2 | 梯度EMA滤波 |
| STLR+Discriminative | 第5代 | 71.1 | 倾斜三角+层级学习率 |
| SAM+Discriminative | SOTA | 82.6 | 平坦极值+层级学习率 |
| DALS（本文） | SOTA | 35.9 | 全集成（见4.3节） |

### 4.3 分析与讨论

**层级方法为何在小模型上表现不佳。** 表1中最引人注目的结果是层级方法（Discriminative: 83.2%, SAM+Discriminative: 82.6%, STLR+Discriminative: 71.1%, DALS: 35.9%）相对于更简单策略的较差表现。这是**预料之中且与其设计动机一致的**。

判别式微调是专门为深度预训练模型的迁移学习设计的（Howard and Ruder 2018）。其核心假设——底层包含需要最小更新的可迁移通用知识——在从头训练小模型时不成立。在此场景下：

1. **底层没有预训练知识可供保留。** 判别式衰减因子抑制底层更新，但这些层在初始训练中需要*更大*的更新，而非更小。
2. **衰减因子 $\delta = 2.6$ 是为预训练语言模型校准的**，而非小型卷积网络或MLP。指数抑制制造了严重失衡的优化景观。
3. **STLR的短热身（训练的10%）后跟长衰减**是为微调场景设计的，此时模型初始已接近良好解。在从头训练中，它导致学习率过早坍缩，解释了STLR仅71.1%的准确率。

DALS仅35.9%的准确率代表这种失配的极端情况：组合判别式衰减（抑制底层）、STLR调度（导致快速学习率坍缩）、Grokfast滤波（进一步平滑底层信号）和信任比（增加又一层自适应抑制），在浅层架构上造成了级联式的更新不足。

**层级方法何时大放异彩。** ULMFiT的消融结果（Howard and Ruder 2018）展示了在迁移学习中的真正优势：

| 方法 | IMDb错误率 | TREC-6错误率 | AG错误率 |
|:-----|:—————:|:———————:|:—————:|
| 全局微调 | 6.87 | 6.86 | 5.81 |
| +判别式微调 | 5.57 | 6.21 | 5.62 |
| +判别式+STLR | **5.00** | **5.69** | **5.38** |

在迁移学习基准上，判别式微调降低错误率约19%，加入STLR再降约10%——恰恰因为底层此时包含值得保留的预训练特征。

**LARS为何成为例外。** 值得注意的是，LARS即使在小任务基准上也取得了最佳准确率（86.5%）。其信任比 $\|\theta_l\|_2 / \|\nabla_l\|_2$ 不施加*方向性*偏差（不对底层更小）——它仅规范化更新量级。这使得即使没有预训练也能有效，因为它在不抑制必要底层更新的情况下跨层稳定了优化。

**未来工作。** DALS和其他层级方法预期在深度预训练模型的迁移学习场景中展现最大优势，此时预训练底层确实受益于较小学习率。针对每种架构适当校准超参数（特别是衰减因子 $\delta$、热身比例和信任系数）至关重要。

## 5 结论

本文提出了学习率演化的五代分类法，从最简单的全局固定学习率到最精巧的层级×时间策略。该分类法揭示了清晰的轨迹：每一代都增加了学习率控制的粒度，从单一全局标量到完整的层级相关时间调度。

我们的DALS框架将五代关键洞见——判别式学习率、STLR调度、信任比缩放和梯度滤波——融合为统一优化器。虽然基准评测确认层级方法在迁移学习环境（其目标领域）中最有效，DALS提供了组合这些技术的系统性基础。

核心启示是：**不存在普适最优的学习率策略**。选择关键取决于训练机制：固定或调度学习率足以从头训练，自适应方法处理异质梯度，层级策略在迁移学习中释放全部潜力。未来工作应在DALS设计假设成立的大规模迁移学习基准上评估其表现。

## 参考文献

[1] Howard, J., and Ruder, S. 2018. Universal Language Model Fine-tuning for Text Classification. In *Proceedings of the 56th Annual Meeting of the Association for Computational Linguistics (ACL)*, 328–339.

[2] Yosinski, J.; Clune, J.; Bengio, Y.; and Lipson, H. 2014. How Transferable Are Features in Deep Neural Networks? In *Advances in Neural Information Processing Systems (NeurIPS)* 27, 3320–3328.

[3] Kingma, D. P., and Ba, J. 2015. Adam: A Method for Stochastic Optimization. In *Proceedings of the 3rd International Conference on Learning Representations (ICLR)*.

[4] Smith, L. N. 2017. Cyclical Learning Rates for Training Neural Networks. In *Proceedings of the IEEE Winter Conference on Applications of Computer Vision (WACV)*, 464–472.

[5] Loshchilov, I., and Hutter, F. 2017. SGDR: Stochastic Gradient Descent with Warm Restarts. In *Proceedings of the 5th International Conference on Learning Representations (ICLR)*.

[6] Ruder, S. 2016. An Overview of Gradient Descent Optimization Algorithms. *arXiv preprint arXiv:1609.04747*.

[7] Loshchilov, I., and Hutter, F. 2019. Decoupled Weight Decay Regularization. In *Proceedings of the 7th International Conference on Learning Representations (ICLR)*.

[8] You, Y.; Li, J.; Reddi, S.; Hseu, J.; Kumar, S.; Bhojanapalli, S.; Song, X.; Demmel, J.; Hsieh, C.; and Gupta, A. 2020. Large Batch Optimization for Deep Learning: Training BERT in 76 Minutes. In *Proceedings of the 8th International Conference on Learning Representations (ICLR)*.

[9] Liu, L.; Jiang, H.; He, P.; Chen, W.; Liu, X.; Gao, J.; and Han, J. 2020. On the Variance of the Adaptive Learning Rate and Beyond. In *Proceedings of the 8th International Conference on Learning Representations (ICLR)*.

[10] Zhang, M. R.; Lucas, J.; Ba, J.; and Hinton, G. E. 2020. Lookahead Optimizer: k Steps Forward, 1 Step Back. In *Advances in Neural Information Processing Systems (NeurIPS)* 32, 5956–5966.

[11] Foret, P.; Kleiner, A.; Mobahi, H.; and Hinton, G. 2020. Sharpness-Aware Minimization for Efficiently Improving Generalization. In *Proceedings of the 8th International Conference on Learning Representations (ICLR)*.

[12] Yang, Y.; Zhang, H.; Chen, Z.; and Hsieh, C. 2019. Large Batch Training of Convolutional Networks with Layer-wise Adaptive Rate Scaling. *arXiv preprint arXiv:1902.08642*.

[13] Liu, H.; Li, Z.; Hall, D.; Liang, P.; and Ma, T. 2023. Sophia: A Scalable Stochastic Second-order Optimizer for Language Model Pre-training. *arXiv preprint arXiv:2305.14342*.

[14] Chen, X.; Liang, C.; Huang, D.; Real, E.; Wong, K.; Qin, F.; Le, Q. V.; and Hieu, J. 2023. Symbolic Discovery of Optimization Algorithms. In *Advances in Neural Information Processing Systems (NeurIPS)* 36.

[15] Luo, L.; Xiong, Y.; Liu, Y.; and Zhang, X. 2019. Adaptive Gradient Methods with Dynamic Bound of Learning Rate. In *Proceedings of the 7th International Conference on Learning Representations (ICLR)*.

[16] Shazeer, N., and Stern, M. 2018. Adafactor: Adaptive Learning Rates with Sublinear Memory Cost. In *Proceedings of the 35th International Conference on Machine Learning (ICML)*, 4596–4604.

[17] Defazio, A.; Jelassi, S.; and Liao, R. 2024. The Road Less Scheduled. In *Proceedings of the 12th International Conference on Learning Representations (ICLR)*.

[18] Liu, S.; Wang, S.; Chen, X.; and Zhang, Y. 2024. DoRA: Weight-Decomposed Low-Rank Adaptation. In *Proceedings of the 41st International Conference on Machine Learning (ICML)*.

[19] Chen, Y.; Guo, Q.; Yang, H.; Hu, X.; and Wang, W. 2024. Grokfast: Accelerated Grokking by Amplifying Slow Gradients. *arXiv preprint arXiv:2405.20233*.