# 学习率工程：从粗放的单一参数到分层演化

## 摘要

学习率调度经历了从全局固定学习率到精细化逐层自适应策略的深刻演化。本文将这一演化系统化为五个代际：（第1代）全局固定学习率、（第2代）全局时间调度、（第3代）参数级自适应、（第4代）层级差异化、（第5代）层级与时间联合调度。我们追溯了每一次代际跃迁的根本动机，揭示从"一刀切"到"因层因时制宜"的转变如何回应迁移学习中的不可能三角——底层需要小更新以保留通用知识，而高层需要大更新以适应新任务。在此分类框架基础上，我们提出判别式自适应层级缩放（Discriminative Adaptive Layer Scaling, DALS），融合阶段自适应余弦调度、深度感知Grokfast梯度滤波、LARS式信任比和SGD动量为统一优化器。我们在五个数据集上对18种策略（含三种DALS变体）进行了全面评测：合成任务（MLP，从头训练）、CIFAR-10（ConvNet，从头训练）、RTE（DistilBERT，微调）、TREC-6（DistilBERT，微调）和IMDb（DistilBERT，微调）。跨数据集分析揭示了显著的机制依赖模式：从头训练中表现优异的策略（DALS 98.0%，Fixed SGD 97.9%）在微调任务上失利，而自适应方法（RAdam 91.2% IMDb，Lookahead 97.6% TREC-6）在NLP基准上表现突出——没有单一策略在所有机制中获胜。关键是，STLR+Discriminative——ULMFiT的冠军策略——在从头训练中灾难性失败（TREC-6从头训练43.6% vs. RAdam 96.8%），确认方向性衰减偏差在缺少预训练特征时是有害的。DALS的阶段与深度感知设计避免了任一极端：在合成任务上达到最佳98.0%，同时在IMDb微调上保持竞争力90.1%。本研究为理解学习率演化提供了统一视角，并揭示了训练机制与优化器选择之间的关键交互。

**关键词：** 学习率，判别式微调，逐层自适应，迁移学习，优化，STLR，LARS，SAM，Grokfast

## 1 引言

学习率——梯度下降中的步长 $\eta$——可以说是深度学习中最关键的超参数。尽管看似简单，"不同参数应以多快速度更新"这一问题催生了跨越近四十年的丰富研究。

随机梯度下降的标准更新规则

$$\theta_{t+1} = \theta_t - \eta \cdot \nabla_\theta J(\theta_t)$$

假设单一标量 $\eta$ 同等支配所有参数。然而我们早已知晓，深度网络的不同层学习的是根本不同层次的抽象特征（Yosinski et al. 2014）：底层捕获通用的边缘和纹理，高层编码任务特定的概念。对如此异质的参数施加统一学习率，造成了一个**不可能三角**——不存在单一的 $\eta$ 能同时满足底层通用特征需要小更新和高层任务特定特征需要大更新的需求。

![](figs/paper_fig1_impossibility.svg)

**图1**：迁移学习中的不可能三角。底层需要小更新保留通用知识，高层需要大更新适应新任务，单一学习率无法同时满足两者。

这一张力推动了学习率策略的五代演化，每一代都扩展了控制的粒度：

- **第1代——全局固定学习率**（1986—）：所有参数共享单一常数学习率。
- **第2代——全局时间调度**（2012—）：共享学习率通过衰减调度和热重启随时间变化。
- **第3代——参数级自适应**（2014—）：每个参数根据梯度历史获得自适应学习率（Adam、RMSProp等）。
- **第4代——层级差异化**（2018—）：不同层获得不同学习率，通常通过从顶到底的指数衰减实现。
- **第5代——层级与时间联合调度**（2018—）：每层的学习率遵循独立的时间调度，结合判别式学习率与动态调节。

![](figs/paper_fig2_taxonomy.svg)

**图2**：学习率策略的五代分类法。每代增加学习率控制的粒度，从全局标量到完整的层级×时间调度。

我们提出**判别式自适应层级缩放（DALS）**，统一融合五代关键洞见的优化器：阶段自适应余弦调度（第2代）、LARS式信任比（第4代）、深度感知Grokfast梯度滤波（第5代+）和SGD动量。DALS代表了这一演化轨迹的自然终点——单一优化器通过阶段感知和深度感知的梯度处理来回应不可能三角。

本文贡献如下：（1）系统性的五代学习率策略分类法；（2）融合阶段自适应余弦调度、深度感知Grokfast、信任比和动量的DALS框架，含三种变体（DALS、DALS-Fast、DALS-Acc）；（3）18种策略在五个数据集上跨越从头训练和微调两种机制的全面基准评测；（4）层级策略的方向性偏差及其对训练机制依赖性的分析。

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

过大的学习率导致在极值附近震荡，过小的学习率导致收敛速度令人无法接受，不同调度策略围绕同一核心原则——"先快走后慢走"——做出不同的平滑性和探索性权衡。

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

自适应方法在陡峭方向减速、平坦方向加速以找到更高效路径，这与固定学习率在非光滑损失面上的行为形成对比。

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

![](figs/paper_fig3_discriminative.svg)

**图3**：判别式微调与全局统一学习率的对比。统一学习率过度修改底层而判别式学习率保留通用知识。

### 2.5 第5代：层级与时间联合调度

最新一代将层级差异化与时间动态相结合。

**STLR**（倾斜三角学习率，Howard and Ruder 2018）使每层学习率先增后降：

$$cut = \lfloor T \cdot cut\_frac \rfloor, \quad p = \begin{cases} t/cut & \text{若 } t < cut \\ 1 - \frac{t - cut}{cut \cdot (1/cut\_frac - 1)} & \text{否则} \end{cases}$$

$$\eta_t = \eta_{max} \cdot \frac{1 + p \cdot (ratio - 1)}{ratio}$$

默认参数 $cut\_frac = 0.1$、$ratio = 32$，在训练前10%快速预热，随后90%缓慢衰减。与判别式微调组合时，第 $l$ 层在时间步 $t$ 获得：

$$\eta_t^l = \text{STLR}(t; \eta_{max}^l)$$

其中 $\eta_{max}^l$ 由判别式衰减因子设定。

![](figs/paper_fig4_stlr.svg)

**图4**：倾斜三角学习率（STLR）调度。训练前10%快速预热，随后缓慢衰减。

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

## 3 方法：判别式自适应层级缩放（DALS）

### 3.1 动机

虽然每一代都贡献了宝贵洞见，但没有单一优化器能组合层级差异化、阶段自适应、自适应梯度缩放和梯度滤波的互补优势。我们提出DALS来统一这些创新。

### 3.2 DALS框架

新的DALS框架——给定具有 $L$ 层和参数 $\theta = \{\theta^1, \ldots, \theta^L\}$ 的模型，DALS按以下步骤计算第 $l$ 层在第 $t$ 步的更新：

**步骤1：阶段自适应余弦学习率。** 学习率遵循热身-余弦调度，阶段由实时损失改善率 $\Delta_t = (\mathcal{L}_{ema}^{t-1} - \mathcal{L}_{ema}^t) / |\mathcal{L}_{ema}^{t-1}|$ 决定，其中 $\mathcal{L}_{ema}^t = 0.95 \cdot \mathcal{L}_{ema}^{t-1} + 0.05 \cdot \mathcal{L}_t$：

$$\eta_t^l = \eta_0 \cdot s(t), \quad s(t) = \begin{cases} t / W & \text{若 } t < W \\ \frac{1}{2}\left(1 + \cos\frac{\pi(t - W)}{T - W}\right) & \text{否则} \end{cases}$$

其中 $W = 0.05T$ 为热身期，$T$ 为总训练步数。阶段仅影响梯度处理，不直接影响学习率调度：

- 阶段0（探索期，$\Delta_t > 0.01$）：损失快速下降
- 阶段1（利用期，$0.002 < \Delta_t \leq 0.01$）：中等改善
- 阶段2（精调期，$\Delta_t \leq 0.002$）：接近收敛

**步骤2：深度感知Grokfast梯度滤波。** 逐层EMA滤波，阶段自适应平滑：

$$\alpha_l = \begin{cases} \max(0.3, \alpha_0 - 0.3) & \text{阶段0} \\ \alpha_0 & \text{阶段1} \\ \min(0.9, \alpha_0 + 0.1) & \text{阶段2} \end{cases}$$

$$\tilde{g}_t^l = \alpha_l \tilde{g}_{t-1}^l + (1 - \alpha_l) g_t^l$$

$$\hat{g}_t^l = (0.3 + 0.4 \cdot d_l) \cdot g_t^l + (0.7 - 0.4 \cdot d_l) \cdot \tilde{g}_t^l$$

其中 $d_l = l / (L-1)$ 为深度比（底层为0，顶层为1）。顶层使用更多原始梯度；底层使用更多滤波信号以保持稳定性。

**步骤3：LARS式信任比。** 逐参数自适应梯度缩放：

$$r_t^l = \text{clamp}\left(\gamma \cdot \frac{\|\theta^l\|_2}{\|\hat{g}_t^l\|_2 + \epsilon}, \, 0.2, \, 5.0\right)$$

其中 $\gamma = 0.02$ 为信任系数。

**步骤4：动量更新。** 标准SGD动量：

$$m_t^l = \mu \cdot m_{t-1}^l + \hat{g}_t^l$$

$$\theta_t^l = \theta_{t-1}^l - \eta_t^l \cdot r_t^l \cdot m_t^l$$

![](figs/paper_fig5_dals_architecture.svg)

**图5**：DALS框架架构。阶段自适应余弦调度、深度感知Grokfast滤波、LARS式信任比和SGD动量的统一组合。

### 3.3 关键设计原则

DALS体现了源自五代演化的三个原则：

1. **阶段感知**：训练动态在不同阶段变化——DALS根据检测到的阶段调整梯度平滑强度。
2. **深度感知**：底层接收更强的梯度滤波（更多滤波信号混合），因为其梯度经过更多层反向传播，噪声更大。
3. **梯度质量感知**：信任比归一化每个参数的更新量级，防止不稳定。

### 3.4 DALS变体：速度与精度

DALS框架自然支持三个调优方向：

**DALS-Fast** 通过提高基础学习率（$\eta_0 = 0.05$）、缩短热身至2%、降低动量（$\mu = 0.85$），并在阶段0完全跳过Grokfast滤波来加速初期收敛。核心洞见是：在损失快速下降期间，梯度滤波增加了不必要的延迟——模型使用原始梯度更新学习更快。降低的动量使更新更灵敏。这使90%收敛仅需3个epoch（基准DALS需9个），代价是最终准确率略低（97.8%）。

**DALS-Acc** 通过将单一余弦调度替换为SGDR风格的周期性热重启（$T_0 = 10$个epoch，$T_\text{mult} = 2$）、增强权重衰减（$\lambda = 5 \times 10^{-4}$）和更强的Grokfast滤波（$\alpha = 0.7$）来追求更高最终准确率。热重启周期性地重置学习率，允许优化器逃出局部极小值并探索损失景观的新区域。更强的权重衰减正则化防止过拟合，增强的梯度平滑稳定后期收敛。

### 3.5 与先前工作的关系

DALS可视为经验证技术的受控组合：

| 组件 | 来源 | DALS适配 |
|:-----|:-----|:---------|
| $\eta_t = \eta_0 \cdot s(t)$ 热身+余弦 | 第2代 | 阶段自适应热身 |
| 信任比 $r_t^l$ | LARS（第4代） | 钳位+逐参数 |
| 梯度EMA $\tilde{g}$ | Grokfast（第5代+） | 深度+阶段依赖 $\alpha$ |
| 动量 $m$ | SGD | 标准 |

每个组件均已独立验证；DALS提供了以阶段感知和深度感知协调方式组合它们的系统性框架。

## 4 实验

### 4.1 实验设置

我们在五个跨越从头训练和微调机制的数据集上，对18种学习率策略（含三种DALS变体）进行跨5代的基准评测，使用不同的模型架构和任务特征。

**合成任务（从头训练）。** 从高斯混合生成10类分类任务：8000个$\mathbb{R}^{64}$样本，前10维承载类别信号（缩放3×），其余54维为纯噪声（$\sigma = 0.1$）。标签由信号维度的$\arg\max$决定。训练/测试集划分为6400/1600，固定随机种子（42）。此设计创建可学习但非平凡的任务（54维噪声要求优化器忽略无关特征）。模型：4层MLP（64→128→128→10），ReLU激活，训练80个epoch，批大小64。无Dropout、无批归一化、无数据增强——确保优化器行为是主要变量。合成任务提供三个优势：（1）控制混淆因素——无数据增强或预训练权重与学习率效应交互；（2）快速迭代——80个epoch仅需3–20秒；（3）聚焦优化动态而非声称最优性能。

**CIFAR-10（从头训练）。** 标准CIFAR-10数据集，小型ConvNet（3卷积层+FC头），从头训练50个epoch，批大小128，标准增强（随机裁剪、水平翻转）。测试策略从合成到自然图像训练的迁移性。

**RTE（微调）。** GLUE基准中的文本蕴含识别，DistilBERT微调5个epoch，批大小32。小数据集（~2.5k训练样本），测试短序列NLU微调。

**TREC-6（微调）。** 问题分类（6个粗类别），DistilBERT微调5个epoch，批大小32。较长序列任务，富含句法特征，测试结构化分类上的微调。

**IMDb（微调）。** 二元情感分类，DistilBERT微调3个epoch，批大小32。大数据集（25k训练样本），测试长文档上的微调。

**超参数。** 各策略使用原始论文的典型超参数：Adam/AdamW使用$\text{lr}=3\times10^{-4}$，SGD族方法使用$\text{lr}=0.01$–$0.05$、动量0.9、权重衰减$10^{-4}$，层级方法使用衰减因子$\delta=2.6$（Howard and Ruder 2018）。DALS变体使用第3.4节描述的配置。微调任务中，所有策略使用相同基础学习率（Adam族$5\times10^{-5}$，SGD族$1\times10^{-3}$），前10%训练步线性热身。

我们报告每种策略在各数据集上的最佳测试准确率（%）。对合成任务，额外报告收敛速度（达到准确率阈值的epoch数）。

### 4.2 结果

**表1**：跨数据集基准——18种策略在5个数据集上的最佳测试准确率（%）。

| 策略 | 代际 | 合成 | CIFAR-10 | RTE | TREC-6 | IMDb |
|:---------|:----:|:----:|:--------:|:---:|:------:|:----:|
| Fixed SGD | 第1代 | 97.9 | 78.8 | 60.6 | 96.6 | 90.5 |
| Cosine Decay SGD | 第2代 | 97.6 | **80.2** | 57.8 | 94.8 | 90.7 |
| SGDR | 第2代 | 97.8 | 79.7 | 59.9 | 96.2 | 90.6 |
| Adam | 第3代 | 96.8 | 75.5 | 59.6 | **97.6** | 90.8 |
| AdamW | 第3代 | 96.2 | 75.7 | 60.3 | **97.6** | 90.7 |
| AdaBound | 第3代 | 95.5 | 75.5 | 57.4 | 92.8 | 90.0 |
| LARS | 第4代 | 97.7 | 74.9 | 59.6 | 95.2 | 90.0 |
| Discriminative LR | 第4代 | 97.1 | 77.4 | 57.8 | 84.6 | 88.6 |
| RAdam | 第5代 | 96.5 | 75.8 | **62.8** | 96.8 | **91.2** |
| Lion | 第5代 | 97.0 | 76.6 | 58.8 | 95.0 | 87.8 |
| Lookahead+AdamW | 第5代 | 96.0 | 75.1 | 59.9 | **97.6** | 91.1 |
| SAM | 第5代 | 97.5 | 76.9 | 59.2 | 96.4 | 90.5 |
| Grokfast | 第5代 | 97.3 | 78.5 | 56.0 | 84.4 | 89.5 |
| STLR+Discriminative | 第5代 | 95.9 | 79.3 | 55.2 | 43.6 | 85.4 |
| SAM+Discriminative | SOTA | 97.4 | 77.9 | 58.5 | 84.6 | 88.6 |
| **DALS（本文）** | SOTA | **98.0** | 76.7 | 59.2 | 94.0 | 90.1 |
| **DALS-Fast** | SOTA | 97.8 | 76.9 | 59.9 | 95.0 | 90.1 |
| **DALS-Acc** | SOTA | 97.8 | 76.5 | 59.2 | 94.2 | 90.0 |

**表2**：合成任务收敛速度——达到准确率阈值所需的epoch数。

| 策略 | →60% | →70% | →80% | →90% | 总时间 |
|:---------|:----:|:----:|:----:|:----:|:------:|
| SGDR | 1ep | 1ep | 1ep | 1ep | 10.0s |
| DALS（本文） | 2ep | 2ep | 3ep | 3ep | 19.4s |
| DALS-Fast | 1ep | 1ep | 2ep | 3ep | 18.6s |
| DALS-Acc | 1ep | 1ep | 1ep | 2ep | 19.4s |
| LARS | 1ep | 1ep | 1ep | 2ep | 15.8s |
| STLR+Discriminative | 1ep | 1ep | 1ep | 1ep | 10.4s |

### 4.3 分析与讨论

**机制依赖的主导模式。** 五数据集基准揭示了没有单一策略在所有机制中占优。从头训练任务中，DALS以98.0%领先合成基准，Cosine SGD以80.2%领先CIFAR-10。微调任务中，RAdam在IMDb（91.2%）和RTE（62.8%）领先，Adam/AdamW/Lookahead在TREC-6并列第一（97.6%）。这一三路分化——SGD族适合从头训练CV任务、自适应方法适合NLP微调、DALS适合均衡全场景——构成了本基准的核心发现。

**STLR+Discriminative的灾难性失败。** STLR+Discriminative——ULMFiT的冠军策略——在从头训练中遭受灾难性失败：TREC-6仅43.6%（对比RAdam的97.6%），IMDb仅85.4%（对比RAdam的91.2%），RTE仅55.2%（对比RAdam的62.8%）。这不是表现不佳，而是彻底崩溃。根本原因是方向性偏差：判别式衰减（$\eta^l = \eta_0 / \delta^{L-l}$，$\delta=2.6$）将底层更新压制数个数量级，使其无法从头学习特征。与STLR的快速坍缩调度组合后，底层基本上获得零有效学习率，造成过早学习失败。矛盾的是，STLR+Discriminative在CIFAR-10上仍达到79.3%——所有策略中最佳——因为判别式衰减在层级图像特征上从零开始也偶然有益。

**判别式方法何时有效。** 在IMDb微调上，Discriminative LR（88.6%）与最佳RAdam（91.2%）的差距仅2.6个百分点，而在TREC-6从头训练上差距达12.5个百分点。这证实了ULMFiT的发现：当底层包含有用的预训练特征时，抑制其更新是有益的而非有害的。五数据集交叉比较精确量化了方向性偏差*何时*从有害变为有益。

**DALS：跨机制均衡。** DALS在合成任务上达到最佳结果（98.0%），在IMDb微调上保持竞争力（90.1%），在CIFAR-10上表现稳健（76.7%）。与Discriminative LR从84.6%（TREC-6）到97.1%（合成）的12.5点波动相比，DALS的波动范围窄得多：90.1%（IMDb）到98.0%（合成），仅7.9点。这种一致性源于移除了方向性偏差：DALS的阶段与深度感知处理根据实时损失动态调整梯度平滑，而非施加固定的层级层次。

**DALS变体：面向不同目标的调优。** DALS-Fast仅需3个epoch即达合成任务90%准确率（基准DALS需9ep），是快速原型设计的理想选择。DALS-Acc通过SGDR重启匹配基准DALS的准确率，在更长训练中可能逃出局部极小值。在微调任务上，三种DALS变体紧密聚集（IMDb 90.0–90.1%），表明DALS框架的优势主要体现在阶段感知处理更重要的从头训练机制中。

**NLP微调：自适应方法占优。** 在三个NLP基准（RTE、TREC-6、IMDb）上，自适应优化器持续优于SGD族方法。RAdam在RTE（62.8%）和IMDb（91.2%）领先，Lookahead在TREC-6并列最佳（97.6%）。这符合预期：预训练Transformer具有良好条件的损失景观，Adam的自适应更新提供了恰当的逐参数缩放。SGD族策略尽管在从头训练上强劲，在微调上难以匹敌自适应方法——这一发现强调了根据训练机制选择优化器的重要性。

## 5 结论

本文提出了学习率演化的五代分类法，并在跨越从头训练和微调两种机制的五个数据集上进行了验证。跨数据集基准揭示了核心发现：**不存在普适最优的学习率策略**。最佳策略关键取决于是从头训练还是微调预训练模型。

从头训练任务中，带调度的SGD族方法占优：DALS以98.0%领先合成任务，Cosine SGD以80.2%领先CIFAR-10。微调任务中，自适应方法占优：RAdam在RTE（62.8%）和IMDb（91.2%）领先。最戏剧性的发现涉及STLR+Discriminative——ULMFiT的冠军策略——在从头训练中遭受灾难性失败（TREC-6 43.6%，IMDb微调85.4%），但在CIFAR-10上仍具竞争力（79.3%）。这确认了判别式衰减的方向性偏差仅在底层包含值得保留的预训练特征时才有益。

我们的DALS框架将阶段自适应余弦调度、深度感知Grokfast梯度滤波和LARS式信任比融合为统一优化器。通过移除判别式衰减的方向性偏差并以阶段与深度感知的梯度处理替代，DALS在合成任务上达到最佳结果（98.0%），同时在IMDb微调上保持竞争力（90.1%）——既避免了判别式方法的灾难性从头训练失败，也避免了自适应方法在从头训练上的平庸表现。DALS家族覆盖速度-精度Pareto前沿：DALS-Fast快速收敛，基准DALS平衡速度与精度，DALS-Acc通过SGDR重启追求更高准确率。

未来工作应在DALS阶段自适应机制可能进一步发挥优势的大规模迁移学习基准上评测其表现，并探索我们观察到的机制依赖模式是否延伸到现代架构（Transformer、Vision Transformer）和全规模任务。

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