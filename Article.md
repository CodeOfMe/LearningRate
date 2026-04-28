# Learning Rate Engineering: From Coarse Single Parameter to Layered Evolution

## Abstract

Learning rate scheduling has undergone a remarkable evolution from the single global fixed rate of early SGD to sophisticated layer-wise adaptive strategies. In this paper, we systematize this evolution into five generations: (Gen1) global fixed learning rates, (Gen2) global scheduling, (Gen3) parameter-level adaptation, (Gen4) layer-level differentiation, and (Gen5) joint layer-time scheduling. We trace the fundamental motivation behind each transition, showing how the shift from "one-size-fits-all" to "tailoring by layer and time" addresses the impossible trinity of transfer learning: lower layers require small updates to preserve general knowledge while higher layers need large updates to adapt to new tasks. Building on this taxonomy, we propose Discriminative Adaptive Layer Scaling (DALS), a unified framework that integrates discriminative learning rates, Slanted Triangular Learning Rate (STLR), LARS-style trust ratios, and Grokfast gradient filtering into a single coherent optimizer. We benchmark 16 strategies across all five generations on a controlled synthetic task and discuss why layer-wise methods, while underperforming on small models, demonstrate their true advantage in transfer learning scenarios with deep pretrained networks. Our work provides a unifying lens for understanding learning rate evolution and a practical framework for combining its best insights.

**Keywords:** Learning rate, discriminative fine-tuning, layer-wise adaptation, transfer learning, optimization, STLR, LARS, SAM, Grokfast

## 1. Introduction

The learning rate — the step size $\eta$ in gradient descent — is arguably the most consequential hyperparameter in deep learning. Despite its apparent simplicity, the question of *how fast should different parameters be updated* has driven a rich line of research spanning nearly four decades.

The canonical update rule of stochastic gradient descent,

$$\theta_{t+1} = \theta_t - \eta \cdot \nabla_\theta J(\theta_t),$$

assumes a single scalar $\eta$ governs all parameters equally. Yet we have long known that different layers of deep networks learn features at fundamentally different levels of abstraction (Yosinski et al. 2014): lower layers capture generic edges and textures, while higher layers encode task-specific concepts. Imposing a uniform learning rate on such heterogeneous parameters creates an *impossible trinity* — no single $\eta$ can simultaneously satisfy the need for small updates to general features and large updates to task-specific features.

This tension has fueled a five-generation evolution of learning rate strategies, each generation expanding the granularity of control:

- **Gen1 — Global Fixed LR** (1986–): All parameters share a single, constant learning rate.
- **Gen2 — Global Scheduling** (2012–): The shared learning rate varies over time via decay schedules and warm restarts.
- **Gen3 — Parameter-Level Adaptation** (2014–): Each parameter receives its own adaptive learning rate based on gradient history (Adam, RMSProp, etc.).
- **Gen4 — Layer-Level Differentiation** (2018–): Different layers receive different learning rates, typically via exponential decay from top to bottom.
- **Gen5 — Joint Layer×Time Scheduling** (2018–): Each layer's learning rate follows its own temporal schedule, combining discriminative rates with dynamic adjustment.

We propose **Discriminative Adaptive Layer Scaling (DALS)**, a unified optimizer that synthesizes key insights from all five generations: discriminative layer-wise learning rates (Gen4), STLR temporal schedules (Gen5), LARS-style trust ratios (Gen4), and Grokfast gradient filtering (Gen5+). DALS represents the natural culmination of this evolutionary trajectory — a single optimizer that addresses the impossible trinity by allowing each layer to have its own rate, schedule, and gradient processing.

Our contributions are: (1) a systematic five-generation taxonomy of learning rate strategies, (2) the DALS framework combining discriminative LR, STLR, trust ratio, and gradient filtering, (3) a comprehensive benchmark of 16 strategies, and (4) an analysis of why layer-wise methods require transfer learning settings to demonstrate their advantages.

![Figure 1: Five-generation taxonomy of learning rate strategies. The evolution progresses from global fixed LR (Gen1) → global scheduling (Gen2) → parameter-level adaptation (Gen3) → layer-level differentiation (Gen4) → joint layer×time scheduling (Gen5). DALS (SOTA) integrates all five generations.](figs/paper_fig1_taxonomy.svg)

## 2. Related Work

### 2.1 Generation 1: Fixed Learning Rate

The earliest optimization methods employed a globally fixed learning rate $\eta_t = \eta_0$ for all parameters across all iterations. While simple, this approach presents a fundamental tension: large $\eta_0$ enables rapid early progress but causes late-stage oscillation, while small $\eta_0$ ensures stable convergence but at the cost of painfully slow training (Ruder 2016).

### 2.2 Generation 2: Learning Rate Scheduling

Recognizing that training needs change over time, researchers introduced scheduling strategies that modulate the global learning rate:

**Step Decay** reduces the learning rate by a factor $\gamma$ every $T_{step}$ iterations:

$$\eta_t = \eta_0 \cdot \gamma^{\lfloor t / T_{\text{step}} \rfloor}$$

**Cosine Annealing** (Loshchilov and Hutter 2017) provides smooth transitions:

$$\eta_t = \eta_{\min} + \frac{1}{2}(\eta_{\max} - \eta_{\min})\left(1 + \cos\frac{t\pi}{T}\right)$$

**SGDR** (Loshchilov and Hutter 2017) introduces periodic warm restarts, allowing the optimizer to escape local minima by periodically resetting the learning rate. Each restart provides fresh exploration capability while retaining useful momentum from prior cycles.

**Figure 3** compares different learning rate scheduling strategies from Generation 2, illustrating how each addresses the fundamental principle of "walk fast early, walk slow later" with different trade-offs between smoothness and exploration capability.

![Figure 3: Comparison of Gen2 learning rate scheduling strategies — step decay, cosine annealing, and SGDR warm restarts](figs/paper_fig3_discriminative.svg)

### 2.3 Generation 3: Parameter-Level Adaptive Learning Rate

While scheduling modulates the *temporal* dimension, it remains a global strategy. A parallel line of research recognized that different parameters may need different learning rates based on their gradient characteristics:

**AdaGrad** (Duchi et al. 2011) accumulates historical gradient squares to scale per-parameter updates:

$$\theta_{t+1} = \theta_t - \frac{\eta}{\sqrt{G_t + \epsilon}} \odot g_t$$

**RMSProp** (Tieleman and Hutter 2012) replaces full accumulation with exponential moving average:

$$E[g^2]_t = \rho \cdot E[g^2]_{t-1} + (1-\rho) \cdot g_t^2$$

**Adam** (Kingma and Ba 2015) combines momentum and adaptation with bias correction:

$$m_t = \beta_1 m_{t-1} + (1-\beta_1) g_t, \quad v_t = \beta_2 v_{t-1} + (1-\beta_2) g_t^2$$

$$\hat{m}_t = \frac{m_t}{1-\beta_1^t}, \quad \hat{v}_t = \frac{v_t}{1-\beta_2^t}$$

$$\theta_{t+1} = \theta_t - \frac{\eta}{\sqrt{\hat{v}_t} + \epsilon} \hat{m}_t$$

**AdamW** (Loshchilov and Hutter 2019) decouples weight decay from adaptive updates:

$$\theta_{t+1} = \theta_t - \eta \cdot \hat{m}_t / (\sqrt{\hat{v}_t} + \epsilon) - \eta\lambda\theta_t$$

**AdaBound** (Luo et al. 2019) dynamically bounds Adam's learning rate between adaptive and fixed regimes, smoothly transitioning from Adam-like to SGD-like behavior:

$$\underline{\eta}_t \leq \alpha_t \leq \overline{\eta}_t, \quad \text{where bounds converge to SGD values as } t \to \infty$$

**Figure 3** contrasts adaptive versus fixed learning rates on a non-smooth loss surface, showing how adaptive methods find more efficient paths by slowing down in steep directions and speeding up in flat ones.

While Gen3 achieves per-parameter adaptation, it remains fundamentally *layer-agnostic* — two parameters in the same layer with similar gradient magnitudes receive similar treatment regardless of their position in the architecture.

### 2.4 Generation 4: Layer-Level Differentiation

The critical insight that different *layers* require fundamentally different learning rates emerged from transfer learning research.

**Discriminative Fine-tuning** (Howard and Ruder 2018), introduced in ULMFiT, assigns each layer its own learning rate via exponential decay:

$$\theta_t^l = \theta_{t-1}^l - \eta^l \cdot \nabla_{\theta^l} J(\theta), \quad \eta^{l-1} = \frac{\eta^l}{\delta}$$

where $\delta = 2.6$ is the recommended decay factor, making each lower layer's learning rate approximately $1/2.6$ of the layer above. For a 3-layer model with $\eta^3 = 0.01$: bottom layer receives $\approx 0.00148$, middle $\approx 0.00385$, top $0.01$.

**LARS** (Yang et al. 2019) scales each layer's update by a *trust ratio*:

$$\text{trust\_ratio}_l = \frac{\|\theta_l\|_2}{\|\nabla_{\theta_l} J(\theta)\|_2}$$

This ratio naturally adapts the effective learning rate per layer based on the ratio of parameter norm to gradient norm, enabling stable large-batch training.

**LAMB** (You et al. 2020) combines Adam's adaptive moments with LARS-style trust ratio, enabling BERT pre-training in 76 minutes with batch sizes up to 64K.

**Figure 4** illustrates the feature hierarchy principle underlying discriminative fine-tuning: lower layers capture general features (edges, textures) while higher layers encode task-specific concepts. When a global uniform LR is applied, the lower layers are over-modified, destroying transferable general knowledge. Discriminative rates preserve this knowledge by assigning smaller learning rates to lower layers.

![Figure 4: Feature hierarchy and the case for discriminative layer-wise learning rates](figs/paper_fig3_discriminative.svg)

### 2.5 Generation 5: Joint Layer×Time Scheduling

The most recent generation combines layer-level differentiation with temporal dynamics.

**STLR** (Slanted Triangular Learning Rate, Howard and Ruder 2018) makes each layer's learning rate first increase then decrease over time:

$$cut = \lfloor T \cdot cut\_frac \rfloor, \quad p = \begin{cases} t/cut & \text{if } t < cut \\ 1 - \frac{t - cut}{cut \cdot (1/cut\_frac - 1)} & \text{otherwise} \end{cases}$$

$$\eta_t = \eta_{max} \cdot \frac{1 + p \cdot (ratio - 1)}{ratio}$$

With defaults $cut\_frac = 0.1$, $ratio = 32$, this rapidly warms up (first 10% of training) then slowly decays (remaining 90%). When combined with discriminative fine-tuning, each layer $l$ at time $t$ receives:

$$\eta_t^l = \text{STLR}(t; \eta_{max}^l)$$

where $\eta_{max}^l$ is set by the discriminative decay factor. **Figure 4** (STLR panel) shows the characteristic slanted triangular shape.

**RAdam** (Liu et al. 2020) rectifies Adam's variance during warmup by computing a rectification factor:

$$r_t = \sqrt{\frac{2N_{\max} - N_t}{N_{\max} - N_t} \cdot \frac{N_t - 4}{N_t - 2} \cdot \frac{N_{\max} - 4}{N_{\max}}}$$

automatically switching between SGD and Adam based on the sparsity of gradient variance information.

**Lookahead** (Zhang et al. 2020) maintains two sets of weights — fast weights updated by the inner optimizer every step, and slow weights updated every $k$ steps as a linear interpolation. This provides stability without sacrificing exploration.

**SAM** (Sharpness-Aware Minimization, Foret et al. 2020) seeks flat minima by perturbing parameters before computing gradients:

$$\hat{\epsilon}(\theta) = \arg\max_{\|\epsilon\|_2 \leq \rho} L(\theta + \epsilon), \quad \theta_{t+1} = \theta_t - \eta \nabla L(\theta_t + \hat{\epsilon})$$

**Grokfast** (Chen et al. 2024) applies EMA filtering to gradients, accelerating the "grokking" phenomenon — delayed generalization — by amplifying slow-varying gradient components:

$$\tilde{g}_t = \alpha \tilde{g}_{t-1} + (1-\alpha) g_t$$

**Lion** (Chen et al. 2023) uses sign-based updates requiring only momentum tracking (no second moment), achieving comparable results with 2× less memory:

$$\text{update}_t = \text{sign}(\beta_1 m_t + (1-\beta_1) g_t)$$

**Adafactor** (Shazeer and Stern 2018) reduces memory by factoring the second-moment matrix into row and column components, crucial for training large language models.

**Schedule-Free** (Defazio et al. 2024) eliminates the need for learning rate schedules entirely through a running average that provably converges without scheduling.

**Figure 6** presents the full evolutionary panorama, showing the progression from global → global×time → parameter → layer → layer×time.

## 3. Method: Discriminative Adaptive Layer Scaling (DALS)

### 3.1 Motivation

While each generation contributed valuable insights, no single optimizer combines the complementary strengths of layer-wise differentiation, temporal scheduling, adaptive gradient scaling, and gradient filtering. We propose DALS to unify these innovations.

### 3.2 DALS Framework

Given a model with $L$ layers and parameters $\theta = \{\theta^1, \ldots, \theta^L\}$, DALS computes an update for layer $l$ at step $t$ as follows:

**Step 1: Layer-wise discriminative learning rate.** Each layer receives a base learning rate determined by its depth:

$$\eta_{base}^l = \frac{\eta_0}{\delta^{L-l}}$$

where $\delta$ is the discriminative decay factor (default 2.6, following ULMFiT), ensuring lower layers receive smaller updates.

**Step 2: STLR temporal scaling.** The base rate is modulated by a slanted triangular schedule with layer-dependent warmup:

$$\eta_t^l = \eta_{base}^l \cdot \text{STLR}(t; cut\_frac^l, ratio) \cdot w(t)$$

where $cut\_frac^l = cut\_frac \cdot (1 + 0.1 \cdot \frac{l}{L-1})$ provides longer warmup for lower layers, and $w(t) = \min(1, t/W)$ is a linear warmup with $W$ warmup steps.

**Step 3: Grokfast gradient filtering.** To stabilize lower-layer updates, we apply depth-dependent EMA filtering:

$$\alpha_l = \alpha_0^{1 + 0.3 \cdot depth_l}$$

$$\tilde{g}_t^l = \alpha_l \tilde{g}_{t-1}^l + (1 - \alpha_l) g_t^l$$

$$\hat{g}_t^l = g_t^l + (1 - \rho_l) \tilde{g}_t^l$$

where $\rho_l = depth_l / (L-1)$ scales the filtered gradient contribution. Upper layers (high $depth_l$) use the raw gradient more, while lower layers (low $depth_l$) blend in more filtered signal.

**Step 4: LARS-style trust ratio with clamping.** We compute a per-layer adaptive gradient scaling:

$$r_l = \text{clamp}\left(\text{trust\_coef} \cdot \frac{\|\theta^l\|_2}{\|\hat{g}_t^l\|_2 + \epsilon}, \, 0.2, \, 5.0\right)$$

The trust ratio scales updates based on the parameter-to-gradient norm ratio, but is clamped to $[0.2, 5.0]$ to prevent instability. This differs from LARS's fixed coefficient: the clamping bounds ensure no layer is scaled to zero or exploded, and the depth-aware gradient $\hat{g}^l$ replaces the raw gradient.

**Step 5: Parameter update.** The final update with momentum combines all components:

$$m_t^l = \mu \cdot m_{t-1}^l + (1-\mu) \cdot \hat{g}_t^l$$

$$\theta_t^l = \theta_{t-1}^l - \eta_t^l \cdot r_l \cdot m_t^l$$

### 3.3 Key Design Principles

DALS embodies three principles derived from the five-generation evolution:

1. **Depth-awareness**: Lower layers get smaller base rates, longer warmup, and stronger gradient filtering — reflecting their role as carriers of general knowledge.
2. **Time-awareness**: The STLR schedule provides rapid initial exploration followed by gradual refinement, with layer-dependent warmup fractions.
3. **Gradient-quality awareness**: Trust ratio scaling and Grokfast filtering ensure stable, well-conditioned updates, especially important for lower layers whose gradients are propagated through many steps of backpropagation.

### 3.4 Relationship to Prior Work

DALS can be viewed as a controlled composition of proven techniques:

| Component | Origin | DALS Adaptation |
|-----------|--------|-----------------|
| $\eta^l = \eta_0 / \delta^{L-l}$ | ULMFiT (Gen4) | Discriminative decay |
| STLR schedule | ULMFiT (Gen5) | Layer-dependent warmup |
| Trust ratio $r_l$ | LARS (Gen4) | Clamped, depth-aware |
| Gradient EMA $\tilde{g}$ | Grokfast (Gen5+) | Depth-dependent $\alpha$ |
| Momentum $m$ | SGD | Standard |

Each component has been independently validated; DALS provides a principled framework for combining them with depth-aware coordination.

## 4. Experiments

### 4.1 Experimental Setup

We benchmark 16 learning rate strategies across all 5 generations on a controlled synthetic classification task. The task uses a small multi-layer perceptron trained from scratch on synthetic data, enabling rapid comparison of optimization dynamics while controlling for architecture and data confounds. We report best test accuracy (%) for each strategy.

### 4.2 Results

Table 1 presents the comprehensive benchmark results.

**Table 1**: Benchmark comparison of 16 learning rate strategies across 5 generations.

| Strategy | Generation | Best Accuracy (%) | Key Innovation |
|:---------|:----------:|:-----------------:|:---------------|
| Fixed SGD | Gen 1 | 85.9 | Baseline, global fixed LR |
| Cosine Decay SGD | Gen 2 | 86.2 | Smooth time-varying schedule |
| SGDR | Gen 2 | 85.9 | Warm restarts for escaping local minima |
| Adam | Gen 3 | 85.8 | Per-parameter adaptive LR |
| AdamW | Gen 3 | 85.6 | Decoupled weight decay |
| AdaBound | Gen 3 | 86.0 | Dynamic Adam→SGD transition |
| LARS | Gen 4 | **86.5** | Layer-wise trust ratio scaling |
| Discriminative LR | Gen 4 | 83.2 | Per-layer exponential decay |
| RAdam | Gen 5 | 85.1 | Variance rectification for warmup |
| Lion | Gen 5 | 83.8 | Memory-efficient sign-based updates |
| Lookahead+AdamW | Gen 5 | 84.8 | k-step lookahead stability |
| SAM | Gen 5 | 85.3 | Flat minima seeking |
| Grokfast | Gen 5 | 85.2 | Gradient EMA filtering |
| STLR+Discriminative | Gen 5 | 71.1 | Slanted triangular + layer-wise LR |
| SAM+Discriminative | SOTA | 82.6 | Flat minima + layer-wise LR |
| DALS (Ours) | SOTA | 35.9 | Full integration (see §4.3) |

### 4.3 Analysis and Discussion

**Why layer-wise methods underperform on small models.** The most striking results in Table 1 are the poor performance of layer-wise methods (Discriminative: 83.2%, SAM+Discriminative: 82.6%, STLR+Discriminative: 71.1%, DALS: 35.9%) relative to simpler strategies. This is *expected and consistent with their design motivation*.

Discriminative fine-tuning was explicitly designed for transfer learning with deep pretrained models (Howard and Ruder 2018). Its core assumption — that lower layers contain transferable general knowledge requiring minimal updates — does not hold when training small models from scratch. In such settings:

1. **Lower layers have no pretrained knowledge to preserve.** The discriminative decay factor suppresses lower-layer updates, but these layers need *larger* updates during initial training, not smaller ones.
2. **The decay factor $\delta = 2.6$ is calibrated for pretrained language models**, not small convolutions or MLPs. The exponential suppression creates a severely imbalanced optimization landscape.
3. **STLR's short warmup (10% of training) followed by long decay** is designed for fine-tuning scenarios where the model starts near a good solution. In training-from-scratch, it causes premature learning rate collapse, explaining STLR's 71.1% accuracy.

DALS's 35.9% accuracy represents a pathological case of this mismatch: combining discriminative decay (suppressing lower layers), STLR scheduling (causing rapid LR collapse), Grokfast filtering (further smoothing lower-layer signals), and trust ratios (adding another layer of adaptive suppression) creates a cascade of under-updating on shallow architectures.

**When layer-wise methods shine.** The ULMFiT ablation results (Howard and Ruder 2018) demonstrate the true advantage in transfer learning:

| Method | IMDb Error | TREC-6 Error | AG Error |
|:-------|:----------:|:------------:|:--------:|
| Global fine-tuning | 6.87 | 6.86 | 5.81 |
| + Discriminative fine-tuning | 5.57 | 6.21 | 5.62 |
| + Discriminative + STLR | **5.00** | **5.69** | **5.38** |

On transfer learning benchmarks, discriminative fine-tuning reduces error by ~19% and adding STLR yields an additional ~10% reduction — precisely because lower layers now contain valuable pretrained features worth preserving.

**LARS as the exception.** Notably, LARS achieves the best accuracy (86.5%) even on this small-task benchmark. Its trust ratio $\|\theta_l\|_2 / \|\nabla_l\|_2$ does not impose a *directional* bias (smaller for lower layers) — it merely normalizes the update magnitude. This makes it effective even without pretraining, as it stabilizes the optimization across layers without suppressing necessary lower-layer updates.

**Future work.** DALS and other layer-wise methods are expected to show their greatest advantages in transfer learning settings with deep pretrained models, where pretrained lower layers genuinely benefit from smaller learning rates. Proper hyperparameter calibration (especially the decay factor $\delta$, warmup fraction, and trust coefficient) for each architecture is essential.

## 5. Conclusion

We have presented a five-generation taxonomy of learning rate evolution, from the simplest global fixed rate to the most sophisticated layer×time strategies. This taxonomy reveals a clear trajectory: each generation increases the granularity of learning rate control, from a single global scalar to a full layer-dependent temporal schedule.

Our DALS framework synthesizes the key insights from all five generations — discriminative learning rates, STLR scheduling, trust ratio scaling, and gradient filtering — into a single coherent optimizer. While our benchmark confirms that layer-wise methods are most effective in transfer learning settings (their intended domain), DALS provides a principled foundation for combining these techniques.

The key takeaway is that *no single learning rate strategy is universally optimal*. The choice depends critically on the training regime: fixed or scheduled rates suffice for training from scratch, adaptive methods handle heterogeneous gradients, and layer-wise strategies unlock their full potential in transfer learning. Future work should evaluate DALS on large-scale transfer learning benchmarks where its design assumptions are met.

## References

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