# arXiv Submission Metadata

*Title
: Learning Rate Engineering: From Coarse Single Parameter to Layered Evolution

*Author(s)
: Ming-Hong Yao, Di Wang, Jian Cui, Jin-Yan Chen, Zi-Hao Cui, Fa Wang, Chen Wei, and Qiu-Ye Yu

*Abstract
: Learning rate scheduling has evolved from the single global fixed rate of early SGD to sophisticated layer-wise adaptive strategies. We systematize this evolution into five generations: (Gen1) global fixed learning rates, (Gen2) global scheduling, (Gen3) parameter-level adaptation, (Gen4) layer-level differentiation, and (Gen5) joint layer-time scheduling. We trace the fundamental motivation behind each transition, showing how the shift from one-size-fits-all to tailoring by layer and time addresses the impossible trinity of transfer learning: lower layers require small updates to preserve general knowledge while higher layers need large updates to adapt to new tasks. Building on this taxonomy, we propose Discriminative Adaptive Layer Scaling (DALS), a unified framework that integrates phase-adaptive cosine scheduling, depth-aware Grokfast gradient filtering, and LARS-style trust ratios into a single coherent optimizer. We benchmark 18 strategies including three DALS variants across all five generations on five datasets: synthetic, CIFAR-10 (from scratch), RTE, TREC-6, and IMDb (fine-tuning). On synthetic, DALS achieves the best accuracy at 98.0%, while DALS-Fast reaches 90% in just 3 epochs. The cross-dataset analysis reveals striking regime-dependent patterns--no single strategy wins across all regimes. Critically, STLR+Discriminative, the ULMFiT champion, catastrophically fails on from-scratch tasks (43.6% on TREC-6 from scratch vs. 96.8% with RAdam), confirming that directional decay biases are harmful without pretrained features. DALS avoids either extreme, achieving the best synthetic result while maintaining competitive fine-tuning performance.

Comments
: 20 pages, 5 figures, 3 tables

Report number
: 

Journal reference
: 

External DOI
: 

ACM class
: I.2.6; F.2.1

MSC class
: 68T05; 68W40