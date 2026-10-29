# DALS component origins and adaptations

| Component | Origin | DALS Adaptation |
|---|---|---|
| LR schedule (warmup + cosine) | Gen2 | Phase-adaptive warmup |
| Trust ratio r_t^l | LARS (Gen4) | Clamped per-parameter |
| Gradient EMA g~ | Grokfast (Gen5+) | Depth + phase dependent alpha |
| Momentum m | SGD | Standard |
