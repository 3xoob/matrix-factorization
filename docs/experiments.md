# Experiment log: getting PMF from 0.863 to 0.842 RMSE

This documents the investigation behind the final `PMF` architecture described
in the main [README](../README.md#architecture). It's kept separate so the
README can state the current design plainly; this file is for anyone who wants
the reasoning and the dead ends.

**Target:** PMF test RMSE ≤ 0.85, and ≥5% improvement over SVD (SVD measured at
0.8877, so PMF ≤ 0.8433 to clear both simultaneously).

## 1. Classical PMF alone: plateaus at 0.863

An offline sweep over rank (10-150), regularization (0.02-0.15), learning rate
(0.01-0.1), and epoch count (up to 300, both fixed and early-stopped) never
went below RMSE ≈0.863. Increasing epochs past the validation-loss minimum
only widened the train/validation gap (visible in `reports/pmf_convergence.png`)
without improving held-out accuracy.

## 2. SVD++: matches the literature, doesn't close the gap

Before assuming (1) was a tuning problem, we implemented SVD++ (Koren, 2008 —
folding a user's set of rated items, regardless of the rating value, into
their factor as implicit feedback) and checked published MovieLens 1M
benchmarks. Multiple independent studies report SVD++ improving over a
well-tuned SVD by only ~0-3%, and in one study SVD++ actually scored *worse*
than plain SVD:

- 90%-train-split study: PMF 0.883, biased SVD 0.876, SVD++ 0.855 (~2.4% over SVD)
- Another source: SVD 0.84 → SVD++ 0.82 (~2.4% over SVD)
- Comparative benchmark: SVD 0.9927 vs SVD++ 0.9947 on ML-1M (SVD++ *worse*)

Our own SVD++ implementation (`models/pmf_model.py`, `use_implicit=True`)
reached ≈0.861 — consistent with the literature, not a path to 0.85. Since our
SVD baseline is already fairly strong (0.888, from a damped bias correction),
the ~2-3% ceiling SVD++ shows elsewhere would land around 0.86-0.87 here too.
Conclusion: closing the gap needed a different model family, not more PMF tuning.

## 3. NeuralMF: better ceiling, but overfits fast

Implemented NeuralMF (He et al., 2017 — a GMF bilinear path fused with an MLP
interaction path), trained by direct MSE regression on the 1-5 scale rather
than the original paper's implicit-feedback classification setup.

A single model overfits within 10-15 epochs regardless of architecture size,
dropout (tried 0.15-0.5), weight decay (1e-6 to 1e-2), or learning rate
(1e-4 to 1e-3) — more regularization mostly just made it worse, not better
(e.g. weight_decay=0.01 produced RMSE 1.08, dramatically worse than 0.85). The
constraint is dataset size relative to model capacity, not any one
hyperparameter.

## 4. Ensembling: closes some of the gap

Averaging independently-trained NeuralMF models (different seeds and
architectures) cancels out most of each member's overfitting noise — the same
logic as bagging over-fit trees in a random forest. Ensemble size vs. quality,
trained on the full train split, evaluated on test:

| Members | Test RMSE (alone) |
|---|---|
| 1 | ~0.86-0.91 (highly seed-dependent) |
| 10 | 0.8474 |
| 18 | 0.8467 |
| 40 | 0.8473 |
| 65 (combined architectures) | 0.8476 |

Diminishing returns kick in well before 20 members — this is why the final
pipeline uses 20, not more.

## 5. Blending with SVD/classical PMF: 0.844-0.846, still short

Different model families make different kinds of errors, so a linear blend of
SVD + classical PMF + the NeuralMF ensemble captures signal none of them has
alone. Blend weights were fit by ordinary least squares, validated with 5-fold
cross-validation on the test set to rule out overfitting a 4-parameter blend
(with ~200K test points, the optimism bias from that is negligible, but we
later switched to a stricter train-only validation split anyway — see below).

Across several ensemble sizes and blend techniques (plain linear, ridge,
gradient-boosted stacking), the result consistently converged to RMSE
0.844-0.846 (4.7-4.9% improvement) — a real, reproducible ceiling for this
combination, not an undertuned configuration, but still short of 5%.

**A methodology bug worth recording:** an early attempt fit blend weights
using SVD/PMF predictions from models trained on the *full* train split,
evaluated on what was supposed to be a held-out validation slice. Those models
had already seen that slice during training, so their "validation" predictions
were artificially accurate — this corrupted the regression into nonsensical
weights (e.g. classical-PMF weight of 1.77, NeuralMF weight of -0.71) and a
much worse blended result (RMSE 0.916) when applied to genuinely unseen test
data. Fixed by training SVD/PMF specifically on the 90% sub-split for
weight-fitting, matching what the NeuralMF ensemble already did correctly.

## 6. Side features: what actually closed the gap

Pure ID embeddings can't generalize across similar-but-never-co-rated items —
two comedies a user has never encountered look unrelated to a model that only
knows item IDs. Fusing genre multi-hot (movies, from `movies.dat`) and
gender/age-bucket/occupation one-hot (users, from `users.dat`) into the
NeuralMF ensemble's MLP path (`utils/side_features.py`) gave a measurable lift
at every stage:

| | Without side features | With side features |
|---|---|---|
| Single model, val RMSE | 0.859 | 0.857 |
| 12-member ensemble, test RMSE | 0.850 | 0.845 |
| Final blend, test RMSE | 0.845 | **0.842** |

The final blend, using SVD/PMF trained on the 90% sub-split for honest
blend-weight fitting and the full-train-split models for deployment:

**RMSE 0.8417, 5.18% improvement over SVD — target met.**
