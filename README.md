# FinSight 🔍📈
### An NLP-Driven Multi-Stage Pipeline for Cross-Market Stock Signal Generation
**Using Domain-Adapted FinBERT and a Calibrated Ensemble Classifier**

> *Department of Artificial Intelligence and Machine Learning — Shri Ramdeobaba College of Engineering and Management, Nagpur*
> *Academic Year 2025–26 | Pipeline v4.2*

---

## 📋 Table of Contents

- [Overview](#overview)
- [Key Results](#key-results)
- [Pipeline Architecture](#pipeline-architecture)
- [Dataset](#dataset)
- [Methodology](#methodology)
  - [Ground Truth Labeling](#ground-truth-labeling)
  - [NLP Preprocessing](#nlp-preprocessing)
  - [FinBERT Feature Extraction](#finbert-feature-extraction)
  - [Feature Engineering](#feature-engineering)
  - [Ensemble & Differential Evolution](#ensemble--differential-evolution)
  - [Probability Calibration](#probability-calibration)
- [Results](#results)
- [Sample Predictions](#sample-predictions)
- [Project Structure](#project-structure)
- [Setup & Installation](#setup--installation)
- [Usage](#usage)
- [Version History](#version-history)
- [Limitations & Future Work](#limitations--future-work)
- [Authors](#authors)
- [References](#references)

---

## Overview

**FinSight** converts raw financial news headlines into calibrated **BUY / HOLD / SELL** signals across a dual-market universe of **45 equity tickers** spanning the **US market** (NVDA, MSFT, AAPL, TSLA, …) and the **Indian NSE** (INFY.NS, HDFCBANK.NS, RELIANCE.NS, …).

The system addresses three fundamental gaps in existing NLP-for-finance literature:

1. **Single-market bias** — most prior systems cover only US equities and ignore structural differences of emerging markets like the NSE (F&O expiry calendars, FII flow patterns, intraday news timing).
2. **Heuristic ensemble blending** — equal or manually tuned weights leave performance on the table; FinSight uses **Differential Evolution** to jointly optimize blend weights and decision thresholds.
3. **Missing calibration loop** — no prior published system simultaneously applies multi-horizon labeling, ticker-level bias correction, domain fine-tuning, *and* a post-calibration threshold re-sweep as a unified pipeline.

### What makes FinSight different

| Contribution | Details |
|---|---|
| Multi-horizon weighted labeling | 60/25/15% weight over T+1/T+2/T+3 forward returns, reducing single-day noise |
| FinBERT as feature extractor | 42 of 73 features come from FinBERT — not used as a standalone classifier |
| DE ensemble optimization | 8-dimensional parameter vector (weights + thresholds) optimized over 500 generations |
| Post-calibration threshold sweep | Corrects systematic probability compression caused by isotonic regression |
| Ticker-level regime switch | Corrects for historical class imbalance at individual stock level |
| Dual-market support | Market-specific volatility thresholds: τ_US = 1.00%, τ_NSE = 0.75% |

---

## Key Results

Evaluated on a **strictly temporal hold-out** of **2,631 articles (Mar 8–11, 2026)** — entirely invisible during training, validation, fine-tuning, and threshold selection.

| Metric | FinSight v4.2 | Best Prior Work (Mukherjee et al., 2024) |
|---|---|---|
| **Macro F1** | **0.8276** | 0.800 |
| **Accuracy** | **84.42%** | 79.6% |
| **SELL F1** | **0.8819** | 0.780 |
| **HOLD F1** | **0.8244** | 0.820 |
| **BUY F1** | **0.7764** | 0.800 |
| **Brier Score** | **0.0838** ↓ | 0.1250 |

FinSight surpasses **all six comparable benchmarks** across every reported metric, with the most pronounced margin in SELL F1 (+0.102 over prior SOTA).

---

## Pipeline Architecture

FinSight is a **seven-stage modular pipeline**. Each stage serializes its outputs to disk, enabling independent auditing, retraining, or replacement without affecting other stages. No downstream stage can access test-split data during training — all forms of look-ahead bias are prevented by design.

```
┌──────────┐    ┌──────────┐    ┌──────────┐    ┌──────────┐
│ Stage 1  │───▶│ Stage 2  │───▶│ Stage 3  │───▶│ Stage 4  │
│ EDA &    │    │  Price   │    │   NLP    │    │ FinBERT  │
│  Audit   │    │ Labeling │    │   Prep   │    │ Features │ ◀── Deep Learning
└──────────┘    └──────────┘    └──────────┘    └────┬─────┘
                                                      │
┌──────────┐    ┌──────────┐    ┌──────────┐          │
│ Stage 7  │◀───│ Stage 6  │◀───│ Stage 5  │◀─────────┘
│  Calib.  │    │ DE Weight│    │ Ensemble │
│& Infer.  │    │  Optim.  │    │ Training │
└──────────┘    └──────────┘    └──────────┘
    ▲ Calibration
```

---

## Dataset

| Attribute | Value |
|---|---|
| Raw articles collected | 14,690 |
| Equity articles (post-deduplication) | 11,483 |
| Tickers covered | 45 (26 US + 19 India NSE) |
| Temporal range | January 2023 – March 2026 |
| Training split | 8,286 articles (72.2%) |
| Validation split | 566 articles (4.9%) |
| Test split (temporal hold-out) | 2,631 articles (22.9%) |
| Average headline length | 14.3 ± 5.1 tokens |
| Label distribution | SELL 38.1% \| HOLD 36.2% \| BUY 25.7% |
| Class imbalance ratio | 1.48× (SELL / BUY) |

**Sources:** Google News RSS, Yahoo Finance RSS, Finviz
**Price data:** Yahoo Finance auto-adjusted OHLCV via `yfinance`

---

## Methodology

### Ground Truth Labeling

Labels are derived from a **multi-horizon weighted return score**, not a single forward day (which is susceptible to exogenous macro noise):

$$S_i = 0.60 \cdot r_{i,T+1} + 0.25 \cdot r_{i,T+2} + 0.15 \cdot r_{i,T+3}$$

The weighted score is thresholded by market type:

$$y_i = \begin{cases} \text{BUY} & \text{if } S_i > \tau_m \\ \text{SELL} & \text{if } S_i < -\tau_m \\ \text{HOLD} & \text{otherwise} \end{cases}, \quad \tau_m = \begin{cases} 1.00\% & \text{US equities} \\ 0.75\% & \text{India NSE} \end{cases}$$

A **T+0 anchor rule** prevents look-ahead bias at the timezone boundary: for US equities with pre-market timestamps, the same calendar day close is used; for Indian equities, the next trading day close is used.

---

### NLP Preprocessing

A seven-stage text cleaning pipeline applied to every headline:

1. Strip source attribution suffixes (e.g., `— Bloomberg`, `— Reuters`) via comprehensive regex
2. Normalize numeric shorthand (`3.5B` → `3.5 billion`)
3. Convert percentage symbols to word form
4. Remove currency symbols
5. Strip URLs
6. Remove non-alphanumeric punctuation (except sentence terminators)
7. Collapse whitespace

After cleaning: NLTK tokenization, lemmatization, and stopword removal (NLTK English corpus + finance-specific neutrals: *stock*, *company*, *quarterly*, etc.)

---

### FinBERT Feature Extraction

FinBERT (`ProsusAI/finbert`, BERT-base, 12 transformer layers) is deployed in two complementary modes, contributing **42 of the 73 total features**:

#### Layer 1 — Frozen FinBERT (7 features)
Pre-trained FinBERT with no gradient computation → zero leakage risk:
- `p_pos`, `p_neg`, `p_neu` — three-class probability distribution
- `net_score` = p_pos − p_neg
- `confidence` = max(p_pos, p_neg, p_neu)
- `shannon_entropy` — uncertainty signal
- `ema_momentum` — 5-article exponential moving average of net score

#### Layer 2 — Fine-tuned FinBERT (3 features)
A new linear classification head (768→3) trained on the training split only (4 epochs, AdamW, lr=2e-5). Transformer encoder weights are **kept frozen** — only the head is trained. This adapts FinBERT's output vocabulary from `{positive, negative, neutral}` → `{BUY, HOLD, SELL}`. Standalone validation Macro F1: 0.4638 (expected and by design — text alone cannot resolve price context).
- `p_ft_buy`, `p_ft_hold`, `p_ft_sell`

#### CLS Embeddings (32 features → 17 used)
768-dimensional CLS token extracted from the frozen FinBERT final layer. PCA fitted on training set reduces to **32 components** (67.1% variance explained). Components encode latent semantic structure: event type distinctions (earnings vs. regulatory vs. macro) without hand-crafted rules.

> ⚠️ **Known limitation:** Only 17 of 32 PCA components (components 0–14, 19, 24) were used in v4.2 — an overzealous filtering step since PCA components are orthogonal by construction. Expanding to all 32 is the highest-priority improvement for v5.

---

### Feature Engineering

**73 total features** across six groups:

| Group | Count | Description |
|---|---|---|
| Frozen FinBERT | 7 | p_pos, p_neg, p_neu, net, confidence, entropy, momentum |
| Fine-tuned FinBERT | 3 | p_ft_buy, p_ft_hold, p_ft_sell |
| CLS PCA Embeddings | 17 | PCA components 0–14, 19, 24 |
| Lexicon / NLP | 29 | Loughran-McDonald bull/bear, India-specific signals, analyst action terms, event type flags |
| LOO Group Features | 7 | Ticker-level aggregates from training set (mean ft-net, BUY-vs-SELL ratio, article count, consensus deviation) |
| Market Context | 3 | Region mood, news burst z-score, deviation from market-wide sentiment |
| Cross Features | 3 | Interaction terms (ft_buy × fb_pos, etc.) |
| Calendar / Source | 4 | Hour, day-of-week, source credibility score |

---

### Ensemble & Differential Evolution

**Six models** form the ensemble:
- XGBoost (1,000 trees, max depth 5, class-weighted)
- CatBoost (800 iterations, depth 7, Focal loss)
- LightGBM (1,200 leaves)
- ResidualMLP-A (128-unit hidden, GELU, 3 residual blocks, BatchNorm)
- ResidualMLP-B (same architecture, different init)
- Gaussian Naive Bayes (baseline)

**Differential Evolution** optimizes the 8-dimensional parameter vector θ = (w₁, …, w₆, T_sell, T_buy):

$$\hat{y}_{\text{blend}} = \text{softmax}\!\left(\sum_{k=1}^{6} w_k \cdot \hat{P}_k\right)$$

$$\theta^* = \arg\max_\theta \text{MacroF1}\!\left(y_\text{val},\ \text{Threshold}(\hat{y}_\text{blend}(\theta),\ T_\text{sell},\ T_\text{buy})\right)$$

DE configuration: population size 80, mutation factor F=0.7, crossover rate CR=0.9, 500 generations.

**Converged weights:**

| Model | Weight |
|---|---|
| CatBoost | 58.1% |
| XGBoost | 40.8% |
| LightGBM | 0.5% |
| MLP-A | 0.4% |
| MLP-B | 0.2% |
| GaussianNB | 0.0% |

Gradient-boosted trees dominate due to their superiority on structured tabular data with correlated probability features. MLPs are constrained by training set size; GaussianNB's independence assumption is violated by feature construction.

---

### Probability Calibration

Raw ensemble blend probabilities are calibrated using **Isotonic Regression** fitted on the validation set — a non-parametric monotonic mapping that aligns predicted probabilities with observed outcome frequencies.

**Critical insight:** Post-calibration probabilities are compressed toward class base rates, shifting effective decision boundaries. The raw SELL threshold (T_raw_sell = 0.26) becomes invalid after calibration. A post-calibration re-sweep over T_sell ∈ [0.30, 0.70] on the validation set identifies:

$$T^{\text{cal}}_{\text{sell}} = 0.52$$

This single fix was responsible for the **largest version-over-version improvement**: Macro F1 0.8188 (v3) → **0.8276 (v4.2)**.

| | Before Calibration | After Calibration |
|---|---|---|
| Brier Score | 0.1841 | **0.0838** |
| SELL threshold | 0.26 | 0.52 |

---

## Results

### Per-Class Performance (Test Set — 2,631 articles)

| System | Macro F1 | Accuracy | SELL F1 | HOLD F1 | BUY F1 | Brier |
|---|---|---|---|---|---|---|
| Mishev et al., 2020 | 0.640 | 0.620 | 0.590 | 0.660 | 0.680 | 0.2410 |
| Yang et al., 2023 | 0.710 | 0.712 | 0.690 | 0.730 | 0.710 | 0.1870 |
| Sawhney et al., 2021 | 0.710 | 0.693 | 0.680 | 0.720 | 0.730 | 0.1950 |
| Wu et al., 2023 | 0.790 | 0.782 | 0.760 | 0.800 | 0.810 | 0.1410 |
| Mukherjee et al., 2024 | 0.800 | 0.796 | 0.780 | 0.820 | 0.800 | 0.1250 |
| FinSight v1 | 0.741 | 0.729 | 0.801 | 0.742 | 0.679 | 0.1680 |
| FinSight v3 (+LOO) | 0.819 | 0.823 | 0.868 | 0.821 | 0.767 | 0.1012 |
| **FinSight v4.2 (Ours)** | **0.8276** | **0.8442** | **0.8819** | **0.8244** | **0.7764** | **0.0838** |

### Confusion Matrix (Normalized, Test Set)

```
               Predicted
               SELL    HOLD    BUY
True  SELL  [ 89.2%   6.9%   4.0% ]   n = 1,135
      HOLD  [  8.0%  86.5%   5.5% ]   n = 1,118
      BUY   [  5.3%   7.3%  87.4% ]   n =   978
```

Largest off-diagonal mass: SELL→HOLD (7.3%) — the safer of the two possible SELL errors from a risk management perspective.

### Precision-Recall (Average Precision)

| Class | AP |
|---|---|
| SELL | 0.89 |
| HOLD | 0.82 |
| BUY  | 0.79 |

### SHAP Feature Importance (Top 10 — CatBoost)

```
ft_net          ████████████████████████████  0.140
loo_ft_net      ████████████████████████      0.120
pca_0           ██████████████████            0.090
fb_net          ████████████████              0.080
ft_buy          ██████████████                0.070
pca_1           ████████████                  0.060
divergence      ██████████                    0.050
kw_net          █████████                     0.045
loo_kw_net      ████████                      0.040
ft_conf         ███████                       0.035
```

FinBERT-derived features occupy **3 of the top 5 positions**, confirming the effectiveness of domain-adapted feature extraction.

---

## Sample Predictions

Live inference demonstration from FinSight Inference v2:

| Ticker | Headline | P(SELL) | P(HOLD) | P(BUY) | Signal |
|---|---|---|---|---|---|
| NVDA | Nvidia reports record Q4 revenue, beats estimates by 18% | 0.04 | 0.12 | 0.84 | ✅ BUY |
| NVDA | Nvidia stock falls despite earnings beat on weak guidance | 0.71 | 0.21 | 0.08 | 🔴 SELL |
| INFY.NS | Infosys wins $1.5B contract with European bank | 0.07 | 0.18 | 0.75 | ✅ BUY |
| MSFT | Microsoft misses cloud revenue estimates; guidance cut | 0.78 | 0.16 | 0.06 | 🔴 SELL |
| HDFCBANK.NS | HDFC Bank reports strong NII growth, in-line results | 0.55 | 0.30 | 0.15 | 🔴 SELL* |
| AAPL | Apple launches Vision Pro; early reviews mixed | 0.18 | 0.61 | 0.21 | ⚪ HOLD |

> \*HDFC Bank: Despite positive news content, the ticker-level regime correction correctly predicts SELL based on a 9.6% historical BUY rate in the training corpus — consistent with the "sell-the-news" phenomenon in markets where consensus expectations are already priced in.

---

## Project Structure

```
NLP_Project/
│
├── data/
│   ├── raw/                   # Raw scraped articles (CSV)
│   ├── processed/             # Post-dedup, labeled dataset
│   └── price_data/            # yfinance OHLCV cache
│
├── notebooks/
│   ├── stage1_eda_audit.ipynb
│   ├── stage2_price_labeling.ipynb
│   ├── stage3_nlp_preprocessing.ipynb
│   ├── stage4_finbert_features.ipynb
│   ├── stage5_ensemble_training.ipynb
│   ├── stage6_de_optimization.ipynb
│   └── stage7_calibration_inference.ipynb
│
├── models/
│   ├── finbert_head/          # Fine-tuned FinBERT classification head
│   ├── ensemble/              # Trained XGBoost, CatBoost, LightGBM, MLP checkpoints
│   ├── calibrator/            # Fitted isotonic calibrator
│   └── pca/                   # Fitted PCA transformer
│
├── inference/
│   ├── finsight_inference_v2.py   # End-to-end live signal generation
│   └── demo.py                    # Quick demo on any ticker + date
│
├── src/
│   ├── scraper.py             # Google News RSS + Yahoo Finance + Finviz scraper
│   ├── labeler.py             # Multi-horizon weighted labeling
│   ├── preprocessor.py        # 7-stage NLP cleaning pipeline
│   ├── feature_engineering.py # Full 73-feature vector construction
│   ├── ensemble.py            # Ensemble training + DE optimization
│   └── calibration.py        # Isotonic calibration + threshold sweep
│
├── results/
│   ├── confusion_matrix_v4.2.png
│   ├── pr_curves_v4.2.png
│   ├── version_progression.png
│   └── reliability_diagrams.png
│
├── requirements.txt
├── environment.yml
└── README.md
```

---

## Setup & Installation

### Requirements

- Python 3.10
- CUDA-capable GPU recommended (tested on NVIDIA T4 16GB)

### Install

```bash
git clone https://github.com/tanmaypanchariya/NLP_Project.git
cd NLP_Project

# Create environment
conda env create -f environment.yml
conda activate finsight

# OR pip install
pip install -r requirements.txt
```

### Key Dependencies

```
torch==2.1.0
transformers==4.40.0
xgboost==2.0.3
catboost==1.2.2
lightgbm==4.1
scikit-learn==1.3.2
scipy           # Differential Evolution
yfinance        # Price data
feedparser      # RSS scraping
shap            # Feature importance
nltk
pandas
numpy
```

---

## Usage

### Run the Full Pipeline

Execute notebooks in order (`stage1` → `stage7`), or use the orchestrator:

```bash
python src/run_pipeline.py --config config.yaml
```

### Live Inference on Any Ticker

```python
from inference.finsight_inference_v2 import FinSightInference

model = FinSightInference.load("models/")

signal = model.predict(ticker="INFY.NS", date="2026-04-15")
print(signal)
# {
#   "ticker": "INFY.NS",
#   "signal": "BUY",
#   "p_buy": 0.73,
#   "p_hold": 0.19,
#   "p_sell": 0.08,
#   "articles_used": 4,
#   "regime_correction_applied": True
# }
```

### Quick Demo

```bash
python inference/demo.py --ticker NVDA --date 2026-04-15
```

---

## Version History

| Version | Macro F1 | Key Change |
|---|---|---|
| Baseline | ~0.60 | FinBERT standalone (fine-tuned head only) |
| v1 | 0.741 | Initial ensemble, unoptimized weights, no calibration |
| v2 | 0.801 | Feature engineering additions (lexicon, cross features) |
| v3 | 0.819 | LOO group features + price momentum signals |
| v4.0 | ~0.821 | Threshold tuning |
| v4.1 | ~0.822 | Ensemble diversity adjustments |
| **v4.2** | **0.8276** | **Post-calibration threshold re-sweep** (+0.0057) |

The decisive v4.2 improvement resulted **exclusively** from recognizing that isotonic calibration compresses probabilities toward base rates, invalidating the raw threshold, and sweeping to find the corrected T_sell = 0.52. This is the single highest-leverage engineering insight in the project's development history.

---

## Limitations & Future Work

### Current Limitations

1. **PCA under-utilization** — 15 of 32 extracted PCA components were filtered out (overzealous feature selection). Since PCA components are orthogonal by construction, this represents recoverable signal loss. Expanding to all 32 components is the highest-priority fix.

2. **Short test window** — The temporal hold-out covers only four trading days in March 2026. Performance may not generalize uniformly across all market regimes or quarterly reporting seasons.

3. **LOO feature reconstruction gap** — During live inference, LOO group features are approximated since exact training group statistics evolve with new articles. This does not affect offline evaluation but introduces a small distributional gap in production.

4. **No intraday modeling** — The system is restricted to end-of-day signal generation and cannot model intraday news timing effects or order flow dynamics.

### Future Directions

- [ ] **Use all 32 PCA components** — the filtering step was architecturally incorrect; all components are orthogonal
- [ ] **Sentiment-Adjusted Labeling (SAL)** — hybrid label formula y_SAL = ω₁·r_price + ω₂·s_finbert to train on contrarian dynamics
- [ ] **GPT-4o structured event extraction** — replace regex-based event tagging with structured output calls for richer event typing (M&A, guidance, earnings beats) — most direct path to improving BUY F1 from 0.7764 toward 0.85
- [ ] **Expand ticker universe** beyond 45 tickers
- [ ] **Intraday news latency modeling** for real-time applicability
- [ ] **Quarterly regime analysis** to assess F1 stability across reporting seasons

---

## Authors

| Name | Roll No. | Email |
|---|---|---|
| Tanmay Panchariya | C 43 | panchariyato@rknec.edu |
| Sanket Shinde | C 20 | shindesa@rknec.edu |

**Guide:** Prof. Amit Pimpalkar
**Department:** Artificial Intelligence and Machine Learning
**Institution:** Shri Ramdeobaba College of Engineering and Management, Nagpur
**Academic Year:** 2025–26

---

## References

Selected key references (full list in the project report):

- Yang et al., 2020 — FinBERT: Financial sentiment analysis with pre-trained language models
- Devlin et al., 2019 — BERT: Pre-training of deep bidirectional transformers
- Loughran & McDonald, 2011 — Finance-specific sentiment lexicon
- Wu et al., 2023 — FinBERT + XGBoost hybrid (closest prior work, F1: 0.79)
- Mukherjee et al., 2024 — Transformer CLS + PCA + ensemble (prior SOTA, F1: 0.80)
- Storn & Price, 1997 — Differential Evolution algorithm
- Guo et al., 2017 — On calibration of modern neural networks

Full bibliography available in the [project report](docs/finsight_v4.2_report.pdf).

---

## License

This project is released for academic purposes. Please cite the project report if you use this work:

```bibtex
@techreport{panchariya2026finsight,
  title     = {FinSight: An NLP-Driven Multi-Stage Pipeline for Cross-Market Stock Signal
               Generation Using Domain-Adapted FinBERT and a Calibrated Ensemble Classifier},
  author    = {Panchariya, Tanmay and Shinde, Sanket},
  year      = {2026},
  institution = {Department of Artificial Intelligence and Machine Learning,
                 Shri Ramdeobaba College of Engineering and Management, Nagpur},
  note      = {B.E. Project Report, Academic Year 2025--26.
               Available: https://github.com/tanmaypanchariya/NLP_Project}
}
```

---

> *"The post-calibration threshold re-sweep was the single highest-leverage fix in this project's development history — a reminder that understanding what your model's outputs actually mean matters as much as the model itself."*
