# METR — Lab Logs & Observations

This file tracks daily experiments, grid searches, and key insights over the course of the METR project.

---

### 2026-03-18 — Triple Barrier Labeling: Grid Analysis (Nifty)

**What was done:**
Implemented `src/tripple_barrier.py` with a forward-scan labeler. Ran a grid search over `k ∈ [0.5, 0.75, 1.0, 1.5]` × `T ∈ [3, 5, 10]` on Nifty training data (~2447 rows).

**Key Results:**

| k | T | -1 (%) | 0 (%) | +1 (%) | Notes |
|---|---|--------|-------|--------|-------|
| 0.5 | 3 | 63.6 | 0.2 | 36.2 | Barriers too tight — nearly all hit same day |
| 0.75 | 3 | 55.1 | 3.5 | 41.4 | Still tight, heavy -1 skew |
| 1.0 | 3 | 47.9 | 10.4 | 41.6 | Moderate — meaningful timeout class appears |
| 1.5 | 3 | 35.6 | 33.9 | 30.5 | Most balanced 3-class distribution |
| 1.0 | 5 | 51.4 | 2.7 | 45.9 | Longer T reduces timeouts |
| 1.5 | 5 | 43.2 | 14.9 | 41.9 | Good balance with more time |

**Observations:**
1. **Persistent -1 bias across all configurations.** Two causes identified:
   - **Same-day tiebreak:** When both barriers are hit in one day (common at low k), the code defaults to `-1`. At `k=0.5`, this dominates the label.
   - **Market microstructure:** Intraday lows tend to be further from Open than Highs (negative skew), so the lower barrier gets hit first naturally.
2. **k=0.5 and k=0.75 are too tight** — barriers fall inside a single day's range, so the label mostly measures intraday skew, not directional signal.
3. **Increasing T beyond 5 adds almost nothing** — most barriers are hit within the first few days regardless.
4. **Best candidates:** `k=1.0, T=3` (if you want some timeout filtering) or `k=1.5, T=3` (if you want balanced classes).

---

### 2026-03-19 — Triple Barrier Labeling: Multi-Asset Grid Analysis

**What was done:**
Refined `src/tripple_barrier.py` to use a symmetric barrier ($k$) and evaluate over a wider grid ($k \in [0.5, 0.75, ..., 2.5, 3.0]$ and $T \in [3, 5, 10]$).
Replaced feature-label correlation with a **Balance Score** metric to evaluate configurations, given the weak predictive power of the current features.
- **Balance Score** measures the sum of squared deviations from a perfectly balanced distribution (33.3% per class). Lower is better.
- Applied a **hard filter**: configurations where any single class exceeds 45% are rejected to prevent the model from learning a strong majority-class bias.

**Key Results (Mathematical Optimal vs Practical Chosen):**

*1. Mathematically Optimal (pure balance)*
| Asset | k | T | Balance Score | -1 (%) | 0 (%) | +1 (%) |
|---|---|---|---|--------|-------|--------|
| **Nifty** | 2.0 | 5 | 0.4 | 33.7% | 33.5% | 32.8% |
| **Gold** | 2.5 | 5 | 15.4 | 34.8% | 35.0% | 30.1% |
| **USDINR** | 2.0 | 5 | 1.6 | 32.3% | 34.0% | 33.7% |

*2. Practically Chosen (realistic market moves)*
| Asset | k | T | -1 (%) | 0 (%) | +1 (%) |
|---|---|---|--------|-------|--------|
| **Nifty** | 1.5 | 3 | 35.6% | 33.9% | 30.5% |
| **Gold** | 1.75 | 3 | 41.6% | 31.2% | 27.3% |
| **USDINR** | 1.5 | 3 | 34.7% | 31.2% | 34.1% |

**Observations & Final Logic:**
1. **Mathematical Optimization Overfits to Outliers:** While the algorithm found near-perfect mathematical balance at $T=5$ and $k \in \{2.0, 2.5\}$, expecting an asset to move $2.5\sigma$ in just 5 days is statistically highly improbable. Using these parameters would force the XGBoost model to hunt for extreme outlier events rather than tradeable signals.
2. **The $T=3$ Practical Choice:** A shorter time horizon ($T=3$) with tighter, more realistic barriers ($k=1.5$ to $k=1.75$) yields a slightly less "perfect" mathematical score but a vastly superior practical trading goal. It still maintains a healthy ~30-34% timeout class, proving the noise filter is functioning.
3. **Asset Volatility Profile:** Gold requires slightly wider barriers ($k=1.75$) than Nifty and USDINR ($k=1.5$) to achieve this balance, reflecting its higher relative intra-period volatility (longer tails).

**Next Steps:**
- Add `TB_Label` to the data pipeline (`src/data_cleaning.ipynb`), computed using the identified best parameters per asset.
- Retrain XGBoost as a 3-class classifier (`multi:softprob`).
