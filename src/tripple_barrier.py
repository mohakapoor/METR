import polars as pl
import numpy as np
import matplotlib.pyplot as plt
import yaml

CONFIG_PATH = "config.yaml"
with open(CONFIG_PATH, "r") as f:
    config = yaml.safe_load(f)
FEATURES = config["Exposure_Features"]

def generate_barriers(df, k, T):
    labels = []
    opens = df['Open'].to_list()
    high = df['High'].to_list()
    low = df['Low'].to_list()
    vol_20d = df['Vol_20d'].to_list()
    for i in range(len(opens)-T-1):
        label = 0 # base case that T days passed without anything
        p0 = opens[i+1]
        sigma = vol_20d[i]
        p_upper = p0*(1+k*sigma)
        p_lower = p0*(1-k*sigma)

        for j in range(1,T+1):
            if(high[i+j] >= p_upper) and (low[i+j] <= p_lower):
                label = -1 # assuming low was first to be safe
                break
            elif(high[i+j] >= p_upper):
                label = 1 # profit
                break
            elif(low[i+j] <= p_lower):
                label = -1 # stop loss
                break
            else:
                pass
        labels.append(label)
    return labels


MAX_CLASS_PCT = 45.0  # hard filter: no single class above this


def balance_score(pct_neg1, pct_0, pct_pos1):
    """Sum of squared deviations from perfect 33.3%. Lower = more balanced."""
    ideal = 100.0 / 3
    return (pct_neg1 - ideal)**2 + (pct_0 - ideal)**2 + (pct_pos1 - ideal)**2


def passes_filter(pct_neg1, pct_0, pct_pos1):
    """True if no class exceeds MAX_CLASS_PCT."""
    return max(pct_neg1, pct_0, pct_pos1) <= MAX_CLASS_PCT


# ============================================
# 3. GRID EVALUATION + PLOT
# ============================================
K_VALUES = [0.5,0.75,1.0,1.25,1.5,1.75,2.0,2.5,3.0]
T_VALUES = [3, 5,10]


def evaluate_triple_barrier_grid(assets):
    for name, df in assets:
        results = []  # (label_str, pct_neg1, pct_0, pct_pos1, bal_score, passed)

        with open(f"reports/tripple_barrier/{name}_grid_results.txt", "w", encoding="utf-8") as f:
            for k in K_VALUES:
                for t in T_VALUES:
                    labels = generate_barriers(df, k=k, T=t)
                    counts = pl.Series("TB_Label", labels).value_counts().sort("TB_Label")
                    total = counts["count"].sum()
                    counts = counts.with_columns(
                        (pl.col("count") / total * 100).round(1).alias("pct")
                    )

                    # extract percentages (handle missing classes)
                    pct_map = dict(zip(
                        counts["TB_Label"].to_list(),
                        counts["pct"].to_list()
                    ))
                    pct_neg1 = pct_map.get(-1, 0.0)
                    pct_0 = pct_map.get(0, 0.0)
                    pct_pos1 = pct_map.get(1, 0.0)

                    # score
                    bal = balance_score(pct_neg1, pct_0, pct_pos1)
                    passed = passes_filter(pct_neg1, pct_0, pct_pos1)
                    tag = "✓" if passed else "✗"

                    combo_label = f"k={k}, T={t}"
                    results.append((combo_label, pct_neg1, pct_0, pct_pos1, bal, passed))

                    header = f"\n{combo_label}: balance={bal:.1f} [{tag}]"
                    print(f"[{name}] {header}")
                    print(counts)
                    f.write(header + "\n")
                    f.write(str(counts) + "\n")

            # find best combo (lowest balance score among those that pass filter)
            passing = [r for r in results if r[5]]
            if passing:
                best = min(passing, key=lambda x: x[4])
                summary = f"\n{'='*50}\nBEST for {name}: {best[0]}  (balance={best[4]:.1f})\n{'='*50}"
            else:
                best = min(results, key=lambda x: x[4])
                summary = f"\n{'='*50}\nBEST for {name} (no combo passed {MAX_CLASS_PCT}% filter): {best[0]}  (balance={best[4]:.1f})\n{'='*50}"
            print(summary)
            f.write(summary + "\n")

        # --- PLOT ---
        labels_list = [r[0] for r in results]
        pct_neg1 = [r[1] for r in results]
        pct_0 = [r[2] for r in results]
        pct_pos1 = [r[3] for r in results]
        scores = [r[4] for r in results]
        passed_list = [r[5] for r in results]

        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 8), gridspec_kw={'height_ratios': [3, 1]})
        fig.suptitle(f"Triple Barrier Grid — {name.upper()} (max class ≤ {MAX_CLASS_PCT}%)", fontsize=14, fontweight='bold')

        # bar chart: label distribution
        x = np.arange(len(labels_list))
        width = 0.25
        ax1.bar(x - width, pct_neg1, width, label='-1 (stop loss)', color='#e74c3c')
        ax1.bar(x, pct_0, width, label='0 (timeout)', color='#95a5a6')
        ax1.bar(x + width, pct_pos1, width, label='+1 (profit)', color='#2ecc71')
        ax1.axhline(y=MAX_CLASS_PCT, color='black', linestyle='--', alpha=0.5, label=f'{MAX_CLASS_PCT}% cap')
        ax1.set_ylabel('Percentage (%)')
        ax1.set_title('Label Distribution')
        ax1.set_xticks(x)
        ax1.set_xticklabels(labels_list, rotation=45, ha='right', fontsize=7)
        ax1.legend()
        ax1.set_ylim(0, 85)

        # bar chart: balance score (lower = better)
        bar_colors = ['#2ecc71' if p else '#e74c3c' for p in passed_list]
        if any(passed_list):
            best_score = min(s for s, p in zip(scores, passed_list) if p)
            best_idx = next(i for i, (s, p) in enumerate(zip(scores, passed_list)) if p and s == best_score)
            bar_colors[best_idx] = '#f39c12'  # highlight best
        ax2.bar(x, scores, color=bar_colors, width=0.5)
        ax2.set_ylabel('Balance Score')
        ax2.set_title('Distance from Perfect Balance (lower = better, green = passed filter, red = failed)')
        ax2.set_xticks(x)
        ax2.set_xticklabels(labels_list, rotation=45, ha='right', fontsize=7)

        plt.tight_layout()
        plt.savefig(f"reports/tripple_barrier/{name}_grid_plot.png", dpi=150)
        plt.close()
        print(f"Plot saved: reports/tripple_barrier/{name}_grid_plot.png")


# ============================================
# 4. RUN
# ============================================
if __name__ == "__main__":
    df1 = pl.read_parquet(r"data\processed\train\nifty.parquet")
    df2 = pl.read_parquet(r"data\processed\train\gold.parquet")
    df3 = pl.read_parquet(r"data\processed\train\usdinr.parquet")

    assets = [
        ("nifty", df1),
        ("gold", df2),
        ("usdinr", df3),
    ]

    evaluate_triple_barrier_grid(assets)
