import polars as pl
import numpy as np
import matplotlib.pyplot as plt
import yaml

# CONFIG_PATH = "../config.yaml"
# with open(CONFIG_PATH, "r") as f:
#     config = yaml.safe_load(f)
# FEATURES = config["Exposure_Features"]

def generate_barriers(df, k, T):
    n = len(df)
    labels = [None] * n  # Pad with None so length matches df exactly
    returns = [None] * n # same
    
    opens = df['Open'].to_numpy()
    close_prices = df['Close'].to_numpy()
    high = df['High'].to_numpy()
    low = df['Low'].to_numpy()
    vol_20d = df['Vol_20d'].to_numpy()
    
    for i in range(n - T - 1):
        if opens[i+1] == 0 or np.isnan(vol_20d[i]):
            continue # Skip invalid rows
            
        label = 0 # base case that T days passed without anything
        p0 = opens[i+1]
        sigma = vol_20d[i]
        p_upper = p0*(1+k*sigma)
        p_lower = p0*(1-k*sigma)
        
        if i+1 < n and opens[i+1] != 0:
            returns[i] = (close_prices[i+T] - opens[i+1]) / opens[i+1]
            
        for j in range(1, T+1):
            if i+j >= n: # Safety bounds check
                break
                
            if (high[i+j] >= p_upper) and (low[i+j] <= p_lower):
                label = 0 # marking as neutral
                break
            elif (high[i+j] >= p_upper):
                label = 1 # profit
                break
            elif (low[i+j] <= p_lower):
                label = -1 # stop loss
                break
                
        labels[i] = label
        
    return labels, returns


MAX_CLASS_PCT = 45.0  # hard filter: no single class above this


def balance_score(pct_neg1, pct_0, pct_pos1):
    """Sum of squared deviations from perfect 33.3%. Lower = more balanced."""
    ideal = 100.0 / 3
    return (pct_neg1 - ideal)**2 + (pct_0 - ideal)**2 + (pct_pos1 - ideal)**2


def passes_filter(pct_neg1, pct_0, pct_pos1):
    """True if no class exceeds MAX_CLASS_PCT."""
    return max(pct_neg1, pct_0, pct_pos1) <= MAX_CLASS_PCT



K_VALUES = [0.5,0.75,1.0,1.25,1.5,1.75,2.0,2.5,3.0]
T_VALUES = [3, 5, 7, 10]  

# After running multiple T values i realised T = 5 is the best because higher values dont offer significantly better improvements and add ambiguity 


def evaluate_triple_barrier_grid(assets):
    for name, df in assets:
        results = []  # (label_str, pct_neg1, pct_0, pct_pos1, bal_score, passed, t, return_spread)

        with open(f"reports/tripple_barrier/{name}_grid_results.txt", "w", encoding="utf-8") as f:
            f.write(name.upper() + ":\n")
            for k in K_VALUES:
                for t in T_VALUES:
                    
                    labels, returns = generate_barriers(df, k=k, T=t)
                    
                    # Compute aggregations skipping None values at the end
                    res_df = pl.DataFrame({"TB_Label": labels, "return": returns}).drop_nulls("TB_Label")
                    
                    counts = (
                        res_df.group_by("TB_Label")
                        .agg([
                            pl.len().alias("count"),
                            (pl.col("return").mean() * 100).alias("mean_return(%)") #make it %
                        ])
                        .sort("TB_Label")
                    )
                    
                    total = res_df.height
                    counts = counts.with_columns(
                        (pl.col("count") / total * 100).round(1).alias("pct")
                    )

                    # Extract percentages
                    pct_map = dict(zip(counts["TB_Label"].to_list(), counts["pct"].to_list()))
                    pct_neg1 = pct_map.get(-1, 0.0)
                    pct_0 = pct_map.get(0, 0.0)
                    pct_pos1 = pct_map.get(1, 0.0)
                    
                    # Extract mean returns for the spread calculation
                    ret_map = dict(zip(counts["TB_Label"].to_list(), counts["mean_return(%)"].to_list()))
                    mean_ret_neg1 = ret_map.get(-1, 0.0)
                    mean_ret_pos1 = ret_map.get(1, 0.0)
                    return_spread = mean_ret_pos1 - mean_ret_neg1

                    # Score
                    bal = balance_score(pct_neg1, pct_0, pct_pos1)
                    passed = passes_filter(pct_neg1, pct_0, pct_pos1)


                    combo_label = f"k={k}, T={t}"
                    results.append((combo_label, pct_neg1, pct_0, pct_pos1, bal, passed, t, return_spread))

                    header = f"\n{combo_label}: balance={bal:.1f} | spread={return_spread:.2f}%"
                    print(f"[{name}] {header}")
                    
                    # Print and format DataFrame nicely with only 3 decimal spots for return
                    out_counts = counts.select([
                        pl.col("TB_Label").alias("label"),
                        pl.col("count"),
                        pl.col("pct").alias("ratio(%)"),
                        pl.col("mean_return(%)").round(3)
                    ])
                    print(out_counts)
                    f.write(header + "\n")
                    f.write(str(out_counts) + "\n")

        # --- PLOT (Individual per T) ---
        for t_val in T_VALUES:
            t_results = [r for r in results if r[6] == t_val]
            if not t_results:
                continue

            labels_list = [r[0] for r in t_results]
            pct_neg1 = [r[1] for r in t_results]
            pct_0 = [r[2] for r in t_results]
            pct_pos1 = [r[3] for r in t_results]
            scores = [r[4] for r in t_results]
            passed_list = [r[5] for r in t_results]
            spreads = [r[7] for r in t_results]

            fig = plt.figure(figsize=(15, 8))
            gs = fig.add_gridspec(2, 2, width_ratios=[1, 1], height_ratios=[1, 1])
            fig.suptitle(f"Triple Barrier Grid (T={t_val}) — {name.upper()} (max class ≤ {MAX_CLASS_PCT}%)", fontsize=16, fontweight='bold')

            ax1 = fig.add_subplot(gs[0, 0])
            ax2 = fig.add_subplot(gs[1, 0])
            ax3 = fig.add_subplot(gs[:, 1])

            # ax1: label distribution
            x = np.arange(len(labels_list))
            width = 0.25
            ax1.bar(x - width, pct_neg1, width, label='-1 (stop loss)', color='#e74c3c')
            ax1.bar(x, pct_0, width, label='0 (timeout)', color='#95a5a6')
            ax1.bar(x + width, pct_pos1, width, label='+1 (profit)', color='#2ecc71')
            ax1.axhline(y=MAX_CLASS_PCT, color='black', linestyle='--', alpha=0.5, label=f'{MAX_CLASS_PCT}% cap')
            ax1.set_ylabel('Percentage (%)')
            ax1.set_title('Label Distribution')
            ax1.set_xticks(x)
            ax1.set_xticklabels(labels_list, rotation=45, ha='right', fontsize=8)
            ax1.legend()
            ax1.set_ylim(0, 85)

            # ax2: balance score
            bar_colors = ['#2ecc71' if p else '#e74c3c' for p in passed_list]
            if any(passed_list):
                passed_scores = [s for s, p in zip(scores, passed_list) if p]
                if passed_scores:
                    best_score = min(passed_scores)
                    best_idx = next(i for i, (s, p) in enumerate(zip(scores, passed_list)) if p and s == best_score)
                    bar_colors[best_idx] = '#f39c12'
            ax2.bar(x, scores, color=bar_colors, width=0.5)
            ax2.set_ylabel('Balance Score')
            ax2.set_title('Distance from Perfect Balance (lower = better)')
            ax2.set_xticks(x)
            ax2.set_xticklabels(labels_list, rotation=45, ha='right', fontsize=8)

            # ax3: return spread
            ax3.plot(x, spreads, marker='o', color='#3498db', linewidth=2)
            ax3.axhline(0, color='gray', linestyle='--', alpha=0.5)
            ax3.set_ylabel('Spread (%)')
            ax3.set_title('Return Spread (+1 mean vs -1 mean)')
            ax3.set_xticks(x)
            ax3.set_xticklabels(labels_list, rotation=45, ha='right', fontsize=8)

            plt.tight_layout()
            out_path = f"reports/tripple_barrier/{name}_grid_results_t{t_val}.png"
            plt.savefig(out_path, dpi=150)
            plt.close()
            print(f"Plot saved: {out_path}")



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
