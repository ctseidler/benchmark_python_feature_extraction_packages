# https://scikit-posthocs.readthedocs.io/en/latest/tutorial.html#critical-difference-diagrams
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scikit_posthocs as sp
import scipy.stats as ss

FILE = Path(r"D:\csr\Projects\04_Benchmarking\results\12_Rankings") / "AutoML_results.csv"

df = pd.read_csv(FILE, index_col=None, sep=";", decimal=",", encoding="utf-8")
print(df)

# cv_fold = dataset, estimator = package, score = balanced_accuracy
avg_rank = df.groupby("dataset").balanced_accuracy.rank(pct=True).groupby(df.package).mean()
print(avg_rank)

# p-value is only reliable for n > 10 and more than 6 repeated samples
print(ss.friedmanchisquare(*[group["balanced_accuracy"].values for name, group in df.groupby("package")]))

# # Average ordinal ranks (1 = best), higher balanced_accuracy should rank better
# df["rank"] = df.groupby("dataset")["balanced_accuracy"].rank(ascending=False, method="average")
# avg_rank = df.groupby("package")["rank"].mean()
# print(avg_rank)

df["block_id_col"] = df["dataset"] + "_" + df["cv_fold"].astype(str)

test_results = sp.posthoc_conover_friedman(
    df,
    melted=True,
    block_col="dataset",
    block_id_col="block_id_col",
    group_col="package",
    y_col="balanced_accuracy",
)
print(test_results)
plt.figure(figsize=(10, 6), dpi=100)
sp.sign_plot(test_results, cbar_ax_bbox=[0.9, 0.35, 0.04, 0.3])

plt.figure(figsize=(10, 6), dpi=100)
plt.title("Critical difference diagram of average balanced accuracy ranks")
sp.critical_difference_diagram(avg_rank, test_results)
plt.show()

####################################################################################################################
# from pathlib import Path

# import matplotlib.pyplot as plt
# import numpy as np
# import pandas as pd
# import scikit_posthocs as sp


# # Optional: SciPy >= 1.9 has studentized_range; fallback to statsmodels if unavailable
# def get_q_alpha(alpha, k):
#     try:
#         from scipy.stats import studentized_range

#         return studentized_range.isf(alpha, k, np.inf)
#     except Exception:
#         try:
#             from statsmodels.stats.libqsturng import qsturng

#             return qsturng(1 - alpha, k, np.inf)
#         except Exception as e:
#             raise ImportError("Need scipy>=1.9 (studentized_range) or statsmodels (qsturng) to compute q_alpha.") from e


# # ----------------- USER SETTINGS -----------------
# FILE = Path(r"D:\csr\Projects\04_Benchmarking\results\12_Rankings") / "AutoML_results.csv"
# dataset_col = "dataset"  # dataset identifier
# algo_col = "package"  # algorithm identifier
# run_col = "cv_fold"  # run/repetition/fold column (5 runs per dataset-algo)
# score_col = "balanced_accuracy"  # higher is better
# sep = ";"  # CSV separator
# decimal = ","  # decimal separator in your CSV (change to "." if standard)
# encoding = "utf-8"

# alpha = 0.05  # significance level for Nemenyi/CD
# title = "Critical difference diagram (Nemenyi) of average ranks"
# # -------------------------------------------------


# def main():
#     # 1) Read data
#     df = pd.read_csv(FILE, sep=sep, decimal=decimal, encoding=encoding)

#     # Basic checks
#     for c in [dataset_col, algo_col, score_col]:
#         if c not in df.columns:
#             raise ValueError(f"Column '{c}' not found in the input file.")

#     # 2) Aggregate the 5 runs per (dataset, algorithm)
#     #    Use mean across runs so each dataset contributes equally.
#     df_agg = df.groupby([dataset_col, algo_col], as_index=False)[score_col].mean().rename(columns={score_col: "score"})

#     # 3) Ensure complete blocks (every dataset has all algorithms)
#     k = df_agg[algo_col].nunique()
#     counts = df_agg.groupby(dataset_col)[algo_col].nunique()
#     complete_datasets = counts[counts == k].index
#     dropped = set(counts.index) - set(complete_datasets)
#     if dropped:
#         print(f"Warning: Dropping {len(dropped)} incomplete dataset(s) (missing algorithms): {sorted(dropped)}")
#     df_agg = df_agg[df_agg[dataset_col].isin(complete_datasets)]
#     N = df_agg[dataset_col].nunique()

#     if k < 2 or N < 2:
#         raise ValueError(f"Not enough algorithms (k={k}) or datasets (N={N}) after filtering for complete blocks.")

#     # 4) Compute per-dataset ordinal ranks (1=best because higher score is better)
#     df_agg["rank"] = df_agg.groupby(dataset_col)["score"].rank(ascending=False, method="average")

#     # Average ranks per algorithm
#     avg_rank = df_agg.groupby(algo_col)["rank"].mean().sort_values()
#     print(avg_rank)

#     # 5) Nemenyi post-hoc pairwise p-values on aggregated data
#     test_results = sp.posthoc_nemenyi(
#         df_agg,
#         group_col=algo_col,
#         val_col="score",
#     )
#     # Align p-values to the avg_rank order
#     test_results = test_results.loc[avg_rank.index, avg_rank.index]

#     # 6) Compute numeric Critical Difference (CD) for Nemenyi
#     q_alpha = get_q_alpha(alpha, k)
#     cd_value = q_alpha * np.sqrt((k * (k + 1)) / (6.0 * N))
#     print(f"Nemenyi critical difference (alpha={alpha}, k={k}, N={N}): {cd_value:.4f}")

#     # 7) Plot CD diagram
#     plt.figure(figsize=(10, 6), dpi=100)
#     plt.title(title)
#     sp.critical_difference_diagram(avg_rank, test_results)
#     plt.tight_layout()
#     plt.show()


# if __name__ == "__main__":
#     main()
