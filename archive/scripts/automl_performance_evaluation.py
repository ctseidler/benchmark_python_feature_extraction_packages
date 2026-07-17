#!/usr/bin/env python3
"""
Pairwise comparison of machine learning algorithms across datasets
using the Demšar-style nonparametric workflow:


- Aggregate repeated runs per dataset x algorithm
- Rank algorithms within each dataset (1 = best)

- Global test: Friedman (Iman-Davenport variant)
- Post-hoc (if global is significant):

  - Default: Holm-adjusted pairwise z-tests on average rank differences (more powerful with small N)
  - Optional: Conover post-hoc (if scikit-posthocs is installed)

  - Optional: Nemenyi critical difference (CD) threshold

Input CSV format (long form) must contain at least:

- dataset: dataset identifier --> dataset
- algorithm: algorithm identifier --> package
- run: run index (repeat number) --> cv_fold
- score: performance metric (consistent direction across datasets) --> balanced_accuracy

Example:
    python compare_algorithms.py data.csv --direction higher --alpha 0.05 --posthoc auto

Dependencies:
    numpy, pandas, scipy
    Optional: statsmodels (for Holm adjustment), scikit-posthocs (for Conover), tabulate (for Markdown tables)

Author: Your friendly stats assistant
"""

import argparse
import sys
from itertools import combinations

import numpy as np
import pandas as pd
from scipy import stats

# Optional imports

try:
    from statsmodels.stats.multitest import multipletests
except ImportError:
    multipletests = None

try:
    # SciPy >= 1.9 has the studentized range distribution
    from scipy.stats import studentized_range
except ImportError:
    studentized_range = None

try:
    import scikit_posthocs as sp
except ImportError:
    sp = None


def main():
    parser = argparse.ArgumentParser(
        description="Compare ML algorithms across datasets using Friedman and post-hoc tests."
    )
    parser.add_argument("csv", help="Path to CSV file with columns: dataset, algorithm, run, score")
    parser.add_argument(
        "--direction",
        choices=["higher", "lower"],
        default="higher",
        help="Metric direction (higher-is-better or lower-is-better)",
    )
    parser.add_argument("--alpha", type=float, default=0.05, help="Significance level")
    parser.add_argument(
        "--agg", choices=["mean", "median"], default="mean", help="Aggregation over runs per dataset x algorithm"
    )
    parser.add_argument(
        "--posthoc",
        choices=["auto", "holm", "conover", "nemenyi"],
        default="auto",
        help="Post-hoc method: 'holm' (pairwise z + Holm), 'conover' (requires scikit-posthocs), 'nemenyi' (CD), or 'auto'",
    )
    args = parser.parse_args()

    # Load data
    df = pd.read_csv(args.csv, sep=";", decimal=",", encoding="utf-8")
    required_cols = {"dataset", "algorithm", "run", "score"}
    if not required_cols.issubset(df.columns):
        sys.exit(f"CSV must contain columns: {required_cols}, found: {df.columns.tolist()}")

    # Aggregate runs
    agg_df = aggregate_runs(df, agg=args.agg)

    # Ranks
    ranks_df, avg_ranks, _ = compute_ranks(agg_df, direction=args.direction)
    N, k = ranks_df.shape
    print(
        f"\nDesign summary: N={N} datasets, k={k} algorithms, aggregated over runs with '{args.agg}', direction='{args.direction}'."
    )

    # Average ranks and mean scores
    mean_scores = agg_df.groupby("algorithm")["score"].mean().reindex(avg_ranks.index)
    summary = pd.DataFrame(
        {
            "algorithm": avg_ranks.index,
            "avg_rank": avg_ranks.values,
            "mean_score": mean_scores.values,
        }
    ).sort_values("avg_rank")
    print_table(summary, title="Average ranks and mean scores")

    # Global test
    fr = friedman_iman_davenport(ranks_df)
    print(
        f"\nGlobal test (Friedman Iman-Davenport): F={fr['F']:.4f}, df1={fr['k']-1}, df2={(fr['k']-1)*(fr['N']-1)}, p-value={fr['pvalue']:.6f}"
    )

    if fr["pvalue"] >= args.alpha:
        print(f"\nNo significant global difference at alpha={args.alpha}. Stop here.")
        return

    # Post-hoc selection
    method = args.posthoc
    if method == "auto":
        # Prefer Conover if scikit-posthocs available; else Holm pairwise z
        method = "conover" if sp is not None else "holm"

    print(f"\nPost-hoc method: {method}")

    if method == "conover":
        con_df = conover_posthoc(agg_df, alpha=args.alpha)
        if con_df is None:
            print("Conover post-hoc not available; falling back to Holm-adjusted pairwise z-tests.")
            method = "holm"
        else:
            # Add rank differences for context
            def rd(a, b):
                return abs(avg_ranks[a] - avg_ranks[b])

            con_df["rank_diff"] = [rd(a, b) for a, b in zip(con_df["alg1"], con_df["alg2"])]
            con_df.sort_values(["p_adj", "rank_diff"], ascending=[True, False], inplace=True)
            print_table(con_df, title="Conover post-hoc (Holm-adjusted) pairwise p-values")
            # Also print significant pairs
            sig = con_df[con_df["significant"]]
            if not sig.empty:
                print_table(sig, title=f"Significant pairs at alpha={args.alpha} (Conover+Holm)")
            else:
                print(f"\nNo significant pairwise differences at alpha={args.alpha} (Conover+Holm).")
            # Also compute and show Nemenyi CD for reference
            CD = nemenyi_cd(N, k, alpha=args.alpha)
            print(f"\nNemenyi critical difference (CD) at alpha={args.alpha}: CD={CD:.4f}")
            return

    if method == "holm":
        pair_df = pairwise_holm_on_ranks(avg_ranks, N, k, alpha=args.alpha)
        print_table(pair_df, title="Pairwise z-tests on average ranks with Holm adjustment")
        sig = pair_df[pair_df["significant"]]
        if not sig.empty:
            print_table(sig, title=f"Significant pairs at alpha={args.alpha} (Holm-adjusted)")
        else:
            print(f"\nNo significant pairwise differences at alpha={args.alpha} (Holm-adjusted).")
        # Also compute Nemenyi CD for reference
        CD = nemenyi_cd(N, k, alpha=args.alpha)
        print(f"\nNemenyi critical difference (CD) at alpha={args.alpha}: CD={CD:.4f}")
        return

    if method == "nemenyi":
        CD = nemenyi_cd(N, k, alpha=args.alpha)
        print(f"\nNemenyi critical difference (CD) at alpha={args.alpha}: CD={CD:.4f}")
        # List pairs exceeding CD
        algs = list(avg_ranks.index)
        pairs = []
        for a, b in combinations(algs, 2):
            diff = abs(avg_ranks[a] - avg_ranks[b])
            pairs.append((a, b, diff, diff > CD))
        nd = pd.DataFrame(pairs, columns=["alg1", "alg2", "rank_diff", "diff_exceeds_CD"])
        nd.sort_values(["diff_exceeds_CD", "rank_diff"], ascending=[False, False], inplace=True)
        print_table(nd, title="Pairs and Nemenyi CD exceedance")
        if not nd[nd["diff_exceeds_CD"]].empty:
            print_table(nd[nd["diff_exceeds_CD"]], title=f"Pairs exceeding CD at alpha={args.alpha}")
        else:
            print("\nNo pairs exceed the Nemenyi CD (results likely conservative with small N).")
        return


def print_table(df, title=None):
    """Print a DataFrame as a Markdown table if possible; otherwise as plain text."""
    if title:
        print(f"\n{title}")
    try:
        # requires 'tabulate' package
        print(df.to_markdown(index=False))
    except ImportError:
        print(df.to_string(index=False))


def holm_adjust(pvals):
    """
    Holm's step-down adjusted p-values without statsmodels.
    Returns array of adjusted p-values preserving input order.
    """
    pvals = np.asarray(pvals, dtype=float)
    m = pvals.size
    order = np.argsort(pvals)
    p_sorted = pvals[order]
    adj_sorted = np.empty(m)

    # Holm step-down: p'_(i) = max_{j<=i} ( (m - j + 1) * p_(j) ), clipped to 1
    cummax = 0.0
    for i in range(m):
        adj_i = (m - i) * p_sorted[i]
        cummax = max(cummax, adj_i)
        adj_sorted[i] = min(cummax, 1.0)

    adj = np.empty(m)
    adj[order] = adj_sorted
    return adj


def aggregate_runs(df, agg="mean"):
    """Aggregate repeated runs per dataset x algorithm."""
    if agg not in ("mean", "median"):
        raise ValueError("agg must be 'mean' or 'median'")
    grouped = df.groupby(["dataset", "algorithm"])["score"]
    agg_df = grouped.mean().reset_index() if agg == "mean" else grouped.median().reset_index()
    return agg_df


def compute_ranks(agg_df, direction="higher"):
    """
    Compute ranks per dataset: 1 = best.
    Returns:
      ranks_df: DataFrame (datasets x algorithms) of ranks
      avg_ranks: Series (algorithms) of average ranks across datasets
    """
    if direction not in ("higher", "lower"):
        raise ValueError("direction must be 'higher' or 'lower'")
    pivot = agg_df.pivot(index="dataset", columns="algorithm", values="score")
    # Check balanced design (no missing)
    if pivot.isna().any().any():
        missing = pivot.isna().sum()
        raise ValueError(f"Missing scores for some dataset x algorithm combinations:\n{missing[missing > 0]}")
    ascending = direction == "lower"
    ranks_df = pivot.rank(axis=1, ascending=not ascending, method="average")
    avg_ranks = ranks_df.mean(axis=0)
    return ranks_df, avg_ranks, pivot


def friedman_iman_davenport(ranks_df):
    """
    Compute Friedman test (Iman-Davenport F).
    Returns dict with N, k, chi2, F, pvalue.
    """
    N = ranks_df.shape[0]  # number of datasets
    k = ranks_df.shape[1]  # number of algorithms
    Rj = ranks_df.sum(axis=0).values  # sum of ranks per algorithm
    chi2 = (12.0 / (N * k * (k + 1))) * np.sum(Rj**2) - 3.0 * N * (k + 1)
    F = ((N - 1) * chi2) / (N * (k - 1) - chi2)
    pval = 1.0 - stats.f.cdf(F, k - 1, (k - 1) * (N - 1))
    return {"N": N, "k": k, "chi2": chi2, "F": F, "pvalue": pval}


def pairwise_holm_on_ranks(avg_ranks, N, k, alpha=0.05):
    """
    Pairwise comparisons using z = rank_diff / SE, with Holm adjustment.
    SE = sqrt(k(k+1)/(6N))
    Returns DataFrame of results.
    """
    SE = np.sqrt(k * (k + 1) / (6.0 * N))
    pairs = []
    algs = list(avg_ranks.index)
    for a, b in combinations(algs, 2):
        diff = abs(avg_ranks[a] - avg_ranks[b])
        z = diff / SE
        p = 2.0 * stats.norm.sf(z)
        pairs.append((a, b, diff, z, p))
    df = pd.DataFrame(pairs, columns=["alg1", "alg2", "rank_diff", "z", "p"])
    if multipletests is not None:
        _, p_adj, _, _ = multipletests(df["p"].values, alpha=alpha, method="holm")
    else:
        p_adj = holm_adjust(df["p"].values)
    df["p_adj"] = p_adj
    df["significant"] = df["p_adj"] < alpha
    df.sort_values(["p_adj", "rank_diff"], ascending=[True, False], inplace=True)
    return df


def nemenyi_cd(N, k, alpha=0.05):
    """
    Compute Nemenyi critical difference: CD = q_alpha * sqrt(k(k+1)/(6N))
    q_alpha is the Studentized range critical value for k groups.
    """
    SE = np.sqrt(k * (k + 1) / (6.0 * N))
    if studentized_range is not None:
        try:
            q = studentized_range.isf(alpha, k, np.inf)  # df = infinity approximation
        except Exception:
            q = studentized_range.isf(alpha, k, 1e9)
    else:
        # Fallback approximation using normal quantile (conservative)
        q = stats.norm.isf(alpha / 2.0) * np.sqrt(2.0)
    CD = q * SE
    return CD


def conover_posthoc(agg_df, alpha=0.05):
    """
    Conover post-hoc after Friedman using scikit-posthocs, if available.
    Returns long-form DataFrame of adjusted p-values with Holm correction.
    """
    if sp is None:
        return None
    try:
        # scikit-posthocs can take long-form data with block design
        pmat = sp.posthoc_conover(
            agg_df,
            y="score",
            x="algorithm",
            block="dataset",
            p_adjust="holm",
        )
        # Melt the symmetric matrix to long form
        p_long = (
            pmat.reset_index()
            .melt(id_vars="index", var_name="alg2", value_name="p_adj")
            .rename(columns={"index": "alg1"})
        )
        # Remove self comparisons and duplicate pairs (keep alg1 < alg2)
        p_long = p_long[p_long["alg1"] < p_long["alg2"]].copy()
        p_long["significant"] = p_long["p_adj"] < alpha
        return p_long
    except ImportError:
        return None


if __name__ == "__main__":
    main()
