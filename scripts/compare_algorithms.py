#!/usr/bin/env python3
"""
Compare ML algorithms across datasets using:

- Nonparametric (Demšar-style): Friedman (Iman–Davenport) + post-hoc
- Mixed-effects model on raw scores: algorithm fixed effect, dataset random intercept

Usage:
    python compare_algorithms.py master_table.csv --mode mixed --direction higher --alpha 0.05
    python compare_algorithms.py master_table.csv --classifier tpot --score-col balanced_accuracy
    python compare_algorithms.py master_table.csv --summary-only --summary-out summary.csv

CSV columns (long form, required):

- dataset: dataset identifier
- algorithm: algorithm identifier (feature-extraction package/config)
- run: run index (repeat number)
- classifier: classifier identifier (tpot, random_forest, xgboost, logreg, svm)
- <score-col>: performance metric (default balanced_accuracy; also accuracy, f1_score)

The analysis is run separately per classifier (use --classifier to select one,
otherwise every classifier is analysed in turn). Algorithms missing on some
datasets are dropped to keep a complete-block design for the Friedman test.

Notes:

- Mixed-effects model uses raw scores (not aggregated).
- If direction='lower', scores are multiplied by -1 so "higher is better".

- Global test for mixed model is LRT (ML fit) comparing full vs null (no algorithm effect).
- Pairwise contrasts use fixed-effects covariance (normal approx) with Holm correction.

"""

import argparse
import sys
from itertools import combinations
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats
from statsmodels.regression.mixed_linear_model import MixedLM

# Optional imports (each is a heavy/optional dep; fall back to None and the code
# branches on availability). The per-line ignores are the standard pattern for
# optional imports: the None assignment is intentional on the fallback branch.
try:
    from statsmodels.stats.multitest import multipletests
except Exception:
    multipletests = None  # type: ignore

try:
    # SciPy >= 1.9 has the studentized range distribution
    from scipy.stats import studentized_range
except Exception:
    studentized_range = None  # type: ignore

try:
    import scikit_posthocs as sp  # type: ignore
except Exception:
    sp = None


def print_table(df, title=None):
    """Print a DataFrame as a Markdown table if possible; otherwise as plain text."""
    if title:
        print(f"\n{title}")
    try:
        print(df.to_markdown(index=False))
    except Exception:
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

    cummax = 0.0
    for i in range(m):
        adj_i = (m - i) * p_sorted[i]
        cummax = max(cummax, adj_i)
        adj_sorted[i] = min(cummax, 1.0)

    adj = np.empty(m)
    adj[order] = adj_sorted
    return adj


def aggregate_runs(df, agg="mean"):
    """Aggregate repeated runs per dataset × algorithm."""
    if agg not in ("mean", "median"):
        raise ValueError("agg must be 'mean' or 'median'")
    grouped = df.groupby(["dataset", "algorithm"])["score"]
    agg_df = grouped.mean().reset_index() if agg == "mean" else grouped.median().reset_index()
    return agg_df


def compute_ranks(agg_df, direction="higher"):
    """
    Compute ranks per dataset: 1 = best.
    Returns:
      ranks_df: DataFrame (datasets × algorithms) of ranks
      avg_ranks: Series (algorithms) of average ranks across datasets
    """
    if direction not in ("higher", "lower"):
        raise ValueError("direction must be 'higher' or 'lower'")
    pivot = agg_df.pivot(index="dataset", columns="algorithm", values="score")
    if pivot.isna().any().any():
        missing = pivot.isna().sum()
        raise ValueError(
            f"Missing scores for some dataset×algorithm combinations:\n{missing[missing > 0]}"
        )
    ascending = direction == "lower"
    ranks_df = pivot.rank(axis=1, ascending=not ascending, method="average")
    avg_ranks = ranks_df.mean(axis=0)
    return ranks_df, avg_ranks, pivot


def friedman_iman_davenport(ranks_df):
    """
    Compute Friedman test (Iman–Davenport F).
    Returns dict with N, k, chi2, F, pvalue.
    """
    N = ranks_df.shape[0]
    k = ranks_df.shape[1]
    Rj = ranks_df.sum(axis=0).values
    chi2 = (12.0 / (N * k * (k + 1))) * np.sum(Rj**2) - 3.0 * N * (k + 1)
    F = ((N - 1) * chi2) / (N * (k - 1) - chi2)
    pval = 1.0 - stats.f.cdf(F, k - 1, (k - 1) * (N - 1))
    return {"N": N, "k": k, "chi2": chi2, "F": F, "pvalue": pval}


def pairwise_holm_on_ranks(avg_ranks, N, k, alpha=0.05):
    """
    Pairwise comparisons using z = rank_diff / SE, with Holm adjustment.
    SE = sqrt(k(k+1)/(6N))
    """
    SE = np.sqrt(k * (k + 1) / (6.0 * N))
    pairs = []
    algs = list(avg_ranks.index)
    for a, b in combinations(algs, 2):
        diff = abs(avg_ranks[a] - avg_ranks[b])
        z = diff / SE
        p = 2.0 * stats.norm.sf(z)
        pairs.append((a, b, diff, z, p))
    df = pd.DataFrame(pairs, columns=pd.Index(["alg1", "alg2", "rank_diff", "z", "p"]))
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
    """
    SE = np.sqrt(k * (k + 1) / (6.0 * N))
    if studentized_range is not None:
        try:
            q = studentized_range.isf(alpha, k, np.inf)
        except Exception:
            q = studentized_range.isf(alpha, k, 1e9)
    else:
        q = stats.norm.isf(alpha / 2.0) * np.sqrt(2.0)
    CD = q * SE
    return CD


def conover_posthoc(agg_df, alpha=0.05):
    """
    Conover post-hoc after Friedman using scikit-posthocs, if available.
    """
    if sp is None:
        return None
    try:
        pmat = sp.posthoc_conover(
            agg_df,
            y="score",
            x="algorithm",
            block="dataset",
            p_adjust="holm",
        )
        p_long = (
            pmat.reset_index()
            .melt(id_vars="index", var_name="alg2", value_name="p_adj")
            .rename(columns={"index": "alg1"})
        )
        p_long = p_long[p_long["alg1"] < p_long["alg2"]].copy()
        p_long["significant"] = p_long["p_adj"] < alpha
        return p_long
    except Exception:
        return None


# -------------------------

# Mixed-effects model block

# -------------------------


def mixed_global_lrt(df, direction="higher"):
    """
    Global likelihood-ratio test for algorithm effect in MixedLM:

    - Full: score ~ C(algorithm) + (1|dataset)
    - Null: score ~ 1 + (1|dataset)

    Fit with ML (reml=False) for valid LRT.
    Returns: dict with llf_full, llf_null, LR, df, pvalue, k_alg
    """
    data = df.copy()
    if direction == "lower":
        data["score"] = -data["score"]

    # Ensure categorical
    data["algorithm"] = data["algorithm"].astype("category")
    data["dataset"] = data["dataset"].astype("category")

    # Full model (ML)
    try:
        full = MixedLM.from_formula(
            "score ~ C(algorithm)", groups="dataset", re_formula="1", data=data
        )
        full_res = full.fit(method="lbfgs", reml=False)
    except Exception as e:
        raise RuntimeError(f"MixedLM full model failed: {e}")

    # Null model (ML)
    try:
        null = MixedLM.from_formula("score ~ 1", groups="dataset", re_formula="1", data=data)
        null_res = null.fit(method="lbfgs", reml=False)
    except Exception as e:
        raise RuntimeError(f"MixedLM null model failed: {e}")

    llf_full = full_res.llf
    llf_null = null_res.llf
    LR = 2.0 * (llf_full - llf_null)
    k_alg = data["algorithm"].nunique()
    df_lr = k_alg - 1
    pval = 1.0 - stats.chi2.cdf(LR, df_lr)
    return {
        "llf_full": llf_full,
        "llf_null": llf_null,
        "LR": LR,
        "df": df_lr,
        "pvalue": pval,
        "k_alg": k_alg,
        "full_res_ml": full_res,
    }


def mixed_emm_and_pairs(df, direction="higher", alpha=0.05, use_reml=True):
    """
    Fit MixedLM and compute:

      - Estimated marginal means (EMM) per algorithm (fixed effects)
      - Pairwise contrasts with Holm-adjusted p-values

    Uses REML by default for estimates; normal approx for p-values.
    Returns (emm_df, pairs_df, model_result)
    """
    data = df.copy()
    if direction == "lower":
        data["score"] = -data["score"]

    data["algorithm"] = data["algorithm"].astype("category")
    data["dataset"] = data["dataset"].astype("category")

    try:
        res = MixedLM.from_formula(
            "score ~ C(algorithm)", groups="dataset", re_formula="1", data=data
        ).fit(method="lbfgs", reml=use_reml)
    except Exception as e:
        raise RuntimeError(f"MixedLM fit failed: {e}")

    params = res.params
    cov = res.cov_params()
    algs = list(data["algorithm"].cat.categories)
    # Identify baseline algorithm (the one without a dummy coefficient)
    dummy_prefix = "C(algorithm)[T."
    dummies = [p for p in params.index if p.startswith(dummy_prefix)]
    dummy_algs = [p[len(dummy_prefix) : -1] for p in dummies]  # strip prefix and trailing ']'
    base_alg = next(a for a in algs if a not in dummy_algs)

    # Helper to build weight vector for algorithm mean (Intercept + dummy if not base)
    def w_for_alg(a):
        w = pd.Series(0.0, index=params.index)
        w["Intercept"] = 1.0
        if a != base_alg:
            key = f"{dummy_prefix}{a}]"
            if key not in w.index:
                raise KeyError(f"Missing parameter for algorithm {a}")
            w[key] = 1.0
        return w

    # EMM per algorithm
    rows = []
    for a in algs:
        w = w_for_alg(a)
        est = float(np.dot(w, params))
        var = float(np.dot(np.dot(w, cov), w))
        se = np.sqrt(max(var, 0.0))
        zcrit = stats.norm.isf(alpha / 2.0)
        ci_lo = est - zcrit * se
        ci_hi = est + zcrit * se
        rows.append((a, est, se, ci_lo, ci_hi))
    emm_df = pd.DataFrame(
        rows, columns=pd.Index(["algorithm", "emm", "se", "ci_lo", "ci_hi"])
    ).sort_values("emm", ascending=False)

    # Pairwise contrasts
    pairs = []
    for a, b in combinations(algs, 2):
        w_a = w_for_alg(a)
        w_b = w_for_alg(b)
        c = w_a - w_b  # intercept cancels
        diff = float(np.dot(c, params))
        var = float(np.dot(np.dot(c, cov), c))
        se = np.sqrt(max(var, 0.0))
        z = diff / se if se > 0 else np.nan
        p = 2.0 * stats.norm.sf(abs(z)) if np.isfinite(z) else np.nan
        pairs.append((a, b, diff, se, z, p))
    pairs_df = pd.DataFrame(pairs, columns=pd.Index(["alg1", "alg2", "diff", "se", "z", "p"]))

    # Holm adjustment
    pvals = pairs_df["p"].values
    if multipletests is not None and np.isfinite(pvals).all():
        _, p_adj, _, _ = multipletests(pvals, alpha=alpha, method="holm")
    else:
        p_adj = holm_adjust(np.nan_to_num(pvals, nan=1.0))
    pairs_df["p_adj"] = p_adj
    pairs_df["significant"] = pairs_df["p_adj"] < alpha
    pairs_df.sort_values(["p_adj", "diff"], ascending=[True, False], inplace=True)

    return emm_df, pairs_df, res, base_alg


def summary_table(df, score_col="balanced_accuracy"):
    """Mean ± std per (classifier, dataset, algorithm) across runs.

    Produces the per-package summary table for the paper: for each combination
    of classifier, dataset and feature-extraction algorithm, the mean and
    standard deviation of ``score_col`` over the repeated runs.
    """
    g = df.groupby(["classifier", "dataset", "algorithm"])[score_col]
    summary = g.agg(["mean", "std", "count"]).reset_index()
    summary["mean_std"] = (
        summary["mean"].map(lambda x: f"{x:.4f}") + " ± " + summary["std"].map(lambda x: f"{x:.4f}")
    )
    summary = summary.sort_values(["classifier", "dataset", "mean"], ascending=[True, True, False])
    return summary


def _enforce_complete_blocks(df):
    """Drop algorithms not present in every dataset (complete-block design).

    The Friedman test and the ranking diagram require a complete block design:
    every algorithm must be evaluated on every dataset. Some algorithms are
    missing on individual datasets (e.g. kats/tsfeatures on Bosch CNC), so this
    helper drops any algorithm that has at least one missing dataset and prints
    which ones were removed.
    """
    pivot = df.pivot_table(index="dataset", columns="algorithm", values="score", aggfunc="first")
    missing = pivot.isna()
    if not missing.any().any():
        return df
    complete_algs = pivot.columns[~missing.any(axis=0)].tolist()
    dropped = sorted(set(pivot.columns) - set(complete_algs))
    print(
        "  NOTE: dropping algorithms not present in all datasets "
        f"(complete-block design): {dropped}"
    )
    return df[df["algorithm"].isin(complete_algs)].copy()


def run_analysis(df, args, classifier_label):
    """Run the nonparametric and/or mixed-effects analysis for ONE classifier.

    ``df`` must already be filtered to a single classifier and must contain a
    ``score`` column. The complete-block constraint is enforced inside.
    """
    df = _enforce_complete_blocks(df)

    n_datasets = df["dataset"].nunique()
    n_algs = df["algorithm"].nunique()
    print(
        f"\nDesign summary: datasets={n_datasets}, algorithms={n_algs}, "
        f"runs per dataset×algorithm (variable), direction='{args.direction}'."
    )

    # -----------------
    # Nonparametric path
    # -----------------
    if args.mode in ("nonparam", "both"):
        agg_df = aggregate_runs(df, agg=args.agg)
        ranks_df, avg_ranks, pivot_scores = compute_ranks(agg_df, direction=args.direction)
        N, k = ranks_df.shape
        print_table(
            pd.DataFrame(
                {
                    "algorithm": avg_ranks.index,
                    "avg_rank": avg_ranks.values,
                    "mean_score": agg_df.groupby("algorithm")["score"]
                    .mean()
                    .reindex(avg_ranks.index)
                    .values,
                }
            ).sort_values("avg_rank"),
            title="Nonparametric: Average ranks and mean aggregated scores",
        )

        fr = friedman_iman_davenport(ranks_df)
        print(
            "\nNonparametric global (Friedman Iman–Davenport): "
            f"F={fr['F']:.4f}, df1={fr['k'] - 1}, "
            f"df2={(fr['k'] - 1) * (fr['N'] - 1)}, "
            f"p-value={fr['pvalue']:.6f}"
        )

        if fr["pvalue"] < args.alpha:
            method = args.posthoc
            if method == "auto":
                method = "conover" if sp is not None else "holm"
            print(f"\nNonparametric post-hoc method: {method}")
            if method == "conover":
                con_df = conover_posthoc(agg_df, alpha=args.alpha)
                if con_df is None:
                    print(
                        "Conover post-hoc not available; falling back to "
                        "Holm-adjusted pairwise z-tests."
                    )
                    method = "holm"
                else:
                    # Add rank differences for context
                    def rd(a, b):
                        return abs(avg_ranks[a] - avg_ranks[b])

                    con_df["rank_diff"] = [rd(a, b) for a, b in zip(con_df["alg1"], con_df["alg2"])]
                    con_df.sort_values(
                        ["p_adj", "rank_diff"], ascending=[True, False], inplace=True
                    )
                    print_table(con_df, title="Conover post-hoc (Holm-adjusted) pairwise p-values")
                    sig = con_df[con_df["significant"]]
                    if not sig.empty:
                        print_table(
                            sig, title=f"Significant pairs at alpha={args.alpha} (Conover+Holm)"
                        )
                    else:
                        print(
                            "\nNo significant pairwise differences at "
                            f"alpha={args.alpha} (Conover+Holm)."
                        )
            if method == "holm":
                pair_df = pairwise_holm_on_ranks(avg_ranks, N, k, alpha=args.alpha)
                print_table(pair_df, title="Pairwise z-tests on average ranks with Holm adjustment")
                sig = pair_df[pair_df["significant"]]
                if not sig.empty:
                    print_table(
                        sig, title=f"Significant pairs at alpha={args.alpha} (Holm-adjusted)"
                    )
                else:
                    print(
                        "\nNo significant pairwise differences at "
                        f"alpha={args.alpha} (Holm-adjusted)."
                    )
            if method == "nemenyi":
                CD = nemenyi_cd(N, k, alpha=args.alpha)
                print(f"\nNemenyi critical difference (CD) at alpha={args.alpha}: CD={CD:.4f}")
                algs = list(avg_ranks.index)
                nd = pd.DataFrame(
                    [
                        (
                            a,
                            b,
                            abs(avg_ranks[a] - avg_ranks[b]),
                            abs(avg_ranks[a] - avg_ranks[b]) > CD,
                        )
                        for a, b in combinations(algs, 2)
                    ],
                    columns=pd.Index(["alg1", "alg2", "rank_diff", "diff_exceeds_CD"]),
                ).sort_values(["diff_exceeds_CD", "rank_diff"], ascending=[False, False])
                print_table(nd, title="Pairs and Nemenyi CD exceedance")
        else:
            print(
                "\nNonparametric global test not significant at "
                f"alpha={args.alpha}; post-hoc comparisons typically "
                "not pursued."
            )

    # ----------------
    # Mixed-effects path
    # ----------------
    if args.mode in ("mixed", "both"):
        data = df.copy()
        if args.direction == "lower":
            data["score"] = -data["score"]
        data["algorithm"] = data["algorithm"].astype("category")
        data["dataset"] = data["dataset"].astype("category")

        res = MixedLM.from_formula(
            "score ~ C(algorithm)", groups="dataset", re_formula="1", data=data
        ).fit()

        # Joint Wald test: all algorithm dummies = 0

        params = res.params
        cov = res.cov_params()
        L_rows = []
        param_names = list(params.index)
        for name in param_names:
            if name.startswith("C(algorithm)[T."):
                row = np.zeros(len(param_names))
                row[param_names.index(name)] = 1.0
                L_rows.append(row)
        L = np.vstack(L_rows)  # shape: (k-1, p)

        wald_stat = float(L @ params @ np.linalg.inv(L @ cov @ L.T) @ (L @ params))
        df_wald = L.shape[0]
        pval = 1.0 - stats.chi2.cdf(wald_stat, df_wald)
        print(
            f"Wald test (algorithms jointly = 0): chi2={wald_stat:.4f}, df={df_wald}, p={pval:.6f}"
        )

        # EMM and pairwise contrasts (REML for estimates)
        try:
            emm_df, pairs_df, res_reml, base_alg = mixed_emm_and_pairs(
                df, direction=args.direction, alpha=args.alpha, use_reml=True
            )
        except RuntimeError as e:
            print(f"\nMixed-effects pairwise contrasts failed: {e}")
            return

        print_table(emm_df, title=f"Mixed-effects EMM per algorithm (baseline='{base_alg}', REML)")

        print_table(
            pairs_df, title=f"Mixed-effects pairwise contrasts (Holm-adjusted), alpha={args.alpha}"
        )

        sig = pairs_df[pairs_df["significant"]]
        if not sig.empty:
            print_table(sig, title=f"Mixed-effects significant pairs at alpha={args.alpha}")
        else:
            print(f"\nNo significant mixed-effects pairwise differences at alpha={args.alpha}.")

        # Optional: print variance components
        try:
            var_comp = pd.DataFrame(
                {
                    "component": ["dataset_sd", "residual_sd"],
                    "estimate": [
                        np.sqrt(float(res_reml.cov_re.iloc[0, 0])),
                        float(res_reml.scale) ** 0.5,
                    ],
                }
            )
            print_table(var_comp, title="Mixed-effects variance components (REML)")
        except Exception:
            pass


def main():
    """Parse args, build the summary table, and run the analysis per classifier."""
    parser = argparse.ArgumentParser(
        description=(
            "Compare ML algorithms across datasets using nonparametric and/or mixed-effects model."
        )
    )
    parser.add_argument(
        "csv",
        help=(
            "Path to CSV master table with columns: dataset, algorithm, "
            "run, classifier, <score-col>"
        ),
    )
    parser.add_argument(
        "--direction",
        choices=["higher", "lower"],
        default="higher",
        help="Metric direction (higher-is-better or lower-is-better)",
    )
    parser.add_argument("--alpha", type=float, default=0.05, help="Significance level")
    parser.add_argument(
        "--agg",
        choices=["mean", "median"],
        default="mean",
        help="Aggregation for nonparametric part",
    )
    parser.add_argument(
        "--posthoc",
        choices=["auto", "holm", "conover", "nemenyi"],
        default="auto",
        help=(
            "Nonparametric post-hoc: 'holm' (pairwise z + Holm), "
            "'conover' (requires scikit-posthocs), 'nemenyi' (CD), or 'auto'"
        ),
    )
    parser.add_argument(
        "--mode",
        choices=["nonparam", "mixed", "both"],
        default="both",
        help="Which analysis to run: nonparametric, mixed-effects, or both",
    )
    parser.add_argument(
        "--classifier",
        default=None,
        help="Run analysis for a single classifier only (e.g. tpot, random_forest, "
        "xgboost, logreg, svm). If omitted, the analysis is run separately for "
        "EVERY classifier in the table.",
    )
    parser.add_argument(
        "--score-col",
        default="balanced_accuracy",
        help=(
            "Column to use as the score (default: balanced_accuracy). "
            "Also valid: accuracy, f1_score."
        ),
    )
    parser.add_argument(
        "--datasets",
        nargs="+",
        default=None,
        help="Subset of dataset keys to include (default: all).",
    )
    parser.add_argument(
        "--summary-out",
        type=Path,
        default=None,
        help="Write a mean±std summary CSV (per classifier x dataset x algorithm) to this path.",
    )
    parser.add_argument(
        "--summary-only",
        action="store_true",
        help="Only produce the mean±std summary table and skip the statistical analysis.",
    )
    args = parser.parse_args()

    # Load data
    df = pd.read_csv(args.csv, sep=";", decimal=",", encoding="utf-8")
    required_cols = {"dataset", "algorithm", "run", "classifier", args.score_col}
    if not required_cols.issubset(df.columns):
        sys.exit(f"CSV must contain columns: {required_cols}, found: {df.columns.tolist()}")

    # Filter datasets
    if args.datasets:
        df = df[df["dataset"].isin(args.datasets)].copy()
        if df.empty:
            sys.exit(f"No rows left after --datasets filter: {args.datasets}")

    # --- Mean ± std summary table for the paper ---
    if args.summary_out or args.summary_only:
        summ = summary_table(df, args.score_col)
        if args.summary_out:
            summ.to_csv(args.summary_out, index=False, sep=";", decimal=",")
            print(f"Wrote summary table to {args.summary_out}")
        print_table(
            summ, title=f"Mean ± std of {args.score_col} per (classifier, dataset, algorithm)"
        )
        if args.summary_only:
            return

    # --- Statistical analysis, per classifier ---
    if args.classifier is not None:
        classifiers = [args.classifier]
    else:
        classifiers = sorted(df["classifier"].unique())

    for clf in classifiers:
        sub = df[df["classifier"] == clf].copy()
        if sub.empty:
            print(f"\nNo rows for classifier '{clf}'; skipping.")
            continue
        sub["score"] = sub[args.score_col]
        print(f"\n{'=' * 70}")
        print(f"Classifier: {clf}   (score = {args.score_col})")
        print(f"{'=' * 70}")
        run_analysis(sub, args, clf)


if __name__ == "__main__":
    main()
