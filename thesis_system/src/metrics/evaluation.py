from __future__ import annotations

from collections.abc import Sequence

import numpy as np
import pandas as pd


SUMMARY_METRICS = (
    "accuracy",
    "macro_f1",
    "top_k_accuracy",
    "loss",
    "training_time_seconds",
    "epochs_completed",
    "final_training_loss",
    "final_validation_loss",
)


def _paired_vectors(
    y_true: Sequence[object] | np.ndarray,
    y_pred: Sequence[object] | np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    true = np.asarray(y_true)
    predicted = np.asarray(y_pred)
    if true.ndim != 1 or predicted.ndim != 1:
        raise ValueError("y_true and y_pred must be one-dimensional.")
    if len(true) != len(predicted):
        raise ValueError("y_true and y_pred must contain the same number of samples.")
    return true, predicted


def accuracy_score(
    y_true: Sequence[object] | np.ndarray,
    y_pred: Sequence[object] | np.ndarray,
) -> float:
    """Return the fraction of exactly matched next-token predictions."""
    true, predicted = _paired_vectors(y_true, y_pred)
    if len(true) == 0:
        return float("nan")
    return float(np.mean(true == predicted))


def macro_f1_score(
    y_true: Sequence[object] | np.ndarray,
    y_pred: Sequence[object] | np.ndarray,
    labels: Sequence[object] | np.ndarray | None = None,
) -> float:
    """Return unweighted mean F1 across present or explicitly requested classes."""
    true, predicted = _paired_vectors(y_true, y_pred)
    if len(true) == 0:
        return float("nan")

    if labels is None:
        class_labels = list(dict.fromkeys([*true.tolist(), *predicted.tolist()]))
    else:
        class_labels = list(labels)
        if len(class_labels) != len(set(class_labels)):
            raise ValueError("labels must contain unique class values.")
    if not class_labels:
        return float("nan")

    scores: list[float] = []
    for label in class_labels:
        true_positive = int(np.sum((true == label) & (predicted == label)))
        false_positive = int(np.sum((true != label) & (predicted == label)))
        false_negative = int(np.sum((true == label) & (predicted != label)))
        denominator = (2 * true_positive) + false_positive + false_negative
        scores.append(0.0 if denominator == 0 else (2 * true_positive) / denominator)
    return float(np.mean(scores))


def _probability_inputs(
    y_true: Sequence[object] | np.ndarray,
    probabilities: Sequence[Sequence[float]] | np.ndarray,
    labels: Sequence[object] | np.ndarray | None,
) -> tuple[np.ndarray, np.ndarray, list[object], np.ndarray]:
    true = np.asarray(y_true)
    probability_array = np.asarray(probabilities, dtype=float)
    if true.ndim != 1:
        raise ValueError("y_true must be one-dimensional.")
    if probability_array.ndim != 2:
        raise ValueError("probabilities must be a two-dimensional sample-by-class array.")
    if len(true) != probability_array.shape[0]:
        raise ValueError("y_true and probabilities must contain the same number of samples.")
    if probability_array.shape[1] == 0:
        raise ValueError("probabilities must contain at least one class column.")
    if not np.all(np.isfinite(probability_array)):
        raise ValueError("probabilities must contain only finite values.")
    if np.any(probability_array < 0):
        raise ValueError("probabilities cannot contain negative values.")

    class_labels = (
        list(range(probability_array.shape[1])) if labels is None else list(labels)
    )
    if len(class_labels) != probability_array.shape[1]:
        raise ValueError("labels must contain one value for each probability column.")
    if len(class_labels) != len(set(class_labels)):
        raise ValueError("labels must contain unique class values.")

    label_to_column = {label: column for column, label in enumerate(class_labels)}
    try:
        true_columns = np.asarray([label_to_column[value] for value in true.tolist()])
    except (KeyError, TypeError) as exc:
        raise ValueError("Every y_true value must appear in labels.") from exc
    return true, probability_array, class_labels, true_columns


def top_k_accuracy(
    y_true: Sequence[object] | np.ndarray,
    probabilities: Sequence[Sequence[float]] | np.ndarray,
    k: int = 3,
    labels: Sequence[object] | np.ndarray | None = None,
) -> float:
    """Return the share of true tokens appearing among the k highest probabilities."""
    if isinstance(k, bool) or not isinstance(k, (int, np.integer)) or k < 1:
        raise ValueError("k must be a positive integer.")
    true, probability_array, _, true_columns = _probability_inputs(
        y_true, probabilities, labels
    )
    if len(true) == 0:
        return float("nan")

    effective_k = min(int(k), probability_array.shape[1])
    ranked_columns = np.argsort(-probability_array, axis=1, kind="stable")[
        :, :effective_k
    ]
    hits = np.any(ranked_columns == true_columns[:, None], axis=1)
    return float(np.mean(hits))


def negative_log_loss(
    y_true: Sequence[object] | np.ndarray,
    probabilities: Sequence[Sequence[float]] | np.ndarray,
    labels: Sequence[object] | np.ndarray | None = None,
    epsilon: float = 1e-15,
) -> float:
    """Return mean cross-entropy from class probabilities for the true tokens."""
    if not 0 < epsilon < 1:
        raise ValueError("epsilon must be between 0 and 1.")
    true, probability_array, _, true_columns = _probability_inputs(
        y_true, probabilities, labels
    )
    if len(true) == 0:
        return float("nan")

    row_totals = probability_array.sum(axis=1)
    if np.any(row_totals <= 0):
        raise ValueError("Each probability row must have a positive total.")
    normalized = probability_array / row_totals[:, None]
    true_probabilities = normalized[np.arange(len(true_columns)), true_columns]
    return float(-np.mean(np.log(np.clip(true_probabilities, epsilon, 1.0))))


def aggregate_algorithm_summary(
    fold_results: pd.DataFrame,
    confidence_level: float = 0.95,
) -> pd.DataFrame:
    """Aggregate real fold results into per-algorithm mean, SD, and Confidence Interval columns."""
    if not isinstance(fold_results, pd.DataFrame):
        raise TypeError("fold_results must be a pandas DataFrame.")
    if "algorithm" not in fold_results.columns:
        raise ValueError("fold_results must contain an 'algorithm' column.")

    output_columns = ["algorithm", "folds_completed"]
    present_metrics = [metric for metric in SUMMARY_METRICS if metric in fold_results]
    for metric in present_metrics:
        output_columns.extend([f"{metric}_mean", f"{metric}_std"])
    if fold_results.empty:
        return pd.DataFrame(columns=output_columns)

    completed = fold_results
    if "status" in completed.columns:
        completed = completed[completed["status"].eq("completed")]
    if completed.empty:
        return pd.DataFrame(columns=output_columns)

    rows: list[dict[str, object]] = []

    # Student's t critical values for two-tailed confidence levels
    t_tables = {
        0.90: {1: 6.314, 2: 2.920, 3: 2.353, 4: 2.132, 5: 2.015, 6: 1.943, 7: 1.895, 8: 1.860, 9: 1.833, 10: 1.812},
        0.95: {1: 12.706, 2: 4.303, 3: 3.182, 4: 2.776, 5: 2.571, 6: 2.447, 7: 2.365, 8: 2.306, 9: 2.262, 10: 2.228},
        0.99: {1: 63.657, 2: 9.925, 3: 5.841, 4: 4.604, 5: 4.032, 6: 3.707, 7: 3.499, 8: 3.355, 9: 3.250, 10: 3.169},
    }
    target_level = 0.95
    if abs(confidence_level - 0.90) < 0.02:
        target_level = 0.90
    elif abs(confidence_level - 0.99) < 0.005:
        target_level = 0.99
    
    t_table = t_tables.get(target_level, t_tables[0.95])
    ci_label_pct = f"{int(target_level * 100)}%"

    for algorithm, algorithm_results in completed.groupby("algorithm", sort=False):
        folds_completed = (
            int(algorithm_results["fold"].nunique())
            if "fold" in algorithm_results
            else int(len(algorithm_results))
        )
        row: dict[str, object] = {
            "algorithm": algorithm,
            "folds_completed": folds_completed,
        }
        for metric in present_metrics:
            values = pd.to_numeric(algorithm_results[metric], errors="coerce").dropna()
            n_samples = len(values)
            mean_val = float(values.mean()) if n_samples > 0 else np.nan
            std_val = float(values.std(ddof=1)) if n_samples > 1 else np.nan
            
            row[f"{metric}_mean"] = mean_val
            row[f"{metric}_std"] = std_val

            if n_samples > 1 and not np.isnan(std_val):
                t_crit = t_table.get(n_samples - 1, 1.96)
                margin = float(t_crit * (std_val / np.sqrt(n_samples)))
                row[f"{metric}_ci_margin"] = margin
                row[f"{metric}_ci_lower"] = mean_val - margin
                row[f"{metric}_ci_upper"] = mean_val + margin
                row[f"{metric}_95ci"] = f"{mean_val:.4f} ± {margin:.4f}"
                row[f"{metric}_ci_display"] = f"{mean_val:.4f} ± {margin:.4f} ({ci_label_pct} CI)"
            else:
                row[f"{metric}_ci_margin"] = np.nan
                row[f"{metric}_ci_lower"] = np.nan
                row[f"{metric}_ci_upper"] = np.nan
                row[f"{metric}_95ci"] = f"{mean_val:.4f}" if not np.isnan(mean_val) else "N/A"
                row[f"{metric}_ci_display"] = f"{mean_val:.4f}" if not np.isnan(mean_val) else "N/A"

        rows.append(row)

    return pd.DataFrame(rows, columns=output_columns + [col for col in rows[0] if col not in output_columns])


def paired_ttest_comparison(
    fold_results: pd.DataFrame,
    metric: str = "macro_f1",
) -> pd.DataFrame:
    """Compute paired t-test statistics comparing all algorithm pairs across matching LORO folds."""

    if not isinstance(fold_results, pd.DataFrame) or fold_results.empty:
        return pd.DataFrame()
    if not {"algorithm", "test_group", metric}.issubset(fold_results.columns):
        return pd.DataFrame()

    algorithms = fold_results["algorithm"].unique().tolist()
    if len(algorithms) < 2:
        return pd.DataFrame()

    pivot = fold_results.pivot(index="test_group", columns="algorithm", values=metric).dropna()
    if len(pivot) < 2:
        return pd.DataFrame()

    rows = []
    for i in range(len(algorithms)):
        for j in range(i + 1, len(algorithms)):
            alg_a = algorithms[i]
            alg_b = algorithms[j]
            if alg_a not in pivot.columns or alg_b not in pivot.columns:
                continue

            vals_a = pivot[alg_a].values
            vals_b = pivot[alg_b].values
            diff = vals_a - vals_b
            mean_diff = float(np.mean(diff))
            std_diff = float(np.std(diff, ddof=1)) if len(diff) > 1 else 0.0
            n = len(diff)
            se_diff = std_diff / np.sqrt(n) if n > 0 else 0.0
            t_stat = mean_diff / se_diff if se_diff > 0 else 0.0

            abs_t = abs(t_stat)
            if abs_t >= 4.604:
                p_text = "p < 0.01 (**)"
                is_sig = True
            elif abs_t >= 2.776:
                p_text = "p < 0.05 (*)"
                is_sig = True
            elif abs_t >= 2.132:
                p_text = "p < 0.10"
                is_sig = False
            else:
                p_text = "p > 0.10 (ns)"
                is_sig = False

            higher_alg = alg_a if mean_diff > 0 else alg_b

            rows.append(
                {
                    "Model Pair": f"{alg_a} vs {alg_b}",
                    "Metric": metric,
                    "Mean Difference": round(mean_diff, 4),
                    "t-Statistic": round(t_stat, 4),
                    "p-Value Range": p_text,
                    "Statistically Significant (p < 0.05)": "Yes (*)" if is_sig else "No (Comparable)",
                    "Finding": (
                        f"{higher_alg} is significantly better"
                        if is_sig
                        else f"{higher_alg} has slight edge (comparable)"
                    ),
                }
            )

    return pd.DataFrame(rows)



