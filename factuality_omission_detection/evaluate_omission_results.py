"""
Score model predictions on the MedExpert dataset for omission detection.
"""
import os
from typing import List, Dict, Optional
import argparse
import logging
import unicodedata

import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
import jsonlines
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix
)

# Set up logging
logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)
logger = logging.getLogger()

# Map severity level to numerical values for comparing
SEVERITY_CATEGORIES = pd.CategoricalDtype(
    categories=["No Omission", "Mild", "Moderate", "Severe", "Life-threatening", "All-Err"],
    ordered=True
)

# Pretty names for models
model_display_order = {
    "llama2": "Llama-2",
    "llama3": "Llama-3.3",
    "olmo2": "OLMO-2",
    "gemma2": "Gemma-2",
    "biollm": "OBioLLM",
    "All": "All"
}


def plot_confusion_matrix(cm: pd.DataFrame, output_path: str):
    # Define the new labels
    labels_x = ["Predicted Omission", "Predicted No Omission"]
    labels_y = ["Actual Omission", "Actual No Omission"]

    plt.figure(figsize=(5, 4), dpi=200)
    sns.heatmap(
        cm,
        annot=True,
        fmt="0.2f",
        cmap="Blues",
        xticklabels=labels_x,
        yticklabels=labels_y,
        vmax=1,
        vmin=0,
        linewidths=1.5,
        linecolor='white'
    )
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()


def format_example(example: pd.Series) -> str:
    """Format a single example for logging."""
    if example.empty:
        return "No example"
    # Format the annotated and predicted omissions as numbered lists
    if example.n_omissions_pred > 0:
        pred = [f"{i + 1}. {omission}" for i, omission in enumerate(example.omissions_pred)]
        pred = "\n\t".join(pred)
    else:
        pred = "No omissions predicted."
    if example.n_omissions > 0:
        gt = [f"{i + 1}. {omission}" for i, omission in enumerate(example.omissions)]
        gt = "\n\t".join(gt)
    else:
        gt = "No annotated omissions."

    out = f"ID: {example.name} Title: {example.title}\n"
    out += f"Question: {example.question}\n"
    out += f"Chatbot Response ({example.model}): {example.response}\n"
    out += f"Severity: {example.omission_severity}\n"
    out += f"Annotated Omissions:\n\t{gt}\n"
    out += f"Predicted Omissions:\n\t{pred}\n"
    return out


def score_predictions(y_true: List[bool], y_pred: List[bool], pos_label: bool = True) -> pd.Series:
    """Compute classification metrics for binary predictions.

    Assumes positive class is True.
    """
    accuracy = accuracy_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred, zero_division=0, pos_label=pos_label, average="binary")
    recall = recall_score(y_true, y_pred, zero_division=0, pos_label=pos_label, average="binary")
    f1 = f1_score(y_true, y_pred, zero_division=0, pos_label=pos_label, average="binary")
    row = pd.Series({
        "Acc.": accuracy,
        "Prec.": precision,
        "Rec.": recall,
        "F1": f1,
        "N": int(len(y_true))
    })
    return row


def load_medexpert_dataset(file_path: str) -> pd.DataFrame:
    """Load the MedExpert dataset from a JSONL file."""
    df = pd.read_json(
        file_path,
        lines=True,
        dtype={"id": str}
    ).set_index("id")

    df.model = df.model.map(lambda x: model_display_order.get(x.lower(), x))

    # Normalize the omission_severity field
    def normalize_severity(severity: Optional[str]) -> str:
        if pd.isna(severity):
            return "No Omission"
        severity = unicodedata.normalize('NFKD', severity).encode('ascii', 'ignore').decode('utf-8').strip()
        severity_map = {
            "No Omission": "No Omission",
            "Mild - no action is required": "Mild",
            "Moderate - may negatively impact the patients health if no action is taken": "Moderate",
            "Severe  may require medical intervention by a doctor": "Severe",
            "Severe - may require medical intervention by a doctor": "Severe",
            "Life-threatening - can be life-threatening without medical intervention": "Life-threatening"
        }
        severity = severity_map[severity]
        return severity
    df.omission_severity = df.omission_severity.map(normalize_severity)
    df.omission_severity = df.omission_severity.astype(SEVERITY_CATEGORIES)
    df["n_omissions"] = df.omissions.map(lambda x: len(x) if isinstance(x, list) else 0)

    # Hack fix
    err = df[(df.omission_severity > "No Omission") & (df.n_omissions == 0)]
    if not err.empty:
        logger.warning(f"Found {err.shape[0]} records with severity > 'No Omission' but 0 omissions. Fixing severity to 'No Omission'.\n{err.index.tolist()}")
        df.loc[err.index, "omission_severity"] = "No Omission"
    return df


def load_predictions(file_path: str) -> pd.DataFrame:
    """Load model predictions from a JSONL file.

    Load HealthBench-ICL output differently than Zero-Shot outputs.
    """
    records = []
    if "healthbench-icl" in file_path.lower():
        # HealthBench-ICL output format
        with jsonlines.open(file_path) as reader:
            for obj in reader:
                omissions = [
                    o["criterion"] for o in obj["completeness-healthbench"]["meta"] if o["criteria_met"] is False
                ]
                record = {
                    "id": str(obj["id"]),
                    "omissions": omissions,
                    "n_omissions": len(omissions),
                    "meta": obj["completeness-healthbench"].copy()
                }
                records.append(record)
    elif "zero-shot" in file_path.lower():
        # Zero-Shot output format
        with jsonlines.open(file_path) as reader:
            for obj in reader:
                omissions = [
                    o["omission"] for o in obj["predicted_omissions"]
                ]
                record = {
                    "id": str(obj["id"]),
                    "omissions": omissions,
                    "n_omissions": len(omissions),
                    "meta": obj.copy()
                }
                records.append(record)
    else:
        raise ValueError("Predictions file format not recognized. Expected 'healthbench-icl' or 'zero-shot' in filename.")
    df = pd.DataFrame(records).set_index("id")
    return df


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--dataset_file",
        type=str,
        required=True,
        help="Path to MedExpert dataset JSONL file."
    )
    parser.add_argument(
        "--predictions_file",
        type=str,
        required=True,
        help="Path to the JSONL file containing model predictions.",
    )
    return parser.parse_args()


def main():
    args = parse_args()

    # Load dataset
    logger.info(f"Loading dataset from {args.dataset_file}")
    dataset_df = load_medexpert_dataset(args.dataset_file)

    # Load predictions
    logger.info(f"Loading predictions from {args.predictions_file}")
    omission_df = load_predictions(args.predictions_file)

    # Merge dataset and predictions
    merged_df = dataset_df.join(omission_df, how="outer", rsuffix="_pred")
    logger.info(f"Merged dataset and predictions: {merged_df.shape[0]} records")

    # Add an "All" domain for overall metrics
    merged_df["label"] = merged_df.n_omissions.map(lambda x: True if x > 0 else False)
    merged_df["pred_label"] = merged_df.n_omissions_pred.map(lambda x: True if x > 0 else False)
    results_all = merged_df.copy()
    results_all["domain"] = "All"
    merged_df = pd.concat([merged_df, results_all], axis=0)

    # Compute and log metrics across domains and severity levels
    results_df = []
    cm = None
    for domain, group_df in merged_df.groupby("domain", observed=True):
        # Compute metrics for each severity level within the domain
        for severity, df in group_df.groupby("omission_severity", observed=True):
            # Handle the "No Omission" case separately
            if severity == "No Omission":
                row = score_predictions(
                    df["label"].tolist(),
                    df["pred_label"].tolist(),
                    pos_label=False
                )
            else:
                row = score_predictions(
                    df["label"].tolist(),
                    df["pred_label"].tolist()
                )
            row["Severity"] = severity
            row["Domain"] = domain
            results_df.append(row)

        # Compute overall metrics for the domain
        row = score_predictions(
            group_df["label"].tolist(),
            group_df["pred_label"].tolist()
        )
        row["Severity"] = "All-Err"
        row["Domain"] = domain
        results_df.append(row)

        # Store confusion matrix for the "All" domain
        if domain == "All":
            cm = confusion_matrix(
                group_df["label"].tolist(),
                group_df["pred_label"].tolist(),
                labels=[True, False],
                normalize="true"
            )
    results_df = pd.DataFrame(results_df)
    results_df["Severity"] = results_df["Severity"].astype(SEVERITY_CATEGORIES)
    results_df["N"] = results_df["N"].astype(int)
    results_df = results_df.set_index(["Domain", "Severity"]).sort_index()

    # Format confusion matrix
    cm = pd.DataFrame(
        cm,
        index=["Actual Positive", "Actual Negative"],
        columns=["Predicted Positive", "Predicted Negative"]
    )
    if "healthbench-icl" in args.predictions_file.lower():
        output_path = "confusion_matrix_healthbench_icl_omission_detection.png"
    else:
        output_path = "confusion_matrix_zeros_shot_omission_detection.png"
    plot_confusion_matrix(cm, output_path=output_path)
    logger.info(f"Confusion Matrix (All Domain), Saved to {output_path}:\n{cm}")

    # Log results
    with pd.option_context('display.float_format', '{:.1%}'.format):
        s = results_df.to_string()
        s = s.replace("%", "")
        logger.info(f"\nOmission Detection Results:\n{s}")

    # Log accuracy per model for All domain and All-Err severity
    model_acc = merged_df[merged_df["domain"] == "All"].groupby("model", observed=True).apply(
        lambda df: accuracy_score(df["label"], df["pred_label"])
    ).sort_values(ascending=False)
    logger.info(f"\nAccuracy per Model (All Domain, All-Err Severity):\n{model_acc.to_string()}")

    # Log examples
    # Pre-selected after manual review of false positive/negative errors
    example_ids = [
        "ovmqy_ltR3dkSSP8WPfQflN7mcpYnEjBuArX7AGcCNw=",
        "0HeXy64xwkdVDfn5oHsNnPhWeN5dnbGkK2EowQnnZVI=",
        "5SSZxZGP-Ld3BbJ6J5tYUJYPP5PdcVZR1rypuOEQ4A8=",
        "4hwBx7ACHagZrJvHyT8ccJkYC3gFdCHcNZcpV7QDZf4="
    ]

    for ex_id in example_ids:
        ex = merged_df.loc[ex_id].iloc[0]
        logger.info(f"\nExample ID {ex_id}:\n{format_example(ex)}")

    # Randomly sample examples of each error type
    shuf_examples = merged_df.sample(frac=1, random_state=42)
    shuf_examples = shuf_examples[shuf_examples.omission_severity > "Mild"]
    fp_example = shuf_examples[(shuf_examples["label"] == False) & (shuf_examples["pred_label"] == True)].head(1)
    tp_example = shuf_examples[(shuf_examples["label"] == True) & (shuf_examples["pred_label"] == True)].head(1)
    fn_example = shuf_examples[(shuf_examples["label"] == True) & (shuf_examples["pred_label"] == False)].head(1)
    if not fp_example.empty:
        logger.info(f"\nExample False Positive:\n{format_example(fp_example.iloc[0])}")
    if not tp_example.empty:
        logger.info(f"\nExample True Positive:\n{format_example(tp_example.iloc[0])}")
    if not fn_example.empty:
        logger.info(f"\nExample False Negative:\n{format_example(fn_example.iloc[0])}")


if __name__ == "__main__":
    main()

