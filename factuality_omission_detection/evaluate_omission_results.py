"""
Score model predictions on the MedExpert dataset for omission detection.
"""
import os
from typing import List, Dict, Optional
import argparse
import logging
import unicodedata

import pandas as pd
import jsonlines
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score
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


def score_predictions(y_true: List[bool], y_pred: List[bool], pos_label: bool = True) -> pd.Series:
    """Compute classification metrics for binary predictions.

    Assumes positive class is True.
    """
    accuracy = accuracy_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred, zero_division=0, pos_label=pos_label)
    recall = recall_score(y_true, y_pred, zero_division=0, pos_label=pos_label)
    f1 = f1_score(y_true, y_pred, zero_division=0, pos_label=pos_label)
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
            "Life-threatening - can be life-threatening without medical intervention": "Life-threatening"
        }
        severity = severity_map[severity]
        return severity
    df.omission_severity = df.omission_severity.map(normalize_severity)
    df.omission_severity = df.omission_severity.astype(SEVERITY_CATEGORIES)
    df["n_omissions"] = df.omissions.map(lambda x: len(x) if isinstance(x, list) else 0)
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
    results_df = pd.DataFrame(results_df)
    results_df["Severity"] = results_df["Severity"].astype(SEVERITY_CATEGORIES)
    results_df["N"] = results_df["N"].astype(int)
    results_df = results_df.set_index(["Domain", "Severity"]).sort_index()

    # Log results
    with pd.option_context('display.float_format', '{:.1%}'.format):
        s = results_df.to_string()
        s = s.replace("%", "")
        logger.info(f"\nOmission Detection Results:\n{s}")

    # Log a false positive and true positive example if available
    shuf_examples = merged_df.sample(frac=1, random_state=42)
    fp_example = shuf_examples[(shuf_examples["label"] == False) & (shuf_examples["pred_label"] == True)].head(1)
    tp_example = shuf_examples[(shuf_examples["label"] == True) & (shuf_examples["pred_label"] == True)].head(1)

    def format_example(example: pd.Series) -> str:
        if example.empty:
            return "No example"
        # Format the annotated and predicted omissions as numbered lists
        if example.n_omissions_pred > 0:
            pred = [f"{i+1}. {omission}" for i, omission in enumerate(example.omissions_pred)]
            pred = "\n\t".join(pred)
        else:
            pred = "No omissions predicted."
        if example.n_omissions > 0:
            gt = [f"{i+1}. {omission}" for i, omission in enumerate(example.omissions)]
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

    if not fp_example.empty:
        logger.info(f"\nExample False Positive:\n{format_example(fp_example.iloc[0])}")
    if not tp_example.empty:
        logger.info(f"\nExample True Positive:\n{format_example(tp_example.iloc[0])}")


if __name__ == "__main__":
    main()

