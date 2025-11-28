#!/usr/bin/env python3
"""
Dataset Quality Check & IAA Calculation Script
====================================================

Usage:
    python data_preprocessing_from_annotation_interface.py

    # With anonymization of annotators
    python data_preprocessing_from_annotation_interface.py --anonymous

Description:
    This script processes raw annotation data from John Snow Labs Annotation interface. 
    A sample file exported from the John Snow Lab annoation interface is given in 
    `sample_data/sample_data_export_from_JSL.json`
    The data topics are in `sample_data/medexpert_questions_with_topics.jsonl`
    It performs post-hoc data cleaning, merges topic information, calculates Inter-Annotator 
    Agreement (IAA) using Krippendorff's Alpha (for omission severity, model certainty, 
    and factuality), and generates Ground Truth (GT) datasets.

    It also performs quality control checks, flagging missing severity labels, 
    descriptions, or comments.

Arguments:
    --anonymous : Optional. If set, redacts specific annotator usernames (e.g., 'agornia1')
                  to generic IDs (e.g., 'Annotator000') in both the logs and the output files. You have to edit this mapping in function `get_anonymized_annotator(.)`

Configuration:
    Update 'project_dir' and file name variables below to point to your specific data version.
    If you want to use `get_anonymized_annotator`, please provide the correct annotator pseudonames in the function

Input Files:
    1. Raw Annotations: {project_dir}/sample_data/sample_data_export_from_JSL.json
    2. Topics File:     {project_dir}/sample_data/medexpert_questions_with_topics.jsonl

Output Files:
    1. Log File:        {project_dir}/sample_data/dataset_quality_check.log
    2. Processed Data:  {project_dir}/sample_data/sample_data_export_from_JSL_processed.jsonl
    3. Benchmark Data:  {project_dir}/sample_data/{file_date}/sample_data_export_from_JSL_gt.jsonl
    4. All Data:        {project_dir}/sample_data/sample_data_export_from_JSL_processed_senticized.jsonl
"""

# Imports
import subprocess
import json
import os
import argparse
from functools import partial

import pandas as pd
import simpledorff
import spacy
nlp = spacy.load("en_core_web_sm")

import logging
import sys
import numpy as np

pd.set_option('display.max_rows', 100)
pd.set_option('display.max_columns', None)

###################
# --- FILE PATHS SAMPLE DATA---
###################
project_dir = os.getcwd()

raw_annotations_file = f"{project_dir}/sample_data/sample_data_export_from_JSL.json"
processed_file = f"{project_dir}/sample_data/sample_data_export_from_JSL_processed.jsonl"
topics_file = f"{project_dir}/sample_data/medexpert_questions_with_topics.jsonl"
only_base = False


# --- ARGUMENT PARSER ---
parser = argparse.ArgumentParser(description="Process annotation results and calculate IAA.")
parser.add_argument(
    "--anonymous", 
    action="store_true", 
    help="Redact annotator usernames to anonymous IDs in output files."
)
args = parser.parse_args()


# --- Start of Logging Configuration ---
log_file_path = os.path.join(project_dir, 'sample_data', 'dataset_quality_check.log')
print(f"Log in : {str(log_file_path)}")

# Configure logging to write to a file, overwriting it on each run
logging.basicConfig(
    level=logging.INFO,
    format='%(message)s',
    filename=log_file_path,
    filemode='w'
)

log = logging.getLogger('stdout_logger')

# Define a stream-like object that redirects writes to the logger
class StreamToLogger(object):
   def __init__(self, logger, log_level=logging.INFO):
      self.logger = logger
      self.log_level = log_level
      self.linebuf = ''

   def write(self, buf):
      # Process buffer line by line and log it
      for line in buf.rstrip().splitlines():
         self.logger.log(self.log_level, line.rstrip())

   def flush(self):
      # This flush method is needed for compatibility.
      pass

# Redirect the standard output to our custom logger
sys.stdout = StreamToLogger(log, logging.INFO)
print("--- LOGGING STARTED ---")
if args.anonymous:
    print("--- ANNOTATOR ANONYMOUS MODE: ACTIVE ---")
# --- End of Logging Configuration ---


# --- Helper functions ---

# Function to anonymize annotators
def get_anonymized_annotator(username: str) -> str:
    """
    Maps a username to its corresponding anonymized annotator ID.

    Args:
        username: The original username string.

    Returns:
        The anonymized annotator ID if found, otherwise the original username.
    """
    annotator_map = {
        'JohnSmith': 'Annotator000',
        'JaneDoe': 'Annotator001',
        'AliceBrown': 'Annotator002',
        'AlexTaylor': 'Annotator003',
    }
    # The .get() method safely returns the value for a key,
    # or a default value (in this case, the original username) if the key doesn't exist.
    return annotator_map.get(username, username)



process_command = [
    "python",
    f"{project_dir}/scripts/data_preprocessing/process_raw_annotations.py",
    raw_annotations_file,
    "--overwrite"
]
try:
    # This will raise an error if the command fails
    process_output = subprocess.check_output(process_command, text=True)
    print(process_output)
except subprocess.CalledProcessError as e:
    print(f"Command failed with error: {e}")
    print(" ".join(process_command))
    exit(1)

##################
# Main
##################

print(r"""# Prepare the Data""")
print(r"""## Post-hoc fixes""")

# -- Load dataset
dataset_df = pd.read_json(processed_file, lines=True)
print(dataset_df.head())


# -- Load topics file, remove domain column as its not necesary for merging
topics_questions_df = pd.read_json(topics_file, lines=True).drop(columns=['domain'])

# -- Add a "Topics" column for each question
dataset_df = pd.merge(
    # Use .assign() to create a temporary key for the merge
    dataset_df.assign(_merge_key=dataset_df['question'].str.lower().str.replace(r'\s+', '', regex=True)), 
    topics_questions_df.assign(_merge_key=topics_questions_df['question'].str.lower().str.replace(r'\s+', '', regex=True)), 

    # Merge settings
    on='_merge_key', 
    suffixes=('_df1', '_df2')

    ).drop(
        # Drop the temporary key and the redundant question column
        columns=['_merge_key', 'question_df2']

    ).rename(
        #  Rename the original question column back
        columns={'question_df1': 'question'}
    )

# -- Separate the base title from "-iaa"
dataset_df["base_title"] = dataset_df.title.map(lambda x: x.replace("-iaa", ""))

########################
# Make annotators anonmyous
########################
if args.anonymous:
    print("Redacting annotator IDs...")
    # Anonymize the main column
    dataset_df["annotator"] = dataset_df["annotator"].apply(get_anonymized_annotator)
    
    # Anonymize annotator within the 'meta' column's dictionary
    dataset_df['meta'] = dataset_df['meta'].apply(
        lambda meta: {**meta, 'annotator': get_anonymized_annotator(meta['annotator'])} 
        if isinstance(meta, dict) and 'annotator' in meta else meta
    )

# -- Fix issues when the omission severity level values are not as expected
COLUMN_TO_CLEAN = 'omission_severity'

# omission severity (*only* values allowed)
TARGET_VALUES = {
    'Mild': 'Mild - no action is required',
    'Moderate': 'Moderate - may negatively impact the patients health if no action is taken',
    'Severe': 'Severe - may require medical intervention by a doctor',
    'Life-threatening': 'Life-threatening - can be life-threatening without medical intervention',
    'None': None  # Use None for null/missing values
}

# create the mapping of all known "bad" values to the "good" target values
CORRECTION_MAP = {
    # values from your "bad" list
    'Severe  may require medical intervention by a doctor': TARGET_VALUES['Severe'],
    '𝗠𝗼𝗱𝗲𝗿𝗮𝘁𝗲 - may negatively impact the patient’s health if no action is taken': TARGET_VALUES['Moderate'],
    '𝗠𝗶𝗹𝗱 - no action is required': TARGET_VALUES['Mild'],
    '𝗟𝗶𝗳𝗲-𝘁𝗵𝗿𝗲𝗮𝘁𝗲𝗻𝗶𝗻𝗴 - can be life-threatening without medical intervention': TARGET_VALUES['Life-threatening'],
    '𝗦𝗲𝘃𝗲𝗿𝗲 – may require medical intervention by a doctor': TARGET_VALUES['Severe'],

    # add common null/empty values 
    '': TARGET_VALUES['None'],
    np.nan: TARGET_VALUES['None'],
    None: TARGET_VALUES['None'],

    # add "good" values mapping to themselves 
    TARGET_VALUES['Mild']: TARGET_VALUES['Mild'],
    TARGET_VALUES['Moderate']: TARGET_VALUES['Moderate'],
    TARGET_VALUES['Severe']: TARGET_VALUES['Severe'],
    TARGET_VALUES['Life-threatening']: TARGET_VALUES['Life-threatening'],
}

print("\n--- Values Before Cleaning ---")
print(dataset_df[COLUMN_TO_CLEAN].value_counts(dropna=False))

# apply the correction
dataset_df[COLUMN_TO_CLEAN] = dataset_df[COLUMN_TO_CLEAN].replace(CORRECTION_MAP)

print("\n--- Values After Cleaning ---")
print(dataset_df[COLUMN_TO_CLEAN].value_counts(dropna=False))

# -- Fix issues when the model certainity level values are not as expected
COLUMN_TO_CLEAN = 'model_certainty'

# severity (*only* values allowed)
TARGET_VALUES = {
    'Low': 'Low - The model does not provide sufficient certainty and information',
    'Moderate' : 'Moderate - The model provides some certainty and information',
    'High' : 'High - The model provides sufficient certainty and information',
    'None': None  # Use None for null/missing values
}

# create the mapping of all known "bad" values to the "good" target values
CORRECTION_MAP = {
    # values from your "bad" list
    '𝗟𝗼𝘄 - The model does not provide sufficient certainty and information': TARGET_VALUES['Low'],
    '𝗠𝗼𝗱𝗲𝗿𝗮𝘁𝗲 - The model provides some certainty and information':  TARGET_VALUES['Moderate'],
    '𝗛𝗶𝗴𝗵 - The model provides sufficient certainty and information': TARGET_VALUES['High'],


    # add common null/empty values 
    '': TARGET_VALUES['None'],
    np.nan: TARGET_VALUES['None'],
    None: TARGET_VALUES['None'],

    # add "good" values mapping to themselves 
    TARGET_VALUES['Low']: TARGET_VALUES['Low'],
    TARGET_VALUES['Moderate']: TARGET_VALUES['Moderate'],
    TARGET_VALUES['High']: TARGET_VALUES['High'],
}

print("\n--- Values Before Cleaning ---")
print(dataset_df[COLUMN_TO_CLEAN].value_counts(dropna=False))

# apply the correction
dataset_df[COLUMN_TO_CLEAN] = dataset_df[COLUMN_TO_CLEAN].replace(CORRECTION_MAP)

print("\n--- Values After Cleaning ---")
print(dataset_df[COLUMN_TO_CLEAN].value_counts(dropna=False))



########################
# Remove hypothesis entries. Only inlcude the base questions.
########################
if only_base:
    dataset_df = dataset_df[dataset_df.hypothesis_id.isna()]

# Remove `hypothesis_id` from dataframe 
if "hypothesis_id" in dataset_df.columns:
    dataset_df.drop(columns=["hypothesis_id"], inplace=True)

# Remove `context_id` from dataframe 
if "context_id" in dataset_df.columns:
    dataset_df.drop(columns=["context_id"], inplace=True)



########################
# Drop duplicate entries. Keep the entry that was most recently updated.
########################
dataset_df["last_updated"] = dataset_df.meta.map(lambda x: x["last_updated"])
dataset_df.sort_values(by="last_updated", inplace=True, ascending=False)
duplicate_annotations = dataset_df[dataset_df.duplicated(subset=["base_title", "annotator"], keep=False)]
print(f"******** Duplicate Entries *********")
print(duplicate_annotations[["id", "annotator", "title", "base_title", "last_updated"]])
dataset_df = dataset_df.sort_values(by="last_updated", ascending=False).drop_duplicates(subset=["base_title", "annotator"], keep="first")


##########################
# Add highest factuality column
# Map severity level to numerical values for comparing
###########################
factuality_severity_levels = ["No Error", "Mild", "Moderate", "Severe", "Life-threatening"]
severity_dtype = pd.CategoricalDtype(categories=factuality_severity_levels, ordered=True)
def get_highest_fact_severity(factuality_list):
    # Check for null
    if not factuality_list:
        return "No Error"
    selected_severities = pd.Categorical(
        [f["severity"] for f in factuality_list],
        dtype=severity_dtype
    )
    return max(selected_severities)
dataset_df["highest_fact_severity"] = dataset_df.factuality.map(get_highest_fact_severity)


########################
# Add "iaa" column
# Mark entries that are used for IAA calculations
########################
iaa_df = dataset_df[dataset_df.duplicated(subset=["base_title"], keep=False)].copy().reset_index(drop=True)
dataset_df["iaa"] = dataset_df.id.map(lambda x: x in iaa_df.id.values)

print(dataset_df, iaa_df)


# Removing annotation_id from meta 
dataset_df['meta'] = dataset_df['meta'].apply(lambda x: {k: v for k, v in x.items() if k != 'annotation_id'} if isinstance(x, dict) else x)

print(r"""### Overwriting file and senticizing""")

#####################
# Save updated files and senticize
#####################
# All entries

#  Drop last_updated from top-level before saving (it remains in meta)
if "last_updated" in dataset_df.columns:
    dataset_df.drop(columns=["last_updated"], inplace=True)

dataset_df.to_json(
    processed_file,
    orient="records",
    lines=True
)
# Senticize the fixed annotations
# It'll use the new, fixed processed_file (dataset_df)
reprocess_command = [
    "python",
    f"{project_dir}/scripts/data_preprocessing/process_raw_annotations.py",
    raw_annotations_file,
    "--senticize",
]
try:
    # This will raise an error if the command fails
    reprocess_output = subprocess.check_output(reprocess_command, text=True)
    print(reprocess_output)
except subprocess.CalledProcessError as e:
    print(f"Command failed with error: {e}")
    exit(1)


########################
# Drop redundant IAA entries for ground-truth dataset
########################
sent_dataset_df = pd.read_json(
    processed_file.replace("_processed.jsonl", "_processed_senticized.jsonl"),
    lines=True
)
# Re-extract `last_updated` for sorting (since we removed it from the source file)
sent_dataset_df["last_updated"] = sent_dataset_df.meta.apply(lambda x: x.get("last_updated"))

ground_truth_df = sent_dataset_df.sort_values(by="last_updated", ascending=False)
ground_truth_df = ground_truth_df.drop_duplicates(subset="base_title", keep="first").copy()

# Remove `last_updated` before saving
if "last_updated" in ground_truth_df.columns:
    ground_truth_df.drop(columns=["last_updated"], inplace=True)

ground_truth_df.to_json(
    processed_file.replace("_processed.jsonl", "_gt.jsonl"),
    orient="records",
    lines=True
)

print(
    r"""
# Basic Stats

Calculated with `annot_result_stats.py`
"""
)

stats_command = [
    "python",
    f"{project_dir}/scripts/data_preprocessing/annot_result_stats.py",
    "--input_file",
    processed_file
]
try:
    # This will raise an error if the command fails
    stats_output = subprocess.check_output(stats_command, text=True)
    print(stats_output)
except subprocess.CalledProcessError as e:
    print(f"Command failed with error: {e}")
    exit(1)

print(
    r"""
# Quality Checking

Flagging tasks with the following issues:

1. Containing marked factuality spans without a comment
2. Identified omissions without an omission severity label
3. Tasks without model certainty
"""
)

print(r"""## Missing Omission Severity""")

df_filtered_no_severity = dataset_df[(~dataset_df.omissions.isna()) & (dataset_df.omission_severity.isna())]
print(f"Number of tasks with omissions without a severity label: {len(df_filtered_no_severity)}")
print(df_filtered_no_severity[['title', 'annotator', 'omissions', 'comments']])

print(r"""## Missing Omission Description""")

df_filtered_no_desc = dataset_df[(dataset_df.omissions.isna()) & (~dataset_df.omission_severity.isna())]
print(f"Number of tasks with an omission severity label but no description: {len(df_filtered_no_desc)}")
print(df_filtered_no_desc[['title', 'annotator', 'omission_severity', 'omissions', 'comments']])

print(df_filtered_no_desc[['title', 'annotator', 'omission_severity', 'omissions', 'comments']]) 

print(r"""## Missing Model Certainty""")

df_filtered_no_certainty = dataset_df[dataset_df.model_certainty.isna()]
print(f"Number of tasks without model certainty: {len(df_filtered_no_certainty)}")
print(f"{df_filtered_no_certainty[['title', 'annotator']]}")

print(
    r"""
## Missing Factuality Comment

Only includes entries with a highlighted span but no explanation.
"""
)

def flag_entries_factuality(row: pd.Series):
    # Check that each marked span has a comment
    for marked_span in row.factuality:
        if not marked_span["comment"]:
            return True
    return False

# Remove null factuality entries
df_filtered_no_fact_comment = dataset_df[~dataset_df.factuality.isna()]
# Filter entries that have marked spans without a comment
df_filtered_no_fact_comment = df_filtered_no_fact_comment[df_filtered_no_fact_comment.apply(flag_entries_factuality, axis="columns")]
print(f"Number of tasks with marked factuality spans without a comment: {len(df_filtered_no_fact_comment)}")
print(f"{df_filtered_no_fact_comment[['title', 'annotator']]}")

print(r"""## Summary of Tasks with Missing Information""")

missing_info_tasks = df_filtered_no_severity.title.values.tolist() + df_filtered_no_fact_comment.title.values.tolist() + df_filtered_no_certainty.title.values.tolist()

missing_info_tasks = list(set(missing_info_tasks))

print(f"Titles with missing information (combined): {len(missing_info_tasks)}")
print("\n".join(missing_info_tasks))

print(r"""Looking at omission comments for tasks without an omission severity label""")

print(df_filtered_no_severity[["id", "title", "omissions"]])

print(r"""Converting the highlighted span annotations to sentence annotations""")

# Expand the sentences into their own dataframe
annotated_sent_df = []
def expand_factuality(row: pd.Series):
    for sent in row.factuality:
        sent["id"] = f"{row.title}_{sent['sentence_id']}"
        sent["annotator"] = row.annotator
        annotated_sent_df.append(sent)
sent_dataset_df.apply(expand_factuality, axis="columns")
annotated_sent_df = pd.DataFrame(annotated_sent_df).set_index("id")

# Reduce to False spans
annotated_sent_df = annotated_sent_df[annotated_sent_df.label==False]

# Fixing factuality severity rendering issue -- maybe not required?

for idx, row in annotated_sent_df.iterrows():
    formatted_annotations = []
    for i, a in enumerate(row.annotations):
        formatted_annotations.append(f"{i}. ({a['span_start']}, {a['span_end']}) [{a['severity']}] <<{a['highlighted_text']}>> {a['comment']}")
    formatted_annotations = "\n".join(formatted_annotations)
    print(f"""{idx} {row.annotator}

Sentence ({row.span_start}, {row.span_end}): <<{row.text}>>

Highlighted span(s):
{formatted_annotations}

---------""")


# [Commenting IAA part for sample files (as there aren't enough files to compute IAA) 
# User can uncomment it later for actual data 
'''

print(
    r"""
# Calculate inter-annotator agreement (IAA)

Reduce the dataset to only the duplicate annotations (those with the same title and/or same title minus "-iaa")
"""
)

raw_label_to_label = {
    "omission_severity": {
        None: "None to Mild",
        "No Error": "None to Mild",
        "Mild - no action is required": "None to Mild",
        "Moderate - may negatively impact the patients health if no action is taken": "Moderate",
        "Severe  may require medical intervention by a doctor": "Severe to Life-threatening",
        "Life-threatening - can be life-threatening without medical intervention": "Severe to Life-threatening"
    },
    "highest_fact_severity": {
        None: "None to Mild",
        "No Error": "None to Mild",
        "Mild": "None to Mild",
        "Moderate": "Moderate",
        "Severe": "Severe to Life-threatening",
        "Life-threatening": "Severe to Life-threatening"
    }
}

def calculate_multichoice_iaa(
    df: pd.DataFrame,
    agreement_col: str,
    id_col: str = "base_title",
    annotator_col: str = "annotator",
    relabel: bool = True
):
    """Calculate Krippendorff's alpha for a multiple-choice column"""
    df = df.copy()
    iaa_results = {}

    def iaa(temp_df, col) -> float:
        try:
            alpha = simpledorff.calculate_krippendorffs_alpha_for_df(
                temp_df,
                experiment_col=id_col,
                annotator_col=annotator_col,
                class_col=col
            )
        except ZeroDivisionError as err:
            alpha = -1

            print(f"Error calculating IAA for column `{col}`: {err}. Setting to {alpha}.\n{temp_df[col].value_counts()=}")
        return alpha

    # Agreement on whether any row has the annotation (assumes Null)
    # Must be done before re-labeling
    df["has_label"] = df[agreement_col].map(lambda x: False if pd.isna(x) or x == "No Error" else True)

    # Check if re-labeling is needed
    if (agreement_col in raw_label_to_label) and relabel:
        df[agreement_col] = df[agreement_col].map(lambda x: raw_label_to_label[agreement_col].get(x, x))
    else:
        df[agreement_col] = df[agreement_col].map(lambda x: "No Error" if pd.isna(x) else x)

    # Overall agreement including labels
    # Most straightforward IAA
    iaa_results["All Labels (Multi)"] = iaa(df, agreement_col)

    # Agreement on whether a document _has_ an annotation (indicative of an error)
    iaa_results["Has Error (Binary)"] = iaa(df, "has_label")

    # Per-label agreement
    all_labels = df[agreement_col].unique()
    for label in all_labels:
        # Create a temporary copy of the dataframe for this calculation
        temp_df = df.copy()

        # This is the key step: Create a new binary column for the current label.
        # It will be True if the severity matches the current label, and False otherwise.
        binary_col_name = f'is_{label}'
        temp_df[binary_col_name] = (temp_df[agreement_col] == label)
        # print(f"{binary_col_name=} {temp_df[binary_col_name].sum() / len(temp_df):.2}")

        # Calculate Krippendorff's alpha on this new binary (True/False) column
        k_alpha = iaa(temp_df, binary_col_name)

        # Store the result
        iaa_results[f"Per-label: {label}"] = k_alpha

    # Display the annotations in an easy-to-compare way
    display_df = df.copy()
    display_df.loc[:, "annotator_rank"] = display_df.groupby(id_col).cumcount().copy() + 1
    display_df = display_df.pivot(
        index=id_col,
        columns='annotator_rank',
        values=['annotator', agreement_col]
    ).copy()
    display_df.columns = [f'{col[0]}_{col[1]}' for col in display_df.columns]    
    display_df = display_df.reset_index()

    results_df = pd.DataFrame.from_dict(
        iaa_results, 
        orient='index', 
        columns=["krippendorffs_alpha"]
    )
    return results_df, display_df

# Loop through each agreement column and generate a combined results table
for RELABEL in [True, False]:
    print("=" * 70)
    print(f"On {RELABEL=}")
    print("=" * 70)
    agreement_columns_to_process = ["model_certainty", "omission_severity", "highest_fact_severity"]
    for agreement_col in agreement_columns_to_process:
        print("=" * 50)
        print(f"Calculating IAA for: '{agreement_col}'")
        print("=" * 50)

        all_results = []

        # --- Calculate IAA for the entire DataFrame ---
        n = iaa_df.base_title.nunique()
        overall_results_df, _ = calculate_multichoice_iaa(iaa_df, agreement_col=agreement_col, relabel=RELABEL)
        overall_results_df['domain'] = f"All (N={n})"
        all_results.append(overall_results_df)

        # --- Calculate IAA for each unique domain ---
        unique_domains = iaa_df['domain'].unique()
        for domain in unique_domains:
            # Filter the DataFrame for the current domain
            domain_df = iaa_df[iaa_df['domain'] == domain].copy()
            n = domain_df.base_title.nunique()

            # Calculate IAA for the domain-specific subset
            domain_results_df, _ = calculate_multichoice_iaa(domain_df, agreement_col=agreement_col, relabel=RELABEL)
            domain_results_df['domain'] = f"{domain} (N={n})"
            all_results.append(domain_results_df)

        # --- Combine all results into a single table ---
        # Concatenate all the individual result DataFrames
        combined_df = pd.concat(all_results)

        # Pivot the table to have domains as columns and metrics as rows
        # We need to reset the index to turn the metric names into a column for pivoting
        combined_df.reset_index(inplace=True)
        final_table = combined_df.pivot(
            index='index',                  # The metric names (e.g., 'All Labels (Multi)')
            columns='domain',               # The domain names ('All Domains', 'PC', 'MH', etc.)
            values='krippendorffs_alpha'    # The values to fill the table
        )

        # Rename the index for clarity
        final_table.index.name = "IAA Metric"

        # Display the final, combined table for the current agreement column
        print(final_table.to_string(float_format="{:.2f}".format))
        print("\n")

print(iaa_df)


#---Raw IAA--- 
print(f"Calculating Raw IAA...")
df = pd.read_json(processed_file, lines=True)
df = df[df.iaa==True]
df["omission_severity"] = df["omission_severity"].fillna("No Error")

# Multi-label
for col in ["highest_fact_severity", "model_certainty", "omission_severity"]:
	# Pivot the dataframe
	pivoted_df = df.pivot(index="base_title", columns="annotator", values=col)
	
	raw_agree = pivoted_df.apply(
		lambda x: x.dropna().nunique() == 1,
		axis="columns"
	)
	# print(pivoted_df.isna().all(axis=1) )
	print(f"====={col} {raw_agree.mean()*100:.2f}======")
    
# Binary
for col in ["omission_severity", "highest_fact_severity"]:
	# Map labels to True/False
	binary_df = df.copy()
	binary_df[col] = binary_df[col].map(lambda x: False if x=="No Error" else True)
	
	# Pivot the dataframe
	pivoted_df = binary_df.pivot(index="base_title", columns="annotator", values=col)
	
	raw_agree = pivoted_df.apply(
		lambda x: x.dropna().nunique() == 1,
		axis="columns"
	)
	print(f"{col} {raw_agree.mean()*100:.1f}")

iaa_df["has_fact_errors"] = ~iaa_df.factuality.isna()

def analyze_overall_fact_disagreement():
    has_error_count = 0
    disagreement_count = 0
    for base_title, df in iaa_df.groupby("base_title"):
        # Skip entries where both annotators do not find issues
        if not df.has_fact_errors.any():
            continue
        has_error_count += 1

        if not df.has_fact_errors.all():
            disagreement_count += 1

        print(f"{base_title}")
        for i, row in df.iterrows():
            print(f"{row.annotator} [{'Has Error' if row.has_fact_errors else 'No Error'}]")
            if not row.factuality:
                continue
            for error_idx, error_desc in enumerate(row.factuality, start=1):
                print(f"\t{error_idx}. [{error_desc['severity']}] {error_desc['comment']}")
                print(f"\t\t<<{error_desc['highlighted_text']}>>")
        print("\n----------\n")

    print(f"{has_error_count=} {disagreement_count=}")

analyze_overall_fact_disagreement()

print(
    r"""
### Factuality Span

Do annotators agree on the spans that contain errors?

Instead of the entire span, we reduce the task to the token-level and the sentence-level.

**Token level:** Is a token in the highlighted span? Label with "Highlighted" or "Not Highlighted"

**Sentence level:** Is the _sentence_ marked as True/False for factuality. This is for the downstream task of MedScore.
"""
)

# Reduce sentence-level DF to IAA tasks
sent_dataset_iaa_df = sent_dataset_df[sent_dataset_df.iaa]
sent_dataset_iaa_df = sent_dataset_iaa_df[["domain", "base_title", "annotator", "factuality"]].copy()

# New function to create one-hot encoded columns
def map_sentence_span_to_one_hot(row: pd.Series):
    """
    Takes a list of annotation dicts and returns a Series (one-hot encoded).
    """
    sentence_annotations = row.factuality
    # Get the labels present in this specific annotation
    new_row = {
        "domain": row.domain,
        "annotator": row.annotator
    }
    new_row.update({
        f"{row.base_title}_sent_{i}": f['label'] for i, f in enumerate(sentence_annotations)
    })
    return new_row

one_hot_df = sent_dataset_iaa_df.apply(map_sentence_span_to_one_hot, axis="columns", result_type="expand")
long_df = pd.melt(
    one_hot_df,
    id_vars=['domain', 'annotator'], # Columns to keep
    var_name='doc_sentence_id',               # Name for the new column of labels
    value_name='label'             # Name for the new column of True/False values
)
# Drop NaN rows (extra sentence labels were added for the shorter responses)
long_df = long_df.dropna(subset=["label"])

def calculate_sentence_agreement():
    all_results = {}
    all_results[f"All"] = simpledorff.calculate_krippendorffs_alpha_for_df(
        long_df,
        experiment_col="doc_sentence_id",
        annotator_col="annotator",
        class_col="label"
    ).item()
    for domain in iaa_df['domain'].unique():
        # Filter the DataFrame for the current domain
        domain_df = long_df[long_df['domain'] == domain].copy()
        all_results[f"{domain}"] = simpledorff.calculate_krippendorffs_alpha_for_df(
            domain_df,
            experiment_col="doc_sentence_id",
            annotator_col="annotator",
            class_col="label"
        ).item()
    df = pd.DataFrame.from_dict(all_results, orient='index', columns=["Sentence IAA"]).T
    return df

fact_sent_iaa = calculate_sentence_agreement()

print(fact_sent_iaa, long_df)

def convert_spans_to_iob(row: pd.Series):
    """
    Converts character-level spans to token-level IOB labels using spaCy.

    Args:
        text (str): The original text.
        spans (list of dicts): A list of spans, where each span is a dictionary
                                with 'start', 'end', and 'label' keys.
                                e.g., [{'start': 10, 'end': 15, 'label': 'PERSON'}]

    Returns:
        list of tuples: A list of (token, iob_label) tuples.
    """
    text, highlighted_spans = row.response, row.factuality
    # Load a spaCy model (the small English model is fine for tokenization)
    doc = nlp(text)
    token_list = [token.text for token in doc]

    # Initialize a list of 'O' (Outside) labels for each token
    iob_labels = ['O'] * len(doc)

    # If there are highlighted spans
    if highlighted_spans:
        # Iterate through the character-level spans
        for span in highlighted_spans:
            start_char = span['span_start']
            end_char = span['span_end']
            label = "highlighted"

            # Create a spaCy Span object to find token boundaries
            # This handles cases where a span might start or end mid-token
            span_obj = doc.char_span(start_char, end_char, label=label)

            if span_obj is not None:
                # Mark the first token as 'B-' (Beginning)
                iob_labels[span_obj.start] = f"B-{label}"
                # Mark subsequent tokens in the span as 'I-' (Inside)
                for i in range(span_obj.start + 1, span_obj.end):
                    iob_labels[i] = f"I-{label}"
                # for i in range(span_obj.start, span_obj.end):
                #     iob_labels[i] = f"I-{label}" 

    # Combine tokens with their IOB labels
    token_annotations = list(zip(token_list, iob_labels))
    new_rows = [{
        "domain": row.domain,
        "annotator": row.annotator,
        "doc_token_id": f"{row.base_title}_token-{i}",
        "token": t,
        "label": l
    } for i, (t, l) in enumerate(token_annotations)]
    return new_rows

# Convert the spans to IOB labels
iaa_token_fact_df = []
for output in iaa_df.apply(convert_spans_to_iob, axis="columns"):
    iaa_token_fact_df.extend(output)
iaa_token_fact_df = pd.DataFrame(iaa_token_fact_df)


def calculate_token_agreement():
    all_results = {}
    all_results[f"All"] = simpledorff.calculate_krippendorffs_alpha_for_df(
        iaa_token_fact_df,
        experiment_col="doc_token_id",
        annotator_col="annotator",
        class_col="label"
    ).item()
    for domain in iaa_df['domain'].unique():
        # Filter the DataFrame for the current domain
        domain_df = iaa_token_fact_df[iaa_token_fact_df['domain'] == domain].copy()
        all_results[f"{domain}"] = simpledorff.calculate_krippendorffs_alpha_for_df(
            domain_df,
            experiment_col="doc_token_id",
            annotator_col="annotator",
            class_col="label"
        ).item()
    df = pd.DataFrame.from_dict(all_results, orient='index', columns=["Token IAA"]).T
    return df


fact_token_iaa = calculate_token_agreement()

print(fact_token_iaa, iaa_token_fact_df)

fact_span_iaa = pd.concat([fact_token_iaa, fact_sent_iaa])
print(fact_span_iaa)
'''