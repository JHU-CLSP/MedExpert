from datasets import load_dataset
import numpy as np
import pandas as pd
from collections import Counter

# -- Load Datasets --
# Primary dataset for experiments
dataset = load_dataset("sonal-ssj/MedExpert", 'medexpert-benchmark')
medexpert_df = dataset['train'].to_pandas()

# Dataset with double annotations (for reference, though not used in calculation below)
dataset_all = load_dataset("sonal-ssj/MedExpert", 'medexpert-all')
medexpert_all_df = dataset_all['train'].to_pandas()

# -- Configuration --
MODEL_ORDER = ['llama2', 'llama3', 'olmo2', 'gemma2', 'biollm']

# Mappings for cleaning up long severity names
SEVERITY_MAPPING = {
    'Life-threatening - can be life-threatening without medical intervention': 'Life-threatening',
    'Mild - no action is required': 'Mild',
    'Moderate - may negatively impact the patients health if no action is taken': 'Moderate',
    'Severe  may require medical intervention by a doctor': 'Severe',
    'Severe - may require medical intervention by a doctor': 'Severe', # Handling potential typo variations
}

COLUMN_ORDER = ['No Error', 'Mild', 'Moderate', 'Severe', 'Life-threatening']
OMISSION_ORDER = ['No omission', 'Mild', 'Moderate', 'Severe', 'Life-threatening']


# ==========================================
# Table 2 :Percentage of responses by factuality and completeness error severity.
# ==========================================
print('='*100)
print(" Table 2: Percentage of responses by factuality and completeness error severity.")
print('='*100)

def create_summary_table(df, target_col, nan_label, sort_order):
    """
    Generates the crosstab summary table with 'All' and 'N.' rows.
    """
    # 1. Create N table (Raw counts)
    # We do this first because 'All' percentages are derived from totals, not average of percentages
    n_table = pd.crosstab(df['model'], df[target_col], dropna=False)
    
    # Rename columns
    rename_map = SEVERITY_MAPPING.copy()
    rename_map[np.nan] = nan_label
    rename_map[None] = nan_label
    
    n_table.rename(columns=rename_map, inplace=True)
    
    # Ensure all expected columns exist (fill missing with 0)
    for col in sort_order:
        if col not in n_table.columns:
            n_table[col] = 0
            
    # Reorder columns and rows
    n_table = n_table[sort_order].reindex(MODEL_ORDER)
    
    # 2. Calculate 'N.' row (Sum of counts per category)
    n_row = n_table.sum(axis=0)
    n_row.name = 'N.'
    
    # 3. Calculate 'All' row (Percentage based on global counts)
    grand_total = n_row.sum()
    if grand_total > 0:
        all_row = (n_row / grand_total * 100)
    else:
        all_row = pd.Series(0, index=n_row.index)
    all_row.name = 'All'
    
    # 4. Create Main Percentage Table (Per model)
    # Normalize along index (row) -> percentages
    pct_table = n_table.div(n_table.sum(axis=1), axis=0) * 100
    
    # 5. Combine
    final_table = pd.concat([pct_table, all_row.to_frame().T, n_row.to_frame().T])
    
    # 6. Formatting
    # Format everything except 'N.' to 1 decimal place
    display_table = final_table.astype(object)
    rows_to_fmt = display_table.index.difference(['N.'])
    display_table.loc[rows_to_fmt] = display_table.loc[rows_to_fmt].astype(float).map('{:.1f}'.format)
    display_table.loc['N.'] = display_table.loc['N.'].astype(int)
    
    return display_table

# --- Run for Factual Errors (per response) ---
fact_table = create_summary_table(
    medexpert_df, 
    target_col='highest_fact_severity', 
    nan_label='No Error', 
    sort_order=COLUMN_ORDER
)
print('\nRatio of Factual Errors per 100 responses, by Model:\n')
print(fact_table)
print('-'*100)

# --- Run for Omission Errors (per response) ---
omission_table = create_summary_table(
    medexpert_df, 
    target_col='omission_severity', 
    nan_label='No omission', 
    sort_order=OMISSION_ORDER
)
print('\nRatio of Omission Errors per 100 responses, by Model:\n')
print(omission_table)
print('-'*100)


# ==========================================
# Table 6: Percentage of sentences by factuality error severity
# ==========================================

print('='*100)
print(" Table 6: Percentage of sentences by factuality error severity")
print('='*100)

def count_severities(row):
    """
    Counts occurrences of severity types in the 'factuality' column.
    """
    severity_counts = Counter()
    
    # FIX: Handle None or empty data safely without ambiguous boolean checks
    factuality_data = row.get('factuality')
    
    # If None or NaN, treat as empty list
    if factuality_data is None or (isinstance(factuality_data, float) and np.isnan(factuality_data)):
        factuality_data = []

    for item in factuality_data:
        # Check for "no error" (empty annotations list)
        annotations = item.get('annotations')
        if annotations is None or len(annotations) == 0:
            severity_counts['no-error'] += 1
        else:
            # Count specific severities
            for annotation in annotations:
                sev = annotation.get('severity')
                if sev:
                    severity_counts[str(sev).lower()] += 1

    # Format keys for the dataframe
    final_counts = {f"n_count_fact-{level}": count for level, count in severity_counts.items()}
    return pd.Series(final_counts)

# Get sentence counts
medexpert_df['n_sentences_in_response'] = medexpert_df['factuality'].apply(len)

# Apply the fixed function
df_counts = medexpert_df.apply(count_severities, axis=1).fillna(0).astype(int)
df = medexpert_df.join(df_counts)

fact_cols = [
    'n_count_fact-no-error', 'n_count_fact-mild', 
    'n_count_fact-moderate', 'n_count_fact-severe', 
    'n_count_fact-life-threatening'
]

# Calculate Ratios
for col in fact_cols:
    ratio_col = col.replace('n_count_fact-', 'ratio_')
    # Avoid division by zero
    df[ratio_col] = df.apply(
        lambda x: (x[col] / x['n_sentences_in_response'] * 100) if x['n_sentences_in_response'] > 0 else 0, 
        axis=1
    )

# Aggregation by Model
ratio_cols = [col.replace('n_count_fact-', 'ratio_') for col in fact_cols]
result_df = df.groupby('model')[ratio_cols].mean()

# Rename columns for display
col_rename = {
    'ratio_no-error': 'No Error',
    'ratio_mild': 'Mild',
    'ratio_moderate': 'Moderate',
    'ratio_severe': 'Severe',
    'ratio_life-threatening': 'Life-threatening'
}
result_df = result_df.rename(columns=col_rename).reindex(MODEL_ORDER)

# Calculate 'All' (Weighted Average)
total_sentences = df['n_sentences_in_response'].sum()
all_row_data = {}
if total_sentences > 0:
    for fact_col, ratio_col in zip(fact_cols, ratio_cols):
        display_name = col_rename[ratio_col]
        all_row_data[display_name] = (df[fact_col].sum() / total_sentences) * 100

# Calculate 'N.' (Raw Counts)
n_row_data = {}
for fact_col, ratio_col in zip(fact_cols, ratio_cols):
    display_name = col_rename[ratio_col]
    n_row_data[display_name] = df[fact_col].sum()

# Add summary rows
if all_row_data:
    result_df = pd.concat([result_df, pd.DataFrame([all_row_data], index=['All'])])
if n_row_data:
    result_df = pd.concat([result_df, pd.DataFrame([n_row_data], index=['N.'])])

# Formatting for Display
display_df = result_df.astype(object)
percentage_rows = display_df.index.difference(['N.'])
display_df.loc[percentage_rows] = display_df.loc[percentage_rows].astype(float).map('{:.1f}'.format)
if 'N.' in display_df.index:
    display_df.loc['N.'] = display_df.loc['N.'].astype(int)

print('-'*100)
print("Ratio of Factual Errors per 100 Sentences, by Model:\n")
print(display_df)
print('-'*100)
