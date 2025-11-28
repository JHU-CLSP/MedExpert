import os
import argparse

import pandas as pd


def count_annotator_tasks_detailed(file_path):
    """
    Count and list how many tasks each annotator has completed,
    broken down by domain, model, and hypothesis.
    
    Args:
        file_path (str): Path to the TSV file or JSONL file containing the results.
    
    Returns:
        tuple: (summary_counts, detailed_breakdown)
    """
    # Read the TSV or JSON file
    if file_path.endswith('.tsv'):
        df = pd.read_csv(file_path, sep='\t')
    elif file_path.endswith('.jsonl'):
        df = pd.read_json(file_path, lines=True)
    else:
        raise ValueError("Unsupported file format. Please provide a TSV or JSON file.")

    # Parse the title column
    def parse_title(title):
        """Parse title in format: domain-model-hypothesis"""
        try:
            parts = title.split('-')
            if len(parts) >= 3:
                domain = parts[0]
                model = parts[1]
                hypothesis_part = '-'.join(parts[2:])  # In case there are more hyphens
                
                # Determine hypothesis category
                if 'h3-iaa' in hypothesis_part.lower():
                    hypothesis = 'h3-iaa'
                elif '-iaa' in hypothesis_part.lower():
                    hypothesis = '-iaa'
                elif 'h3' in hypothesis_part.lower():
                    hypothesis = 'h3'
                else:
                    hypothesis = 'Qonly'
                
                return domain, model, hypothesis
            else:
                return 'Unknown', 'Unknown', 'Unknown'
        except:
            return 'Unknown', 'Unknown', 'Unknown'
    
    # Apply parsing to create new columns
    df[['domain', 'model', 'hypothesis']] = df['title'].apply(
        lambda x: pd.Series(parse_title(x))
    )
    
    # Remove duplicate tasks per annotator (keep first occurrence)
    df_dedup = df.drop_duplicates(subset=['annotator', 'title'], keep='first')
    
    # Basic count per annotator
    basic_counts = df_dedup['annotator'].value_counts().sort_index()
    
    # Detailed breakdown
    detailed_counts = df_dedup.groupby(['annotator', 'domain', 'model', 'hypothesis']).size().reset_index(name='count')
    
    # Display basic results
    print("=== BASIC TASK COUNTS PER ANNOTATOR ===")
    print("-" * 40)
    for annotator, count in basic_counts.items():
        print(f"{annotator}: {count} tasks")
    
    print(f"\nTotal annotators: {len(basic_counts)}")
    print(f"Total tasks: {basic_counts.sum()}")
    
    # Display detailed breakdown
    print("\n=== DETAILED BREAKDOWN BY ANNOTATOR ===")
    print("-" * 50)
    
    for annotator in sorted(df['annotator'].unique()):
        print(f"\n{annotator}:")
        annotator_data = detailed_counts[detailed_counts['annotator'] == annotator]
        
        for _, row in annotator_data.iterrows():
            print(f"  {row['domain']}-{row['model']}-{row['hypothesis']}: {row['count']} tasks")
        
        total = annotator_data['count'].sum()
        print(f"  Total: {total} tasks")
    
    return basic_counts, detailed_counts, df

def create_summary_tables(detailed_counts, df):
    """Create summary tables for different breakdowns"""
    
    print("\n=== SUMMARY TABLES ===")
    
    # By Domain
    print("\n1. Tasks by Domain:")
    domain_summary = detailed_counts.groupby(['annotator', 'domain'])['count'].sum().unstack(fill_value=0)
    print(domain_summary)
    
    # By Model
    print("\n2. Tasks by Model:")
    model_summary = detailed_counts.groupby(['annotator', 'model'])['count'].sum().unstack(fill_value=0)
    print(model_summary)
    
    # By Hypothesis
    print("\n3. Tasks by Hypothesis:")
    hypothesis_summary = detailed_counts.groupby(['annotator', 'hypothesis'])['count'].sum().unstack(fill_value=0)
    print(hypothesis_summary)
    
    # Overall distribution
    print("\n4. Overall Distribution:")
    overall_dist = detailed_counts.groupby(['domain', 'model', 'hypothesis'])['count'].sum().reset_index(name='total_assignments')
    print(overall_dist.to_string(index=False))
    
    return domain_summary, model_summary, hypothesis_summary, overall_dist

def print_annotator_tasks(df):
    """
    Print the task titles that each annotator has submitted.
    
    Args:
        df (pandas.DataFrame): The dataframe with parsed data
    """
    # Remove duplicates to get unique tasks per annotator
    df_dedup = df.drop_duplicates(subset=['annotator', 'title'], keep='first')
    directory_path = "TMPDONE/" 
    os.makedirs(directory_path, exist_ok=True)
    for annotator in sorted(df_dedup['annotator'].unique()):
        annotator_tasks = df_dedup[df_dedup['annotator'] == annotator]['title'].sort_values()
        # Create filename
        filename = f"{directory_path}"f"{annotator}.txt"
        
        # Save to file
        with open(filename, 'w', encoding='utf-8') as f:
            for title in annotator_tasks:
                f.write(f"{title}\n")
        
        print(f"Saved {len(annotator_tasks)} tasks for {annotator} to {filename}")

# Main function
def analyze_annotator_tasks(file_path):
    """
    Complete analysis of annotator tasks with domain/model/hypothesis breakdown
    
    Args:
        file_path (str): Path to the TSV file
    
    Returns:
        dict: All analysis results
    """
    
    # Get basic and detailed counts
    basic_counts, detailed_counts, df = count_annotator_tasks_detailed(file_path)
    
    # Create summary tables
    domain_summary, model_summary, hypothesis_summary, overall_dist = create_summary_tables(detailed_counts, df)
    
    # Print task titles for each annotator
    print_annotator_tasks(df)
    
    return {
        'basic_counts': basic_counts,
        'detailed_counts': detailed_counts,
        'domain_summary': domain_summary,
        'model_summary': model_summary,
        'hypothesis_summary': hypothesis_summary,
        'overall_distribution': overall_dist
    }


##########
# Main
##########
def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_file", type=str, help="Input file in JSON format", default="results.tsv")
    return parser.parse_args()

if __name__ == "__main__":
    args = parse_args()
    results = analyze_annotator_tasks(args.input_file)

