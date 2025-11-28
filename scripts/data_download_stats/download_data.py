from datasets import load_dataset
import os

def download_medexpert(data_dir="./data"):
    """
    Dpwnload MedExpert and save as JSONL file in data_dir
    """
    os.makedirs(data_dir, exist_ok=True)
    
    
    # -- Load MedExpert-Benchmark dataset (N=540) --
    # Note: We primarily use this data in our experiments (except for IAA computation)
    dataset = load_dataset("sonal-ssj/MedExpert", "medexpert-benchmark")
    dataset['train'].to_json(f"{data_dir}/medexpert-benchmark.jsonl")
    print(f"✓ Saved MedExpert-Benchmark: {len(dataset['train'])} examples at {data_dir}/medexpert-benchmark.jsonl")
    
    # -- Load MedExpert-Benchmark dataset (N=572) --
    # consisting of 540 primary examples plus 32 cases with double annotations for inter-annotator agreement
    dataset_all = load_dataset("sonal-ssj/MedExpert", "medexpert-all")
    dataset_all['train'].to_json(f"{data_dir}/medexpert-all.jsonl")
    print(f"✓ Saved MedExpert-All: {len(dataset_all['train'])} examples at {data_dir}/medexpert-all.jsonl")


download_medexpert()
