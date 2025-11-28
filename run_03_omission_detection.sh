#!/bin/bash
# Description: Omission evaluation
# Usage: ./run_omission_detection.sh

# Set variables from the .env file
set -a
source .env
set +a

# Define the output directory
OUTPUT_DIR="results"
mkdir -p "${OUTPUT_DIR}"

# Run the Zero-shot Omission model
echo "Starting omission detection..."
python "${MEDEXPERT_REPO}/scripts/factuality_omission_detection/omission_grader.py" \
  --input_file "${MEDEXPERT_DATA}/medexpert-benchmark.jsonl" \
  --model_name "gpt-4o-2024-11-20" \
  --server_path "https://api.openai.com/v1" \
  --api_key "${OPENAI_API_KEY}" \
  --output_dir "${OUTPUT_DIR}"

if [ $? -ne 0 ]; then
  echo "Zero-shot Omission Detection evaluation failed."
  exit 1
fi
echo "Zero-shot Omission Detection completed."


# Run the HealthBench ICL model
echo "Starting HealthBench ICL evaluation..."

# Prepare the HealthBench ICL dataset file
if [ ! -f "${MEDEXPERT_REPO}/scripts/factuality_omission_detection/healthbench_icl_dataset.jsonl" ]; then
  echo "Preparing HealthBench ICL dataset..."
  python "${MEDEXPERT_REPO}/scripts/factuality_omission_detection/prepare_healthbench_icl_data.py"
fi

python "${MEDEXPERT_REPO}/scripts/factuality_omission_detection/healthbench_icl_grader.py" \
  --input_file "${MEDEXPERT_DATA}/medexpert-benchmark.jsonl" \
  --model_name "gpt-4o-2024-11-20" \
  --server_path "https://api.openai.com/v1" \
  --api_key "${OPENAI_API_KEY}" \
  --similarity_model "all-mpnet-base-v2" \
  --icl_dataset_file "${MEDEXPERT_REPO}/scripts/factuality_omission_detection/healthbench_icl_dataset.jsonl" \
  --cache_embeddings \
  --output_dir "${OUTPUT_DIR}"

if [ $? -ne 0 ]; then
  echo "HealthBench ICL evaluation failed."
  exit 1
fi
echo "HealthBench ICL evaluation completed."
