# MedExpert
Code for the "MedExpert: An Expert-Annotated Dataset for Medical Chatbot Evaluation" paper at Machine Learning for Health (ML4H) 2025.

Paper Link: 

Dataset release on Hugging Face: 

## Overview


### Setup Instructions

1. Clone the repository:
   ```bash
   git@github.com:JHU-CLSP/MedExpert.git
   cd MedExpert
   ```

2. Copy `example.env` to `.env` and fill in the required environment variables.
   ```bash
   cp example.env .env
   ```

3. Install the required packages:
   ```bash
   conda env create -f environment.yml
   conda activate medexpert
   pip install git+https://github.com/Heyuan9/MedScore.git --no-deps
   ```

## Data Preparation

1. Download the MedExpert dataset from the Hugging Face link provided above.
2. Unzip the dataset and place it in the `data/` directory.


---

## MedExpert Benchmark: Factuality and Omission Detection Systems

We include the code and instructions to run the factuality and omission detection systems discussed in the MedExpert paper.

The following commands assume you have set up the `medexpert` environment as described above.

### Factuality / Hallucination Detection

We evaluate two factuality detection systems:

1. **MedScore+GPT-4o Knowledge**
1. **MedScore+MedRAG**

Both can be run with 

```bash
./run_factuality_detection.sh
```

### Omission Detection

We evaluate two omission detection systems:
1. **Zero-shot Omission Detector**
1. **HealthBench-ICL**

Both can be run with 

```bash
./run_omission_detection.sh
```

Note that the `HealthBench-ICL` dataset automatically downloads with the script.
