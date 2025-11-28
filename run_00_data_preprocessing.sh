#!/bin/bash

# Option 1: Data pre-processing without annotator anonymization
python scripts/data_preprocessing/data_preprocessing_from_annotation_interface.py

# Option 2: Data pre-processing with anonymization of annotators
python scripts/data_preprocessing/data_preprocessing_from_annotation_interface.py --anonymous