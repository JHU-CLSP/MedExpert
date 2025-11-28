#!/bin/bash

# Download data to default `data` directory
python scripts/data_download_stats/download_data.py

# Run data statistics as in paper
python scripts/data_download_stats/get_data_stats.py
