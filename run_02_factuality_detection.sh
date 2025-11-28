#!/bin/bash

# 1. Load environment variables from .env file
# This ignores comments (#) and exports keys to the shell environment
if [ -f .env ]; then
    export $(grep -v '^#' .env | xargs)
else
    echo "WARNING: .env file not found in current directory!"
fi

# 2. Run the python module for factuality detection
python -m medscore.medscore --config ./scripts/factuality_omission_detection/medscore_config.yml --debug

# If the decomposition step is completed, run only the verification step:
# python -m medscore.medscore --config ./scripts/factuality_omission_detection/medscore_config.yml --debug --verify_only
