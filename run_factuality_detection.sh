#!/bin/bash

python -m medscore.medscore --config ./factuality_omission_detection/medscore_config.yml --debug

# If the decomposition step is completed, run only the verification step:
# python -m medscore.medscore --config ./factuality_omission_detection/medscore_config.yml --debug --verify_only
