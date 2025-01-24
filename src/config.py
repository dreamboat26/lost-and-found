import ruamel.yaml as yaml
import os
from pyprojroot import here
from box import Box
import logging
import logging.config
from pathlib import Path

"""
This file is going to look for the closest .here file.
The structure of your code should be
project_folder/
.
.
.
|---configs/
|    |---config.yml
|    |---
|
|---src
|---.here
.
.
.
To test: Run the code from the project root directory (eg. python src/config.py)
"""
# importing the logging.yaml file
_ROOT = Path(here())
_CONFIG_FOLDER = "config"
_CONFIG_FILENAME = "logging.yml"
log_config_path = _ROOT / _CONFIG_FOLDER / _CONFIG_FILENAME

os.makedirs(_ROOT / "logs", exist_ok=True)

if os.path.exists(log_config_path):
    log_config = yaml.safe_load(open(log_config_path, "r"))
    logging.config.dictConfig(log_config)
else:
    raise FileNotFoundError(
        f"Log yaml configuration file not found in {log_config_path}"
    )
