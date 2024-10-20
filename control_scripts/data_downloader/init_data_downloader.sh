#!/bin/sh

set -e

# source the virtual environment
source .venv/bin/activate

# set the PYTHONPATH environment variable to the current directory
export PYTHONPATH=$PWD

# Trigger the data downloader script
python3 src/downloaders/data_downloader.py