#!/bin/bash

# In case of an error stop the script
set -e

echo "Setting up your environment..."

# Uninstall any existing python virtual environments
if [ -d ".venv" ]; then
    echo "Removing existing virtual environment..."
    rm -rf .venv
fi

# Create a new python virtual environment using python3.8
echo "Creating a new virtual environment..."
python3.10 -m venv .venv

# Update pip
echo "Updating pip..."
source .venv/bin/activate
pip install --upgrade pip

# Install the wheel package, needed for the requirements.txt file
echo "Installing wheel package..."
pip install wheel

# Finally install the requirements.txt file
echo "Installing requirements..."

pip install -r requirements.txt

# Delete the datasets folder if it exists
if [ -d "./datasets" ]; then
    echo "Removing existing datasets folder..."
    rm -rf ./datasets
fi

# Create a new datasets folder
echo "Creating datasets folder..."
mkdir datasets

# Set the PYTHONPATH environment variable
export PYTHONPATH=$PWD

# Finally activate the virtual environment
source .venv/bin/activate

# Print completion message
echo "Setup complete!"