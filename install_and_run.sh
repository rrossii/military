#!/bin/bash

echo "Creating environment and installing dependencies..."
export ENV_NAME=military-klym
conda create --name ENV_NAME --file requirements.txt -c conda-forge
conda install -c conda-forge opencv
conda install -c anaconda scikit-learn

echo "Activating environment"
conda activate ENV_NAME


echo "Run the program"
python3 military.py
