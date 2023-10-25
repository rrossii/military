#!/bin/bash

export ENV_NAME="military"
if conda env list | grep -q "$ENV_NAME"; then
    echo "Conda environment '$ENV_NAME' already exists."
else
    echo "Creating environment and installing dependencies"
    conda create --name $ENV_NAME --file requirements.txt -c conda-forge
    conda install -c conda-forge opencv
    conda install -c anaconda scikit-learn
fi
echo "Creating environment and installing dependencies"
conda create --name $ENV_NAME --file requirements.txt -c conda-forge
conda install -c conda-forge opencv
conda install -c anaconda scikit-learn

echo "Activating environment"
conda activate $ENV_NAME
