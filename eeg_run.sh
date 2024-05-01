#!/bin/bash

echo "TESTING"

# Function to check if a command exists
command_exists() {
    command -v "$1" >/dev/null 2>&1
}

# Prompt the user to run the commands or exit
read -p "Press ENTER to install dependencies. Or, any other key to exit: " user_input

# check for response
if [ -z "$user_input" ]; then
    # Install dependencies
    echo "Installing dependencies..."
    pip install pandas numpy tqdm joblib tensorflow scikit-learn
    echo "Finished.\n"
else
    echo "Exiting..."
    exit 0
fi

# Prompt the user to run the command or exit
read -p "Press ENTER to download the dataset. Or, any other key to exit: " user_input

# check for responese
if [ -z "$user_input" ]; then
    # Check if Kaggle CLI is installed
    if ! command_exists kaggle; then
        echo "Kaggle CLI not found. Please install it to continue."
        exit 1
    fi

    # Download dataset
    echo "Downloading dataset..."
    kaggle competitions download -c hms-harmful-brain-activity-classification
    echo "Finished.\n"
else
    echo "Exiting..."
    exit 0
fi

# Prompt the user to run the model or exit
read -p "Press ENTER to run the model. Or, any other key to exit: " user_input

# check for response
if [ -z "$user_input" ]; then
    echo "Running the model...\n"
    python test.py
else
    echo "Exiting..."
    exit 0
fi
