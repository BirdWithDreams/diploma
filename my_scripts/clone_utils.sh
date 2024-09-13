#!/bin/bash

# Set variables
REPO_URL="https://github.com/Edresson/ZS-TTS-Evaluation.git"
DIRECTORY_TO_CLONE="utils"
BRANCH_NAME="main"
NEW_DIR_NAME="utils"

# Create a new directory and navigate into it
mkdir "$NEW_DIR_NAME"
cd "$NEW_DIR_NAME"

# Initialize a new Git repository
git init

# Add the remote repository
git remote add -f origin "$REPO_URL"

# Enable sparse checkout
git config core.sparseCheckout true

# Specify the directory to clone
echo "$DIRECTORY_TO_CLONE" >> .git/info/sparse-checkout

# Pull the specified directory from the main branch
git pull origin "$BRANCH_NAME"

echo "Successfully cloned the '$DIRECTORY_TO_CLONE' directory from $REPO_URL"