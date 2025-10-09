#!/bin/bash
# This is a basic build script for this project. Locally I use uv to manage my virtual environments.
# Basic Ubuntu server does not have uv installed, so we'll make due with python3 -m venv for now.

# Create virtual environment if it doesn't exist
if [ ! -d ".venv" ]; then
    python3 -m venv .venv
fi
echo "Virtual environment folder is created."
# Activate the virtual environment
source .venv/bin/activate
# Install the package
pip install -e .