#!/bin/bash

# Usage: ./generate_submit_script.sh MAX_CPU
# Example: ./generate_submit_script.sh 3

if [[ $# -ne 1 ]]; then
    echo "Usage: $0 MAX_CPU"
    exit 1
fi

MAX_CPU=$1
HOST_NAME=$(hostname)

INPUT_TEMPLATE="submit.template"
OUTPUT_SCRIPT="submit.pbs"

if [[ ! -f "$INPUT_TEMPLATE" ]]; then
    echo "Error: $INPUT_TEMPLATE not found!"
    exit 1
fi

# Replace placeholders and generate output
sed -e "s/#MAX_CPU#/${MAX_CPU}/g" -e "s/#HOST_NAME#/${HOST_NAME}/g" "$INPUT_TEMPLATE" > "$OUTPUT_SCRIPT"

chmod +x "$OUTPUT_SCRIPT"

echo "Generated $OUTPUT_SCRIPT successfully with node $HOST_NAME."
