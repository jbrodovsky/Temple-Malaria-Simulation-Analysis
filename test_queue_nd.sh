#!/bin/bash

set -e  # Exit if any command fails
set -o pipefail

# Check argument
if [ $# -ne 1 ]; then
    echo "Usage: $0 <parallel_jobs>"
    exit 1
fi

P_JOBS=$1
HOST_NAME=$(hostname)

# 1. Download quietly
echo "Downloading openpbs_script_python.zip..."
wget --quiet --show-progress --output-document=openpbs_script_python.zip "https://www.dropbox.com/scl/fi/n1ele9c9q0orh96urfbz3/openpbs_script_python.zip?rlkey=w6gy9d85ibrkhr8vuua4fp19u&dl=0"

# 2. Unzip
echo "Unzipping..."
unzip -o openpbs_script_python.zip

# 3. Change directory
echo "cd to openpbs_script_python/script/"
cd openpbs_script_python/script/

# 4. Make generator script executable
chmod +x gen_submit_pbs.sh

# 5. Generate PBS submit script with given node name
echo "Generating submit.pbs for ${P_JOBS} jobs on node ${HOST_NAME}..."
./gen_submit_pbs.sh ${P_JOBS}

# 6. Submit the job
echo "Submitting submit.pbs..."
qsub submit.pbs

# 7. Show job queue
echo "Current qstat:"
qstat -answ1
