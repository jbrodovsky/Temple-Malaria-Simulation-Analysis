# Find an submit all jobs in the current directory
# Usage: ./launch_jobs.sh

for i in $(ls *.sh); do
    echo "Submitting job $i"
    qsub $i
done