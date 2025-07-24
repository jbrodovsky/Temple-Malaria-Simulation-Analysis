#!/bin/sh
#PBS -l walltime=48:00:00
#PBS -N moz_validation
#PBS -q nd
#PBS -l select=1:ncpus=20:host=hpc
#PBS -o ../log/submit_jobs.output
#PBS -e ../log/submit_jobs.error
cd $PBS_O_WORKDIR

MAX_PARALLEL=20
CURRENT=0
INDEX=0

function wait_for_available_slot() {
    while (( $(jobs -r | wc -l) >= MAX_PARALLEL )); do
        sleep 0.5  # <<< check every 0.5s if a slot is free
    done
}

while IFS= read -r line || [[ -n "$line" ]]; do
    if [[ -n "$line" ]]; then
        wait_for_available_slot  # <<< wait if too many jobs

        echo "Running: $line"
        eval "$line" > ../log/${INDEX}.output 2> ../log/${INDEX}.error &

        ((INDEX++))
    fi
done < ./moz_validation3.txt

# Final wait for all remaining jobs
wait

echo "All jobs completed."
