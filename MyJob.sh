#!/bin/bash
#PBS -l walltime=48:00:00
#PBS -N MyJob
#PBS -q nd
#PBS -l select=1:ncpus=12:host=nd01
#PBS -o ./MyJob.output
#PBS -e ./MyJob.error

cd $PBS_O_WORKDIR
function wait_for_available_slot() {
while (( $(jobs -r | wc -l) >= 12 )); do
    sleep 0.5 
done
}

while IFS= read -r line || [[ -n "$line" ]]; do
    if [[ -n "$line" ]]; then
        wait_for_available_slot
        echo "Running: $line" 
        eval "$line" > ./MyJob_${INDEX}.output 2> ./MyJob_${INDEX}.error & 
        ((INDEX++))
    fi
done < moz_37.txt

