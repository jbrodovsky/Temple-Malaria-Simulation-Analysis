# This script is used to generate the commands.txt file used for torque-launch
# The commands.txt file is used to launch the jobs on the cluster as such:
# ```shell
# torque-launch commands.txt
# ```
# Commands should be in the following format:
# ./bin/MaSim -i ./input/<input_file>.yml -o ./output/<output_folder> -r SQLiteDistrictReporter
# ----
# Check if the number of arguments is correct
if [ "$#" -ne 3 ]; then
    echo "Generate commands for running MaSim on the cluster."
    echo "Usage: $0 <input_file> <output_folder_root> <num_runs>"
    echo "Example: $0 ./input/kenya/strategy.yml ./output/kenya 100"
    exit 1
fi
# Get the desired input configuration file
input_file=$1 # e.g. ./input/<country>/<strategy>.yml
# Validate that the file exists
if [ ! -f $input_file ]; then
    echo "File $input_file does not exist."
    exit 1
fi
# Parse the strategy
strategy=$(basename $input_file .yml)
# Get the desired output folder
output_folder=$2 # e.g. ./output/<country>
output_folder=$output_folder/$strategy
if [[ $output_folder != "${output_folder//\/\//\/}" ]]; then
    echo "Output folder $output_folder is not valid. Did you forget to remove the trailing '/'?"
    exit 1
fi
# Create the output folder if it does not exist
if [ ! -d $output_folder ]; then
    mkdir -p $output_folder
else
    echo "Output folder $output_folder already exists. Overwrite? (y/n)"
    read -r response
    if [ "$response" != "y" ]; then
        exit 1
    fi
fi
# Get the number of runs
num_runs=$3
# Generate the commands file name based on the input file
commands_file=$(basename $input_file .yml)_${num_runs}reps.txt
# Generate the commands.txt file
for i in $(seq 1 $num_runs); do
    echo "./bin/MaSim -i $input_file -o $output_folder/run_$i.db -r SQLiteDistrictReporter"
done > $commands_file
# Create the job script
job_script=$(basename $input_file .yml)_${num_runs}reps_job.sh
echo "#!/bin/sh" > $job_script
echo "#PBS -l walltime=48:00:00" >> $job_script
echo "#PBS -N $(basename $input_file .yml)_${num_runs}reps" >> $job_script
echo "#PBS -q normal" >> $job_script
echo "#PBS -l nodes=1:ppn=28" >> $job_script
echo "cd \$PBS_O_WORKDIR" >> $job_script
echo "torque-launch $commands_file" >> $job_script
# Display the commands file
echo "Job file and commands file generated for $input_file with $num_runs runs."
echo "Commands file: $commands_file"
cat $commands_file
echo "Job script: $job_script"
cat $job_script
echo "To submit the job, run: qsub $job_script"