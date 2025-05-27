#!/bin/bash
#PBS -l walltime=100:00:00
#PBS -N validation
#PBS -q nd
#PBS -l select=1:ncpus=5:host=nd01
#PBS -o submit_jobs.output
#PBS -e submit_jobs.error

cd $PBS_O_WORKDIR

./bin/MaSim -i conf/moz/moz_test.yml -o moz_test_ -r SQLiteDistrictReporter -j 98