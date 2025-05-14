#!/bin/sh
#PBS -l walltime=48:00:00
#PBS -N validation
#PBS -q normal
#PBS -m bae
#PBS -M jbrodovsky@temple.edu
#PBS -l nodes=1:ppn=28
cd $PBS_O_WORKDIR

./bin/MaSim -i conf/moz/calibration/moz_birth_rate_cal.yml -o output/moz/birth_validation/birth_calibration_ -r SQLiteDistrictReporter