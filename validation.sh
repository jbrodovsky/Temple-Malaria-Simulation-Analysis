#!/bin/sh
#PBS -l walltime=48:00:00
#PBS -N validation
#PBS -q normal
#PBS -m bae
#PBS -M jbrodovsky@temple.edu
#PBS -l nodes=1:ppn=28
cd $PBS_O_WORKDIR

# module load gcc/12.2.0

./bin/MaSim -i conf/moz/test.yml -o moz_test_ -r SQLiteDistrictReporter -j 98