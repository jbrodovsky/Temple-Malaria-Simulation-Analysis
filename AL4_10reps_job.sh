#!/bin/sh
#PBS -l walltime=48:00:00
#PBS -N AL4_10reps
#PBS -q normal
#PBS -l nodes=1:ppn=28
cd $PBS_O_WORKDIR
torque-launch AL4_10reps.txt
