#!/bin/sh
#PBS -l walltime=48:00:00
#PBS -N moz_100_jobs
#PBS -q normal
#PBS -l nodes=3:ppn=15
cd $PBS_O_WORKDIR
torque-launch missing_calibration_runs_100_0.847.txt
