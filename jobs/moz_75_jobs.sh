#!/bin/sh
#PBS -l walltime=48:00:00
#PBS -N moz_75_jobs
#PBS -q normal
#PBS -l nodes=3:ppn=15
cd $PBS_O_WORKDIR
torque-launch missing_calibration_runs_75_0.847.txt
