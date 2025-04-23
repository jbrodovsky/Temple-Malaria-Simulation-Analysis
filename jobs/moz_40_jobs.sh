#!/bin/sh
#PBS -l walltime=48:00:00
#PBS -N moz_40_jobs
#PBS -q normal
#PBS -l nodes=3:ppn=15
cd $PBS_O_WORKDIR
torque-launch missing_calibration_runs_40_0.847.txt
