#!/bin/sh
#PBS -l walltime=48:00:00
#PBS -N moz_5000_jobs
#PBS -q normal
#PBS -l nodes=3:ppn=15
cd $PBS_O_WORKDIR
torque-launch missing_calibration_runs_5000_0.847.txt
