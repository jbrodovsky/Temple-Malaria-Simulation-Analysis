#!/bin/sh
#PBS -l walltime=48:00:00
#PBS -N MyJob
#PBS -q normal
#PBS -l nodes=4:ppn=28
cd $PBS_O_WORKDIR
torque-launch missing_calibration_runs_75_0.7.txt
