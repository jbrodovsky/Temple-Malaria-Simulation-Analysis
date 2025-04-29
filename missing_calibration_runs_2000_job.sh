#!/bin/sh
#PBS -l walltime=48:00:00
#PBS -N MissingCalibrationRuns_2000
#PBS -q normal
#PBS -l nodes=4:ppn=28
cd $PBS_O_WORKDIR
torque-launch missing_calibration_runs_2000.txt
