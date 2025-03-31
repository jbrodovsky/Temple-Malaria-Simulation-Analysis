#!/bin/sh
#PBS -l walltime=48:00:00
#PBS -N moz_5000_0.787_0.5_jobs
#PBS -q normal
#PBS -l nodes=4:ppn=28
cd $PBS_O_WORKDIR
torque-launch moz_5000_0.787_0.5_cmds.txt
