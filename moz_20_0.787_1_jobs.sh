#!/bin/sh
#PBS -l walltime=48:00:00
#PBS -N moz_20_0.787_1_jobs
#PBS -q normal
#PBS -l nodes=4:ppn=28
cd $PBS_O_WORKDIR
torque-launch moz_20_0.787_1_cmds.txt
