#!/bin/sh
#PBS -l walltime=48:00:00
#PBS -N moz_10000_0.787_0.8_jobs
#PBS -q normal
#PBS -l nodes=4:ppn=28
cd $PBS_O_WORKDIR
torque-launch moz_10000_0.787_0.8_cmds.txt
