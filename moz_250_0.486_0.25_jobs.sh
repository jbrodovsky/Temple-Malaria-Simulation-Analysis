#!/bin/sh
#PBS -l walltime=48:00:00
#PBS -N moz_250_0.486_0.25_jobs
#PBS -q normal
#PBS -l nodes=4:ppn=28
cd $PBS_O_WORKDIR
torque-launch moz_250_0.486_0.25_cmds.txt
