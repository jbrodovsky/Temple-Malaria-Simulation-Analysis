#!/bin/sh
#PBS -l walltime=48:00:00
#PBS -N moz_75_jobs
#PBS -q normal
#PBS -l nodes=8:ppn=28
cd $PBS_O_WORKDIR
torque-launch scripts/moz/moz_75_cmds.txt
