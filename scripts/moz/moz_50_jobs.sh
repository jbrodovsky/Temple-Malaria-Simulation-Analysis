#!/bin/sh
#PBS -l walltime=48:00:00
#PBS -N moz_50_jobs
#PBS -q normal
#PBS -l nodes=8:ppn=28
cd $PBS_O_WORKDIR
torque-launch scripts/moz/moz_50_cmds.txt
