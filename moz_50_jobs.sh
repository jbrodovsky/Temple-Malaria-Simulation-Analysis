#!/bin/sh
#PBS -l walltime=48:00:00
#PBS -N moz_50_jobs
#PBS -q normal
#PBS -l nodes=1:ppn=15
cd $PBS_O_WORKDIR
torque-launch moz_50_cmds.txt
