#!/bin/sh
#PBS -l walltime=48:00:00
#PBS -N moz_1000_jobs
#PBS -q normal
#PBS -l nodes=1:ppn=15
cd $PBS_O_WORKDIR
torque-launch moz_1000_cmds.txt
