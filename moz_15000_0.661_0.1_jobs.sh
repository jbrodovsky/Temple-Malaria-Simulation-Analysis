#!/bin/sh
#PBS -l walltime=48:00:00
#PBS -N moz_15000_0.661_0.1_jobs
#PBS -q normal
#PBS -l nodes=4:ppn=28
cd $PBS_O_WORKDIR
torque-launch moz_15000_0.661_0.1_cmds.txt
