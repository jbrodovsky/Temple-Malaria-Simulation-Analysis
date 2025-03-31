#!/bin/sh
#PBS -l walltime=48:00:00
#PBS -N moz_1000_0.603_0.25_jobs
#PBS -q normal
#PBS -l nodes=4:ppn=28
cd $PBS_O_WORKDIR
torque-launch moz_1000_0.603_0.25_cmds.txt
