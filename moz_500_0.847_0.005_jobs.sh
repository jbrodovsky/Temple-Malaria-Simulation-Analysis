#!/bin/sh
#PBS -l walltime=48:00:00
#PBS -N moz_500_0.847_0.005_jobs
#PBS -q normal
#PBS -l nodes=4:ppn=28
cd $PBS_O_WORKDIR
torque-launch moz_500_0.847_0.005_cmds.txt
