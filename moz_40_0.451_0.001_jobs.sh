#!/bin/sh
#PBS -l walltime=48:00:00
#PBS -N moz_40_0.451_0.001_jobs
#PBS -q normal
#PBS -l nodes=4:ppn=28
cd $PBS_O_WORKDIR
torque-launch moz_40_0.451_0.001_cmds.txt
