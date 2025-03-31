#!/bin/sh
#PBS -l walltime=48:00:00
#PBS -N moz_100_0.451_0.5_jobs
#PBS -q normal
#PBS -l nodes=4:ppn=28
cd $PBS_O_WORKDIR
torque-launch moz_100_0.451_0.5_cmds.txt
