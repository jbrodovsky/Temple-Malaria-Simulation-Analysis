#!/bin/sh
#PBS -l walltime=48:00:00
#PBS -N moz_1000_jobs
#PBS -q normal
#PBS -l nodes=8:ppn=28
cd $PBS_O_WORKDIR
torque-launch moz_1000_cmds.txt
