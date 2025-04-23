#!/bin/sh
#PBS -l walltime=48:00:00
#PBS -N moz_30_jobs
#PBS -q normal
#PBS -l nodes=8:ppn=28
cd $PBS_O_WORKDIR
torque-launch jobs/jobs/moz_30_cmds.txt
