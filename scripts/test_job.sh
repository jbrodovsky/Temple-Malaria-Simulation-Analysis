#!/bin/sh
#PBS -l walltime=48:00:00
#PBS -N test_job
#PBS -q normal
#PBS -l nodes=1:ppn=28
cd $PBS_O_WORKDIR

echo pwd: $(pwd)
echo launch: $(cat scripts/test_cmds.txt)

torque-launch scripts/test_cmds.txt
