#!/bin/sh
#PBS -l walltime=48:00:00
#PBS -N Test_Job
#PBS -q normal
#PBS -l nodes=1:ppn=1
cd $PBS_O_WORKDIR
torque-launch test_cmds.txt
