#!/bin/sh
#PBS -l walltime=48:00:00
#PBS -N MyJob
#PBS -q normal
#PBS -l nodes=620:ppn=15
cd $PBS_O_WORKDIR
torque-launch batch_commands.txt
