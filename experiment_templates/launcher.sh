#!/bin/bash

##PBS -l procs=1
#PBS nodes=1:ppn={{ppn}}
#PBS -l walltime={{walltime}}
#PBS -l pmem={{pmem}}
#PBS -j oe
#PBS -o {{job_dir}}/$PBS_JOBID.out
#PBS -N txtnets_{{job_id}}

set -e

echo "START: $(date +%s)"

cd "{{code_root}}"
echo "Code root is: {{code_root}}"
echo "Job dir is: {{job_dir}}"
echo "Current working directory is $(pwd)"

bash run.sh python "{{job_dir}}/train.py"

echo "FINISH: $(date +%s)"

touch "{{job_dir}}/DONE"
