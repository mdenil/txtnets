#!/bin/bash

set -e

# time matlab -nodisplay -nosplash -nodesktop -r "run('verify_forward_pass/run_matlab_forward_pass.m')"
time python verify_forward_pass/run_compare_to_matlab.py
