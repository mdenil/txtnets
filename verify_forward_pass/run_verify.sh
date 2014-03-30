#!/bin/bash

matlab -nodisplay -nosplash -nodesktop -r "run('verify_forward_pass/run_matlab_forward_pass.m')"
python verify_forward_pass/run_compare_to_python.py
