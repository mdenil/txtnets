#!/bin/bash

set -e

ROOT="$(pwd)"
ENV="$ROOT/env"
EXTERNAL="$ROOT/external"
LIB="$ROOT/lib"
PLATFORM="$(uname)"

function safe_call {
    # usage:
    #   safe_call function param1 param2 ...

    HERE="$(pwd)"
    "$@"
    cd "$HERE"
}

function conda_install {
    conda install --yes "$@"
}

function pip_install {
    pip install "$@"
}

function install_pycuda {
    cd "$EXTERNAL"

    safe_call pip_install Mako

    git clone --recursive http://git.tiker.net/trees/pycuda.git
    cd pycuda

    python configure.py
    python setup.py build
    python setup.py install
}

function install_scikits-cuda {
    cd "$EXTERNAL"

    git clone https://github.com/lebedov/scikits.cuda.git
    cd scikits.cuda

    python setup.py install
}

mkdir -p "$EXTERNAL"
mkdir -p "$LIB"

export LD_LIBRARY_PATH="$LIB"

conda create --yes --prefix "$ENV" python pip
source activate "$ENV"

# you also need to install fftw

safe_call conda_install numpy scipy mkl
safe_call conda_install matplotlib
safe_call conda_install psutil
safe_call conda_install nose
safe_call pip_install pyprind
safe_call pip_install pyfftw
safe_call pip_install --pre line_profiler
