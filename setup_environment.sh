#!/bin/bash

set -e

ROOT="$(pwd)"
DATA="$ROOT/data"
RESULTS="$ROOT/results"
PLATFORM="$(uname)"
LOCAL_TAG="$PLATFORM"
EXTERNAL="$ROOT/venvs/$LOCAL_TAG/external"
LIB="$ROOT/venvs/$LOCAL_TAG/lib"
BIN="$ROOT/venvs/$LOCAL_TAG/bin"
ENV="$ROOT/venvs/$LOCAL_TAG/env"

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

function install_cld2 {
    cd "$EXTERNAL"

    # build cld2
    svn checkout http://cld2.googlecode.com/svn/trunk/ cld2

    cd cld2/internal/
    if [[ "$PLATFORM" == "Darwin" ]]; then
        sed -i -e 's/^g++/g++-4.8/' compile_libs.sh
    fi
    ./compile_libs.sh
    mv libcld2_full.so libcld2.so "$LIB"

    # build python bindings for cld2
    cd "$EXTERNAL"

    hg clone https://code.google.com/p/chromium-compact-language-detector/
    cd chromium-compact-language-detector

    sed -i -e "/include_dirs/ s~$~library_dirs=\['$LIB'\],~" setup.py setup_full.py

    python setup.py install
    python setup_full.py install
}

function install_word2vec {
    cd "$EXTERNAL"

    svn checkout http://word2vec.googlecode.com/svn/trunk/ word2vec
    cd word2vec
    make

    cp word2vec "$BIN/."
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

function install_reikna {
    cd "$EXTERNAL"

    git clone https://github.com/Manticore/reikna.git

    cd reikna

    python setup.py install
}


mkdir -p "$EXTERNAL"
mkdir -p "$LIB"
mkdir -p "$BIN"
mkdir -p "$DATA"
mkdir -p "$RESULTS"

export LD_LIBRARY_PATH="$LIB"

conda create --yes --prefix "$ENV" python=2.7 pip
source activate "$ENV"

# you also need to install fftw

safe_call conda_install numpy scipy mkl
safe_call conda_install matplotlib
safe_call conda_install psutil
safe_call conda_install nose
safe_call conda_install ipython pyzmq jinja2 tornado
safe_call conda_install nltk
safe_call conda_install pandas
safe_call conda_install beautiful-soup
safe_call conda_install requests
safe_call conda_install statsmodels patsy
safe_call conda_install cython
safe_call conda_install gensim
#safe_call conda_install -c https://conda.binstar.org/richli fftw
#safe_call conda_install -c https://conda.binstar.org/richli pyfftw
safe_call pip_install pyfftw
safe_call pip_install pyprind
safe_call pip_install --pre line_profiler
safe_call pip_install ruffus
safe_call pip_install sh
safe_call pip_install simplejson
safe_call pip_install seaborn
#safe_call install_cld2
safe_call install_word2vec

if [ "$1" == "--cuda" ]; then
    safe_call install_pycuda
    safe_call install_scikits-cuda
    safe_call install_reikna
fi
