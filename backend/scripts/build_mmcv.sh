#!/bin/bash
export CUDA_HOME=$CONDA_PREFIX
export TORCH_CUDA_ARCH_LIST="12.0"
export MMCV_WITH_OPS=1 MAX_JOBS=20 FORCE_CUDA=1
export CC=$CONDA_PREFIX/bin/gcc CXX=$CONDA_PREFIX/bin/g++
pip install -v --no-build-isolation --no-binary mmcv mmcv==2.2.0
echo MMCV_BUILD_EXIT=$?
