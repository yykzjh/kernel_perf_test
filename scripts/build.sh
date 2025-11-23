#!/bin/bash

# sync submodules
git submodule sync --recursive
git submodule update --init --recursive

# install dependencies
uv sync
source ./.venv/bin/activate

pushd third_party

# build deep_gemm
pushd deep_gemm
rm -rf build dist
rm -rf *.egg-info
python setup.py bdist_wheel
uv pip install dist/*.whl --force-reinstall
popd # deep_gemm

popd # third_party
