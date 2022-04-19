#!/bin/bash

export TVM_LIBRARY_PATH=${TVM_HOME}/build-debug_gnu
for i in {1..10}
do
    echo "#.$i testing:"
    clear && python ${TVM_HOME}/tests/python/contrib/test_dnnl.py
done