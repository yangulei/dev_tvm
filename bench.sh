#!/bin/bash
export OMP_NUM_THREADS=28
export KMP_AFFINITY=granularity=fine,noduplicates,compact,1,0

<<COMMENT
COMMENT
echo "benchmarking tvm ..."
#--profiling \
#--network=resnet50\
TVM_LIBRARY_PATH=${TVM_HOME}/build_release_native \
numactl --physcpubind=0-27 -l \
python ${TVM_HOME}/dev_tvm/bench_tvm.py \
 --batch-size=58 \
 --warmup=20 \
 --repeat=5 \
 --steps=20 \
 1>bench_tvm.log \
 2>bench_tvm_debug.log

echo "benchmarking byoc ..."
#DNNL_VERBOSE=1 \
#--network=resnet50\
TVM_LIBRARY_PATH=${TVM_HOME}/build_release_gnu \
numactl --physcpubind=0-27 -l \
python ${TVM_HOME}/dev_tvm/bench_byoc.py \
 --batch-size=58 \
 --warmup=20 \
 --repeat=5 \
 --steps=20 \
 1>bench_byoc.log \
 2>bench_byoc_debug.log

echo "benchmarking byoc usmp ..."
#DNNL_VERBOSE=1 \
#--network=resnet50\
TVM_LIBRARY_PATH=${TVM_HOME}/build_release_gnu \
numactl --physcpubind=0-27 -l \
python ${TVM_HOME}/dev_tvm/bench_byoc_usmp.py \
 --batch-size=58 \
 --warmup=20 \
 --repeat=5 \
 --steps=20 \
 1>bench_byoc_usmp.log \
 2>bench_byoc_usmp_debug.log
