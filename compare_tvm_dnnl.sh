#!/bin/bash
export OMP_NUM_THREADS=56
export KMP_AFFINITY=granularity=fine,noduplicates,compact,1,0

<<COMMENT
COMMENT
echo "benchmarking tvm ..."
#--profiling=0 \
#--network=resnet50\
TVM_LIBRARY_PATH=${TVM_HOME}/build_release_native \
numactl --physcpubind=0-55 --membind=0 \
python ${TVM_HOME}/dev_tvm/bench_tvm.py \
 --batch-size=56 \
 --warmup=20 \
 --repeat=5 \
 --steps=20 \
 1>bench_tvm.log \
 2>bench_tvm_debug.log

echo "benchmarking byoc ..."
# DNNL_VERBOSE=1 \
#  --profiling=0 \
#  --network=resnet50\
TVM_LIBRARY_PATH=${TVM_HOME}/build_release_gnu \
numactl --physcpubind=0-55 --membind=0 \
python ${TVM_HOME}/dev_tvm/bench_dnnl.py \
 --batch-size=56 \
 --warmup=20 \
 --repeat=5 \
 --steps=20 \
 1>bench_dnnl.log \
 2>bench_dnnl_debug.log

echo "collecting results"
grep 'mxnet-dnnl\ fps' bench_tvm.log > res_mxnet-dnnl_fps.csv
grep 'fp32-tvm\ fps' bench_tvm.log > res_fp32-tvm_fps.csv
grep 'bf16-tvm\ fps' bench_tvm.log > res_bf16-tvm_fps.csv
grep 'fp32-tvm\ MSE' bench_tvm.log > res_fp32-tvm_mse.csv
grep 'bf16-tvm\ MSE' bench_tvm.log > res_bf16-tvm_mse.csv
grep 'fp32-dnnl\ fps' bench_dnnl.log > res_fp32-dnnl_fps.csv
grep 'bf16-dnnl\ fps' bench_dnnl.log > res_bf16-dnnl_fps.csv
grep 'fp32-dnnl\ MSE' bench_dnnl.log > res_fp32-dnnl_mse.csv
grep 'bf16-dnnl\ MSE' bench_dnnl.log > res_bf16-dnnl_mse.csv
sed -i 's/\:/\,/g' *.csv
sed -i 's/\±/\,/g' *.csv