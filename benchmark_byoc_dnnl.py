'''
for resnet50 and mobilenet
# BENCHMARKING SCRIPT FOR GLUON MXNET 2.0
'''
import argparse

import time
from tkinter.tix import Tree
import mxnet as mx
import gluoncv

import warnings
warnings.filterwarnings("ignore")
import tvm
from tvm.relay.op.contrib import dnnl
from tvm import relay

import numpy as np
import os

def transform_image(image):
    image = np.array(image) - np.array([123.0, 117.0, 104.0])
    image /= np.array([58.395, 57.12, 57.375])
    image = image.transpose((2, 0, 1))
    image = image[np.newaxis, :]
    return image

def benchmark(network, batch_size, profiling=False, check_acc=False, warmup=100, batches=400, dtype="float32", target="llvm"):
    ctx = tvm.cpu()
    input_shape = (batch_size, 3, 224, 224)
    if network=="InceptionV3":
        input_shape = (batch_size, 3, 300, 300)
    if network=="i3d_resnet50_v1_kinetics400":
        input_shape = (batch_size, 3, 20, 224, 224)
    
    block = gluoncv.model_zoo.get_model(network, pretrained=True)
    mod, params = relay.frontend.from_mxnet(block, shape={"data": input_shape}, dtype=dtype)
    sample = np.random.rand(input_shape[0], input_shape[1],input_shape[2], input_shape[3])
    if network=="i3d_resnet50_v1_kinetics400":
        sample = np.random.rand(input_shape[0], input_shape[1],input_shape[2], input_shape[3], input_shape[4])

    processed_mod = dnnl.partition_for_dnnl(mod, params, alter_layout=True)
    # print(processed_mod)
    with tvm.transform.PassContext(opt_level=3):
        json, lib, params = relay.build(processed_mod, target=target, params=params)
        print(json)

    import tvm.contrib.graph_executor as graph_executor
    # from tvm.contrib.debugger import debug_executor as graph_executor
    rt_mod = graph_executor.create(json, lib, ctx)
    
    rt_mod.set_input("data", tvm.nd.array(sample.astype("float32")))
    rt_mod.set_input(**params)
    # out = rt_mod.run()
    
    for i in range(batches+warmup):
        if i == warmup:
            tic = time.time()
        # print("================start run=========================")
        rt_mod.run()
        # print(rt_mod.profile())
    with_fuse_fps = batches * batch_size / (time.time() - tic)
    print("{}: with_fuse_fps: {:.4f} fps".format(network, with_fuse_fps))

if __name__ == "__main__":
    # os.environ["TVM_LOG_DEBUG"]="DEFAULT=1;ir/transform.cc=1;relay/ir/transform.cc=1"
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--network",
        type=str,
        default="ResNext50_32x4d",
        help="The name of the neural network.",
    )
    parser.add_argument("--batch_size", type=int, default=1, help="The batch size")
    parser.add_argument(
        "--target",
        type=str,
        default="llvm -mcpu=cascadelake -model=platinum-8280",
        help="The compilation target.",
    )
    parser.add_argument("--dtype", type=str, default="float32", help="The data type.")
    
    parser.add_argument("--warmup", type=int, default=20)
    parser.add_argument("--batches", type=int, default=100)
    parser.add_argument("--profiling", type=bool, default=False)
    parser.add_argument("--check_acc", type=bool, default=False)
    args = parser.parse_args()

    if args.network == "all":
        networks = [
                    "resnet18", "resnet34", "resnet50", "resnet101", "resnet152",
                    "vgg11", "vgg13", "vgg16", "vgg19", 
                    "vgg11_bn", "vgg13_bn", "vgg16_bn", "vgg19_bn",
                    "densenet121", 
                    "InceptionV3",
                    ]
    else:
        networks = [args.network]

    target = tvm.target.Target(args.target)

    for network in networks:
        benchmark(network, args.batch_size, profiling=args.profiling,check_acc=args.check_acc,\
        warmup=args.warmup, batches=args.batches, dtype=args.dtype, target=args.target)
