'''
for resnet50 and mobilenet
# BENCHMARKING SCRIPT FOR GLUON MXNET 2.0
'''
import argparse

import time
import mxnet as mx
import gluoncv

import warnings

from tvm.relay.op.transform import repeat
warnings.filterwarnings("ignore")
import tvm
from tvm.relay.op.contrib.dnnl import *
from tvm import relay

import numpy as np
import os
from tvm.contrib import utils
from tvm.relay.build_module import bind_params_by_name
from tvm.contrib.download import download_testdata
from PIL import Image

# %% TVM settings
# os.environ["TVM_BACKTRACE"] = "1"
# os.environ["TVM_LOG_DEBUG"] = ""
# os.environ["TVM_LOG_DEBUG"] = "DEFAULT=1"

# %%
cnt_conv_num = 0
network_dict = {"resnet18":"ResNet18_v1b",
                "resnet34":"ResNet34_v1b",
                "resnet50":"ResNet50_v1b",
                "resnet101":"ResNet101_v1b",
                "resnet152":"ResNet152_v1b",
                "vgg11":"VGG11",
                "vgg13":"VGG13",
                "vgg16":"VGG16",
                "vgg19":"VGG19",
                "vgg11_bn":"VGG11_bn",
                "vgg13_bn":"VGG13_bn",
                "vgg16_bn":"VGG16_bn",
                "vgg19_bn":"VGG19_bn",
                "densenet121":"DenseNet121",
                "InceptionV3":"InceptionV3",
                "MobileNet1.0":"MobileNet1.0",
                "i3d_resnet50_v1_kinetics400":"i3d_resnet50_v1_kinetics400"}

data_dic = {"a":"n",
            "b":"c",
            "c":"h",
            "d":"w",}

weight_dic = {"a":"o",
              "b":"i",
              "c":"h",
              "d":"w",
              "e":"g",}

@tvm.instrument.pass_instrument
class PrintIR:
    """Print the name of the pass, the IR, only before passes execute."""

    def run_before_pass(self, mod, info):
        print("Running pass: {}", info)
        print(mod)
        
def transform_image(image):
    image = np.array(image) - np.array([123.0, 117.0, 104.0])
    image /= np.array([58.395, 57.12, 57.375])
    image = image.transpose((2, 0, 1))
    image = image[np.newaxis, :]
    return image

dnnl_seq = tvm.transform.Sequential(
    [
        # tvm.transform.PrintIR(),
        relay.transform.CanonicalizeOps(),
        relay.transform.InferType(),
        relay.transform.SimplifyInference(),
        relay.transform.FoldConstant(),
        relay.transform.FoldScaleAxis(),
        # tvm.transform.PrintIR(),

        relay.transform.SimplifyExpr(),
        relay.transform.FoldConstant(),
        # tvm.transform.PrintIR(),

        relay.transform.AlterOpLayout(),
        relay.transform.FoldConstant(),
        # tvm.transform.PrintIR(),

        relay.transform.MergeComposite(pattern_table()),
        # tvm.transform.PrintIR(),
        relay.transform.AnnotateTarget("dnnl"),
        relay.transform.MergeCompilerRegions(),
        relay.transform.PartitionGraph(),
        tvm.transform.PrintIR(),
    ]
)

# %% helper functions for float32 <--> bf16 conversions
def np_float2np_bf16(arr):
    """Convert a numpy array of float to a numpy array
    of bf16 in uint16"""
    orig = arr.view("<u4")
    bias = np.bitwise_and(np.right_shift(orig, 16), 1) + 0x7FFF
    return np.right_shift(orig + bias, 16).astype("uint16")

def np_float2tvm_bf16(arr):
    """Convert a numpy array of float to a TVM array
    of bf16"""
    nparr = np_float2np_bf16(arr)
    return tvm.nd.empty(nparr.shape, "uint16").copyfrom(nparr)

def np_bf162np_float(arr):
    """Convert a numpy array of bf16 (uint16) to a numpy array
    of float"""
    u32 = np.left_shift(arr.astype("uint32"), 16)
    return u32.view("<f4")

def np_bf16_cast_and_cast_back(arr):
    """Convert a numpy array of float to bf16 and cast back"""
    return np_bf162np_float(np_float2np_bf16(arr))

def benchmark(network, batch_size, warmup=20, repeat=5, steps=20, target="llvm", profiling=False):
    
    if profiling:
        from tvm.contrib.debugger import debug_executor as graph_executor
    else:
        from tvm.contrib import graph_executor
    ctx = tvm.cpu()
    input_shape = (batch_size, 3, 224, 224)
    if network=="InceptionV3":
        input_shape = (batch_size, 3, 300, 300)
    if network=="i3d_resnet50_v1_kinetics400":
        input_shape = (batch_size, 3, 20, 224, 224)
    input_data = np.random.uniform(-1, 1, size=input_shape).astype("float32")
    output_shape = (batch_size, 1000)
    dev = tvm.cpu()

    net = gluoncv.model_zoo.get_model(network_dict[network], pretrained=True)
    print("importing {} from gluoncv ... ".format(network))

    print("running mxnet ...")
    mxnet_input = mx.ndarray.array(input_data)
    for i in range(steps+warmup):
        if i == warmup:
            tic = time.time()
        net(mxnet_input).wait_to_read()
    mxnet_dnnl_fps = steps * batch_size / (time.time() - tic)
    print("{}: mxnet-dnnl fps: {}".format(network, round(mxnet_dnnl_fps)))
    mxnet_output = net(mxnet_input).asnumpy()
    out_fp32 = mxnet_output
    print("{}: mxnet-dnnl result[0:12]: \n{}".format(network, mxnet_output.flatten()[0:12]))
    
    print("importing to fp32 graph ... ")
    mod_fp32, params = relay.frontend.from_mxnet(
        net, shape={"data": input_shape}, dtype="float32"
    )
    mod_fp32["main"] = bind_params_by_name(mod_fp32["main"], params)
    with open('mod_{}_fp32.swift'.format(network), 'w') as fout:
        fout.write(mod_fp32.astext(show_meta_data=False))
    print("done")

    print("building fp32-tvm lib ...")
    with tvm.transform.PassContext(opt_level=3):
        json_built, lib_built, params_built = relay.build(mod_fp32, target=target, params=params)
    print("running fp32-tvm module ...")
    module_runtime = graph_executor.create(json_built, lib_built, dev)
    module_runtime.set_input("data", input_data, **params_built)
    for i in range(warmup):
        module_runtime.run()
    perf_timer = module_runtime.benchmark(dev, "run", repeat=repeat, number=steps, end_to_end=True)
    infer_times = np.array(perf_timer.results)
    time_mean = np.mean(infer_times)/batch_size
    time_std = np.std(infer_times)/batch_size
    print("{}: fp32-tvm time: {}??{}ms".format(network, round(time_mean*1000), round(time_std*1000)))
    fps_mean = 1/time_mean
    fps_std = fps_mean*(time_std/time_mean)
    print("{}: fp32-tvm fps: {}??{}".format(network, round(fps_mean), round(fps_std)))
    if profiling:
        print('Profiling results:')
        print(module_runtime.profile())
    output = module_runtime.get_output(0, tvm.nd.empty(output_shape)).numpy()
    print('{}: fp32-tvm MSE: {}'.format(network, np.square(np.subtract(output, mxnet_output)).mean()))
    print("{}: fp32-tvm result[0:12]: \n{}".format(network, output.flatten()[0:12]))

    '''
    print("converting to bf16 graph ... ")
    mod_bf16 = relay.transform.ToMixedPrecision('bfloat16')(mod_fp32)
    mod_bf16["main"] = bind_params_by_name(mod_bf16["main"], params)
    with open('mod_{}_bf16.swift'.format(network), 'w') as fout:
        fout.write(mod_bf16.astext(show_meta_data=False))
    print("done")

    print("building bf16-tvm lib ...")
    with tvm.transform.PassContext(opt_level=3):
        json_built, lib_built, params_built = relay.build(mod_bf16, target=target, params=params)
    print("running bf16-tvm module ...")
    module_runtime = graph_executor.create(json_built, lib_built, dev)
    module_runtime.set_input("data", input_data, **params_built)
    for i in range(warmup):
        module_runtime.run()
    perf_timer = module_runtime.benchmark(dev, "run", repeat=repeat, number=steps, end_to_end=True)
    infer_times = np.array(perf_timer.results)
    time_mean = np.mean(infer_times)/batch_size
    time_std = np.std(infer_times)/batch_size
    print("{}: bf16-tvm time: {}??{}ms".format(network, round(time_mean*1000), round(time_std*1000)))
    fps_mean = 1/time_mean
    fps_std = fps_mean*(time_std/time_mean)
    print("{}: bf16-tvm fps: {}??{}".format(network, round(fps_mean), round(fps_std)))
    if profiling:
        print('Profiling results:')
        print(module_runtime.profile())
    output = module_runtime.get_output(0, tvm.nd.empty(output_shape, "uint16")).numpy()
    output = np_bf162np_float(output)
    print('{}: bf16-tvm MSE: {}'.format(network, np.square(np.subtract(output, mxnet_output)).mean()))
    print("{}: bf16-tvm result[0:12]: \n{}".format(network, output.flatten()[0:12]))
    '''

if __name__ == "__main__":
    # os.environ["TVM_LOG_DEBUG"]="DEFAULT=1;ir/transform.cc=1;relay/ir/transform.cc=1"
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--network",
        type=str,
        choices=["resnet18", "resnet34", "resnet50", "resnet101", "resnet152",
                "vgg11", "vgg13", "vgg16", "vgg19",
                "vgg11_bn", "vgg13_bn", "vgg16_bn", "vgg19_bn",
                "densenet121", "InceptionV3", "MobileNet1.0", "i3d_resnet50_v1_kinetics400", "all"],
        default="all",
        help="The name of the neural network.",
    )
    parser.add_argument("--batch-size", type=int, default=1, help="The batch size")
    parser.add_argument(
        "--target",
        type=str,
        default="llvm -mcpu=cooperlake -model=platinum-8369",
        help="The compilation target.",
    )
    parser.add_argument("--dtype", type=str, default="float32", help="The data type.")

    parser.add_argument("--warmup", type=int, default=20)
    parser.add_argument("--repeat", type=int, default=5)
    parser.add_argument("--steps", type=int, default=20)
    parser.add_argument("--profiling", type=bool, default=False)
    args = parser.parse_args()

    if args.network == "all":
        networks = ["resnet18", "resnet34", "resnet50", "resnet101", "resnet152",
                    "vgg11", "vgg13", "vgg16", "vgg19",
                    "vgg11_bn", "vgg13_bn", "vgg16_bn", "vgg19_bn",
                    "densenet121",
                    "InceptionV3"]
    else:
        networks = [args.network]

    target = tvm.target.Target(args.target)

    for network in networks:
        benchmark(network, args.batch_size, warmup=args.warmup, repeat=args.repeat, \
            steps=args.steps, target=args.target, profiling=args.profiling)

