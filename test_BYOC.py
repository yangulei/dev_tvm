'''
for resnet50 and mobilenet
# BENCHMARKING SCRIPT FOR GLUON MXNET 2.0
'''
import time
import mxnet as mx
import warnings

# from torch._C import T
warnings.filterwarnings("ignore")
from mxnet.gluon.model_zoo.vision import *
import tvm
from tvm.relay.op.contrib.dnnl import *
from tvm import relay
import tvm.contrib.graph_executor as runtime
import numpy as np
from tvm.relay.testing import *
import os
from tvm.contrib import utils

# model_dict = {'resnet50_v1': resnet50_v1}#{'mobilenet_v2_1_0': mobilenet_v2_1_0}
model_dict = {'resnet50_v1': resnet}
# model_dict = {'resnet50_v1': resnet, 'mobilenet_v2_1_0': mobilenet}

def make_pattern(with_bias=True, with_bn=False):
    from tvm.relay.dataflow_pattern import is_op, wildcard
    data = wildcard()
    weight = wildcard()
    bias = wildcard()
    gamma, beta, moving_mean, moving_var = wildcard(), wildcard(), wildcard(), wildcard()
    conv = is_op("nn.conv2d")(data, weight)
    if with_bias:
        conv_out = is_op("nn.bias_add")(conv, bias)
    else:
        conv_out = conv
    if with_bn:
        bn_out = is_op("nn.batch_norm")(conv_out, gamma, beta, moving_mean, moving_var)
    else:
        bn_out = conv_out
    return is_op("nn.relu")(bn_out)

# def make_pattern(with_bias=True):
#     from tvm.relay.dataflow_pattern import is_op, wildcard
#     data = wildcard()
#     weight = wildcard()
#     conv = is_op('nn.conv2d')(data, weight)
#     return wildcard()(conv)

conv2d_bias_relu_pat = ("dnnl.conv2d_relu_with_bias", make_pattern(with_bias=True))
conv2d_bias_bn_relu_pat = ("dnnl.conv2d_bn_relu_with_bias", make_pattern(with_bias=True, with_bn=True))
conv2d_relu_pat = ("dnnl.conv2d_relu_wo_bias", make_pattern(with_bias=False))
conv2d_bn_relu_pat = ("dnnl.conv2d_bn_relu_wo_bias", make_pattern(with_bias=False, with_bn=True))
patterns = [conv2d_bias_relu_pat, conv2d_relu_pat, conv2d_bias_bn_relu_pat, conv2d_bn_relu_pat]#

def update_lib(lib):
    # Include the path of src/runtime/contrib/dnnl/dnnl.cc
    test_dir = os.path.dirname(os.path.realpath(os.path.expanduser(__file__)))
    # source_dir = os.path.join(test_dir, "..", "..", "..")
    # contrib_path = os.path.join(source_dir, "src", "runtime", "contrib")
    source_dir = os.path.join(test_dir, "..", "tvm")
    contrib_path = os.path.join(source_dir, "src", "runtime", "contrib")
    # Setup the gcc flag to compile DNNL code.
    kwargs = {}
    kwargs["options"] = ["-O2", "-std=c++14", "-I" + contrib_path]
    tmp_path = utils.tempdir()
    lib_name = 'lib.so'
    lib_path = tmp_path.relpath(lib_name)
    # The generated C code with DNNL APIs is compiled to a binary lib.so.
    lib.export_library(lib_path, fcompile=False, **kwargs)
    # Load the lib.so back to a runtime module.
    lib = tvm.runtime.load_module(lib_path)
    return lib

def benchmark(batch_size=1, batches=10, warmup=2):
    mx.random.seed(0)
    sample = mx.nd.random.uniform(-1.0, 1.0, shape=(batch_size,3,224,224))
    target = "llvm -model=platinum-8124m -mcpu=skylake-avx512"
    ctx = tvm.cpu()
    input_shape = (batch_size, 3, 224, 224)
    for model_name in model_dict.keys():
        # net = model_dict[model_name](pretrained=True)
        # net.hybridize(static_alloc=True, static_shape=True)
        # mod, params = relay.frontend.from_mxnet(net, shape={"data": input_shape}, dtype="float32")#port the Gluon model to a por
        mod, params = model_dict[model_name].get_workload(batch_size=batch_size, dtype="float32")
        print('==================0 relayed model ==================')
        print(mod["main"].astext(show_meta_data=False))

        mod1 = relay.transform.ToMixedPrecision('bfloat16')(mod)
        print('==================1 fp16 model ==================')
        print(mod1["main"].astext(show_meta_data=False))

        mod2 = relay.transform.MergeComposite(pattern_table())(mod1)
        mod2 = relay.transform.Legalize()(mod2)
        print('==================2 MergeComposite ==================')
        print(mod2["main"].astext(show_meta_data=False))

        mod3 = relay.transform.AnnotateTarget(["dnnl"])(mod2)
        print('==================3 AnnotateTarget ==================')
        print(mod3["main"].astext(show_meta_data=False))

        mod4 = relay.transform.MergeCompilerRegions()(mod3)
        print('==================4 MergeCompilerRegions ==================')
        print(mod4["main"].astext(show_meta_data=False))

        mod5 = relay.transform.PartitionGraph()(mod4)
        print('==================5 PartitionGraph ==================')
        print(mod5["main"].astext(show_meta_data=False))

        # with tvm.transform.PassContext(opt_level=3, disabled_pass=["FoldConstant","DynamicToStatic"]):#compile the graph
        with tvm.transform.PassContext(opt_level=1):#compile the graph
            print('================== building ==================')
            json, lib, param = tvm.relay.build(mod5, target="llvm", params=params)
            print('================== built ==================')
            print(json)
        # lib = update_lib(lib)
        rt_mod = tvm.contrib.graph_executor.create(json, lib, ctx)#Create a runtime executor module given a graph and module.
        data = np.random.uniform(size=input_shape)
        # rt_mod.set_input("data", sample)
        rt_mod.set_input("data", tvm.nd.array(data.astype("float32")))
        for i in range(batches+warmup):
            if i == warmup:
                tic = time.time()
            out = rt_mod.run()
        with_fuse_ms = (time.time() - tic) / (batches) * 1000
        print("{}: with_fuse_ms: {:.4f} ms".format(model_name, with_fuse_ms))
benchmark(batch_size=1)