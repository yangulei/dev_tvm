'''
for resnet50 and mobilenet
# BENCHMARKING SCRIPT FOR GLUON MXNET 2.0
'''
import argparse
import time
import re

import mxnet as mx
import gluoncv

import tvm
from tvm.relay import transform
from tvm import relay
from tvm.relay.build_module import bind_params_by_name
from tvm.relay.testing.temp_op_attr import TempOpAttr
from tvm.relay.op.contrib import dnnl
from tvm.contrib import graph_executor

import numpy as np
import os

# %% TVM settings
# os.environ["TVM_BACKTRACE"] = "1"
# os.environ["TVM_LOG_DEBUG"] = ""
# os.environ["TVM_LOG_DEBUG"] = "DEFAULT=1"


def partition_for_dnnl(mod, params=None, alter_layout=True):
    """Partition the graph greedily offloading supported operators to DNNL.

    Parameters
    ----------
    mod : Module
        The module to run passes on.
    params : Optional[Dict[str, NDArray]]
        Constant input parameters.
    Returns
    -------
    mod : Module
        Annotated and partitioned module.
    """
    if params:
        mod["main"] = bind_params_by_name(mod["main"], params)

    with TempOpAttr("nn.conv2d", "FTVMLegalize", dnnl.legalize_group_conv):
        with TempOpAttr("nn.conv2d_transpose", "FTVMLegalize", dnnl.legalize_group_conv):
            seq = tvm.transform.Sequential(
                [
                    transform.CanonicalizeOps(),
                    transform.InferType(),
                    transform.SimplifyInference(),
                    transform.FoldConstant(),
                    transform.FoldScaleAxis(),
                    # fold consecutive add ops to simplify pattern `conv2d-bias_add-bn-relu`
                    transform.SimplifyExpr(),
                    transform.FoldConstant(),
                    # alter group conv /conv_transpose layout to `GOIHW` / `GIOHW`
                    transform.Legalize(),
                    transform.FoldConstant(),
                ]
            )
            with tvm.transform.PassContext(opt_level=3):
                mod = seq(mod)
    if alter_layout:
        with TempOpAttr("nn.conv1d", "FTVMAlterOpLayout", dnnl.alter_conv):
            with TempOpAttr("nn.conv2d", "FTVMAlterOpLayout", dnnl.alter_conv):
                with TempOpAttr("nn.conv3d", "FTVMAlterOpLayout", dnnl.alter_conv):
                    with TempOpAttr(
                        "nn.conv2d_transpose", "FTVMAlterOpLayout", dnnl.alter_conv_transpose
                    ):
                        with TempOpAttr(
                            "nn.conv3d_transpose", "FTVMAlterOpLayout", dnnl.alter_conv_transpose
                        ):
                            alter_layout_seq = tvm.transform.Sequential(
                                [
                                    transform.AlterOpLayout(),
                                    transform.FoldConstant(),
                                ]
                            )
                            with tvm.transform.PassContext(opt_level=3):
                                mod = alter_layout_seq(mod)

    byoc_seq = tvm.transform.Sequential(
        [
            transform.MergeComposite(dnnl.pattern_table()),
            transform.AnnotateTarget("dnnl"),
            transform.MergeCompilerRegions(),
            transform.PartitionGraph(),
        ]
    )
    with tvm.transform.PassContext(opt_level=3):
        mod = byoc_seq(mod)
    return mod

#%%


def get_input_shape(model_name, batch_size):
    input_shape = list()
    if model_name.endswith('_int8'):
        return list()
    # classification
    elif re.match('(^resnet)+[18,34,50,101,152]+_+[v1,v1b,v1c,v1d,v1s,v2]+(_\d.\d\d$|$)', model_name):
        input_shape = (batch_size, 3, 224, 224)
    elif re.match('(^resnext|^se_resnext)+[50,101]+_', model_name):
        input_shape = (batch_size, 3, 224, 224)
    elif model_name.startswith('resnest'):
        input_shape = (batch_size, 3, 224, 224)
    elif model_name.startswith('mobilenet'):
        input_shape = (batch_size, 3, 224, 224)
    elif re.match('^[vgg]+[11,13,16,19]+($|_bn)', model_name):
        input_shape = (batch_size, 3, 224, 224)
    elif model_name.startswith('squeezenet'):
        input_shape = (batch_size, 3, 224, 224)
    elif model_name.startswith('densenet'):
        input_shape = (batch_size, 3, 224, 224)
    elif model_name in ('alexnet', 'darknet53', 'googlenet', 'xception', 'senet_154'):
        input_shape = (batch_size, 3, 224, 224)
    elif model_name == 'inceptionv3':
        input_shape = (batch_size, 3, 299, 299)
    elif re.match('(^cifar_)(resnet|wideresnet|resnext29_16x64d)', model_name):
        input_shape = (batch_size, 3, 224, 224)

    # # detection
    # elif re.match('(^ssd_300_)+(resnet34_v1b|vgg16_atrous)+(_voc$|_coco$)', model_name):
    #     input_shape = (batch_size, 3, 300, 300)
    # elif re.match('(^ssd_512_)+(vgg16_atrous|resnet50_v1|mobilenet1.0)+(_voc|_coco)+$', model_name):
    #     input_shape = (batch_size, 3, 512, 512)
    # elif re.match('^faster_rcnn_*', model_name):
    #     input_shape = (1, 3, 600, 600)
    # elif re.match('^[yolo3_]+(darknet53|mobilenet1.0)+(_voc$|_coco$)', model_name):
    #     input_shape = (batch_size, 3, 416, 416)
    # elif re.match('(^center_net_resnet)+(18|50|101)+_v1b_', model_name):
    #     input_shape = (batch_size, 3, 512, 512)

    # # segmentation
    # elif re.match('(^fcn|^psp|^deeplab)+_resnet+(50|101|200|269)+(_ade$|_coco$)', model_name):
    #     input_shape = (batch_size, 3, 480, 480)
    # elif model_name.endswith('citys'):
    #     input_shape = (batch_size, 3, 480, 480)
    # elif model_name == 'icnet_resnet50_mhpv1':
    #     input_shape = (batch_size, 3, 480, 480)
    # elif model_name.startswith('mask_rcnn'):
    #     input_shape = (1, 3, 480, 480)

    # # pose estimation
    # elif re.match('(^simple_pose_resnet)', model_name):
    #     input_shape = (batch_size, 3, 256, 192)
    # elif model_name.startswith('mobile_pose'):
    #     input_shape = (batch_size, 3, 256, 192)
    # elif model_name.startswith('alpha_pose'):
    #     input_shape = (batch_size, 3, 256, 192)

    # # action recognition
    # elif model_name in ('inceptionv3_kinetics400', 'inceptionv3_ucf101'):
    #     input_shape = (32, 3, 299, 299)
    # elif re.match('(^resnet)+(18|34|50|101|152)+_v1b_kinetics400$', model_name):
    #     input_shape = (32, 3, 224, 224)
    # elif re.match('(^c3d|^r2plus1d)(\S*)kinetics400$', model_name):
    #     input_shape = (batch_size, 3, 32, 112, 112)
    # elif model_name in ('p3d_resnet50_kinetics400', 'p3d_resnet101_kinetics400'):
    #     input_shape = (batch_size, 3, 32, 112, 112)
    # elif re.match('(^i3d_)(\S*)(kinetics400$|ucf101$)', model_name):
    #     input_shape = (batch_size, 3, 32, 224, 224)
    # elif re.match('(^slowfast_)(4x16_resnet50|8x8_resnet50|8x8_resnet101)_kinetics400$', model_name): # got DNNL error
    #     input_shape = (batch_size, 3, 4, 224, 224)
    # elif model_name == 'i3d_slow_resnet101_f16s4_kinetics700':
    #     input_shape = (batch_size, 3, 4, 224, 224)
    # elif model_name == 'vgg16_ucf101':
    #     input_shape = (32, 3, 224, 224)
    # elif model_name == 'resnet50_v1b_hmdb51':
    #     input_shape = (32, 3, 224, 224)
    # elif model_name == 'resnet50_v1b_sthsthv2':
    #     input_shape = (32, 3, 224, 224)
    # elif model_name == 'i3d_resnet50_v1_sthsthv2':
    #     input_shape = (batch_size, 3, 32, 224, 224)

    # # depth prediction
    # elif model_name.startswith('monodepth2_resnet18_kitti_'):
    #     input_shape = (batch_size, 3, 640, 192)
    # elif model_name.startswith('monodepth2_resnet18_posenet_'):
    #     input_shape = (batch_size, 6, 640, 192)
    else:
        return list()

    # print("input shape for {}: {}".format(model, input_shape))
    return input_shape

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

def benchmark(name, batch_size, warmup=20, repeat=5, steps=20, target="llvm", profiling=False):

    if profiling:
        from tvm.contrib.debugger import debug_executor as graph_executor
    else:
        from tvm.contrib import graph_executor

    input_shape = get_input_shape(name, batch_size)
    input_data = np.random.uniform(-1, 1, size=input_shape).astype("float32")
    output_shape = (batch_size, 1000)
    dev = tvm.cpu()

    print("importing {} from gluoncv ... ".format(name))
    cv_model = gluoncv.model_zoo.get_model(name, pretrained=True)
    print("done")

    print("running mxnet ...")
    mxnet_input = mx.ndarray.array(input_data)
    mxnet_times = [None] * steps
    for i in range(steps+warmup):
        if i < warmup:
            cv_model(mxnet_input).wait_to_read()
            tic = time.time()
        else:
            cv_model(mxnet_input).wait_to_read()
            mxnet_times[i-warmup] = time.time() - tic
            tic = time.time()
    mxnet_times = np.array(mxnet_times)*1000    # infer time in ms
    mxnet_time_mean = np.mean(mxnet_times)/batch_size
    mxnet_time_std = np.std(mxnet_times)/batch_size
    mxnet_time_rel_err = mxnet_time_std/mxnet_time_mean
    print("{}: mxnet time: {}±{}ms".format(name,
          round(mxnet_time_mean), round(mxnet_time_std)))
    mxnet_dnnl_fps = 1000 / mxnet_time_mean
    print("{}: mxnet fps: {}±{}".format(name,
          round(mxnet_dnnl_fps), round(mxnet_dnnl_fps*mxnet_time_rel_err)))

    mxnet_output = cv_model(mxnet_input).asnumpy()
    print("{}: mxnet result[0:12]: \n{}".format(
        name, mxnet_output.flatten()[0:12]))

    print("importing to fp32 graph ... ")
    mod_fp32, params = relay.frontend.from_mxnet(
        cv_model, shape={"data": input_shape}, dtype="float32")
    mod_fp32["main"] = bind_params_by_name(mod_fp32["main"], params)
    # with open('mod_{}_fp32.swift'.format(name), 'w') as fout:
    #     fout.write(mod_fp32.astext(show_meta_data=False))
    # print("done")

    print("converting to bf16 graph ... ")
    mod_bf16 = relay.transform.ToMixedPrecision('bfloat16')(mod_fp32)
    mod_bf16["main"] = bind_params_by_name(mod_bf16["main"], params)
    # with open('mod_{}_bf16.swift'.format(name), 'w') as fout:
    #     fout.write(mod_bf16.astext(show_meta_data=False))
    # print("done")

    for engine, dtype in [('tvm', 'fp32'), ('tvm', 'bf16'), ('byoc', 'fp32'), ('byoc', 'bf16')]:
        print("building {}-{} lib ...".format(engine, dtype))
        if dtype == "fp32":
            mod = mod_fp32
        elif dtype == "bf16":
            mod = mod_bf16
        else:
            print("unsupported datatype: {}, use 'fp32' or 'bf16'".format(dtype))
            exit()
        if engine == "byoc":
            mod = partition_for_dnnl(mod, params, alter_layout=True)
        with open('mod_{}_{}-{}.swift'.format(name, engine, dtype), 'w') as fout:
            fout.write(mod.astext(show_meta_data=False))
        with tvm.transform.PassContext(opt_level=3):
            json_built, lib_built, params_built = relay.build(
                mod, target=target, params=params)
        with open('mod_{}_{}-{}_built.json'.format(name, engine, dtype), 'w') as fout:
            fout.write(json_built)
        print("running {}-{} module ...".format(engine, dtype))
        module_runtime = graph_executor.create(json_built, lib_built, dev)
        module_runtime.set_input("data", input_data, **params_built)
        for i in range(warmup):
            module_runtime.run()
        perf_timer = module_runtime.benchmark(
            dev, "run", repeat=repeat, number=steps, end_to_end=True)
            
        infer_times = np.array(perf_timer.results)*1000    # infer time in ms
        infer_time_mean = np.mean(infer_times)/batch_size
        infer_time_std = np.std(infer_times)/batch_size
        infer_time_rel_err = infer_time_std/infer_time_mean
        print("{}: {}-{} time: {}±{}ms".format(name, engine, dtype,
            round(infer_time_mean), round(infer_time_std)))
        infer_fps = 1000 / infer_time_mean
        print("{}: {}-{} fps: {}±{}".format(name, engine, dtype,
            round(infer_fps), round(infer_fps*infer_time_rel_err)))

        if profiling:
            print('Profiling results:')
            print(module_runtime.profile())

        if dtype == "fp32":
            output = module_runtime.get_output(
                0, tvm.nd.empty(output_shape)).numpy()
        elif dtype == "bf16":
            output = module_runtime.get_output(
                0, tvm.nd.empty(output_shape, "uint16")).numpy()
            output = np_bf162np_float(output)
        print('{}: {}-{} MSE: {}'.format(name, engine, dtype,
                                         np.square(np.subtract(output, mxnet_output)).mean()))
        print("{}: {}-{} result[0:12]: \n{}".format(name,
              engine, dtype, output.flatten()[0:12]))

'''
    print("building tvm-bf16 lib ...")
    with tvm.transform.PassContext(opt_level=3):
        json_built, lib_built, params_built = relay.build(
            mod_bf16, target=target, params=params)
    with open('mod_{}_bf16_tvm_built.json'.format(name), 'w') as fout:
        fout.write(json_built)
    print("running tvm-bf16 module ...")
    module_runtime = graph_executor.create(json_built, lib_built, dev)
    module_runtime.set_input("data", input_data, **params_built)
    for i in range(warmup):
        module_runtime.run()
    perf_timer = module_runtime.benchmark(
        dev, "run", repeat=repeat, number=steps, end_to_end=True)
        
    tvm_bf16_times = np.array(perf_timer.results)*1000    # infer time in ms
    tvm_bf16_time_mean = np.mean(tvm_bf16_times)/batch_size
    tvm_bf16_time_std = np.std(tvm_bf16_times)/batch_size
    tvm_bf16_time_rel_err = tvm_bf16_time_std/tvm_bf16_time_mean
    print("{}: tvm-bf16 time: {}±{}ms".format(name,
          round(tvm_bf16_time_mean), round(tvm_bf16_time_std)))
    tvm_bf16_dnnl_fps = 1000 / tvm_bf16_time_mean
    print("{}: tvm-bf16 fps: {}±{}".format(name,
          round(tvm_bf16_dnnl_fps), round(tvm_bf16_dnnl_fps*tvm_bf16_time_rel_err)))

    if profiling:
        print('Profiling results:')
        print(module_runtime.profile())

    tvm_bf16_output = module_runtime.get_output(
        0, tvm.nd.empty(output_shape, "uint16")).numpy()
    tvm_bf16_output = np_bf162np_float(tvm_bf16_output)
    print('{}: tvm-bf16 MSE: {}'.format(name,
          np.square(np.subtract(tvm_bf16_output, mxnet_output)).mean()))
    print(
        "{}: tvm-bf16 result[0:12]: \n{}".format(name, tvm_bf16_output.flatten()[0:12]))

    print("building fp32-dnnl lib ...")
    mod_fp32_dnnl = partition_for_dnnl(mod_fp32, params, alter_layout=True)
    with open('mod_{}_fp32_dnnl.swift'.format(name), 'w') as fout:
        fout.write(mod_fp32_dnnl.astext(show_meta_data=False))
    with tvm.transform.PassContext(opt_level=3):
        json_built, lib_built, params_built = relay.build(
            mod_fp32_dnnl, target=target, params=params)
    print("running fp32-dnnl module ...")
    module_runtime = graph_executor.create(json_built, lib_built, dev)
    module_runtime.set_input("data", input_data, **params_built)
    for i in range(warmup):
        module_runtime.run()
    perf_timer = module_runtime.benchmark(
        dev, "run", repeat=repeat, number=steps, end_to_end=True)
    infer_times = np.array(perf_timer.results)
    time_mean = np.mean(infer_times)/batch_size
    time_std = np.std(infer_times)/batch_size
    print("{}: fp32-dnnl time: {}±{}ms".format(name,
          round(time_mean*1000), round(time_std*1000)))
    fps_mean = 1/time_mean
    fps_std = fps_mean*(time_std/time_mean)
    print("{}: fp32-dnnl fps: {}±{}".format(name,
          round(fps_mean), round(fps_std)))
    if profiling:
        print('Profiling results:')
        print(module_runtime.profile())
    output = module_runtime.get_output(0, tvm.nd.empty(output_shape)).numpy()
    print('{}: fp32-dnnl MSE: {}'.format(name,
          np.square(np.subtract(output, mxnet_output)).mean()))
    print(
        "{}: fp32-dnnl result[0:12]: \n{}".format(name, output.flatten()[0:12]))

    print("building bf16-dnnl lib ...")
    mod_bf16_dnnl = partition_for_dnnl(mod_bf16, params, alter_layout=True)
    with open('mod_{}_bf16_dnnl.swift'.format(name), 'w') as fout:
        fout.write(mod_bf16_dnnl.astext(show_meta_data=False))
    with tvm.transform.PassContext(opt_level=3):
        json_built, lib_built, params_built = relay.build(
            mod_bf16_dnnl, target=target, params=params)
    print("running bf16-dnnl module ...")
    module_runtime = graph_executor.create(json_built, lib_built, dev)
    module_runtime.set_input("data", input_data, **params_built)
    for i in range(warmup):
        module_runtime.run()
    perf_timer = module_runtime.benchmark(
        dev, "run", repeat=repeat, number=steps, end_to_end=True)
    infer_times = np.array(perf_timer.results)
    time_mean = np.mean(infer_times)/batch_size
    time_std = np.std(infer_times)/batch_size
    print("{}: bf16-dnnl time: {}±{}ms".format(name,
          round(time_mean*1000), round(time_std*1000)))
    fps_mean = 1/time_mean
    fps_std = fps_mean*(time_std/time_mean)
    print("{}: bf16-dnnl fps: {}±{}".format(name,
          round(fps_mean), round(fps_std)))
    if profiling:
        print('Profiling results:')
        print(module_runtime.profile())
    output = module_runtime.get_output(
        0, tvm.nd.empty(output_shape, "uint16")).numpy()
    output = np_bf162np_float(output)
    print('{}: bf16-dnnl MSE: {}'.format(name,
          np.square(np.subtract(output, mxnet_output)).mean()))
    print(
        "{}: bf16-dnnl result[0:12]: \n{}".format(name, output.flatten()[0:12]))
'''

if __name__ == "__main__":
    # os.environ["TVM_LOG_DEBUG"]="DEFAULT=1;ir/transform.cc=1;relay/ir/transform.cc=1"
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--name",
        type=str,
        default="all",
        help="The name of the neural name.",
    )
    parser.add_argument("--batch-size", type=int,
                        default=1, help="The batch size")
    parser.add_argument(
        "--target",
        type=str,
        default="llvm -mcpu=cooperlake -model=platinum-8369",
        help="The compilation target.",
    )
    parser.add_argument("--dtype", type=str,
                        default="float32", help="The data type.")

    parser.add_argument("--warmup", type=int, default=20)
    parser.add_argument("--repeat", type=int, default=5)
    parser.add_argument("--steps", type=int, default=20)
    parser.add_argument("--profiling", type=bool, default=False)
    args = parser.parse_args()

    if args.name == "all":
        names = []
        for name in list(gluoncv.model_zoo.get_model_list()):
            if len(get_input_shape(name, args.batch_size)):
                names.append(name)
    else:
        names = [args.name]

    target = tvm.target.Target(args.target)

    for name in names:
        benchmark(name, args.batch_size, warmup=args.warmup, repeat=args.repeat,
                  steps=args.steps, target=args.target, profiling=args.profiling)

# %%
