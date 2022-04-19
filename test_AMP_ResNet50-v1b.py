# %% imports
import tvm
from tvm import relay
from tvm import te
from tvm.contrib.debugger import debug_executor as runtime
from tvm.relay.build_module import bind_params_by_name
from tvm.contrib import graph_executor
import tvm.testing
import numpy as np
import gluoncv
import os

# %%
@tvm.instrument.pass_instrument
class PrintIR:
    """Print the name of the pass, the IR, only before passes execute."""
    def run_before_pass(self, mod, info):
        print("Running pass: {}", info)
        print(mod)

# %% TVM settings
os.environ["TVM_BACKTRACE"] = "1"
os.environ["TVM_LOG_DEBUG"] = ""
# os.environ["TVM_LOG_DEBUG"] = "DEFAULT=1"
target = "llvm"
opt_level = 3
dev = tvm.cpu()

# %% import resnet
input_name = "data"
batch_size = 1
dtype = 'float32'
input_shape = (batch_size, 3, 224, 224)
output_shape = (batch_size, 1000)
input_data = np.random.uniform(-1, 1, size=input_shape).astype("float32")
block = gluoncv.model_zoo.get_model("ResNet50_v1b", pretrained=True)
print("importing ResNet50_v1b from gluoncv ... ")
mod_fp32, params = relay.frontend.from_mxnet(
    block, shape={input_name: input_shape}, dtype=dtype
)
with open('mod_fp32.swift', 'w') as fout:
    fout.write(mod_fp32.astext(show_meta_data=False).replace(', %resnetv1b', ', \n%resnetv1b'))
print("done")

# # %% convert to fp16
# print("converting to fp16 ... ")
# mod_fp16 = relay.transform.ToMixedPrecision('float16')(mod_fp32)
# # mod_fp16 = relay.frontend.ChangeDatatype('float32', 'float16')(mod_fp32)
# with open('mod_fp16.swift', 'w') as fout:
#     fout.write(mod_fp16.astext(show_meta_data=False).replace(', %resnetv1b', ', \n%resnetv1b'))
# print("done")

# # %% bn folding
# print("folding BN ... ")
# bn_seq = tvm.transform.Sequential(
#     [   
#         # tvm.transform.PrintIR(),
#         relay.transform.CanonicalizeOps(),
#         relay.transform.InferType(),
#         relay.transform.SimplifyInference(),
#         relay.transform.FoldConstant(),
#         relay.transform.FoldScaleAxis(),
#         relay.transform.InferType(),
#         ])
# mod_fp32["main"] = bind_params_by_name(mod_fp32["main"], params)
# mod_fp32_bn = bn_seq(mod_fp32)
# with open('mod_fp32_bn.swift', 'w') as fout:
#     fout.write(mod_fp32_bn.astext(show_meta_data=False).replace(', %resnetv1b', ', \n%resnetv1b'))
# print("done")

# %% convert to bfloat16
print("converting to bf16 ... ")
mod_bf16 = relay.transform.ToMixedPrecision('bfloat16')(mod_fp32)
mod_bf16["main"] = bind_params_by_name(mod_bf16["main"], params)
# mod_bf16 = relay.frontend.ChangeDatatype('float32', 'bfloat16')(mod_fp32)
with open('mod_bf16.swift', 'w') as fout:
    fout.write(mod_bf16.astext(show_meta_data=False).replace(', %resnetv1b', ', \n%resnetv1b'))
print("done")

# %% oneDNN dnnl
from tvm.relay.op.contrib.dnnl import pattern_table
dnnl_seq = tvm.transform.Sequential(
    [   
        # tvm.transform.PrintIR(),
        relay.transform.CanonicalizeOps(),
        relay.transform.InferType(),
        relay.transform.SimplifyInference(),
        relay.transform.FoldConstant(),
        relay.transform.FoldScaleAxis(),
        tvm.transform.PrintIR(),

        # CustomPipeline(),
        # relay.transform.FoldConstant(),
        # # tvm.transform.PrintIR(),
        
        # relay.transform.AlterOpLayout(),
        # relay.transform.FoldConstant(),
        # # tvm.transform.PrintIR(),

        relay.transform.MergeComposite(pattern_table()),
        # tvm.transform.PrintIR(),
        relay.transform.AnnotateTarget("dnnl"),
        relay.transform.MergeCompilerRegions(),
        relay.transform.PartitionGraph(),
        tvm.transform.PrintIR(),
    ]
)
# # %%
# print("converting to fp32-dnnl ... ")
# mod_fp32_dnnl = dnnl_seq(mod_fp32)
# with open('mod_fp32_dnnl.swift', 'w') as fout:
#     fout.write(mod_fp32_dnnl.astext(show_meta_data=False).replace(', %resnetv1b', ', \n%resnetv1b'))
# print("done")

# # %%
# print("converting to bf16-dnnl ... ")
# mod_bf16_dnnl = dnnl_seq(mod_bf16)
# with open('mod_bf16_dnnl.swift', 'w') as fout:
#     fout.write(mod_bf16_dnnl.astext(show_meta_data=False).replace(', %resnetv1b', ', \n%resnetv1b'))
# print("done")

# # %% optimize fp32 graph
# print("optimizing fp32 model ... ")
# opt_mod_fp32, opt_params_fp32 = relay.build_module.optimize(mod_fp32, target='llvm', params=params)
# with open('opt_mod_fp32.swift', 'w') as fout:
#     fout.write(opt_mod_fp32.astext(show_meta_data=False).replace(', %resnetv1b', ', \n%resnetv1b'))
# print("done")

# # %% optimize fp16 graph
# print("optimizing fp16 model ... ")
# opt_mod_fp16, opt_params_fp16 = relay.build_module.optimize(mod_fp16, target='llvm', params=params)
# with open('opt_mod_fp16.swift', 'w') as fout:
#     fout.write(opt_mod_fp16.astext(show_meta_data=False).replace(', %resnetv1b', ', \n%resnetv1b'))
# print("done")

# # %% optimize bf16 graph
# print("optimizing bf16 model ... ")
# with tvm.transform.PassContext(opt_level=3):
#     opt_mod_bf16, opt_params_bf16 = relay.build_module.optimize(mod_bf16, target='llvm', params=params)
# with open('opt_mod_bf16.swift', 'w') as fout:
#     fout.write(opt_mod_bf16.astext(show_meta_data=False).replace(', %resnetv1b', ', \n%resnetv1b'))
# print("done")

# %% optimize bf16-dnnl graph
# print("optimizing bf16-dnnl model ... ")
# with tvm.transform.PassContext(opt_level=3):
#     opt_mod_bf16_dnnl, opt_params_bf16_dnnl = relay.build_module.optimize(mod_bf16_dnnl, target='llvm', params=params)
# with open('opt_mod_bf16_dnnl.swift', 'w') as fout:
#     fout.write(opt_mod_bf16_dnnl.astext(show_meta_data=False).replace(', %resnetv1b', ', \n%resnetv1b'))
# print("done")

# # %% build fp32 module
# built_graph_fp32, built_mod_fp32, built_params_fp32 = relay.build(mod_fp32, target='llvm', params=params)
# with open('built_graph_fp32.json', 'w') as json:
#     json.write(built_graph_fp32)

# # %% build fp16 module
# built_graph_fp16, built_mod_fp16, built_params_fp16 = relay.build(mod_fp16, target='llvm', params=params)
# with open('built_graph_fp16.json', 'w') as json:
#     json.write(built_graph_fp16)

# # %% build bf16 module
# print("building bf16 model ... ")
# with tvm.transform.PassContext(opt_level=3):
#     built_graph_bf16, built_mod_bf16, built_params_bf16 = relay.build(mod_bf16, target='llvm', params=params)
# with open('built_graph_bf16.json', 'w') as json:
#     json.write(built_graph_bf16)
# print("done")

# # %% build bf16-dnnl module
# print("building bf16-dnnl model ... ")
# with tvm.transform.PassContext(opt_level=3):
#     built_graph_bf16_dnnl, built_mod_bf16_dnnl, built_params_bf16_dnnl = relay.build(mod_bf16_dnnl, target='llvm', params=params)
# with open('built_graph_bf16_dnnl.json', 'w') as json:
#     json.write(built_graph_bf16_dnnl)
# print("done")

# %%
print("building fp32 lib ...")
with tvm.transform.PassContext(opt_level=opt_level):
    lib_fp32 = relay.build(mod_fp32, target, params=params)

print("running fp32 module ...")
# create module
module_fp32 = graph_executor.GraphModule(lib_fp32["default"](dev))
# set input and parameters
module_fp32.set_input("data", input_data)
# run
module_fp32.run()
# get output
out_fp32 = module_fp32.get_output(0, tvm.nd.empty(output_shape)).numpy()
out_fp32.tofile('out_fp32.csv',sep=',')

# Print first 10 elements of output
print("fp32 result[0:10]: \n{}".format(out_fp32.flatten()[0:10]))

# # %%
# print("building fp32-dnnl lib ...")
# with tvm.transform.PassContext(opt_level=opt_level):
#     lib_fp32_dnnl = relay.build(mod_fp32_dnnl, target, params=params)

# print("running fp32-dnnl module ...")
# # create module
# module_fp32_dnnl = graph_executor.GraphModule(lib_fp32_dnnl["default"](dev))
# # set input and parameters
# module_fp32_dnnl.set_input("data", input_data)
# # run
# module_fp32_dnnl.run()
# # get output
# out_fp32_dnnl = module_fp32_dnnl.get_output(0, tvm.nd.empty(output_shape)).numpy()

# # Print first 10 elements of output
# print("fp32-dnnl result: \n{}".format(out_fp32_dnnl.flatten()[0:10]))

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

# # %%
# print("building bf16 lib ...")
# with tvm.transform.PassContext(opt_level=opt_level):
#     lib_bf16 = relay.build(mod_bf16, target, params=params)

# print("running bf16 module ...")
# # create module
# module_bf16 = graph_executor.GraphModule(lib_bf16["default"](dev))
# # set input and parameters
# module_bf16.set_input("data", input_data)
# # run
# module_bf16.run()
# # get output
# out_bf16 = module_bf16.get_output(0, tvm.nd.empty(output_shape, "uint16")).numpy()

# # Print first 10 elements of output
# print("bf16 result: \n{}".format(np_bf162np_float(out_bf16).flatten()[0:10]))

# %%
print("building bf16-dnnl lib ...")
with tvm.transform.PassContext(opt_level=opt_level):
    lib_bf16_dnnl = relay.build(dnnl_seq(mod_bf16), target, params=params)

print("running bf16-dnnl module ...")
# create module
module_bf16_dnnl = graph_executor.GraphModule(lib_bf16_dnnl["default"](dev))
# set input and parameters
module_bf16_dnnl.set_input("data", input_data)
# run
module_bf16_dnnl.run()

# get output
out_bf16_dnnl = module_bf16_dnnl.get_output(0, tvm.nd.empty(output_shape, "uint16")).numpy()
out_bf16_dnnl = np_bf162np_float(out_bf16_dnnl).flatten()
out_bf16_dnnl.tofile('out_bf16_dnnl.csv',sep=',')

# Print first 10 elements of output
print("bf16_dnnl result[0:10]: \n{}".format(out_bf16_dnnl[0:10]))

# %% compute MSE

print('MSE:', np.square(np.subtract(out_fp32, out_bf16_dnnl)).mean())