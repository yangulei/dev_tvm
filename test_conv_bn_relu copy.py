# %% imports
import tvm
import tvm.testing
from tvm import relay
from tvm import te
import numpy as np
import os

# %% helper func for fp32-bf16 conversion
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
    
# %% tell out the pid for debugging
print("current pid: {}".format(os.getpid()))

# %% conv-bn-relu
print("construct conv-bn-relu pattern ... ")
conv_dtype = 'float32'
data = relay.var("data", shape=(1, 3, 224, 224), dtype=conv_dtype)
weight = relay.var("weight", shape=(64, 3, 7, 7), dtype=conv_dtype)
conv = relay.nn.conv2d(data, weight, strides=(2, 2), padding=(3, 3, 3, 3), channels=64, kernel_size=(7, 7), out_dtype=conv_dtype)

bn_dtype = 'float32'
bn_in = relay.var('bn_in', shape=(1, 64, 112, 112), dtype=bn_dtype)
bn_gamma = relay.var("gamma", shape=(64,), dtype=bn_dtype)
bn_beta = relay.var("beta", shape=(64,), dtype=bn_dtype)
bn_mean = relay.var("mean", shape=(64,), dtype=bn_dtype)
bn_var = relay.var("var", shape=(64,), dtype=bn_dtype)
bn = relay.nn.batch_norm(conv, bn_gamma, bn_beta, bn_mean, bn_var)

relu = relay.nn.relu(bn[0])

mod_conv_bn_relu = tvm.IRModule.from_expr(relu)
with open('mod_conv_bn_relu.txt', 'w') as fout:
    fout.write(mod_conv_bn_relu.astext())
print("done")

# %% prepare weights
print("preparing weights ... ")
weights_type = 'float32'
params = {
    # "data": np.random.uniform(-1, 1, size=(1, 3, 224, 224)).astype(weights_type),
    "weight": np.random.uniform(-1, 1, size=(64, 3, 7, 7)).astype(weights_type),
    "gamma": np.random.uniform(-1, 1, size=(64, )).astype(weights_type),
    "beta": np.random.uniform(-1, 1, size=(64, )).astype(weights_type),
    "mean": np.random.uniform(-1, 1, size=(64, )).astype(weights_type),
    "var": np.random.uniform(-1, 1, size=(64, )).astype(weights_type),
    }
print("done")
# %% convert to fp32 model
print("converting to fp32 model ... ")
mod_fp32 = tvm.relay.transform.InferType()(mod_conv_bn_relu)
with open('mod_fp32.txt', 'w') as fout:
    fout.write(mod_fp32.astext().replace(', %resnetv1b', ', \n%resnetv1b'))
print("done")

# %% convert to fp16 model
print("converting to fp16 model ... ")
mod_fp16 = relay.transform.ToMixedPrecision('float16')(mod_fp32)
with open('mod_fp16.txt', 'w') as fout:
    fout.write(mod_fp16.astext().replace(', %resnetv1b', ', \n%resnetv1b'))
print("done")

# %% convert to bf16 model
print("converting to bf16 model ... ")
mod_bf16 = relay.transform.ToMixedPrecision('bfloat16')(mod_fp32)
with open('mod_bf16.txt', 'w') as fout:
    fout.write(mod_bf16.astext().replace(', %resnetv1b', ', \n%resnetv1b'))
print("done")

# %% opotimize fp32 model
print("optimizing fp32 model ... ")
opt_mod_fp32, opt_params_fp32 = relay.build_module.optimize(mod_fp32, target='llvm', params=params)
with open('opt_mod_fp32.txt', 'w') as fout:
    fout.write(opt_mod_fp32.astext().replace(', %resnetv1b', ', \n%resnetv1b'))
print("done")

# %% potimize fp16 model
print("optimizing fp16 model ... ")
opt_mod_fp16, opt_params_fp16 = relay.build_module.optimize(mod_fp16, target='llvm', params=params)
with open('opt_mod_fp16.txt', 'w') as fout:
    fout.write(opt_mod_fp16.astext().replace(', %resnetv1b', ', \n%resnetv1b'))
print("done")

# %% potimize bf16 model
print("optimizing bf16 model ... ")
opt_mod_bf16, opt_params_bf16 = relay.build_module.optimize(mod_bf16, target='llvm', params=params)
with open('opt_mod_bf16.txt', 'w') as fout:
    fout.write(opt_mod_bf16.astext().replace(', %resnetv1b', ', \n%resnetv1b'))
print("done")

# %%
seq = tvm.transform.Sequential([
    relay.transform.FuseOps(),
    # relay.transform.FoldConstant(),
    ])
with tvm.transform.PassContext(opt_level=3):
    with tvm.target.Target("llvm"):
        seq_mod_bf16 = seq(mod_fp32)
print(seq_mod_bf16)
with open('seq_mod_bf16.txt', 'w') as fout:
    fout.write(opt_mod_bf16.astext())
print("done")
# %%
