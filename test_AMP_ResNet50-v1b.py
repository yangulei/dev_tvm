# %% imports
import tvm
from tvm import relay
from tvm import te
import numpy as np
import gluoncv

# %% import resnet
input_name = "data"
batch_size = 1
dtype = 'float32'
input_shape = (batch_size, 3, 224, 224)
output_shape = (batch_size, 1000)
block = gluoncv.model_zoo.get_model("ResNet50_v1b", pretrained=True)
print("importing ResNet50_v1b from gluoncv ... ")
mod_fp32, params = relay.frontend.from_mxnet(
    block, shape={input_name: input_shape}, dtype=dtype
)
with open('mod_fp32.txt', 'w') as fout:
    fout.write(mod_fp32.astext().replace(', %resnetv1b', ', \n%resnetv1b'))
print("done")

# %% convert to fp16
print("converting to fp16 ... ")
mod_fp16 = relay.transform.ToMixedPrecision('float16')(mod_fp32)
# mod_fp16 = relay.frontend.ChangeDatatype('float32', 'float16')(mod_fp32)
with open('mod_fp16.txt', 'w') as fout:
    fout.write(mod_fp16.astext().replace(', %resnetv1b', ', \n%resnetv1b'))
print("done")

# %% convert to bfloat16
print("converting to bf16 ... ")
mod_bf16 = relay.transform.ToMixedPrecision('bfloat16')(mod_fp32)
# mod_bf16 = relay.frontend.ChangeDatatype('float32', 'bfloat16')(mod_fp32)
with open('mod_bf16.txt', 'w') as fout:
    fout.write(mod_bf16.astext().replace(', %resnetv1b', ', \n%resnetv1b'))
print("done")

# %% potimize graph
# print("optimizing fp32 model ... ")
# opt_mod_fp32, opt_params_fp32 = relay.build_module.optimize(mod_fp32, target='llvm', params=params)
# with open('opt_mod_fp32.txt', 'w') as fout:
#     fout.write(opt_mod_fp32.astext().replace(', %resnetv1b', ', \n%resnetv1b'))
# print("done")

# %% potimize graph
# print("optimizing fp16 model ... ")
# opt_mod_fp16, opt_params_fp16 = relay.build_module.optimize(mod_fp16, target='llvm', params=params)
# with open('opt_mod_fp16.txt', 'w') as fout:
#     fout.write(opt_mod_fp16.astext().replace(', %resnetv1b', ', \n%resnetv1b'))
# print("done")
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
    return tvm.nd.empty(nparr.shape, "bfloat16").copyfrom(nparr)

def np_bf162np_float(arr):
    """Convert a numpy array of bf16 (uint16) to a numpy array
    of float"""
    u32 = np.left_shift(arr.astype("uint32"), 16)
    return u32.view("<f4")

def np_bf16_cast_and_cast_back(arr):
    """Convert a numpy array of float to bf16 and cast back"""
    return np_bf162np_float(np_float2np_bf16(arr))

# %% potimize graph
print("optimizing bf16 model ... ")
params_bf16 = {}
for key in params.keys():
    # params_bf16[key] = np_float2np_bf16(params[key].numpy())
    params_bf16[key] = tvm.nd.empty(params[key].shape,'bfloat16')
opt_mod_bf16, opt_params_bf16 = relay.build_module.optimize(mod_bf16, target='llvm', params=params)
with open('opt_mod_bf16.txt', 'w') as fout:
    fout.write(opt_mod_bf16.astext().replace(', %resnetv1b', ', \n%resnetv1b'))
print("done")

# %% build fp32 module
json, lib, params = relay.build(mod_fp32, target='llvm', params=params)
with open('built_fp32.json', 'w') as fout:
    fout.write(json)

# %% build fp16 module
json, lib, params = relay.build(mod_fp16, target='llvm', params=params)
with open('built_fp16.json', 'w') as fout:
    fout.write(json)

# %% build bf16 module
json, lib, params = relay.build(mod_bf16, target='llvm', params=params)
with open('built_bf16.json', 'w') as fout:
    fout.write(json)

# %%
