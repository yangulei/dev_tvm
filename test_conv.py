# %% imports
import tvm
import tvm.testing
from tvm import relay
from tvm import te
import numpy as np
import os
    
# %% tell out the pid for debugging
print("current pid: {}".format(os.getpid()))

# %% conv
print("construct conv pattern ... ")
conv_dtype = 'float32'
data = relay.var("data", shape=(1, 3, 224, 224), dtype=conv_dtype)
weight = relay.var("weight", shape=(64, 3, 7, 7), dtype=conv_dtype)
conv = relay.nn.conv2d(data, weight, strides=(2, 2), padding=(3, 3, 3, 3), channels=64, kernel_size=(7, 7), out_dtype=conv_dtype)

mod_conv = tvm.IRModule.from_expr(conv)
with open('mod_conv.swift', 'w') as fout:
    fout.write(mod_conv.astext(show_meta_data=False))
print("done")

# %% prepare weights
print("preparing weights ... ")
weights_type = 'float32'
params = {
    # "data": np.random.uniform(-1, 1, size=(1, 3, 224, 224)).astype(weights_type),
    "weight": np.random.uniform(-1, 1, size=(64, 3, 7, 7)).astype(weights_type)
    }
print("done")
# %% convert to fp32 model
print("converting to fp32 model ... ")
mod_fp32 = tvm.relay.transform.InferType()(mod_conv)
with open('mod_fp32.swift', 'w') as fout:
    fout.write(mod_fp32.astext(show_meta_data=False))
print("done")

# %% convert to fp16 model
print("converting to fp16 model ... ")
mod_fp16 = relay.transform.ToMixedPrecision('float16')(mod_fp32)
with open('mod_fp16.swift', 'w') as fout:
    fout.write(mod_fp16.astext(show_meta_data=False))
print("done")

# %% convert to bf16 model
print("converting to bf16 model ... ")
mod_bf16 = relay.transform.ToMixedPrecision('bfloat16')(mod_fp32)
with open('mod_bf16.swift', 'w') as fout:
    fout.write(mod_bf16.astext(show_meta_data=False))
print("done")

# %% opotimize fp32 model
print("optimizing fp32 model ... ")
opt_mod_fp32, opt_params_fp32 = relay.build_module.optimize(mod_fp32, target='llvm', params=params)
with open('opt_mod_fp32.swift', 'w') as fout:
    fout.write(opt_mod_fp32.astext(show_meta_data=False))
print("done")

# %% potimize fp16 model
print("optimizing fp16 model ... ")
opt_mod_fp16, opt_params_fp16 = relay.build_module.optimize(mod_fp16, target='llvm', params=params)
with open('opt_mod_fp16.swift', 'w') as fout:
    fout.write(opt_mod_fp16.astext(show_meta_data=False))
print("done")

# %% potimize bf16 model
print("optimizing bf16 model ... ")
opt_mod_bf16, opt_params_bf16 = relay.build_module.optimize(mod_bf16, target='llvm', params=params)
with open('opt_mod_bf16.swift', 'w') as fout:
    fout.write(opt_mod_bf16.astext(show_meta_data=False))
print("done")