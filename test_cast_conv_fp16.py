# %% imports
import tvm
from tvm import relay
import numpy as np
import os

# %% set debug level
os.environ["TVM_BACKTRACE"] = "1"
os.environ["TVM_LOG_DEBUG"] = "DEFAULT=1"
    
# # %% tell out the pid for debugging
# print("current pid: {}".format(os.getpid()))

# # %%
# print("testing cast only")
# src_dtype = 'float32'
# data = relay.var("data", shape=(1, 3, 224, 224), dtype=src_dtype)
# dst_dtype = 'float16'
# dst = relay.cast(data, dst_dtype)
# mod_cast = tvm.IRModule.from_expr(dst)
# print("mod cast:\n", mod_cast)

# opt_mod_cast, opt_params_cast = relay.build_module.optimize(mod_cast, target='llvm')
# print("optimized mod cast:\n", opt_mod_cast)

# # %%
# print("test conv only")
# data = relay.var("data", shape=(1, 3, 224, 224), dtype='float16')
# weight = relay.var("weight", shape=(64, 3, 7, 7), dtype='float16')
# params = {
#     "weight": tvm.nd.empty((64, 3, 7, 7), 'float16')
#     }
# conv = relay.nn.conv2d(data, weight, strides=(2, 2), padding=(3, 3, 3, 3), \
#     channels=64, kernel_size=(7, 7), out_dtype='float16')
# mod_conv = tvm.IRModule.from_expr(conv)
# print("mod conv:\n", mod_conv)

# opt_mod_conv, opt_params_conv = relay.build_module.optimize( \
#     mod_conv, target='llvm', params=params)
# print("optimized mod conv:\n", opt_mod_conv)

# %%
print("testing cast+conv")
data = relay.var("data", shape=(1, 3, 224, 224), dtype='float32')
weight = relay.var("weight", shape=(64, 3, 7, 7), dtype='float32')
data_bf16 = relay.cast(data, 'float16')
weight_bf16 = relay.cast(weight, 'float16')
params = {
    "weight": tvm.nd.empty((64, 3, 7, 7), 'float32')
    }
conv = relay.nn.conv2d(data_bf16, weight_bf16, strides=(2, 2), padding=(3, 3, 3, 3), \
    channels=64, kernel_size=(7, 7), out_dtype='float32')
mod_conv = tvm.IRModule.from_expr(conv)
print("mod cast+conv:\n", mod_conv)

# fold_mod_conv = relay.transform.FoldConstant()(mod_conv)
# print("folded mod cast+conv:\n", fold_mod_conv)

with tvm.transform.PassContext(opt_level=3):
    opt_mod_conv, opt_params_conv = relay.build_module.optimize( \
    # opt_mod_conv, opt_params_conv = relay.build( \
        mod_conv, target='llvm', params=params)
    print("optimized mod cast+conv:\n", opt_mod_conv)

# %%