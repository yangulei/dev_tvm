# %% imports
import tvm
from tvm import relay
from tvm import te
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
# dst_dtype = 'bfloat16'
# dst = relay.cast(data, dst_dtype)
# mod_cast = tvm.IRModule.from_expr(dst)
# print("mod cast:\n", mod_cast)

# opt_mod_cast, opt_params_cast = relay.build_module.optimize(mod_cast, target='llvm')
# print("optimized mod cast:\n", opt_mod_cast)

# # %%
# print("test conv only")
# data = relay.var("data", shape=(1, 3, 224, 224), dtype='bfloat16')
# weight = relay.var("weight", shape=(64, 3, 7, 7), dtype='bfloat16')
# params = {
#     "weight": tvm.nd.empty((64, 3, 7, 7), 'bfloat16')
#     }
# conv = relay.nn.conv2d(data, weight, strides=(2, 2), padding=(3, 3, 3, 3), \
#     channels=64, kernel_size=(7, 7), out_dtype='bfloat16')
# mod_conv = tvm.IRModule.from_expr(conv)
# print("mod conv:\n", mod_conv)

# opt_mod_conv, opt_params_conv = relay.build_module.optimize( \
#     mod_conv, target='llvm', params=params)
# print("optimized mod conv:\n", opt_mod_conv)

# %%
print("testing cast+conv")
data = relay.var("data", shape=(1, 3, 224, 224), dtype='float32')
weight = relay.var("weight", shape=(64, 3, 7, 7), dtype='float32')
data_bf16 = relay.cast(data, 'bfloat16')
weight_bf16 = relay.cast(weight, 'bfloat16')
params = {
    "weight": tvm.nd.empty((64, 3, 7, 7), 'float32')
    }
# weight_bf16 = relay.sqrt(weight_bf16)
conv = relay.nn.conv2d(data_bf16, weight_bf16, strides=(2, 2), padding=(3, 3, 3, 3), \
    channels=64, kernel_size=(7, 7), out_dtype='bfloat16')

# sqrt = relay.sqrt(conv)
mod_conv = tvm.IRModule.from_expr(conv)
print("mod cast+conv:\n", mod_conv)

# fold_mod_conv = relay.transform.FoldConstant()(mod_conv)
# print("folded mod cast+conv:\n", fold_mod_conv)

with tvm.transform.PassContext(opt_level=3):
    opt_mod_conv, opt_params_conv = relay.build_module.optimize( \
    # opt_mod_conv, opt_params_conv = relay.build( \
        mod_conv, target='llvm', params=params)
    print("optimized mod cast+conv:\n", opt_mod_conv)

# %% build bf16 module
print("building bf16 model ... ")
with tvm.transform.PassContext(opt_level=3):
    built_graph_bf16, built_mod_bf16, built_params_bf16 = relay.build(mod_conv, target='llvm', params=params)
with open('built_graph_bf16.json', 'w') as json:
    json.write(built_graph_bf16)
print("done")

# %% np <--> TVM helperfunctions
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

# %%
np.random.seed(122)
do_vectorize = True
A = te.placeholder((32,), dtype="bfloat16")
B = te.placeholder((32,), dtype="bfloat16")
d = te.compute((32,), lambda x: A[x] + B[x])
sch = te.create_schedule(d.op)
print(tvm.lower(sch, [A, B, d]))
if do_vectorize:
    sch[d].vectorize(d.op.axis[0])
module = tvm.build(sch, [A, B, d])
npa = np.random.rand(32).astype("float32")
npb = np.random.rand(32).astype("float32")
va = np_bf16_cast_and_cast_back(npa)
vb = np_bf16_cast_and_cast_back(npb)
res = np_bf16_cast_and_cast_back(va + vb)
a_ = np_float2tvm_bf16(npa)
b_ = np_float2tvm_bf16(npb)
c_ = tvm.nd.empty((32,), "uint16")
module(a_, b_, c_)
tvm.testing.assert_allclose(np_bf162np_float(c_.numpy()), res)