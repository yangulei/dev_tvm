# %% imports
import tvm
from tvm import relay
from tvm import te
import numpy as np
import gluoncv

# %% define tensors
n = 1024
A = tvm.te.placeholder((n,), name='A')
B = tvm.te.placeholder((n,), name='B')
C = tvm.te.compute(A.shape, lambda i: A[i] + B[i], name='C')
# %% define schedule
s = tvm.te.create_schedule(C.op)

# %% create runtime module
target = "llvm"
fadd = tvm.build(s, [A, B, C], target)

# %% run the module
dev = tvm.device(target, 0)
a = tvm.nd.array(np.random.uniform(size=n).astype(A.dtype), dev)
b = tvm.nd.array(np.random.uniform(size=n).astype(B.dtype), dev)
c = tvm.nd.array(np.zeros(n, dtype=C.dtype), dev)
fadd(a, b, c)
output= c.numpy()

# %%
data_shape = (1, 3, 224, 224)
weight_shape = (64, 3, 7, 7)
data = relay.var("data", shape=data_shape, dtype="float32")
weight = relay.var("weight", shape=weight_shape, dtype="float32")
conv = relay.nn.conv2d(data, weight, strides=(1, 1), padding=(1, 1), out_dtype="float32")
mod = tvm.IRModule.from_expr(conv)
mod = tvm.relay.transform.InferType()(mod)

mod_params = {
    "data": np.random.uniform(-1, 1, size=data_shape).astype("float32"),
    "weight": np.random.uniform(-1, 1, size=weight_shape).astype("float32"),
}
# %%
mod = tvm.relay.transform.InferType()(mod)
mod_bf16 = relay.transform.ToMixedPrecision('bfloat16', 0)(mod)
# %%
A = te.placeholder((32,), dtype="bfloat16")
B = te.placeholder((32,), dtype="bfloat16")
d = te.compute((32,), lambda x: A[x] + B[x])
sch = te.create_schedule(d.op)
print(tvm.lower(sch, [A, B, d]))
module = tvm.build(sch, [A, B, d])
# %% conv
conv_dtype = 'float32'
data = relay.var("data", shape=(1, 3, 224, 224), dtype=conv_dtype)
weight = relay.var("weight", shape=(64, 3, 7, 7), dtype=conv_dtype)
conv = relay.nn.conv2d(data, weight, strides=(2, 2), padding=(3, 3, 3, 3), channels=64, kernel_size=(7, 7), out_dtype=conv_dtype)
mod = tvm.IRModule.from_expr(conv)
mod = tvm.relay.transform.InferType()(mod)
mod_bf16 = relay.transform.ToMixedPrecision('bfloat16', 0)(mod)

# %% bn
bn_dtype = 'float32'
bn_gamma = relay.var("gamma", shape=(64,), dtype=bn_dtype)
bn_beta = relay.var("beta", shape=(64,), dtype=bn_dtype)
bn_mean = relay.var("mean", shape=(64,), dtype=bn_dtype)
bn_var = relay.var("var", shape=(64,), dtype=bn_dtype)
bn = relay.nn.batch_norm(conv, bn_gamma, bn_beta, bn_mean, bn_var)

# %%
mod = tvm.IRModule.from_expr(bn[0])
mod = tvm.relay.transform.InferType()(mod)
mod_bf16 = relay.transform.ToMixedPrecision('bfloat16', 0)(mod)

# %%
bn_dtype = 'float32'
bn_in = relay.var('bn_in', shape=(1, 64, 112, 112), dtype=bn_dtype)
bn_gamma = relay.var("gamma", shape=(64,), dtype=bn_dtype)
bn_beta = relay.var("beta", shape=(64,), dtype=bn_dtype)
bn_mean = relay.var("mean", shape=(64,), dtype=bn_dtype)
bn_var = relay.var("var", shape=(64,), dtype=bn_dtype)
bn = relay.nn.batch_norm(bn_in, bn_gamma, bn_beta, bn_mean, bn_var)
mod = tvm.IRModule.from_expr(bn[0])
mod = tvm.relay.transform.InferType()(mod)
print('FP32 model: ', mod.astext())
mod_bf16 = relay.transform.ToMixedPrecision('bfloat16', 0)(mod)
print('BF16 model: ', mod_bf16.astext())

# %%
conv_dtype = 'float32'
data = relay.var("data", shape=(1, 3, 224, 224), dtype=conv_dtype)
weight0 = relay.var("weight0", shape=(3, 3, 3, 3), dtype=conv_dtype)
conv0 = relay.nn.conv2d(data, weight0, strides=(1, 1), padding=(1, 1, 1, 1), channels=3, kernel_size=(3, 3), out_dtype=conv_dtype)
weight1 = relay.var("weight1", shape=(3, 3, 3, 3), dtype=conv_dtype)
conv1 = relay.nn.conv2d(conv0, weight1, strides=(1, 1), padding=(1, 1, 1, 1), channels=3, kernel_size=(3, 3), out_dtype=conv_dtype)
mod = tvm.IRModule.from_expr(conv1)
mod = tvm.relay.transform.InferType()(mod)
print('FP32 model: ', mod.astext())
mod_bf16 = relay.transform.ToMixedPrecision('bfloat16', 0)(mod)
print('BF16 model: ', mod_bf16.astext())
# %%
bn_dtype = 'float32'
bn_gamma = relay.var("gamma", shape=(3,), dtype=bn_dtype)
bn_beta = relay.var("beta", shape=(3,), dtype=bn_dtype)
bn_mean = relay.var("mean", shape=(3,), dtype=bn_dtype)
bn_var = relay.var("var", shape=(3,), dtype=bn_dtype)
bn = relay.nn.batch_norm(conv1, bn_gamma, bn_beta, bn_mean, bn_var)
mod = tvm.IRModule.from_expr(bn[0])
mod = tvm.relay.transform.InferType()(mod)
mod_bf16 = relay.transform.ToMixedPrecision('bfloat16', 0)(mod)
# %%
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

relu = relay.nn.relu(bn)

mod = tvm.IRModule.from_expr(conv)
mod = tvm.relay.transform.InferType()(mod)
mod_bf16 = relay.transform.ToMixedPrecision('bfloat16', 0)(mod)