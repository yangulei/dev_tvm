import tvm
from tvm import relay
import numpy as np
import os

os.environ["TVM_BACKTRACE"] = "1"
os.environ["TVM_LOG_DEBUG"] = ""
os.environ["TVM_LOG_DEBUG"] = "DEFAULT=1"

xshape = (2, 2, 1, 1)
inp = np.random.uniform(size=xshape).astype(np.int64)

x = relay.var("x", shape=xshape, dtype='float32')

v1 = relay.nn.conv2d(x, weight=relay.const(value=np.random.random((1, 2, 3, 1))), strides=[2, 2], padding=[3, 3, 3, 3], kernel_size=[3, 1], channels=1)
out = relay.squeeze(v1, axis=[1])

func = relay.Function([x], out)
mod = tvm.IRModule.from_expr(func)
print(mod.astext(show_meta_data=True))

with tvm.transform.PassContext(opt_level=4):
    relay.build_module.create_executor("graph", mod, tvm.cpu(), target='llvm').evaluate()