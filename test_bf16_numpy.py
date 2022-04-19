# %%
import numpy as np
import matplotlib.pyplot as plt

# %% helper functions for float32 <--> bf16 conversions
def np_float2np_bf16(arr):
    """Convert a numpy array of float to a numpy array
    of bf16 in uint16"""
    orig = arr.view("<u4")
    bias = np.bitwise_and(np.right_shift(orig, 16), 1) + 0x7FFF
    return np.right_shift(orig + bias, 16).astype("uint16")

def np_bf162np_float(arr):
    """Convert a numpy array of bf16 (uint16) to a numpy array
    of float"""
    u32 = np.left_shift(arr.astype("uint32"), 16)
    return u32.view("<f4")

def np_bf16_cast_and_cast_back(arr):
    """Convert a numpy array of float to bf16 and cast back"""
    return np_bf162np_float(np_float2np_bf16(arr))

# %%
# shape = (1024, 1024)
total = 1024*1024
np.random.seed(2022)
x_fp32 = np.linspace(-1, 1, total).astype("float32")
x_cast = np_bf16_cast_and_cast_back(x_fp32)
abs_diff = np.abs(x_fp32-x_cast)
rel_diff = abs_diff/np.abs(x_fp32)
# %%
abs_hist, abs_edges = np.histogram(abs_diff, bins=100)
# abs_hist = abs_hist/total*100
# plt.hist(abs_diff, bins=100, histtype='step')
plt.plot(abs_edges[1:], abs_hist)
plt.show()

# %%
rel_hist, rel_edges = np.histogram(rel_diff, bins=100)
plt.plot(rel_edges[1:], rel_hist)
plt.show()
# %%
