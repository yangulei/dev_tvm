#%%
import json
import numpy as np
import base64

#%%
network = "resnet18_v1b"

#%%
with open('mod_{}_fp32_bind.json'.format(network), 'rb') as fin:
    mod_json = json.load(fin)

# %%
arrays = mod_json['b64ndarrays']
for array in arrays:
    nparray=np.frombuffer(base64.decodebytes(array.encode('utf-8')))
    print(nparray)
# %%
