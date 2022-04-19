#%%
from tvm import relay
from tvm.relay.build_module import bind_params_by_name
import gluoncv
import json

#%%
network = "resnet18_v1b"
input_shape = (1, 3, 224, 224)

#%%
print("importing {} from gluoncv ... ".format(network))
net = gluoncv.model_zoo.get_model(network, pretrained=True)
print("done")

#%%
print("importing to fp32 graph ... ")
mod_fp32, params = relay.frontend.from_mxnet(
    net, shape={"data": input_shape}, dtype="float32"
)
print("done")

#%%
print("saving fp32 model ...")
with open('mod_{}_fp32.swift'.format(network), 'w') as fout:
    fout.write(mod_fp32.astext(show_meta_data=True))
print("done")

#%%
print("binding the params ...")
mod_fp32["main"] = bind_params_by_name(mod_fp32["main"], params)
print("done")

#%%
print("saving binded fp32 model ...")
with open('mod_{}_fp32_bind.swift'.format(network), 'w') as fout:
    fout.write(mod_fp32.astext(show_meta_data=True))
print("done")

#%%
with open('mod_{}_fp32_bind.json'.format(network), 'rb') as fin:
    mod_fp32_json = json.load(fin)
print(mod_fp32_json)