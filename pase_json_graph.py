# %%
import json

# %%
json_file = '/home/youlei/dev_tvm/test_post_sum/mod_resnet50_v1b_byoc-fp32_built.json'
with open(json_file, 'r') as f:
    json_str = f.read()

# %%
json_dict = json.loads(json_str)

# %%
nodes = json_dict['nodes']
arg_nodes = json_dict['arg_nodes']
heads = json_dict['heads']
node_row_ptr = json_dict['node_row_ptr']

dltypes = json_dict['attrs']['dltype'][1]   # list_str
device_indexes = json_dict['attrs']['device_index'][1]  # list_int
shapes = json_dict['attrs']['shape'][1] # list_shape
storage_ids = json_dict['attrs']['storage_id'][1]   # list_int

# %%
num_nodes = len(nodes)
num_entries = node_row_ptr[-1]

# %%
cmp_nodes = []
for node in nodes:
    if node['op'] == 'tvm_op':
        num_inputs = int(node['attrs']['num_inputs'])
        num_outputs = int(node['attrs']['num_outputs'])

        inputs = node['inputs']
        input_types = []
        input_shapes = []
        input_storage_ids = []
        for input in inputs:
            nid = input[0]
            indx = input[1]
            eid = node_row_ptr[nid] + indx
            input_types.append(dltypes[eid])
            input_shapes.append(shapes[eid])
            input_storage_ids.append(storage_ids[eid])
        cmp_node = node
        cmp_node['input_types'] = input_types
        cmp_node['input_shapes'] = input_shapes
        cmp_node['input_storage_ids'] = input_storage_ids
        cmp_nodes.append(cmp_node)
    else:
        cmp_nodes.append(node)


# %%
with open(json_file.replace('built', 'cmp'), 'w') as f:
    f.write(json.dumps(cmp_nodes, indent=4))

# %%
