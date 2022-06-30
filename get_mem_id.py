
log_files=['/home/youlei/dev_tvm/test_set_handle/resnet50_v1b.log','/home/youlei/dev_tvm/test_post_sum/resnet50_v1b.log']

def replace_mem_with_id(log_file):
    mem_list=[]
    with open(log_file,'r') as fin:
        with open(log_file.replace('.log', '_id.log'),'w') as fout:
            for line in fin:
                if 'address of scratch' in line:
                    continue
                elif 'address of' in line:
                    address = line.strip().split(' ')[-1]
                    if address not in mem_list:
                        mem_list.append(address)
                    mem_idx = mem_list.index(address)
                    line = line.replace(address, str(mem_idx))
                elif 'onednn_verbose' in line:
                    exe_time = line.strip().split(',')[-1]
                    if exe_time.replace('.','',1).isdigit():
                        line = ','.join(line.split(',')[:-1]) + '\n'
                fout.write(line)

for log_file in log_files:
    replace_mem_with_id(log_file)