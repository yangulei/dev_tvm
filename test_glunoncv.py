#%%
import mxnet as mx
import gluoncv
import numpy as np
import re

#%%
model_list = sorted(list(gluoncv.model_zoo.get_model_list()))
print(model_list)

#%%
def get_input_shape(model, batch_size):
    input_shape = list()
    if model.endswith('_int8'):
        return list()
    # classification
    elif re.match('(^resnet)+[18,34,50,101,152]+_+[v1,v1b,v1c,v1d,v1s,v2]+(_\d.\d\d$|$)', model):
        input_shape = (batch_size, 3, 224, 224)
    elif re.match('(^resnext|^se_resnext)+[50,101]+_', model):
        input_shape = (batch_size, 3, 224, 224)
    elif model.startswith('resnest'):
        input_shape = (batch_size, 3, 224, 224)
    elif model.startswith('mobilenet'):
        input_shape = (batch_size, 3, 224, 224)
    elif re.match('^[vgg]+[11,13,16,19]+($|_bn)', model):
        input_shape = (batch_size, 3, 224, 224)
    elif model.startswith('squeezenet'):
        input_shape = (batch_size, 3, 224, 224)
    elif model.startswith('densenet'):
        input_shape = (batch_size, 3, 224, 224)
    elif model in ('alexnet', 'darknet53', 'googlenet', 'xception', 'senet_154'):
        input_shape = (batch_size, 3, 224, 224)
    elif model == 'inceptionv3':
        input_shape = (batch_size, 3, 299, 299)
    elif re.match('(^cifar_)(resnet|wideresnet|resnext29_16x64d)', model):
        input_shape = (batch_size, 3, 224, 224)

    # # detection
    # elif re.match('(^ssd_300_)+(resnet34_v1b|vgg16_atrous)+(_voc$|_coco$)', model):
    #     input_shape = (batch_size, 3, 300, 300)
    # elif re.match('(^ssd_512_)+(vgg16_atrous|resnet50_v1|mobilenet1.0)+(_voc|_coco)+$', model):
    #     input_shape = (batch_size, 3, 512, 512)
    # elif re.match('^faster_rcnn_*', model):
    #     input_shape = (1, 3, 600, 600)
    # elif re.match('^[yolo3_]+(darknet53|mobilenet1.0)+(_voc$|_coco$)', model):
    #     input_shape = (batch_size, 3, 416, 416)
    # elif re.match('(^center_net_resnet)+(18|50|101)+_v1b_', model):
    #     input_shape = (batch_size, 3, 512, 512)

    # # segmentation
    # elif re.match('(^fcn|^psp|^deeplab)+_resnet+(50|101|200|269)+(_ade$|_coco$)', model):
    #     input_shape = (batch_size, 3, 480, 480)
    # elif model.endswith('citys'):
    #     input_shape = (batch_size, 3, 480, 480)
    # elif model == 'icnet_resnet50_mhpv1':
    #     input_shape = (batch_size, 3, 480, 480)
    # elif model.startswith('mask_rcnn'):
    #     input_shape = (1, 3, 480, 480)

    # # pose estimation
    # elif re.match('(^simple_pose_resnet)', model):
    #     input_shape = (batch_size, 3, 256, 192)
    # elif model.startswith('mobile_pose'):
    #     input_shape = (batch_size, 3, 256, 192)
    # elif model.startswith('alpha_pose'):
    #     input_shape = (batch_size, 3, 256, 192)

    # # action recognition
    # elif model in ('inceptionv3_kinetics400', 'inceptionv3_ucf101'):
    #     input_shape = (32, 3, 299, 299)
    # elif re.match('(^resnet)+(18|34|50|101|152)+_v1b_kinetics400$', model):
    #     input_shape = (32, 3, 224, 224)
    # elif re.match('(^c3d|^r2plus1d)(\S*)kinetics400$', model):
    #     input_shape = (batch_size, 3, 32, 112, 112)
    # elif model in ('p3d_resnet50_kinetics400', 'p3d_resnet101_kinetics400'):
    #     input_shape = (batch_size, 3, 32, 112, 112)
    # elif re.match('(^i3d_)(\S*)(kinetics400$|ucf101$)', model):
    #     input_shape = (batch_size, 3, 32, 224, 224)
    # elif re.match('(^slowfast_)(4x16_resnet50|8x8_resnet50|8x8_resnet101)_kinetics400$', model): # got DNNL error
    #     input_shape = (batch_size, 3, 4, 224, 224)
    # elif model == 'i3d_slow_resnet101_f16s4_kinetics700':
    #     input_shape = (batch_size, 3, 4, 224, 224)
    # elif model == 'vgg16_ucf101':
    #     input_shape = (32, 3, 224, 224)
    # elif model == 'resnet50_v1b_hmdb51':
    #     input_shape = (32, 3, 224, 224)
    # elif model == 'resnet50_v1b_sthsthv2':
    #     input_shape = (32, 3, 224, 224)
    # elif model == 'i3d_resnet50_v1_sthsthv2':
    #     input_shape = (batch_size, 3, 32, 224, 224)

    # # depth prediction
    # elif model.startswith('monodepth2_resnet18_kitti_'):
    #     input_shape = (batch_size, 3, 640, 192)
    # elif model.startswith('monodepth2_resnet18_posenet_'):
    #     input_shape = (batch_size, 6, 640, 192)
    else:
        return list()

    # print("input shape for {}: {}".format(model, input_shape))
    return input_shape

#%%
batch_size = 1
shape_dict = {}
for model in model_list:
    input_shape = get_input_shape(model, batch_size)
    if len(input_shape):
        print("processing ", model)
        input_data = np.random.uniform(-1, 1, size=input_shape).astype("float32")
        mxnet_input = mx.ndarray.array(input_data)
        net = gluoncv.model_zoo.get_model(model, pretrained=True)
        net(mxnet_input)
        # mxnet_output = net(mxnet_input).asnumpy()
        shape_dict[model] = input_shape
        try:
            input_data = np.random.uniform(-1, 1, size=input_shape).astype("float32")
            mxnet_input = mx.ndarray.array(input_data)
            net = gluoncv.model_zoo.get_model(model, pretrained=True)
            net(mxnet_input)
            # mxnet_output = net(mxnet_input).asnumpy()
            shape_dict[model] = input_shape
            print("{}: {}".format(model, input_shape))
        except:
            print("cannot run ", model)
print(len(shape_dict), " models found.")
print(shape_dict)

# %%
# net = gluoncv.model_zoo.get_model("model", pretrained=True)

# %%
