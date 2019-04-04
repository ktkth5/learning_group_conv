import argparse
from pretrainedmodels import *
from utils.benchmark import add_flops_counting_methods
import torch
from torch.autograd import Variable

# all_models = [fbresnet152, cafferesnet101, bninception, resnext101_32x4d, resnext101_64x4d, inceptionv4, inceptionresnetv2, nasnetalarge, nasnetamobile, alexnet, densenet121, densenet169, densenet201, densenet161, resnet18, resnet34, resnet50, resnet101, resnet152, inceptionv3, squeezenet1_0, squeezenet1_1, vgg11, vgg11_bn, vgg13, vgg13_bn, vgg16, vgg16_bn, vgg19_bn, vgg19, dpn68, dpn68b, dpn92, dpn98, dpn131, dpn107, xception, senet154, se_resnet50, se_resnet101, se_resnet152, se_resnext50_32x4d, se_resnext101_32x4d, pnasnet5large, polynet]
# names = ["fbresnet152", "cafferesnet101", "bninception", "resnext101_32x4d", "resnext101_64x4d", "inceptionv4", "inceptionresnetv2", "nasnetalarge", "nasnetamobile", "alexnet", "densenet121", "densenet169", "densenet201", "densenet161", "resnet18", "resnet34", "resnet50", "resnet101", "resnet152", "inceptionv3", "squeezenet1_0", "squeezenet1_1", "vgg11", "vgg11_bn", "vgg13", "vgg13_bn", "vgg16", "vgg16_bn", "vgg19_bn", "vgg19", "dpn68", "dpn68b", "dpn92", "dpn98", "dpn131", "dpn107", "xception", "senet154", "se_resnet50", "se_resnet101", "se_resnet152", "se_resnext50_32x4d", "se_resnext101_32x4d", "pnasnet5large", "polynet"]
#
# model_dict = {}
# for (model,name) in zip(all_models,names):
#     print ("'%s':%s,"%(name,name))


"""
_models = {
    'squeezenet1_0': { 'n_params': 1248424, 'feature_l_name': 'last_conv', 'acc1':58.1},
    'squeezenet1_1': {'n_params': 1235496, 'feature_l_name': 'last_conv', "acc1":58.3},
    'vgg11_bn': { 'n_params': 123642856, 'feature_l_name': 'last_linear', 'model_out_size': 4096, 'acc1':70.5},
    'vgg13_bn': { 'n_params': 123642856, 'feature_l_name': 'last_linear', 'model_out_size': 4096, 'acc1':71.5},
    'vgg16_bn': { 'n_params': 123642856, 'feature_l_name': 'last_linear', 'model_out_size': 4096, 'acc1':73.5},
    'vgg19_bn': { 'n_params': 123642856, 'feature_l_name': 'last_linear', 'model_out_size': 4096, 'acc1':74.3},
    'cafferesnet101':{'acc1':76.2},
    'vgg11':{'acc1':68.9},
    'vgg13':{'acc1':69.6},
    'vgg16':{'acc1':71.6},
    'vgg19':{'acc1':72.0},
    'dpn68b':{'acc1':77.0},
    'dpn92':{'acc1':79.4},
    'dpn107':{'acc1':79.7},
    'pnasnet5large':{'acc1':82.7},
    'polynet':{'acc1':81.0},
    'densenet121': { 'n_params': 7978856, 'feature_l_name': 'last_linear', 'model_out_size': 1024, 'acc1':74.6},
    'densenet161': {'n_params': 28681000, 'feature_l_name': 'last_linear', 'model_out_size': 2208, 'acc1':77.6},
    'densenet169':{ 'n_params': 14149480, 'feature_l_name': 'last_linear', 'model_out_size': 1664, 'acc1':76.0},
    'densenet201': { 'n_params': 20013928, 'feature_l_name': 'last_linear', 'model_out_size': 1920, 'acc1':77.2},
    'inceptionresnetv2': { 'n_params': 55843464, 'feature_l_name': 'last_linear', 'model_out_size': 1536, 'acc1':80.2},
    'inceptionv3': { 'n_params': 27161264, 'feature_l_name': 'last_linear', 'model_out_size': 2048, 'acc1':77.3},
    'inceptionv4': { 'n_params': 42679816, 'feature_l_name': 'last_linear', 'model_out_size': 1536, 'acc1':80.1},
    'bninception': {'n_params': 11295240, 'feature_l_name': 'last_linear', 'model_out_size': 1024, 'acc1':73.5},
    'xception': { 'n_params': 22855952, 'feature_l_name': 'last_linear', 'model_out_size': 2048, 'acc1':79.9},
    'dpn68': { 'n_params': 12611602, 'feature_l_name': 'classifier', 'acc1':75.9},
    'dpn98': { 'n_params': 61570728, 'feature_l_name': 'classifier', 'acc1':79.2},
    'dpn131': {'n_params': 79254504, 'feature_l_name': 'classifier', 'acc1':79.4},
    'alexnet': { 'n_params': 61100840, 'feature_l_name': 'last_linear', 'model_out_size': 4096, 'acc1':56.4},
    'resnet18': { 'n_params': 11689512, 'feature_l_name': 'last_linear', 'model_out_size': 512, 'acc1':70.1},
    'resnet34': {'n_params': 21797672, 'feature_l_name': 'last_linear', 'model_out_size': 512, 'acc1':73.6},
    'resnet50': {'n_params': 25557032, 'feature_l_name': 'last_linear', 'model_out_size': 2048, 'acc1':76.0},
    'resnet101': { 'n_params': 44549160, 'feature_l_name': 'last_linear', 'model_out_size': 2048, 'acc1':77.4},
    'resnet152': { 'n_params': 60192808, 'feature_l_name': 'last_linear', 'model_out_size': 2048, 'acc1':78.4},
    'resnext101_32x4d': { 'n_params': 44177704, 'feature_l_name': 'last_linear', 'model_out_size': 2048, 'acc1':78.8},
    'resnext101_64x4d': { 'n_params': 83455272, 'feature_l_name': 'last_linear', 'model_out_size': 2048, 'acc1':79.6},
    'senet154': { 'n_params': 115088984, 'feature_l_name': 'last_linear', 'model_out_size': 2048, 'acc1':81.3},
    'fbresnet152': {'n_params': 60268520, 'feature_l_name': 'last_linear', 'model_out_size': 2048, 'acc1':77.8},
    'se_resnet50': { 'n_params': 28088024, 'feature_l_name': 'last_linear', 'model_out_size': 2048, 'acc1':77.6},
    'se_resnet101': { 'n_params': 49326872, 'feature_l_name': 'last_linear', 'model_out_size': 2048, 'acc1':78.3},
    'se_resnet152': { 'n_params': 66821848, 'feature_l_name': 'last_linear', 'model_out_size': 2048, 'acc1':78.7},
    'se_resnext50_32x4d': { 'n_params': 27559896, 'feature_l_name': 'last_linear',  'model_out_size': 2048, 'acc1':79.1},
    'se_resnext101_32x4d': {'n_params': 48955416, 'feature_l_name': 'last_linear', 'model_out_size': 2048, 'acc1':80.2},
    'nasnetalarge': {'n_params': 88753150, 'feature_l_name': 'last_linear', 'model_out_size': 4032, 'acc1':82.6},
    'nasnetamobile': {'n_params': 5289978, 'feature_l_name': 'last_linear', 'model_out_size': 1056, 'acc1':74.1},
}
for (model,name) in zip(all_models,names):
    m = model(pretrained=None)
    m = add_flops_counting_methods(m)
    m = m.train()
    m.start_flops_count()
    imsize = list(pretrained_settings[name].values())[0]["input_size"]
    batch = torch.FloatTensor(bs, imsize[0], imsize[1], imsize[2])
    _ = m(batch)
    print (name, m.compute_average_flops_cost() / 1e9 / 2 , _models[name]["acc1"])
    print (m)
"""

import test_cifar

bs = 2
img_sizes = [27]
for img_size in img_sizes:
    m = test_cifar.test_cifar()
    m = add_flops_counting_methods(m)
    m = m.train()
    m.start_flops_count()
    batch = torch.FloatTensor(bs, 3, img_size, img_size)
    _ = m(batch)
    print("original YOLOv3", f"img size: {img_size} flops: {m.compute_average_flops_cost() / 1e9 / 2}")




"""
fbresnet152 11.52285568 77.8
cafferesnet101 7.568146432 76.2
bninception 2.033444336 73.5
resnext101_32x4d 22.751002624 78.8
resnext101_64x4d 75.544182784 79.6
inceptionv4 12.252438624 80.1
inceptionresnetv2 13.160027936 80.2
nasnetalarge 213.149422296 82.6
nasnetamobile 4.95378964 74.1
alexnet 0.655809024 56.4
densenet121 2.833137664 74.6
densenet169 3.358179328 76.0
densenet201 4.289445888 77.2
densenet161 7.725699072 77.6
resnet18 1.813561344 70.1
resnet34 3.663249408 73.6
resnet50 4.087136256 76.0
resnet101 7.79935744 77.4
resnet152 11.511578624 78.4
inceptionv3 5.716083296 77.3
squeezenet1_0 0.8210977 58.1
squeezenet1_1 0.350446612 58.3
vgg11 7.489169408 68.9
vgg11_bn 7.489169408 70.5
vgg13 11.190953984 69.6
vgg13_bn 11.190953984 71.5
vgg16 15.353404416 71.6
vgg16_bn 15.353404416 73.5
vgg19_bn 19.515854848 74.3
vgg19 19.515854848 72.0
dpn68 12.181533172 75.9
dpn68b 12.181533172 77.0
dpn92 14.061200884 79.4
dpn98 34.214290932 79.2
dpn131 46.305904116 79.4
dpn107 57.022092788 79.7
xception 71.417836448 79.9
senet154 66.206008152 81.3
se_resnet50 3.858448216 77.6
se_resnet101 7.572906872 78.3
se_resnet152 11.286971224 78.7
se_resnext50_32x4d 11.398496088 79.1
se_resnext101_32x4d 22.755763064 80.2
pnasnet5large 293.76969174 82.7
polynet 34.697445472 81.0
"""