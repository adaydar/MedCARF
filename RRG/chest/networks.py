# -*- coding: utf-8 -*-

from network_pretrain import Densenet121
import torch


def init_net(net, pretrain, init_type, gpu_ids, init_gain=0.02):
    assert(torch.cuda.is_available())
    if len(gpu_ids) > 1:                
        net = torch.nn.DataParallel(net, gpu_ids)
    net.to('cuda')
    if not pretrain:
        init_weights(net, init_type, init_gain=init_gain)
    else:
        print('initialize network with pretrained net')
    return net
    
def define_network(init_type, gpu_ids, network, pretrain=True, avg=0, weight=1, truncated=0, num_classes=14):
    net = Densenet121(pretrain, avg, weight, truncated, num_classes)
    return net
