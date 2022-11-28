import numpy as np
from numpy.core.fromnumeric import reshape, sort
from numpy.lib.function_base import select
import torch
import torch.nn as nn
import torch.optim as optim
import heapq
import copy

from math import *


def krum(args, nets, selected):
    n = len(selected)
    f = args.n_Byzantine 
    m = 1 # number of how many vectors to output
    if f == 0:
        return m, selected

    score_list = []
    for i in range(n):
        net_i = nets[i]
        net_i_para = net_i.state_dict()
        net_i_score_list = []
        for j in range(n):
            net_j = nets[j]
            tmp = 0
            if i == j: continue
            else:
                net_j_para = net_j.state_dict()
                for key in net_i_para:
                    tmp += torch.sum((net_i_para[key] - net_j_para[key])**2)
                net_i_score_list.append(tmp)
        # choose m smallest scores of net_i
        select_score_list = heapq.nsmallest(m, net_i_score_list)
        score_i = sum(select_score_list)
        score_list.append(score_i)
    select_net_list_idx = np.argpartition(score_list, m)
    select_net_list_idx = select_net_list_idx[:m]

    return m, select_net_list_idx

def trim(arr, f):
    n = len(arr)
    sorted_res = sorted(enumerate(arr), key=lambda x: x[1])
    sorted_idx = [i[0] for i in sorted_res]

    trimmed_idx = sorted_idx[f:n-f]
    return trimmed_idx

def trimmed_mean(args, nets, selected):
    n = len(selected)
    f = args.n_Byzantine # f=0: check OK
    m = n - 2*f # number of how many vectors to output
    # if f == 0:
    #     return m, nets

    net_dict = copy.deepcopy(nets)
    tmp = net_dict[0]
    tmp_para = tmp.state_dict()
    for key in tmp_para:
        # shape_rec = tmp_para[key].shape
        net_dict_key = []

        # reshape the para
        for i in range(n):
            net = net_dict[i]
            net_para = net.state_dict()
            net_para_key = net_para[key]
            net_para_key = net_para_key.reshape(-1,) # type(net_para_key): 1-d tensor
            net_dict_key.append(net_para_key)
        para_len = len(net_dict_key[0])

        for k in range(para_len):
            current_para = []
            for i in range(n):
                current_para.append(net_dict_key[i][k])
                
            # choose n-2f - check OK
            current_para_list = [current_para[_].item() for _ in range(len(current_para))]
            select_net_list_idx = trim(current_para_list, f)
            # res_para_len = len(select_net_list_idx)
            for i in range(n):
                if i not in select_net_list_idx:
                    net_dict_key[i][k] = 0

    return m, net_dict

def median(args, nets, selected):
    n = len(selected)
    f = args.n_Byzantine 
    m = 1
    # if f == 0:
    #     return m, nets

    net_dict = copy.deepcopy(nets)
    tmp = net_dict[0]
    tmp_para = tmp.state_dict()
    for key in tmp_para:
        # shape_rec = tmp_para[key].shape
        net_dict_key = []

        # reshape the para
        for i in range(n):
            net = net_dict[i]
            net_para = net.state_dict()
            net_para_key = net_para[key]
            net_para_key = net_para_key.reshape(-1,) # type(net_para_key): 1-d tensor
            net_dict_key.append(net_para_key)
        para_len = len(net_dict_key[0])

        for k in range(para_len):
            current_para = []
            for i in range(n):
                current_para.append(net_dict_key[i][k])
                
            # choose median
            current_para_list = [current_para[_].item() for _ in range(len(current_para))]
            if n%2 == 1: # odd
                select_net_list_idx = trim(current_para_list, int(n/2))
            else: #odd
                select_net_list_idx = trim(current_para_list, n/2-1)
            # res_para_len = len(select_net_list_idx)
            for i in range(n):
                if i not in select_net_list_idx:
                    net_dict_key[i][k] = 0

    return m, net_dict
    
def zeno(args, nets, selected, test_dl_global, global_model, zeno_weight=1e-2):
    n = len(selected)
    f = args.n_Byzantine 
    m = n - f # number of how many vectors to output
    # if f == 0:
        # return m, selected    

    device = args.device
    lr = args.lr # gamma
    if type(test_dl_global) == type([1]):
        pass
    else:
        test_dl_global = [test_dl_global]
    
    # calculate calculate the 1st part of Stochasic Descendant Score
    if args.optimizer == 'adam':
        optimizer = optim.Adam(filter(lambda p: p.requires_grad, global_model.parameters()), lr=lr, weight_decay=args.reg)
    elif args.optimizer == 'amsgrad':
        optimizer = optim.Adam(filter(lambda p: p.requires_grad, global_model.parameters()), lr=lr, weight_decay=args.reg,
                            amsgrad=True)
    elif args.optimizer == 'sgd': # sgd：args default
        optimizer = optim.SGD(filter(lambda p: p.requires_grad, global_model.parameters()), lr=lr, momentum=args.rho, weight_decay=args.reg)
    criterion = nn.CrossEntropyLoss().to(device)

    cnt = 0
    epoch_loss_collector = []
    for tmp in test_dl_global:
        for batch_idx, (x, target) in enumerate(tmp):
            x, target = x.to(device), target.to(device)

            optimizer.zero_grad()
            x.requires_grad = True
            target.requires_grad = False
            target = target.long()

            out = global_model(x)
            loss = criterion(out, target)
            cnt += 1
            epoch_loss_collector.append(loss.item())

    loss_1 = sum(epoch_loss_collector) / len(epoch_loss_collector)
    
    # for every net, calculate the 2nd part of Stochasic Descendant Score
    loss_2 = []
    for net_id, net in nets.items():
        if net_id not in selected:
            continue
        if args.optimizer == 'adam':
            optimizer = optim.Adam(filter(lambda p: p.requires_grad, net.parameters()), lr=lr, weight_decay=args.reg)
        elif args.optimizer == 'amsgrad':
            optimizer = optim.Adam(filter(lambda p: p.requires_grad, net.parameters()), lr=lr, weight_decay=args.reg,
                                amsgrad=True)
        elif args.optimizer == 'sgd': # sgd：args default
            optimizer = optim.SGD(filter(lambda p: p.requires_grad, net.parameters()), lr=lr, momentum=args.rho, weight_decay=args.reg)
        criterion = nn.CrossEntropyLoss().to(device)

        cnt = 0
        epoch_loss_collector = []
        for tmp in test_dl_global:
            for batch_idx, (x, target) in enumerate(tmp):
                x, target = x.to(device), target.to(device)

                optimizer.zero_grad()
                x.requires_grad = True
                target.requires_grad = False
                target = target.long()

                out = net(x)
                loss = criterion(out, target)
                cnt += 1
                epoch_loss_collector.append(loss.item())

        loss_2.append(sum(epoch_loss_collector) / len(epoch_loss_collector))

    # calculate the magnitude of the update
    loss_3 = []

    global_model_para = global_model.state_dict()
    for i in range(n):
        net = nets[i]
        net_para = net.state_dict()
        tmp_val = 0
        for key in net_para:
            tmp_val += torch.sum((global_model_para[key] - net_para[key])**2)
        loss_3.append(tmp_val)

    score = []
    for i in range(n):
        score.append(loss_1 - loss_2[i] - zeno_weight*loss_3[i])

    # choose m highest scores 
    idx = np.argpartition(score, f)
    select_net_list_idx = idx[f:]
    return select_net_list_idx

def center_clipping(args, nets, selected, global_model, tau=100, L=1): # default tau=100, clipping iterations L=1
    n = len(selected)
    f = args.n_Byzantine 
    m = 1

    net_dict = copy.deepcopy(nets)
    tmp = net_dict[0]
    tmp_para = tmp.state_dict()
    global_para = global_model.state_dict()

    for key in tmp_para:
        shape_rec = tmp_para[key].shape
        net_dict_key = []

        # reshape the para
        for i in range(n):
            net = net_dict[i]
            net_para = net.state_dict()
            net_para_key = net_para[key]
            net_para_key = net_para_key.reshape(-1,) # type(net_para_key): 1-d tensor
            net_dict_key.append(net_para_key)
        para_len = len(net_dict_key[0])

        # center clipping
        # reshape the global para
        # v = global_para
        v_key = global_para[key]
        v_key = v_key.reshape(-1,) # type(net_para_key): 1-d tensor

        for l in range(L): # for each clipping iteration
            c_list = []
            for i in range(n): # for every client
                c = (net_dict_key[i] - v_key) * min(1, tau/(torch.norm(net_dict_key[i] - v_key)))
                c_list.append(c)
            
            c_sum = 0
            for i in range(n):
                c_sum += c_list[i]
            v_key = v_key + 1/n * c_sum
        
        v_key = v_key.reshape(shape_rec)
        global_para[key] = v_key

    return global_para

        


        