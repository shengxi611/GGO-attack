from numpy.lib.function_base import average
from defences import median
import numpy as np 
import torch
import copy
import scipy.stats as st 

def reshape_to_vec(net_dict, num, key):
    # reshape the para to vector
    net_dict_key = []
    for i in range(num):
        net_para = net_dict[i].state_dict()
        net_para_key = net_para[key]
        shape = net_para_key.shape
        net_para_key = net_para_key.reshape(-1,) # type(net_para_key): 1-d tensor
        net_dict_key.append(net_para_key)
    return net_dict_key, shape

def reconstruct_to_tensor(net_dict, net_dict_key, num, key, shape):
    # reconstruct the para to tensor
    for i in range(num):
        net_para_key = net_dict_key[i]
        net_para_key = net_para_key.reshape(shape)
        tmp = net_dict[i].state_dict()
        tmp[key] = net_para_key
        net_dict[i].load_state_dict(tmp)
    return net_dict

def vec_mean(vec, num): # num: how many to average
    # row, col = len(vec), len(vec[0])
    col = len(vec[0])
    mean = [0 for _ in range(col)]
    for j in range(col):
        for i in range(num): mean[j] += vec[i][j]
        mean[j] /= num
    return mean

def vec_sub(a, b):
    row, col = len(a), len(a[0])
    if row != len(b) or col != len(b[0]):
        print("The size of two lists must be the same.")
        return -1
    res = [[i for i in range(col)] for _ in range(row)]
    for i in range(row):
        for j in range(col):
            res[i][j] = a[i][j] - b[i][j]
    return res

def sign_trick(para, old_para, per_w):
    return para + per_w * np.sign(para - old_para)

def element_wise_trick(para_list, old_para_list, per_w, f, num, threshold=0.5): # default threshold == 0.5
    '''
        para_list:      parameter list of this iteration
        old_para_list:  parameter list of last iteration
        num:            # total received workers
        f:              # Byzantine workers
        threshold:      poison threshold
    '''
    para_gap = vec_sub(para_list, old_para_list)
    para_mean = vec_mean(para_list, num)
    for i in range(f): # for every Byzantine worker
        for k in range(len(para_list[0])): # for every dimension
            # ############### * find more important element ###############
            # ? approach 1: threshold
            if np.abs(para_gap[i][k]/old_para_list[i][k]) > threshold:

            # ? approach 2: sign
            # if np.sign(para_list[i][k]) != np.sign(old_para_list[i][k]):
            # ############### * find more important element ###############

                # ############### * poison ###############
                # ? approach 1: avg
                # para_list[i][k] = -para_mean[k] # idea of IPM    

                # ? approach 2: weighted avg
                para_list[i][k] = -per_w * para_mean[k]
                # ############### * poison ############### 
    return para_list

def our_with_collusion(nets, args, old_nets, bar_selected):
    n = args.n_parties
    f = args.n_Byzantine
    m = n - f
    per_w = args.perturb_weight
    threshold = args.threshold

    old_net_dict = copy.deepcopy(old_nets)
    net_dict = copy.deepcopy(nets)
    tmp = net_dict[0]
    tmp_para = tmp.state_dict()
    for key in tmp_para:
        # reshape the para to vector
        net_dict_key, shape = reshape_to_vec(net_dict, n, key) # whitebox
        old_net_dict_key, shape = reshape_to_vec(old_net_dict, n, key)
        # get the number of parameters
        para_len = len(net_dict_key[0])
        # search |max| of each row
        max_current_para = []
        for i in range(f):
            row_max = np.max(np.abs(net_dict_key[i].numpy()))
            max_current_para.append(row_max)
        # i: #worker, k: #dimension
        for k in range(para_len): 
            current_para = []
            std = 0 
            for i in range(f):
                current_para.append(net_dict_key[i][k])
            current_para_list = [current_para[_].item() for _ in range(len(current_para))]
            # search median, max, min, mean of each col(all poisoned workers)
            if f%2 == 1: # odd
                median_idx = trim(current_para_list, int(f/2))
                med_val = current_para_list[median_idx[0]]
            else: # even
                median_idx = trim(current_para_list, int(f/2-1))
                med_val = (current_para_list[median_idx[0]] + current_para_list[median_idx[1]]) / 2
            max_val = max(current_para_list)
            min_val = min(current_para_list)
            mean_val = np.average(current_para_list)

        # ############### * attack ###############
        # ? idea 1: sign
        # for i in range(f):
            # net_dict_key[i] = sign_trick(net_dict_key[i], old_net_dict_key[i], per_w)

        # ? idea 2: element-wise
        net_dict_key = element_wise_trick(net_dict_key, old_net_dict_key, per_w, f, f, threshold)
        # ############### * attack ###############

        net_dict = reconstruct_to_tensor(net_dict, net_dict_key, f, key, shape)

    return net_dict

def trim(arr, f):
    n = len(arr)
    sorted_res = sorted(enumerate(arr), key=lambda x: x[1])
    sorted_idx = [i[0] for i in sorted_res]

    trimmed_idx = sorted_idx[f:n-f]
    return trimmed_idx

def inner_product_manipulation(nets, args):
    n = args.n_parties
    f = args.n_Byzantine
    m = n - f
    per_w = args.perturb_weight

    net_dict = copy.deepcopy(nets)
    tmp = net_dict[0]
    tmp_para = tmp.state_dict()
    for key in tmp_para:
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
            avg = 0
            for i in range(f, n):
                avg += net_dict_key[i][k]
            avg /= m
            for i in range(f):
                net_dict_key[i][k] = -per_w * avg

    return net_dict

def inner_product_manipulation_black(nets, args):
    n = args.n_parties
    f = args.n_Byzantine
    m = n - f
    per_w = args.perturb_weight

    seed = args.init_seed
    np.random.seed(seed)


    net_dict = copy.deepcopy(nets)
    tmp = net_dict[0]
    tmp_para = tmp.state_dict()
    for key in tmp_para:
        net_dict_key = []

        # reshape the para
        for i in range(n):
            net = net_dict[i]
            net_para = net.state_dict()
            net_para_key = net_para[key]
            net_para_key = net_para_key.reshape(-1,) # type(net_para_key): 1-d tensor
            net_dict_key.append(net_para_key)
        para_len = len(net_dict_key[0])

        rand_ipm = np.random.randn(para_len) # random ipm
        rand_ipm = torch.from_numpy(rand_ipm)
        for i in range(f):
            net_dict_key[i] = rand_ipm
                
    return net_dict
