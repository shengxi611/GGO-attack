import numpy as np
import json
import torch
import torch.optim as optim
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
from torch.autograd import Variable
import torch.utils.data as data
import argparse
import logging
import os
import copy
from math import *

import datetime
#from torch.utils.tensorboard import SummaryWriter

import random
from model import *
from utils import *
from attacks import *
from defences import *

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, default='simple-cnn', help='neural network used in training')
    parser.add_argument('--dataset', type=str, default='generated', help='dataset used for training')
    parser.add_argument('--net_config', type=lambda x: list(map(int, x.split(', '))))
    parser.add_argument('--partition', type=str, default='noniid-labeldir', help='the data partitioning strategy')
    parser.add_argument('--batch-size', type=int, default=64, help='input batch size for training (default: 64)')
    parser.add_argument('--lr', type=float, default=0.01, help='learning rate (default: 0.01)')
    parser.add_argument('--epochs', type=int, default=5, help='number of local epochs')
    parser.add_argument('--n_parties', type=int, default=2, help='number of workers in a distributed cluster')
    parser.add_argument('--alg', type=str, default='robust-bar',
                            help='communication strategy: fedavg/fedprox')
    parser.add_argument('--comm_round', type=int, default=30, help='number of maximum communication roun')
    parser.add_argument('--is_same_initial', type=int, default=1, help='Whether initial all the models with the same parameters in fedavg')
    parser.add_argument('--init_seed', type=int, default=0, help="Random seed")
    parser.add_argument('--dropout_p', type=float, required=False, default=0.0, help="Dropout probability. Default=0.0")
    parser.add_argument('--datadir', type=str, required=False, default="./data/", help="Data directory")
    parser.add_argument('--reg', type=float, default=1e-5, help="L2 regularization strength")
    # parser.add_argument('--logdir', type=str, required=True, default="./logs/", help='Log directory path')
    parser.add_argument('--logdir', type=str, default="./logs/", help='Log directory path')
    parser.add_argument('--modeldir', type=str, required=False, default="./models/", help='Model directory path')
    parser.add_argument('--beta', type=float, default=0.5, help='The parameter for the dirichlet distribution for data partitioning')
    parser.add_argument('--device', type=str, default='cpu', help='The device to run the program')
    parser.add_argument('--log_file_name', type=str, default=None, help='The log file name')
    parser.add_argument('--optimizer', type=str, default='sgd', help='the optimizer')
    parser.add_argument('--mu', type=float, default=1, help='the mu parameter for fedprox')
    parser.add_argument('--noise', type=float, default=0, help='how much noise we add to some party')
    parser.add_argument('--noise_type', type=str, default='level', help='Different level of noise or different space of noise')
    parser.add_argument('--rho', type=float, default=0.9, help='Parameter controlling the momentum SGD')
    parser.add_argument('--sample', type=float, default=1, help='Sample ratio for each communication round')
    parser.add_argument('--atk_type', type=str, default='Gaussian', help='type of Byzantine attack')
    parser.add_argument('--def_type', type=str, default='zeno', help='type of Byzantine robust method')
    parser.add_argument('--n_Byzantine', type=int, default=0, help='number of Byzantine workers')
    parser.add_argument('--perturb_weight', type=float, default=1, help='perturbation weight of our method')
    parser.add_argument('--threshold', type=float, default=0.5, help='threshold')
    parser.add_argument('--tau', type=float, default=1, help='parameter of CC')
    parser.add_argument('--L', type=int, default=5, help='parameter of CC')
    parser.add_argument('--n_samples', type=int, default=40000, help='number of samples of generated dataset')
    args = parser.parse_args()
    return args

def init_nets(net_configs, dropout_p, n_parties, args):

    nets = {net_i: None for net_i in range(n_parties)}

    for net_i in range(n_parties):
        if args.dataset == "generated":
            net = PerceptronModel()
        elif args.model == "simple-cnn":
            if args.dataset in ("cifar10"):
                net = SimpleCNN(input_dim=(16 * 5 * 5), hidden_dims=[120, 84], output_dim=10)
        else:
            print("not supported yet")
            exit(1)
        nets[net_i] = net

    model_meta_data = []
    layer_type = []
    for (k, v) in nets[0].state_dict().items():
        model_meta_data.append(v.shape)
        layer_type.append(k)

    return nets, model_meta_data, layer_type 


def train_net(net_id, net, train_dataloader, test_dataloader, epochs, lr, args_optimizer, device="cpu"):
    logger.info('Training network %s' % str(net_id))

    train_acc = compute_accuracy(net, train_dataloader, device=device)
    test_acc, conf_matrix = compute_accuracy(net, test_dataloader, get_confusion_matrix=True, device=device)

    logger.info('>> Pre-Training Training accuracy: {}'.format(train_acc))
    logger.info('>> Pre-Training Test accuracy: {}'.format(test_acc))

    if args_optimizer == 'adam':
        optimizer = optim.Adam(filter(lambda p: p.requires_grad, net.parameters()), lr=lr, weight_decay=args.reg)
    elif args_optimizer == 'amsgrad':
        optimizer = optim.Adam(filter(lambda p: p.requires_grad, net.parameters()), lr=lr, weight_decay=args.reg,
                               amsgrad=True)
    elif args_optimizer == 'sgd': # sgd：args default
        optimizer = optim.SGD(filter(lambda p: p.requires_grad, net.parameters()), lr=lr, momentum=args.rho, weight_decay=args.reg)
    criterion = nn.CrossEntropyLoss().to(device)

    cnt = 0
    if type(train_dataloader) == type([1]):
        pass
    else:
        train_dataloader = [train_dataloader]

    for epoch in range(epochs):
        epoch_loss_collector = []
        for tmp in train_dataloader:
            for batch_idx, (x, target) in enumerate(tmp):
                x, target = x.to(device), target.to(device)

                optimizer.zero_grad()
                x.requires_grad = True
                target.requires_grad = False
                target = target.long()

                out = net(x)
                loss = criterion(out, target)

                loss.backward()
                optimizer.step()

                cnt += 1
                epoch_loss_collector.append(loss.item())

        epoch_loss = sum(epoch_loss_collector) / len(epoch_loss_collector)
        logger.info('Epoch: %d Loss: %f' % (epoch, epoch_loss))

    train_acc = compute_accuracy(net, train_dataloader, device=device)
    test_acc, conf_matrix = compute_accuracy(net, test_dataloader, get_confusion_matrix=True, device=device)

    logger.info('>> Training accuracy: %f' % train_acc)
    logger.info('>> Test accuracy: %f' % test_acc)


    logger.info(' ** Training complete **')
    return train_acc, test_acc

def log_loss(args, test_dl_global, global_model):
    # log loss
    device = args.device
    lr = args.lr # gamma
    if args.optimizer == 'adam':
        optimizer = optim.Adam(filter(lambda p: p.requires_grad, global_model.parameters()), lr=lr, weight_decay=args.reg)
    elif args.optimizer == 'amsgrad':
        optimizer = optim.Adam(filter(lambda p: p.requires_grad, global_model.parameters()), lr=lr, weight_decay=args.reg, amsgrad=True)
    elif args.optimizer == 'sgd': # sgd：args default
        optimizer = optim.SGD(filter(lambda p: p.requires_grad, global_model.parameters()), lr=lr, momentum=args.rho, weight_decay=args.reg)
    criterion = nn.CrossEntropyLoss().to(device)

    epoch_loss_collector = []
    # for tmp in test_dl_global:
        # for x, target in enumerate(tmp):
    for tmp in test_dl_global:
        x = tmp[0]
        target = tmp[1]
        x, target = x.to(device), target.to(device)

        optimizer.zero_grad()
        x.requires_grad = True
        target.requires_grad = False
        target = target.long()

        out = global_model(x)
        loss = criterion(out, target)
        epoch_loss_collector.append(loss.item())

    loss = sum(epoch_loss_collector) / len(epoch_loss_collector)
    logger.info('>> Global Model Loss: %f' % loss)

def view_image(train_dataloader):
    for (x, target) in train_dataloader:
        np.save("img.npy", x)
        print(x.shape)
        exit(0)


def local_train_net(nets, selected, args, net_dataidx_map, bar_selected, test_dl = None, device="cpu"):
    avg_acc = 0.0

    old_nets = copy.deepcopy(nets)

    for net_id, net in nets.items():
        if net_id not in selected:
            continue
        dataidxs = net_dataidx_map[net_id]

        logger.info("Training network %s. n_training: %d" % (str(net_id), len(dataidxs)))
        # move the model to cuda device:
        net.to(device)

        noise_level = args.noise
        if net_id == args.n_parties - 1:
            noise_level = 0

        if args.noise_type == 'space':
            train_dl_local, test_dl_local, _, _ = get_dataloader(args.dataset, args.datadir, args.batch_size, 32, dataidxs, noise_level, net_id, args.n_parties-1)
        else:
            noise_level = args.noise / (args.n_parties - 1) * net_id
            train_dl_local, test_dl_local, _, _ = get_dataloader(args.dataset, args.datadir, args.batch_size, 32, dataidxs, noise_level)
        train_dl_global, test_dl_global, _, _ = get_dataloader(args.dataset, args.datadir, args.batch_size, 32)
        n_epoch = args.epochs


        trainacc, testacc = train_net(net_id, net, train_dl_local, test_dl, n_epoch, args.lr, args.optimizer, device=device)
        logger.info("net %d final test acc %f" % (net_id, testacc))
        avg_acc += testacc
        # saving the trained models here
        # save_model(net, net_id, args)
        # else:
        #     load_model(net, net_id, device=device)
    avg_acc /= len(selected)
    if args.alg == 'local_training':
        logger.info("avg test acc %f" % avg_acc)

    n_Byzantine = args.n_Byzantine
    atk_type = args.atk_type
    # TODO: attack
    if atk_type != 'none':
        for net_id, net in nets.items():
            if net_id < n_Byzantine:
                net_para = net.state_dict()
                if atk_type == 'sign-flipping':
                    for key in net_para:
                        net_para[key] *= -10
                    net.load_state_dict(net_para)
                elif atk_type == 'Gaussian':
                    for key in net_para:
                        net_para_key_shape = net_para[key].shape
                        Gau_array = np.random.normal(loc=0,scale=1,size=net_para_key_shape) # loc: mean, scale: standard deviation
                        Gau_tensor = torch.from_numpy(Gau_array)
                        Gau_tensor = Gau_tensor.to(args.device, torch.double)
                        net_para[key] += Gau_tensor
                    net.load_state_dict(net_para)
                elif atk_type == 'our-with-collusion':
                    nets = our_with_collusion(nets, args, old_nets, bar_selected)
                    break
                elif atk_type == 'IPM':
                    nets = inner_product_manipulation(nets, args)
                    break
                elif atk_type == 'IPM-black':
                    nets = inner_product_manipulation_black(nets, args)
                    break
                else:
                    print("The atk_type is not supported yet.")
                    exit(1)
            else: continue

    # nets_list = list(nets.values())
    # return nets_list
    return nets


def get_partition_dict(dataset, partition, n_parties, init_seed=0, datadir='./data', logdir='./logs', beta=0.5):
    seed = init_seed
    np.random.seed(seed)
    torch.manual_seed(seed)
    X_train, y_train, X_test, y_test, net_dataidx_map, traindata_cls_counts = partition_data(
        dataset, datadir, logdir, partition, n_parties, args, beta=beta)

    return net_dataidx_map

if __name__ == '__main__':
    # torch.set_printoptions(profile="full")
    args = get_args()
    mkdirs(args.logdir)
    mkdirs(args.modeldir)
    if args.log_file_name is None:
        argument_path='experiment_arguments-%s.json' % datetime.datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
    else:
        argument_path=args.log_file_name+'.json'
    with open(os.path.join(args.logdir, argument_path), 'w') as f:
        json.dump(str(args), f)
    device = torch.device(args.device)
    # logging.basicConfig(filename='test.log', level=logger.info, filemode='w')
    # logging.info("test")
    for handler in logging.root.handlers[:]:
        logging.root.removeHandler(handler)

    if args.log_file_name is None:
        args.log_file_name = 'experiment_log-%s' % (datetime.datetime.now().strftime("%Y-%m-%d-%H-%M-%S"))
    log_path=args.log_file_name+'.log'
    logging.basicConfig(
        filename=os.path.join(args.logdir, log_path),
        # filename='/home/qinbin/test.log',
        format='%(asctime)s %(levelname)-8s %(message)s', 
        datefmt='%m-%d %H:%M', level=logging.DEBUG, filemode='w') 

    logger = logging.getLogger()
    logger.setLevel(logging.DEBUG)
    logger.info(device) # cpu

    seed = args.init_seed
    logger.info("#" * 100)
    np.random.seed(seed)
    torch.manual_seed(seed)
    random.seed(seed)
    logger.info("Partitioning data")
    X_train, y_train, X_test, y_test, net_dataidx_map, traindata_cls_counts = partition_data(
        args.dataset, args.datadir, args.logdir, args.partition, args.n_parties, args, beta=args.beta)

    n_classes = len(np.unique(y_train))

    train_dl_global, test_dl_global, train_ds_global, test_ds_global = get_dataloader(args.dataset, 
    # get_dataloader(): from utils.py
                                                                                        args.datadir,
                                                                                        args.batch_size,
                                                                                        32)

    print("len train_dl_global:", len(train_ds_global))


    data_size = len(test_ds_global)

    # test_dl = data.DataLoader(dataset=test_ds_global, batch_size=32, shuffle=False)
    train_all_in_list = []
    test_all_in_list = []
    if args.noise > 0:
        for party_id in range(args.n_parties):
            dataidxs = net_dataidx_map[party_id]

            noise_level = args.noise
            if party_id == args.n_parties - 1: # ?why
                noise_level = 0

            if args.noise_type == 'space':
                train_dl_local, test_dl_local, train_ds_local, test_ds_local = get_dataloader(args.dataset, args.datadir, args.batch_size, 32, dataidxs, noise_level, party_id, args.n_parties-1)
            else:
                noise_level = args.noise / (args.n_parties - 1) * party_id
                train_dl_local, test_dl_local, train_ds_local, test_ds_local = get_dataloader(args.dataset, args.datadir, args.batch_size, 32, dataidxs, noise_level)
            train_all_in_list.append(train_ds_local)
            test_all_in_list.append(test_ds_local)
        train_all_in_ds = data.ConcatDataset(train_all_in_list)
        train_dl_global = data.DataLoader(dataset=train_all_in_ds, batch_size=args.batch_size, shuffle=True)
        test_all_in_ds = data.ConcatDataset(test_all_in_list)
        test_dl_global = data.DataLoader(dataset=test_all_in_ds, batch_size=32, shuffle=False)

    
    if args.alg == 'robust-bar': # robust Byzantine aggregation rules: basic process is the same as fedavg
        logger.info("Initializing nets")
        nets, local_model_meta_data, layer_type = init_nets(args.net_config, args.dropout_p, args.n_parties, args)
        global_models, global_model_meta_data, global_layer_type = init_nets(args.net_config, 0, 1, args)
        global_model = global_models[0] 

        global_para = global_model.state_dict() 
        if args.is_same_initial:
            for net_id, net in nets.items():
                net.load_state_dict(global_para)

        for round in range(args.comm_round):
            logger.info("in comm round:" + str(round))
            # select a set of parties
            arr = np.arange(args.n_parties)
            np.random.shuffle(arr)
            selected = arr[:int(args.n_parties * args.sample)]
            # send the global model to the parties
            global_para = global_model.state_dict()
            if round == 0:
                if args.is_same_initial:
                    for idx in selected:
                        nets[idx].load_state_dict(global_para)
            else:
                for idx in selected:
                    nets[idx].load_state_dict(global_para)
            # local training
            if round == 0: bar_selected = selected # ? if attacker knows whether it is chosen, initialize chosen result
            nets = local_train_net(nets, selected, args, net_dataidx_map, bar_selected, test_dl = test_dl_global, device=device)
            # local_train_net(nets, args, net_dataidx_map, local_split=False, device=device)

            # update global model
            total_data_points = sum([len(net_dataidx_map[r]) for r in selected])
            # fed_avg_freqs = [len(net_dataidx_map[r]) / total_data_points for r in selected] 
            # fed_avg_freqs = [1/len(selected) for r in selected]
            # fed_avg_freqs = [1/len(bar_selected) for r in bar_selected]
            fed_avg_freqs = []

            # TODO: defence
            if args.def_type == 'krum': 
                bar_selected_len, bar_selected = krum(args, nets, selected)
                bar_nets = copy.deepcopy(nets)
                fed_avg_freqs = [1 for r in bar_selected]
            elif args.def_type == 'trimmed-mean':
                bar_selected_len, bar_nets = trimmed_mean(args, nets, selected)
                bar_selected = selected
                fed_avg_freqs = [1/(args.n_parties-2*args.n_Byzantine) for r in bar_selected]
            elif args.def_type == 'median':
                bar_selected_len, bar_nets = median(args, nets, selected)
                bar_selected = selected 
                fed_avg_freqs = [1 for r in bar_selected]
            elif args.def_type == 'zeno': 
                bar_selected = zeno(args, nets, selected, test_dl_global, global_model)
                bar_nets = copy.deepcopy(nets)
                fed_avg_freqs = [1/(args.n_parties-args.n_Byzantine) for r in bar_selected]
            elif args.def_type == 'center-clipping': 
                global_para = center_clipping(args, nets, selected, global_model,tau=args.tau,L=args.L)
                # bar_nets = copy.deepcopy(nets)
                # fed_avg_freqs = [1 for r in bar_selected]
            elif args.def_type == 'none': # 'none'
                bar_selected = selected
                bar_nets = copy.deepcopy(nets)
                fed_avg_freqs = [1/len(bar_selected) for r in bar_selected]
            else:
                print("The defence method is not supported yet.")
                exit(1)
            logger.info(args.def_type + ' selects workers' + str(bar_selected) + ' under ' + args.atk_type + ' attack with ' + str(args.n_Byzantine) + ' Byzantine workers.' )

            if args.def_type != 'center-clipping':
                # for idx in range(len(selected)):
                for idx in range(len(bar_selected)):
                    # net_para = nets[selected[idx]].cpu().state_dict()
                    net_para = bar_nets[bar_selected[idx]].cpu().state_dict()
                    if idx == 0:
                        for key in net_para:
                            global_para[key] = net_para[key] * fed_avg_freqs[idx]
                    else:
                        for key in net_para:
                            global_para[key] += net_para[key] * fed_avg_freqs[idx]

            global_model.load_state_dict(global_para)

            logger.info('global n_training: %d' % len(train_dl_global))
            logger.info('global n_test: %d' % len(test_dl_global))


            train_acc = compute_accuracy(global_model, train_dl_global)
            test_acc, conf_matrix = compute_accuracy(global_model, test_dl_global, get_confusion_matrix=True)


            logger.info('>> Global Model Train accuracy: %f' % train_acc)
            logger.info('>> Global Model Test accuracy: %f' % test_acc)

            log_loss(args, test_dl_global, global_model)
