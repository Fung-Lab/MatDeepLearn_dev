################################################
'''
This file is to construct the geo_CGNN model and finish the training process
necessary input:

--cutoff, default=8A: It defines the radius of the neighborhood  
--max_nei, default=12 : The number of max neighbors of each node
--lr, default=8e-3 : Learning rate
--test_ratio, default=0.2 : The ratio of test set and validate set
--num_epochs, default=5 : The number of epochs 
--dataset_path, default='database' : The root of dataset
--datafile_name, default="my_graph_data_OQMD_8_12_100" : The first X letters of the data file name
--database, default="OQMD" : The file name of the target
--target_name, default='formation_energy_per_atom' : target name

output:
trained model will output to "./model"
training history/ test predictions / graph vector of test data will output to "./data"
'''
################################################
import time
import json
import os
import copy
import numpy as np
import pandas as pd
import random
import torch
import torch.optim as optim
from torch.utils.data import DataLoader, Subset, SubsetRandomSampler
from data_utils import AtomGraphDataset, Atomgraph_collate
from scipy.stats import spearmanr


def use_setpLR(param):
    ms = param["milestones"]
    return ms[0] < 0

def create_model(device, model_param, optimizer_param, scheduler_param, load_model, scaler=None, freeze_gnn=0):

    if options['my'] >=44 :
        model=CrystalGraphConvNet(model_param['orig_atom_fea_len'], model_param['nbr_fea_len'], model_param['atom_fea_len'], model_param['n_conv'], model_param['h_fea_len'], model_param['n_h'], model_param['k'])
    else:
        model=CrystalGraphConvNet(model_param['orig_atom_fea_len'], model_param['nbr_fea_len'], model_param['atom_fea_len'], model_param['n_conv'], model_param['h_fea_len'], model_param['n_h'] )

    clip_value = optimizer_param.pop("clip_value")
    optim_name = optimizer_param.pop("optim")
    if optim_name == "sgd":
        optimizer = optim.SGD(filter(lambda p: p.requires_grad, model.parameters()), momentum=0.9,
                              nesterov=True, **optimizer_param)
    elif optim_name == "adam":
        optimizer = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), **optimizer_param)
    elif optim_name == "amsgrad":
        optimizer = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), amsgrad=True,
                               **optimizer_param)
    elif optim_name == "adagrad":
        optimizer = optim.Adagrad(filter(lambda p: p.requires_grad, model.parameters()), lr_decay=0.1, **optimizer_param)
    else:
        raise NameError("optimizer {} is not supported".format(optim_name))
    use_cosine_annealing = scheduler_param.pop("cosine_annealing")
    if use_cosine_annealing:
        params = dict(T_max=scheduler_param["milestones"][0],
                      eta_min=scheduler_param["gamma"])
        scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, **params)
    
    elif use_setpLR(scheduler_param):
        scheduler_param["step_size"] = abs(scheduler_param["milestones"][0])
        scheduler_param.pop("milestones")
        scheduler = optim.lr_scheduler.StepLR(optimizer, **scheduler_param)
    else:
        scheduler = optim.lr_scheduler.MultiStepLR(optimizer, **scheduler_param)
    cutoff=model_param.pop('cutoff')
    max_nei=model_param.pop('max_nei')
    #name='my-'+str(options['my'])+'-'+str(options['atom_fea_len'])+'-'+str(options['nbr_fea_len'])+str(cutoff)+'_'+str(max_nei)
    name='my-'+str(options['my'])+'-atom_fea_len'+str(options['atom_fea_len'])+'-nbr_fe_len'+str(options['nbr_fea_len'])+'cutoff'+str(cutoff)+'-maxnei'+str(max_nei)+'nconvs'+str(model_param['n_conv'])+'-target-'+options['target_name']+'-dataset-'+options['dataset_path'][2:]
    if freeze_gnn == 1:
        for p in model.convs.parameters():
            p.requires_grad = False
    return Model(device, model, name, optimizer, scheduler, clip_value=clip_value, l=model_param['l'], scaler=scaler)

def main(device, model_param, optimizer_param, scheduler_param, dataset_param, dataloader_param,
         num_epochs, seed, load_model,pred,pre_trained_model_path):
    N_block=model_param.pop('N_block')
    cutoff=model_param['cutoff']
    max_nei=model_param['max_nei']
    with_levels = model_param['with_levels']
    seed1=options['seed1']
    print("Seed:", seed, "seed1", seed1)
    print()
    torch.manual_seed(seed1)
    # Create dataset
    print('before AtomGraphDataset with_levels', with_levels)
    dataset = AtomGraphDataset(dataset_param["dataset_path"],dataset_param['datafile_name'],dataset_param["database"], dataset_param["target_name"],model_param['cutoff'],model_param['N_shbf'],model_param['N_srbf'],model_param['n_grid_K'],model_param['n_Gaussian'], with_levels) 
    scaler = dataset.scaler 
    scaler = None
    dataloader_param["collate_fn"] = Atomgraph_collate

    model_param['orig_atom_feat_len'] = dataset.graph_data[0].nodes.shape[1]

    test_ratio=dataset_param['test_ratio']
    n_graph=len(dataset.graph_data)
    random.seed(seed)
    graphs = dataset.graph_names
    indices=list(range(n_graph))
    random.shuffle(indices) 
    exclude_list=[]
   

    # special case, manually setting
    if test_ratio==-1:
        split = {"train": indices[0:180000], "val": indices[180000:200000], "test": indices[200000: 430000]}
    elif test_ratio==-2:
        split = {"train": indices[0:60000], "val": indices[60000:64619], "test": indices[64619: 69239]}
    elif test_ratio==-3:
        split = {"train": indices[0:7000], "val": indices[7000:7032], "test": indices[7032: -1]}
    elif test_ratio==-33:
        split = {"train": list(set(indices[0:7000])-set(exclude_list)), "val": list(set(indices[7000:7032])-set(exclude_list)), "test": list(set(indices[7032: -1])-set(exclude_list))}
    elif test_ratio==-4:
        split = {"train": indices[0:7000], "val": indices[7000:7032], "test": indices[7032: -1]}
    elif test_ratio==-5:
        split = {"train": indices[0:2], "val": indices[2:3], "test": indices[3: -1]}
    elif test_ratio==-6:
        indices=list(range(n_graph))
        a=indices[19379:-1]
        random.shuffle(a)
        for i in range(len(a)):
            indices[19379+i] = a[i]
        split = {"train": indices[:26379], "val": indices[26379:26479], "test": indices[26479: -1]}
    elif test_ratio==-7:
        indices=list(range(n_graph))
        a=indices[9379:-1]
        random.shuffle(a)
        for i in range(len(a)):
            indices[9379+i] = a[i]
        split = {"train": indices[0:16379], "val": indices[16379:16479], "test": indices[16479: -1]}
    elif test_ratio==-8:
        print('special test ratio')
        split = {"train": indices[0:27824], "val": indices[27825:27825+9274], "test": indices[27825+9275: 27825+9275+9274]}
        #print(split)
    elif test_ratio==-9:
        print('special test ratio')
        trn=13912
        vn = 13912//8
        tn = vn
        split = {"train": indices[0:trn], "val": indices[trn:trn+vn], "test": indices[trn+vn: trn+vn+tn]}
        #print(split)
    elif test_ratio==-10:
        print('special test ratio')
        trn=6956
        vn = 6956//8
        tn = vn
        split = {"train": indices[0:trn], "val": indices[trn:trn+vn], "test": indices[trn+vn: trn+vn+tn]}
        #print(split)
    elif test_ratio==-11:
        print('special test ratio')
        trn=3478
        vn = 3478//8
        tn = vn
        split = {"train": indices[0:trn], "val": indices[trn:trn+vn], "test": indices[trn+vn: trn+vn+tn]}
        #print(split)
    elif test_ratio == -12:
        print('special test ratio')
        trn=106736
        vn = 13342
        tn = 13342
        split = {"train": indices[0:trn], "val": indices[trn:trn+vn], "test": indices[trn+vn: trn+vn+tn]}
    elif test_ratio == -13:
        print('special test ratio')
        trn= 36720
        vn = 4590
        tn = 4590
        split = {"train": indices[0:trn], "val": indices[trn:trn+vn], "test": indices[trn+vn: trn+vn+tn]}
    elif test_ratio == -14:
        print('special test ratio')
        trn= 16266
        vn = 5422
        tn = 5422
        split = {"train": indices[0:trn], "val": indices[trn:trn+vn], "test": indices[trn+vn: trn+vn+tn]}
    elif test_ratio == -15:
        '''
        print("using special case: mp-2018-06-1 ")
        include_list = []
        with open('mp2018-6-1.csv') as f:
            include_mofs = f.read().splitlines()
            print('len of included', len(include_mofs))
        for i in range(n_graph):
            if graphs[indices[i]] in include_mofs:
                include_list.append(i)
        for mof in include_mofs:
            if mof not in graphs:
                print('not in', mof)
        '''
        #split 60,000–5000–4239
        split = {"train": indices[:60000], "val": indices[60000:65000], "test": indices[65000:]}
    elif test_ratio< -20:
        print('special test ratio')
        trn= int(-test_ratio)
        vn = trn//8
        tn = vn
        split = {"train": indices[0:trn], "val": indices[trn:trn+vn], "test": indices[trn+vn: trn+vn+tn]}
        #print(split)
    else:
    # normal case
        n_val=int(n_graph*test_ratio)
        n_train=n_graph-2*n_val
        split = {"train": list(set(indices[0:n_train])-set(exclude_list)), "val": indices[n_train:n_train+n_val], "test": indices[n_train+n_val: n_graph]}
    print(" ".join(["{}: {}".format(k, len(x)) for k, x in split.items()]))

    # Create a DFTGN model
    model = create_model(device, model_param, optimizer_param, scheduler_param, load_model, scaler, freeze_gnn = options['freeze_gnn'])
    if load_model:
        pre_trained_model_path=options['pre_trained_model_path']
        if pre_trained_model_path != 'none':
            print("Loading weights from ", pre_trained_model_path)
            model.load(model_path=pre_trained_model_path)
            print("Model loaded at: {}".format(pre_trained_model_path)) 

    test_set = Subset(dataset, split["test"])
    test_dl = DataLoader(test_set,pin_memory=True, **dataloader_param)
    if not pred:
    # Train
        train_sampler = SubsetRandomSampler(split["train"])
        val_sampler = SubsetRandomSampler(split["val"])
        train_dl = DataLoader(dataset, sampler=train_sampler,pin_memory=True, **dataloader_param)
        val_dl = DataLoader(dataset, sampler=val_sampler,pin_memory=True, **dataloader_param)
        #trainD=[n for n in train_dl]
        print('start training', flush=True)
        model.train(train_dl, val_dl, test_dl, num_epochs)


        if num_epochs > 0:
            model.save()

    # Test
    outputs, targets,_  = model.evaluate(test_dl)

    print(outputs)
    print(targets)
    coef, p = spearmanr(outputs, targets)
    print('coef', coef)
    print('', flush=True)
    names = [dataset.graph_names[i] for i in split["test"]]
    df_predictions = pd.DataFrame({"name": names, "prediction": outputs.squeeze(), "target": targets.squeeze()})
    name=str(cutoff)+'_'+str(max_nei)+options['dataset_path'][2:]+"-"+str(options['seed'])+"-"+str(options['seed1'])+'-'+str(options['my'])+'-'+str(options['n_conv']) + '-n'+str(options['n']) + '-m'+str(options['m']) + options['optim'] + str(options['weight_decay']) + str(options['seed']) +'-h'+str(options['n_h'])
    df_predictions.to_csv("data/test_predictions_{}.csv".format(name), index=False)

    static_train_set = Subset(dataset, split["train"])
    names = [dataset.graph_names[i] for i in split["train"]]
    static_train_dl = DataLoader(static_train_set,pin_memory=True, **dataloader_param)
    outputs, targets, attentions = model.evaluate(static_train_dl)
    n = min(options['n'],len(static_train_set))
    m = options['m']
    attention_values=[]
    attention_indices=[]
    for attention in attentions:
        (values, indices) = torch.topk(attention.squeeze(), m)
        attention_values.append(values.detach().cpu().unsqueeze(0))
        attention_indices.append(indices[:m].detach().cpu().unsqueeze(0))
    attention_values=torch.cat(attention_values, dim=0)
    attention_indices=torch.cat(attention_indices, dim=0)
    ind = np.argpartition(np.squeeze(targets), -n)[-n:]
    orig_outputs = outputs
    orig_targets = targets
    orig_names = names
    outputs = outputs[ind]
    targets = targets[ind]
    ind=torch.LongTensor(ind)
    #print('values size', attention_values.size(), type(attention_values))
    #print('indices size', attention_indices.size(), type(attention_indices))
    attention_values = attention_values[ind]
    attention_indices= attention_indices[ind]
    print('attention_indices size', attention_indices.size())
    print('attention_values size', attention_values.size())
    orig_names = names
    names = [ names[i] for i in ind]
    sizes = [ dataset.graph_data[i].nodes.shape[0] for i in split["train"]]
    sizes = [sizes[i] for i in ind]
    #if(len(attentions)>0):
    #    print('attentions', attentions[0])
    mydict = {"name": names, "prediction": outputs.squeeze(), "target": targets.squeeze(), 'sizes':sizes}
    for i in range(m):
        mydict['i'+str(i)] = attention_indices[:,i]
        mydict['v'+str(i)] = attention_values[:,i]
    #print(mydict)
    df_predictions = pd.DataFrame(mydict)
    df_predictions.to_csv("data/train_attention_{}.csv".format(name), index=False)
    outputs = orig_outputs
    targets = orig_targets
    names = orig_names
    n = len(static_train_set)
    sizes = [ dataset.graph_data[i].nodes.shape[0] for i in split["train"]]
    #new_attentions = [attention.squeeze().detach().cpu().tolist() for attention in attentions]
    new_attentions=[]
    for attention in attentions:
        a = attention.squeeze().detach().cpu().tolist()
        a = [ round(x,3) for x in a]
        new_attentions.append(a)

    mydict = {"name": names, "prediction": outputs.squeeze(), "target": targets.squeeze(), 'sizes':sizes, 'attention_list':new_attentions}
    df_predictions = pd.DataFrame(mydict)
    df_predictions.to_csv("data/train_attention_{}_all.csv".format(name), float_format = '%.3f', index=False)
    print("\nEND")

if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description="Crystal Graph Neural Networks")

    parser.add_argument("--atom_fea_len", type=int, default=64,
        help='the dimension of node features')
    parser.add_argument("--nbr_fea_len", type=int, default=64) 
    parser.add_argument("--orig_atom_fea_len", type=int, default=64) 
    parser.add_argument("--h_fea_len", type=int, default=128) 
    parser.add_argument("--n_h", type=int, default=1) 
    parser.add_argument("--k", type=int, default=3) 
    parser.add_argument("--n_conv", type=int, default=4)
    parser.add_argument("--max_nei", type=int, default=12) 
    parser.add_argument("--cutoff", type=int, default=8) 
    parser.add_argument("--N_block", type=int, default=6)
    parser.add_argument("--N_shbf", type=int, default=6)
    parser.add_argument("--N_srbf", type=int, default=6)
    parser.add_argument("--n_MLP_LR", type=int, default=3)
    parser.add_argument("--n_grid_K", type=int, default=4)
    parser.add_argument("--n_Gaussian", type=int, default=64)
    parser.add_argument("--node_activation", type=str, default="Sigmoid")
    parser.add_argument("--MLP_activation", type=str, default="Elu")
    parser.add_argument("--batch_size", type=int, default=512)
    parser.add_argument("--optim", type=str, default="adam")
    parser.add_argument("--lr", type=float, default=8e-3)
    parser.add_argument("--test_ratio", type=float, default=0.3)
    parser.add_argument("--weight_decay", type=float, default=0)
    parser.add_argument("--clip_value", type=float, default=0)
    parser.add_argument("--milestones", nargs='+', type=int, default=[20])
    parser.add_argument("--gamma", type=float, default=0)
    parser.add_argument("--cosine_annealing", action='store_true')
    parser.add_argument("--num_epochs", type=int, default=5)
    parser.add_argument("--dataset_path", type=str, default='database')
    parser.add_argument("--datafile_name", type=str, default="my_graph_data_OQMD_8_12_100")
    parser.add_argument("--database", type=str, default="OQMD")
    parser.add_argument("--target_name", type=str, default='formation_energy_per_atom')
    parser.add_argument("--num_workers", type=int, default=0)
    parser.add_argument("--seed", type=int, default=12345)
    parser.add_argument("--seed1", type=int, default=34821)
    parser.add_argument("--my", type=int, default=0)
    parser.add_argument("--l", type=int, default=0)
    parser.add_argument("--with_levels", type=int, default=0)
    parser.add_argument("--n", type=int, default=5)
    parser.add_argument("--m", type=int, default=5)
    parser.add_argument("--freeze_gnn", type=int, default=0)
    parser.add_argument("--load_model", action='store_true')
    parser.add_argument("--pred", action='store_true')
    #parser.add_argument("--pre_trained_model_path", type=str, default='../GPG/model/model_my-4-atom_fea_len192-nbr_fe_len64cutoff8-maxnei12nconvs12.pth')
    parser.add_argument("--pre_trained_model_path", type=str, default='none')
    options = vars(parser.parse_args())
    print(options)

    from models import Model

    if options['my']==0:
        from model import  CrystalGraphConvNet
    if options['my']==2:
        from model_my2 import  CrystalGraphConvNet
    if options['my']==3:
        from model_my3 import  CrystalGraphConvNet
    if options['my']==5:
        from model_my5 import  CrystalGraphConvNet
    if options['my']==6:
        from model_my6 import  CrystalGraphConvNet
    if options['my']==38:
        from model_my38 import  CrystalGraphConvNet
    elif options['my']==4:
        from model_my4 import  CrystalGraphConvNet
    elif options['my']==45:
        from model_my45 import  CrystalGraphConvNet
    elif options['my']==458:
        from model_my458 import  CrystalGraphConvNet
    elif options['my']==4582:
        from model_my4582 import  CrystalGraphConvNet
    elif options['my']==4588:
        from model_my4588 import  CrystalGraphConvNet
    elif options['my']==4589:
        from model_my4589 import  CrystalGraphConvNet
    elif options['my']==45810:
        from model_my45810 import  CrystalGraphConvNet
    elif options['my']==459:
        from model_my459 import  CrystalGraphConvNet
    elif options['my']==460:
        from model_my460 import  CrystalGraphConvNet
    elif options['my']==468:
        from model_my468 import  CrystalGraphConvNet
    elif options['my']==478:
        from model_my478 import  CrystalGraphConvNet
    elif options['my']==7:
        from model_my7 import  CrystalGraphConvNet
    elif options['my']==8:
        from model_my8 import  CrystalGraphConvNet
    elif options['my']==9:
        from model_my9 import  CrystalGraphConvNet

    # set cuda
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print("Device:",device)

    # Model parameters
    model_param_names = ['orig_atom_fea_len', 'nbr_fea_len',
                 'atom_fea_len', 'n_conv', 'h_fea_len', 'n_h', 'k','N_block','N_shbf','N_srbf','cutoff','max_nei','n_MLP_LR','n_grid_K','n_Gaussian', 'l', 'with_levels']
    model_param = {k : options[k] for k in model_param_names if options[k] is not None}
    print("Model_param:", model_param)
    print()

    # Optimizer parameters
    optimizer_param_names = ["optim", "lr", "weight_decay", "clip_value"]
    optimizer_param = {k : options[k] for k in optimizer_param_names if options[k] is not None}
    if optimizer_param["clip_value"] == 0.0:
        optimizer_param["clip_value"] = None
    print("Optimizer:", optimizer_param)
    print()

    # Scheduler parameters
    scheduler_param_names = ["milestones", "gamma", "cosine_annealing"]
    #scheduler_param_names = ["milestones", "gamma"]
    scheduler_param = {k : options[k] for k in scheduler_param_names if options[k] is not None}
    print("Scheduler:", scheduler_param)
    print()

    # Dataset parameters
    dataset_param_names = ["dataset_path",'datafile_name','database', "target_name","test_ratio"]
    dataset_param = {k : options[k] for k in dataset_param_names if options[k] is not None}
    print("Dataset:", dataset_param)
    print()

    # Dataloader parameters
    dataloader_param_names = ["num_workers", "batch_size"]
    dataloader_param = {k : options[k] for k in dataloader_param_names if options[k] is not None}
    print("Dataloader:", dataloader_param)
    print()

    main(device, model_param, optimizer_param, scheduler_param, dataset_param, dataloader_param,
         options["num_epochs"], options["seed"], options["load_model"],options["pred"],options["pre_trained_model_path"])
