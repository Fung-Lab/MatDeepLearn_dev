import sys
import torch
import time
import json
import os
import copy
import numpy as np
import pandas as pd
import torch.optim as optim
from torch.utils.data import DataLoader, Subset, SubsetRandomSampler
import torch.nn as nn
from torch.nn import ( Linear, Bilinear, Sigmoid, Softplus, ELU, ReLU, SELU,
                       CELU, BatchNorm1d, ModuleList, Sequential,Tanh, Softmax )
from torch.nn.modules.module import Module
import torch.nn.functional as F
from torch.nn.utils import clip_grad_value_
import re
from scipy.stats import spearmanr
from sklearn.preprocessing import StandardScaler

class Metric(object):
    def __init__(self, metric_fn, name):
        self.metric_fn = metric_fn
        self.name = name
        self.total_metric = 0.0
        self.total_count = 0

    def __call__(self, predictoins, targets):
        return self.metric_fn(predictoins, targets)

    def add_batch_metric(self, predictoins, targets):
        metric_tensor = self.metric_fn(predictoins, targets)
        self.total_metric += metric_tensor.item() * targets.size(0)
        self.total_count += targets.size(0)
        return metric_tensor

    def get_total_metric(self):
        score = self.total_metric / self.total_count
        self.total_metric = 0.0
        self.total_count = 0
        return score

class Checkpoint(object):
    def __init__(self, model):
        self.model = model
        self.best_metric = None
        self.best_weights = model.weights

    def check(self, metric):
        if self.best_metric is None or metric < self.best_metric:
            self.best_metric = metric
            self.best_weights = self.model.weights

    def restore(self):
        self.model.weights = self.best_weights

class Model(object):
    def __init__(self, device, model, name, optimizer, scheduler, l=0, clip_value=None, scaler=None, export_attention=0,
                 metrics=[('loss', nn.MSELoss()), ('mae', nn.L1Loss()) ]):
        self.name=name
        self.model=model.to(device)
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.metrics = [Metric(metric, name) for name, metric in metrics]
        self.clip_value = clip_value
        self.device=device
        self.scaler = scaler
        self.l = l
        self.export_attention = export_attention

    def _set_mode(self, phase):
        if phase == 'train':
            self.model.train()  # Set model to training mode
        else:
            self.model.eval()   # Set model to evaluate mode

    def _process_batch(self, input, targets, phase):
     
        if len(input) == 4:
            atom_fea, nbr_fea, nbr_fea_idx, crystal_atom_idx = input
        else:
            atom_fea, atom_levels, nbr_fea, nbr_fea_idx, crystal_atom_idx = input
            #print('atom_fea size', atom_fea.size(), 'atom_levels size', atom_levels.size())
        atom_fea = atom_fea.to(self.device)
        if len(input) == 5:
            atom_levels = atom_levels.to(self.device)

        nbr_fea = nbr_fea.to(self.device)
        nbr_fea_idx = nbr_fea_idx.to(self.device)
        orig_targets = targets
        attention=[]
        if self.scaler is not None:
            targets = torch.Tensor(self.scaler.transform(targets))
        targets = targets.to(self.device)        
        self.optimizer.zero_grad()
        with torch.set_grad_enabled(phase == 'train'):
            if len(input) == 4:
                outputs, attention = self.model(atom_fea, nbr_fea, nbr_fea_idx, crystal_atom_idx) 
            else:
                outputs, attention = self.model(atom_fea, atom_levels, nbr_fea, nbr_fea_idx, crystal_atom_idx) 
            #print('outputs size', outputs.size())
            #print('targets size', targets.size())
            if self.scaler is not None:
                metric_tensors = [metric.add_batch_metric(torch.Tensor(self.scaler.inverse_transform(outputs.detach().cpu())), orig_targets) for metric in self.metrics]
            else:
                metric_tensors = [metric.add_batch_metric(outputs, targets) for metric in self.metrics]
            if phase == 'train':
                loss = metric_tensors[self.l]
                loss.backward()
                if self.clip_value is not None:
                    clip_grad_value_(self.model.parameters(), self.clip_value)
                self.optimizer.step()

        return metric_tensors, outputs, attention

    def train(self, train_dl, val_dl, test_dl, num_epochs):
        since = time.time()

        dataloaders = {'train': train_dl, 'val': val_dl, 'test':test_dl}
        checkpoint = Checkpoint(self)
        for epoch in range(num_epochs):
            epoch_since = time.time()
            print('Epoch {}/{}'.format(epoch, num_epochs - 1), flush=True)

            #self.scheduler.step()
            print('current lr:', self.optimizer.param_groups[0]['lr'])

            train_val_metrics = []
            for phase in ['train', 'val', 'test']:
                self._set_mode(phase)
                for input, targets in dataloaders[phase]:
                    _, outputs, _ = self._process_batch(input, targets, phase)
                epoch_metrics = [(metric.name, metric.get_total_metric()) for metric in self.metrics]
                text = ' '.join(['{}: {:.4f}'.format(name, metric) for name, metric in epoch_metrics])
                print('{} {}'.format(phase, text))

                if phase == 'val':
                    metric = epoch_metrics[1][1]
                    checkpoint.check(metric)

                train_val_metrics += [('_'.join([phase, name]), metric) for name, metric in epoch_metrics]
            self.scheduler.step()

            time_elapsed = time.time() - epoch_since
            print('Elapsed time (sec.): {:.3f}'.format(time_elapsed))
            print()

        if num_epochs > 0:
            time_elapsed = time.time() - since
            print('Total elapsed time (sec.): {:.3f}'.format(time_elapsed))
            print('The best val metric: {:4f}'.format(checkpoint.best_metric))
            print()
            print('using the model at the end of traing')

            outputs, targets, attentions = self.evaluate(test_dl)
            coef, p = spearmanr(outputs, targets)
            print('coef', coef)
            # load the best model weights
            checkpoint.restore()


    def evaluate(self, dataloader):
        self.model.eval()   # Set model to evaluate mode

        # Iterate over data.
        all_outputs = []
        all_targets = []
        all_graph_vec = []
        attentions = []
        for input, targets in dataloader:
            if len(input) == 4:
                atom_fea, nbr_fea, nbr_fea_idx, crystal_atom_idx = input
            else:
                atom_fea, atom_levels, nbr_fea, nbr_fea_idx, crystal_atom_idx = input

            atom_fea = atom_fea.to(self.device)
            if len(input) == 5:
                atom_levels = atom_levels.to(self.device)

            nbr_fea = nbr_fea.to(self.device)
            nbr_fea_idx = nbr_fea_idx.to(self.device)
            targets = targets.to(self.device)
            
            with torch.set_grad_enabled(False):
                outputs, attention = self.model(atom_fea, nbr_fea, nbr_fea_idx, crystal_atom_idx)
            attentions += attention
            outputs = outputs.to(torch.device("cpu")).numpy()
            targets = targets.to(torch.device("cpu")).numpy()
            all_outputs.append(outputs)
            all_targets.append(targets)

        all_outputs = np.concatenate(all_outputs)
        all_targets = np.concatenate(all_targets)

        all_outputs = torch.FloatTensor(all_outputs).to(self.device)
        all_targets = torch.FloatTensor(all_targets).to(self.device)

        total_metrics = [(metric.name, metric(all_outputs, all_targets).item()) for metric in self.metrics]
        text = ' '.join(['{}: {:.4f}'.format(name, metric) for name, metric in total_metrics])
        print('test {}'.format(text))

        all_outputs = all_outputs.to(torch.device("cpu")).numpy()
        all_targets = all_targets.to(torch.device("cpu")).numpy()

        return all_outputs, all_targets, attentions

    def save(self, model_path="model"):
        model_path="model/model_{}.pth".format(self.name)
        torch.save(self.model.state_dict(), model_path)

    def load(self, model_path):
        self.model.load_state_dict(torch.load(model_path))

    @property
    def weights(self):
        return copy.deepcopy(self.model.state_dict())

    @weights.setter
    def weights(self, state):
        self.model.load_state_dict(state)


def _bn_act(num_features, activation, use_batch_norm=False):
    # batch normal + activation
    if use_batch_norm:
        if activation is None:
            return BatchNorm1d(num_features)
        else:
            return Sequential(BatchNorm1d(num_features), activation)
    else:
        return activation

class NodeEmbedding(Module):
    """
    Node Embedding layer
    """
    def __init__(self, in_features, out_features, activation=Sigmoid(),
                 use_batch_norm=False, bias=False):
        super(NodeEmbedding, self).__init__()
        self.linear = Linear(in_features, out_features, bias=bias)
        self.activation = _bn_act(out_features, activation, use_batch_norm)

    def forward(self, input):
        output=self.linear(input)
        output = self.activation(output)
        return output


class OLP(Module):
    def __init__(self, in_features, out_features, activation=ELU(),
                use_batch_norm=False, bias=False):
        # One layer Perceptron
        super(OLP, self).__init__()
        self.linear = Linear(in_features, out_features, bias=bias)
        self.activation = _bn_act(out_features, activation, use_batch_norm)

    def forward(self,  input):
        z = self.linear(input)
        if self.activation:
            z = self.activation(z)
        return z

class Gated_pooling(Module):

    def __init__(self, in_features, out_features, activation=ELU(),
                 use_batch_norm=False, bias=False):
        super(Gated_pooling, self).__init__()
        self.linear1 = Linear(in_features, out_features, bias=bias)
        self.activation1 = _bn_act(out_features, activation, use_batch_norm)
        self.linear2 = Linear(in_features, out_features, bias=bias)
        self.activation2 = _bn_act(out_features, activation, use_batch_norm)

    def forward(self,  input,graph_indices,node_counts):

        z = self.activation1(self.linear1(input))*self.linear2(input)
        graphcount=len(node_counts)
        device=z.device
        blank=torch.zeros(graphcount,z.shape[1]).to(device)
        blank.index_add_(0, graph_indices, z)/node_counts.unsqueeze(1)
        #output = self.activation2(self.linear2(blank)) ################对每个图加起来
        return blank
 
