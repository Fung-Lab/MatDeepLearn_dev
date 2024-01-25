import numpy as np
import torch
import torch.nn.functional as F
import torch_geometric
from torch import Tensor
import torch.nn as nn
from torch_geometric.nn import (
    CGConv,
    Set2Set,
    global_add_pool,
    global_max_pool,
    global_mean_pool,
)
from torch_scatter import scatter, scatter_add, scatter_max, scatter_mean

from matdeeplearn.common.registry import registry
from matdeeplearn.models.base_model import BaseModel, conditional_grad
from matdeeplearn.preprocessor.helpers import GaussianSmearing, node_rep_one_hot

@registry.register_model("attention_loss")
class AttentionLoss(nn.Module):
    def __init__(self, 
                 teacher_node_dim, 
                 teacher_edge_dim, 
                 teacher_vec_dim,
                 attention_dim,
        ):
        super(AttentionLoss, self).__init__()
        # Student Model Layers
        self.student_n2n_mapping = nn.Linear(teacher_node_dim, attention_dim)
        self.student_e2n_mapping = nn.Linear(teacher_edge_dim, attention_dim)
        self.student_e2e_mapping = nn.Linear(teacher_edge_dim, attention_dim)
        self.student_v2v_mapping = nn.Linear(teacher_vec_dim, attention_dim)

        # Teacher Model Layers
        self.teacher_n2n_mapping = nn.Linear(teacher_node_dim, attention_dim)
        self.teacher_e2n_mapping = nn.Linear(teacher_edge_dim, attention_dim)
        self.teacher_e2e_mapping = nn.Linear(teacher_edge_dim, attention_dim)
        self.teacher_v2v_mapping = nn.Linear(teacher_vec_dim, attention_dim)

    def compute_attention(self, output, batch_list = None):
        n2n_q = self.student_n2n_mapping(output['s_out']['n2n_mapping'])
        e2n_q = self.student_e2n_mapping(output['s_out']['e2n_mapping'])
        e2e_q = self.student_e2e_mapping(output['s_out']['e2e_mapping'])
        v2v_q = self.student_v2v_mapping(output['s_out']['v2v_mapping'])

        output['s_out']['n2n_attention'] = n2n_q
        output['s_out']['e2n_attention'] = e2n_q
        output['s_out']['e2e_attention'] = e2e_q
        output['s_out']['v2v_attention'] = v2v_q

        if batch_list:
            batch_list = batch_list.embedding
            n2n_ks = []
            e2n_ks = []
            e2e_ks= []
            v2v_ks = []

            for batch in batch_list:
                for item in batch:
                    n2n_k = item['n2n_mapping']
                    e2n_k = item['e2n_mapping']
                    e2e_k= item['e2e_mapping']
                    v2v_k = item['v2v_mapping']
                    n2n_ks.append(n2n_k.to(output['s_out']['n2n_mapping'].device))
                    e2n_ks.append(e2n_k.to(output['s_out']['e2n_mapping'].device))
                    e2e_ks.append(e2e_k.to(output['s_out']['e2e_mapping'].device))
                    v2v_ks.append(v2v_k.to(output['s_out']['v2v_mapping'].device))

            n2n_k = torch.cat(n2n_ks, dim=0)
            e2n_k = torch.cat(e2n_ks, dim=0)
            e2e_k = torch.cat(e2e_ks, dim=0)
            v2v_k = torch.cat(v2v_ks, dim=0)

            n2n_k = self.teacher_n2n_mapping(n2n_k)
            e2n_k = self.teacher_e2n_mapping(e2n_k)
            e2e_k = self.teacher_e2e_mapping(e2e_k)
            v2v_k = self.teacher_v2v_mapping(v2v_k)

            if 't_out' not in output:
                output['t_out'] = {}
                
            output['t_out']['n2n_attention'] = n2n_k
            output['t_out']['e2n_attention'] = e2n_k
            output['t_out']['e2e_attention'] = e2e_k
            output['t_out']['v2v_attention'] = v2v_k
        else:
            n2n_k = self.teacher_n2n_mapping(output['t_out']['n2n_mapping'])
            e2n_k = self.teacher_e2n_mapping(output['t_out']['e2n_mapping'])
            e2e_k = self.teacher_e2e_mapping(output['t_out']['e2e_mapping'])
            v2v_k = self.teacher_v2v_mapping(output['t_out']['v2v_mapping'])

            output['t_out']['n2n_attention'] = n2n_k
            output['t_out']['e2n_attention'] = e2n_k
            output['t_out']['e2e_attention'] = e2e_k
            output['t_out']['v2v_attention'] = v2v_k 

        return output