import torch
from torch import Tensor
import torch.nn.functional as F
from torch.nn import Sequential, Linear, ReLU, BatchNorm1d
import torch_geometric
from torch_geometric.nn import (
    Set2Set,
    global_mean_pool,
    global_add_pool,
    global_max_pool,
    MetaLayer,
)
from torch_scatter import scatter_mean, scatter_add, scatter_max, scatter

from matdeeplearn.common.registry import registry
from matdeeplearn.models.base_model import BaseModel, conditional_grad
from matdeeplearn.preprocessor.helpers import GaussianSmearing, node_rep_one_hot

@registry.register_model("MEGNet")
# Megnet
class MEGNet(BaseModel):
    def __init__(
        self,
        node_dim,
        edge_dim,
        output_dim,        
        dim1=64,
        dim2=64,
        dim3=64,
        pre_fc_count=1,
        gc_count=3,
        gc_fc_count=2,
        post_fc_count=1,
        pool="global_mean_pool",
        pool_order="early",
        batch_norm=True,
        batch_track_stats=True,
        act="relu",
        dropout_rate=0.0,
        **kwargs
    ):
        super(MEGNet, self).__init__()
        
        self.batch_track_stats = batch_track_stats
        self.batch_norm = batch_norm
        self.pool = pool
        if pool == "global_mean_pool":
            self.pool_reduce="mean"
        elif pool== "global_max_pool":
            self.pool_reduce="max" 
        elif pool== "global_sum_pool":
            self.pool_reduce="sum"             
        self.act = act
        self.pool_order = pool_order
        self.node_dim = node_dim
        self.edge_dim = edge_dim
        self.output_dim = output_dim           
        self.dropout_rate = dropout_rate
        self.pre_fc_count = pre_fc_count
        self.dim1 = dim1
        self.dim2 = dim2
        self.dim3 = dim3
        self.gc_count = gc_count
        self.post_fc_count = post_fc_count
        
        self.distance_expansion = GaussianSmearing(0.0, self.cutoff_radius, self.edge_dim, 0.2)
        
        ##Determine gc dimension dimension
        assert gc_count > 0, "Need at least 1 GC layer"
        if pre_fc_count == 0:
            self.gc_dim = self.node_dim
        else:
            self.gc_dim = dim1
        ##Determine post_fc dimension
        self.post_fc_dim = dim3

        # setup layers
        self.pre_lin_list = self._setup_pre_gnn_layers()
        self.conv_list, self.bn_list = self._setup_gnn_layers()
        self.post_lin_list, self.lin_out = self._setup_post_gnn_layers()


        ##Set up GNN layers
        self.e_embed_list = torch.nn.ModuleList()
        self.x_embed_list = torch.nn.ModuleList()
        self.u_embed_list = torch.nn.ModuleList()   
        self.conv_list = torch.nn.ModuleList()
        self.bn_list = torch.nn.ModuleList()
        for i in range(gc_count):
            if i == 0:
                e_embed = Sequential(
                    Linear(data.num_edge_features, dim3), ReLU(), Linear(dim3, dim3), ReLU()
                )
                x_embed = Sequential(
                    Linear(gc_dim, dim3), ReLU(), Linear(dim3, dim3), ReLU()
                )
                u_embed = Sequential(
                    Linear((data[0].u.shape[1]), dim3), ReLU(), Linear(dim3, dim3), ReLU()
                )
                self.e_embed_list.append(e_embed)
                self.x_embed_list.append(x_embed)
                self.u_embed_list.append(u_embed)
                self.conv_list.append(
                    MetaLayer(
                        Megnet_EdgeModel(dim3, self.act, self.batch_norm, self.batch_track_stats, self.dropout_rate, gc_fc_count),
                        Megnet_NodeModel(dim3, self.act, self.batch_norm, self.batch_track_stats, self.dropout_rate, gc_fc_count),
                        Megnet_GlobalModel(dim3, self.act, self.batch_norm, self.batch_track_stats, self.dropout_rate, gc_fc_count),
                    )
                )
            elif i > 0:
                e_embed = Sequential(Linear(dim3, dim3), ReLU(), Linear(dim3, dim3), ReLU())
                x_embed = Sequential(Linear(dim3, dim3), ReLU(), Linear(dim3, dim3), ReLU())
                u_embed = Sequential(Linear(dim3, dim3), ReLU(), Linear(dim3, dim3), ReLU())
                self.e_embed_list.append(e_embed)
                self.x_embed_list.append(x_embed)
                self.u_embed_list.append(u_embed)
                self.conv_list.append(
                    MetaLayer(
                        Megnet_EdgeModel(dim3, self.act, self.batch_norm, self.batch_track_stats, self.dropout_rate, gc_fc_count),
                        Megnet_NodeModel(dim3, self.act, self.batch_norm, self.batch_track_stats, self.dropout_rate, gc_fc_count),
                        Megnet_GlobalModel(dim3, self.act, self.batch_norm, self.batch_track_stats, self.dropout_rate, gc_fc_count),
                    )
                )

        ##Set up post-GNN dense layers (NOTE: in v0.1 there was a minimum of 2 dense layers, and fc_count(now post_fc_count) added to this number. In the current version, the minimum is zero)
        if post_fc_count > 0:
            self.post_lin_list = torch.nn.ModuleList()
            for i in range(post_fc_count):
                if i == 0:
                    ##Set2set pooling has doubled dimension
                    if self.pool_order == "early" and self.pool == "set2set":
                        lin = torch.nn.Linear(post_fc_dim * 5, dim2)
                    elif self.pool_order == "early" and self.pool != "set2set":
                        lin = torch.nn.Linear(post_fc_dim * 3, dim2)
                    elif self.pool_order == "late":
                        lin = torch.nn.Linear(post_fc_dim, dim2)
                    self.post_lin_list.append(lin)
                else:
                    lin = torch.nn.Linear(dim2, dim2)
                    self.post_lin_list.append(lin)
            self.lin_out = torch.nn.Linear(dim2, self.output_dim)

        elif post_fc_count == 0:
            self.post_lin_list = torch.nn.ModuleList()
            if self.pool_order == "early" and self.pool == "set2set":
                self.lin_out = torch.nn.Linear(post_fc_dim * 5, self.output_dim)
            elif self.pool_order == "early" and self.pool != "set2set":
                self.lin_out = torch.nn.Linear(post_fc_dim * 3, self.output_dim)              
            else:
                self.lin_out = torch.nn.Linear(post_fc_dim, self.output_dim)   

        ##Set up set2set pooling (if used)
        if self.pool_order == "early" and self.pool == "set2set":
            self.set2set_x = Set2Set(post_fc_dim, processing_steps=3)
            self.set2set_e = Set2Set(post_fc_dim, processing_steps=3)
        elif self.pool_order == "late" and self.pool == "set2set":
            self.set2set_x = Set2Set(self.output_dim, processing_steps=3, num_layers=1)
            # workaround for doubled dimension by set2set; if late pooling not reccomended to use set2set
            self.lin_out_2 = torch.nn.Linear(self.output_dim * 2, self.output_dim)

    @property
    def target_attr(self):
        return "y"
        
    ## Set up pre-GNN dense layers (NOTE: in v0.1 this is always set to 1 layer)
    def _setup_pre_gnn_layers(self):
        pre_lin_list = torch.nn.ModuleList()
        if self.pre_fc_count > 0:
            pre_lin_list = torch.nn.ModuleList()
            for i in range(self.pre_fc_count):
                if i == 0:
                    lin = torch.nn.Linear(self.node_dim, self.dim1)

                else:
                    lin = torch.nn.Linear(self.dim1, self.dim1)
                pre_lin_list.append(lin)

        return pre_lin_list        

    def _setup_post_gnn_layers(self):
        """Sets up post-GNN dense layers (NOTE: in v0.1 there was a minimum of 2 dense layers, and fc_count(now post_fc_count) added to this number. In the current version, the minimum is zero)."""
        post_lin_list = torch.nn.ModuleList()
        if self.post_fc_count > 0:
            for i in range(self.post_fc_count):
                if i == 0:
                    # Set2set pooling has doubled dimension
                    if self.pool_order == "early" and self.pool == "set2set":
                        lin = torch.nn.Linear(self.post_fc_dim * 5, self.dim2)
                    else:
                        lin = torch.nn.Linear(self.post_fc_dim * 3, self.dim2)
                else:
                    lin = torch.nn.Linear(self.dim2, self.dim2)
                post_lin_list.append(lin)
            lin_out = torch.nn.Linear(self.dim2, self.output_dim)
            # Set up set2set pooling (if used)

        # else post_fc_count is 0
        else:
            if self.pool_order == "early" and self.pool == "set2set":
                lin_out = torch.nn.Linear(self.post_fc_dim * 2, self.output_dim)
            else:
                lin_out = torch.nn.Linear(self.post_fc_dim, self.output_dim)

        return post_lin_list, lin_out
    
    def forward(self, data):

        ##Pre-GNN dense layers
        for i in range(0, len(self.pre_lin_list)):
            if i == 0:
                out = self.pre_lin_list[i](data.x.float())
                out = getattr(F, self.act)(out)
            else:
                out = self.pre_lin_list[i](out)
                out = getattr(F, self.act)(out)

        ##GNN layers        
        for i in range(0, len(self.conv_list)):
            if i == 0:
                if len(self.pre_lin_list) == 0:
                    e_temp = self.e_embed_list[i](data.edge_attr)
                    x_temp = self.x_embed_list[i](data.x)
                    u_temp = self.u_embed_list[i](data.u)
                    x_out, e_out, u_out = self.conv_list[i](
                        x_temp, data.edge_index, e_temp, u_temp, data.batch
                    )
                    x = torch.add(x_out, x_temp)
                    e = torch.add(e_out, e_temp)
                    u = torch.add(u_out, u_temp)
                else:
                    e_temp = self.e_embed_list[i](data.edge_attr)
                    x_temp = self.x_embed_list[i](out)
                    u_temp = self.u_embed_list[i](data.u)
                    x_out, e_out, u_out = self.conv_list[i](
                        x_temp, data.edge_index, e_temp, u_temp, data.batch
                    )
                    x = torch.add(x_out, x_temp)
                    e = torch.add(e_out, e_temp)
                    u = torch.add(u_out, u_temp)
                    
            elif i > 0:
                e_temp = self.e_embed_list[i](e)
                x_temp = self.x_embed_list[i](x)
                u_temp = self.u_embed_list[i](u)
                x_out, e_out, u_out = self.conv_list[i](
                    x_temp, data.edge_index, e_temp, u_temp, data.batch
                )
                x = torch.add(x_out, x)
                e = torch.add(e_out, e)
                u = torch.add(u_out, u)          

        ##Post-GNN dense layers
        if self.pool_order == "early":
            if self.pool == "set2set":
                x_pool = self.set2set_x(x, data.batch)
                e = scatter(e, data.edge_index[0, :], dim=0, reduce="mean")
                e_pool = self.set2set_e(e, data.batch)
                out = torch.cat([x_pool, e_pool, u], dim=1)                
            else:
                x_pool = scatter(x, data.batch, dim=0, reduce=self.pool_reduce)
                e_pool = scatter(e, data.edge_index[0, :], dim=0, reduce=self.pool_reduce)
                e_pool = scatter(e_pool, data.batch, dim=0, reduce=self.pool_reduce)
                out = torch.cat([x_pool, e_pool, u], dim=1)
            for i in range(0, len(self.post_lin_list)):
                out = self.post_lin_list[i](out)
                out = getattr(F, self.act)(out)
            out = self.lin_out(out)

        ##currently only uses node features for late pooling
        elif self.pool_order == "late":
            out = x
            for i in range(0, len(self.post_lin_list)):
                out = self.post_lin_list[i](out)
                out = getattr(F, self.act)(out)
            out = self.lin_out(out)
            if self.pool == "set2set":
                out = self.set2set_x(out, data.batch)
                out = self.lin_out_2(out)
            else:
                out = getattr(torch_geometric.nn, self.pool)(out, data.batch)
                
        if out.shape[1] == 1:
            return out.view(-1)
        else:
            return out


class Megnet_EdgeModel(torch.nn.Module):
    def __init__(self, dim, act, batch_norm, batch_track_stats, dropout_rate, fc_layers=2):
        super(Megnet_EdgeModel, self).__init__()
        self.act=act
        self.fc_layers = fc_layers
        if batch_track_stats == "False":
            self.batch_track_stats = False 
        else:
            self.batch_track_stats = True 
        self.batch_norm = batch_norm
        self.dropout_rate = dropout_rate
                
        self.edge_mlp = torch.nn.ModuleList()
        self.bn_list = torch.nn.ModuleList()
        for i in range(self.fc_layers + 1):
            if i == 0:
                lin = torch.nn.Linear(dim * 4, dim)
                self.edge_mlp.append(lin)       
            else:      
                lin = torch.nn.Linear(dim, dim)
                self.edge_mlp.append(lin) 
            if self.batch_norm == "True":
                bn = BatchNorm1d(dim, track_running_stats=self.batch_track_stats)
                self.bn_list.append(bn)            

    def forward(self, src, dest, edge_attr, u, batch):
        comb = torch.cat([src, dest, edge_attr, u[batch]], dim=1)
        for i in range(0, len(self.edge_mlp)):
            if i == 0:
                out = self.edge_mlp[i](comb)
                out = getattr(F, self.act)(out)  
                if self.batch_norm == "True":
                    out = self.bn_list[i](out)
                out = F.dropout(out, p=self.dropout_rate, training=self.training)
            else:    
                out = self.edge_mlp[i](out)
                out = getattr(F, self.act)(out)    
                if self.batch_norm == "True":
                    out = self.bn_list[i](out)
                out = F.dropout(out, p=self.dropout_rate, training=self.training)   
        return out


class Megnet_NodeModel(torch.nn.Module):
    def __init__(self, dim, act, batch_norm, batch_track_stats, dropout_rate, fc_layers=2):
        super(Megnet_NodeModel, self).__init__()
        self.act=act
        self.fc_layers = fc_layers
        if batch_track_stats == "False":
            self.batch_track_stats = False 
        else:
            self.batch_track_stats = True 
        self.batch_norm = batch_norm
        self.dropout_rate = dropout_rate
                
        self.node_mlp = torch.nn.ModuleList()
        self.bn_list = torch.nn.ModuleList()
        for i in range(self.fc_layers + 1):
            if i == 0:
                lin = torch.nn.Linear(dim * 3, dim)
                self.node_mlp.append(lin)       
            else:      
                lin = torch.nn.Linear(dim, dim)
                self.node_mlp.append(lin) 
            if self.batch_norm == "True":
                bn = BatchNorm1d(dim, track_running_stats=self.batch_track_stats)
                self.bn_list.append(bn)    

    def forward(self, x, edge_index, edge_attr, u, batch):
        # row, col = edge_index
        v_e = scatter_mean(edge_attr, edge_index[0, :], dim=0)
        comb = torch.cat([x, v_e, u[batch]], dim=1)
        for i in range(0, len(self.node_mlp)):
            if i == 0:
                out = self.node_mlp[i](comb)
                out = getattr(F, self.act)(out)  
                if self.batch_norm == "True":
                    out = self.bn_list[i](out)
                out = F.dropout(out, p=self.dropout_rate, training=self.training)
            else:    
                out = self.node_mlp[i](out)
                out = getattr(F, self.act)(out)    
                if self.batch_norm == "True":
                    out = self.bn_list[i](out)
                out = F.dropout(out, p=self.dropout_rate, training=self.training)   
        return out


class Megnet_GlobalModel(torch.nn.Module):
    def __init__(self, dim, act, batch_norm, batch_track_stats, dropout_rate, fc_layers=2):
        super(Megnet_GlobalModel, self).__init__()
        self.act=act
        self.fc_layers = fc_layers
        if batch_track_stats == "False":
            self.batch_track_stats = False 
        else:
            self.batch_track_stats = True 
        self.batch_norm = batch_norm
        self.dropout_rate = dropout_rate
                
        self.global_mlp = torch.nn.ModuleList()
        self.bn_list = torch.nn.ModuleList()
        for i in range(self.fc_layers + 1):
            if i == 0:
                lin = torch.nn.Linear(dim * 3, dim)
                self.global_mlp.append(lin)       
            else:      
                lin = torch.nn.Linear(dim, dim)
                self.global_mlp.append(lin) 
            if self.batch_norm == "True":
                bn = BatchNorm1d(dim, track_running_stats=self.batch_track_stats)
                self.bn_list.append(bn)  

    def forward(self, x, edge_index, edge_attr, u, batch):
        u_e = scatter_mean(edge_attr, edge_index[0, :], dim=0)
        u_e = scatter_mean(u_e, batch, dim=0)
        u_v = scatter_mean(x, batch, dim=0)
        comb = torch.cat([u_e, u_v, u], dim=1)
        for i in range(0, len(self.global_mlp)):
            if i == 0:
                out = self.global_mlp[i](comb)
                out = getattr(F, self.act)(out)  
                if self.batch_norm == "True":
                    out = self.bn_list[i](out)
                out = F.dropout(out, p=self.dropout_rate, training=self.training)
            else:    
                out = self.global_mlp[i](out)
                out = getattr(F, self.act)(out)    
                if self.batch_norm == "True":
                    out = self.bn_list[i](out)
                out = F.dropout(out, p=self.dropout_rate, training=self.training)   
        return out


