import os.path
import json
import scipy
from scipy.special import jn_zeros,jn,sph_harm
import numpy as np
import pandas as pd
import glob
import torch
from torch.utils.data import Dataset
import tqdm
from sklearn.preprocessing import StandardScaler

def load_graph_data(file_path):
    Total={}
    for path in file_path:
        print('loading : {}'.format(path))
        try:
            graphs = np.load(path,allow_pickle=True)['graph_dict'].item()
        except UnicodeError:
            graphs = np.load(path, encoding='latin1',allow_pickle=True)['graph_dict'].item()
            graphs = { k.decode() : v for k, v in graphs.items() }
        Total={**Total,**graphs}
    print('load successed, final volume : {}'.format(len(Total)))
    return Total

# for MP edge_max 8.000000000000004  edge_min 0.7249742783999998

edge_max = 8.0
edge_min = 0.724

def scale_edge(d):
    return (d-edge_min)/(edge_max-edge_min)

def Atomgraph_collate(batch):
    nodes = []
    edge_distance=[]
    edge_targets=[]
    edge_sources = []
    graph_indices = []
    node_counts = []
    targets = []
    combine_sets =[]
    plane_wave = []
    total_count = 0

    atom_fea=[]
    nbr_fea =[]
    nbr_fea_idx=[]
    crystal_atom_idx=[]
    atom_levels=[]
    for i, (graph, target) in enumerate(batch):
        # Numbering for each batch
        #atom_fea.append(graph.nodes) #atom fea done
        #nodes.append(graph.nodes) #atom fea done
        #print(graph.nodes.shape)
        #print('neighbor', graph.nei.shape)
        n_i = len(graph)
        atom_fea.append(graph.nodes)
        max_neighbors = (graph.nei+total_count).size//n_i
        nbr_fea_idx.append((graph.nei+total_count).reshape(n_i,max_neighbors))
        combine_sets.append(graph.combine_sets.reshape(n_i,max_neighbors,-1))
        plane_wave.append(graph.plane_wave.reshape(n_i, max_neighbors,-1))
        #node_counts.append(len(graph))
        crystal_atom_idx.append(torch.LongTensor(np.arange(n_i)+total_count)) #for each crystal where are its atoms in this batch
        targets.append(target)
        if graph.with_levels == 1:
            atom_levels.append(graph.levels)
        #graph_indices += [i] * len(graph)
        total_count += n_i

    #combine_sets=np.concatenate(combine_sets,axis=0)
    #plane_wave=np.concatenate(plane_wave,axis=0)
    #nodes = np.concatenate(nodes,axis=0)
    #edge_distance = np.concatenate(edge_distance,axis=0)
    #edge_sources = np.concatenate(edge_sources,axis=0)
    #edge_targets = np.concatenate(edge_targets,axis=0)
    #input = geo_CGNN_Input(nodes,edge_distance, edge_sources, edge_targets, graph_indices, node_counts,combine_sets,plane_wave)
    atom_fea = np.concatenate(atom_fea,axis=0)
    atom_fea = torch.Tensor(atom_fea)
    nbr_fea_idx = np.concatenate(nbr_fea_idx,axis=0)
    nbr_fea_idx = torch.LongTensor(nbr_fea_idx)
    combine_sets = np.concatenate(combine_sets, axis=0)
    plane_wave = np.concatenate(plane_wave, axis=0)
    #nbr_fea = np.concatenate([combine_sets,plane_wave], axis=2)
    nbr_fea = combine_sets
    #nbr_fea = plane_wave
    nbr_fea = torch.Tensor(nbr_fea) 
    targets = torch.Tensor(targets).unsqueeze(1)
    #print('targets size', targets.size())
    #print('total_count', total_count)
    #print('atom_fea', atom_fea.size())
    #print('nbr_fea', nbr_fea.size())
    #print('nbr_idx', nbr_fea_idx.size())
    #print(targets.size())
    if graph.with_levels <= 0 :
        return (atom_fea, nbr_fea, nbr_fea_idx,crystal_atom_idx), targets
    else:
        atom_levels = np.concatenate(atom_levels, axis=0)
        atom_levels = torch.Tensor(atom_levels).unsqueeze(1)
        return (atom_fea, atom_levels, nbr_fea, nbr_fea_idx,crystal_atom_idx), targets

# here we use with_levels to indicate whether we have levels information.
# if with_levels is negative, then that indicate we do not want to have elaborate combine_sets
class AtomGraph(object):
    def __init__(self, graph,cutoff,N_shbf,N_srbf,n_grid_K,n_Gaussian, with_levels=0):
        if with_levels == 1:
            lattice, self.nodes, neighbors,volume,levels = graph #gcong add levels for each atoms
        else:
            lattice, self.nodes, neighbors,volume = graph[0], graph[1], graph[2], graph[3] #gcong add levels for each atoms
        nei=neighbors[0]  #nbr_fea_idx, 
        distance=neighbors[1] #nbr_distance
        #print('EDGE DISTANCE', max([max(l) for l in neighbors[1]]), min([min(l) for l in neighbors[1]]))
        vector=neighbors[2]  #nbr_subtract
        n_nodes = len(self.nodes)
        self.nei = nei
        self.edge_sources = np.concatenate([[i] * len(nei[i]) for i in range(n_nodes)])
        #print('edge_sources', self.edge_sources)
        self.edge_targets=np.concatenate(nei)
        #print('edge_targets', self.edge_targets)
        self.nodes = np.array(self.nodes, dtype=np.float32) #atom_fea
        edge_vector = np.array(vector, dtype=np.float32) #nbr_subtract
        self.edge_index = np.concatenate([range(len(nei[i])) for i in range(n_nodes)])
        self.vectorij= edge_vector[self.edge_sources,self.edge_index]
        edge_distance = np.array(distance, dtype=np.float32) #nbr_distance
        self.distance= edge_distance[self.edge_sources,self.edge_index]
        self.with_levels = with_levels
        if with_levels == 1:
            self.levels = np.array(levels, dtype = np.float32) # gcong add levels
        combine_sets=[]
        # gaussian radial
        N=n_Gaussian
        for n in range(1,N+1):
            phi=Phi(self.distance,cutoff)
            if with_levels == -1:
                combine_sets.append(phi)
            elif with_levels == -2:
                combine_sets.append(scale_edge(self.distance))
            else:
                G=gaussian(self.distance,miuk(n,N,cutoff),betak(N,cutoff))
                combine_sets.append(phi*G) # original implementation
        self.combine_sets=np.array(combine_sets, dtype=np.float32).transpose()

        #print('ALERT WITH LEVBELS', with_levels)
        # plane wave
        grid=n_grid_K
        kr=np.dot(self.vectorij,get_Kpoints_random(grid,lattice,volume).transpose()) 
        self.plane_wave=np.cos(kr)/np.sqrt(volume)
        #self.plane_wave = self.plane_wave.astype(np.float32) #gcong

    def __len__(self):
        return len(self.nodes)

class AtomGraphDataset(Dataset):
    def __init__(self, path, filename,database, target_name,cutoff,N_shbf,N_srbf,n_grid_K,n_Gaussian, with_levels=0):
        super(AtomGraphDataset, self).__init__()
        
        target_path = os.path.join(path, "targets_"+database+".csv")

        '''
        if target_name == 'band_gap' and database=='MP':
            target_path = os.path.join(path, "targets_"+database+'_Eg'+".csv")
        elif target_name == 'formation_energy_per_atom' and database=='MP':
            target_path = os.path.join(path, "targets_"+database+'_Ef'+".csv")
        '''
        print('target_path', target_path)
        df = pd.read_csv(target_path).dropna(axis=0,how='any') 

        if target_name == 'band_gap' and (database=='OQMD' or database=='MEGNet_2018'):
            df=df[df['band_gap']!=0]
        '''
        if target_name == 'band_gap':
            print('only predicting for non zero band gap')
            df=df[df['band_gap']!=0]
        '''
                    
        if with_levels == 1:
            print('Using data with atom level information')
            graph_data_path = sorted(glob.glob(os.path.join(path, 'levelnpz/'+filename+'*.npz')))
        else:
            graph_data_path = sorted(glob.glob(os.path.join(path, 'networxnpz/'+filename+'*.npz')))
        print('The number of files = {}'.format(len(graph_data_path)))
        self.graph_data = load_graph_data(graph_data_path)
        graphs=self.graph_data.keys()
        #print('graphs', graphs)
        '''
        kgraphs=list(graphs)
        print('num graphs loades is', len(list(graphs)))
        for g in kgraphs:
            print(' ', g)
        print(df['id'])
        '''
        try:
            self.graph_names=df.loc[df['id'].isin(graphs)].id.values.tolist()
        except:
            df.columns = ['id', 'formation_energy']
            self.graph_names=df.loc[df['id'].astype(str).isin(graphs)].id.values.tolist()
        #self.graph_names=list(graphs)
        #print('graph_names', self.graph_names, 'len', len(self.graph_names))
        print(df)
        self.targets=np.array(df.loc[df['id'].astype(str).isin(graphs)][target_name].values.tolist())
        print(self.targets)
        scaler = StandardScaler()
        scaler.fit(np.array(self.targets).reshape(-1,1))
        print('scaler mean', scaler.mean_)
        self.scaler = scaler
        self.with_levels = with_levels
        #self.targets=np.array(df.loc[self.graph_names][target_name].tolist())

        #print('targets', self.targets)
        print('the number of valid targets = {}'.format(len(self.targets)))
        print('start to constructe AtomGraph')
        graph_data=[]
        print(self.graph_data['1'])
        for i,name in enumerate(self.graph_names):
            graph_data.append(AtomGraph(self.graph_data[str(name)],cutoff,N_shbf,N_srbf,n_grid_K,n_Gaussian, with_levels))
            if i%2000==0 and i>0:
                print('{} graphs constructed'.format(i))
        print('finish constructe the graph')
        self.graph_data=graph_data
        print('self.graph_data[0] len', len(self.graph_data[0]))
                           
        assert(len(self.graph_data)==len(self.targets))
        print('The number of valid graphs = {}'.format(len(self.targets)))
                
    def __getitem__(self, index):
        return self.graph_data[index], self.targets[index]

    def __len__(self):
        return len(self.graph_names)



def a_SBF(alpha,l,n,d,cutoff):
    root=float(jn_zeros(l,n)[n-1])
    return jn(l,root*d/cutoff)*sph_harm(0,l,np.array(alpha),0).real*np.sqrt(2/cutoff**3/jn(l+1,root)**2)

def a_RBF(n,d,cutoff):
    return np.sqrt(2/cutoff)*np.sin(n*np.pi*d/cutoff)/d

def get_Kpoints_random(q,lattice,volume):
    a0=lattice[0,:]
    a1=lattice[1,:]
    a2=lattice[2,:]
    unit=2*np.pi*np.vstack((np.cross(a1,a2),np.cross(a2,a0),np.cross(a0,a1)))/volume
    ur=[(2*r-q-1)/2/q for r in range(1,q+1)]
    points=[]
    for i in ur:
        for j in ur:
            for k in ur:
                points.append(unit[0,:]*i+unit[1,:]*j+unit[2,:]*k)
    points=np.array(points) 
    return points  


def Phi(r,cutoff):
    return 1-6*(r/cutoff)**5+15*(r/cutoff)**4-10*(r/cutoff)**3
def gaussian(r,miuk,betak):
    return np.exp(-betak*(np.exp(-r)-miuk)**2)
def miuk(n,K,cutoff):
    # n=[1,K]
    return np.exp(-cutoff)+(1-np.exp(-cutoff))/K*n
def betak(K,cutoff):
    return (2/K*(1-np.exp(-cutoff)))**(-2)


