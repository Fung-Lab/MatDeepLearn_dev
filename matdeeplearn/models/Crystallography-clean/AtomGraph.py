################################################
'''
This file is to extract all the information in need to construct the atom graph from .cif files
input:
--data_dir : the .cif file path, the cif files' name should be mp-XXXX or oqmd-XXXX
--output_path 
--name_database : MP or OQMD, 
--cutoff : radius of neighbourhood
--max_num_nbr : max numbers of neighbors
--compress_ratio : percentage of data you want to use

output:
Several .npz files to store the name, lattice vector, nodes features ,neighbors , cell volume
Will output to "output_path"
'''
################################################
from __future__ import print_function, division
from tqdm import tqdm
import csv
import functools
import json
import os
import random
import warnings
import glob
import numpy as np
from operator import itemgetter
from pymatgen.core.structure import Structure
from pydash import py_
import pandas as pd
import networkx as nx

def load_materials(filepath):
    try:
        data = np.load(filepath,allow_pickle=True)['materials']
    except UnicodeError:
        data = np.load(filepath, encoding='latin1')['materials']
    return data


def build_config(my_path,config_path):
    # 输入所有cubic.cif数据
    # 建立one-hot编码以及保存设置
    atoms=[]
    all_files = sorted(glob.glob(os.path.join(my_path,'*.cif')))
    for path in tqdm(all_files):
        crystal = Structure.from_file(path)
        atoms += list(crystal.atomic_numbers)
    unique_z = np.unique(atoms)
    num_z = len(unique_z)
    print('unique_z:', num_z)
    print('min z:', np.min(unique_z))
    print('max z:', np.max(unique_z))
    z_dict = {z:i for i, z in enumerate(unique_z)}
    # Configuration file
    config = dict()
    config["atomic_numbers"] = unique_z.tolist()
    config["node_vectors"] = np.eye(num_z,num_z).tolist() # One-hot encoding
    with open(config_path, 'w') as f:
        json.dump(config, f)
    return config

def compute_BFS_levels(G, starting_nodes):
    current_distance = 0
    current_layer = {-1}
    visited = {-1}
    for n in starting_nodes:
        current_layer.add(n)
        visited.add(n)
    current_layer = starting_nodes
    levels=[0]*G.number_of_nodes()
    for n in starting_nodes:
        levels[n] = 0
    l = 0
    while len(current_layer) > 0:
        next_layer = set()
        l += 1
        for node in current_layer:
            for child in G[node]:
                if child not in visited:
                    visited.add(child)
                    next_layer.add(child)
                    levels[child] = l
        current_layer = next_layer
    return levels

#nbr_fea_idx: each node's neighbors ids
#nbr_w: weights
def build_graph(nbr_fea_idx, nbr_w):
    g = nx.Graph()
    for i, nbrs in enumerate(nbr_fea_idx):
        nbrw = nbr_w[i]
        for j, idx in enumerate(nbrs):
            g.add_edge(i, idx, weight=nbrw[j])
    return g

def process(config,data_path,radius,max_num_nbr, compute_levels, keep_networkx):
    crystal = Structure.from_file(data_path)
    volume=crystal.lattice.volume
    coords=crystal.cart_coords
    lattice=crystal.lattice.matrix
    atoms=crystal.atomic_numbers
    material_id=data_path[:-4]
    atomnum=config['atomic_numbers']
    z_dict = {z:i for i, z in enumerate(atomnum)}
    #one_hotvec=np.array(config["node_vectors"])
    #atom_fea = np.vstack([one_hotvec[z_dict[atoms[i]]] for i in range(len(crystal))])

    hot_vects = []
    for i in range (len(crystal)):
        vect = np.zeros(95)
        vect[atoms[i]] = 1.0
        hot_vects.append(vect)
    atom_fea = np.vstack(hot_vects)

    all_nbrs = crystal.get_all_neighbors(radius, include_index=True)
    all_nbrs = [sorted(nbrs, key=lambda x: x[1]) for nbrs in all_nbrs]
    nbr_fea_idx, nbr_fea, nbr_w = [], [], []

    for i,nbr in enumerate(all_nbrs):
        if len(nbr) < max_num_nbr:
            nbr_fea_idx.append(list(map(lambda x: x[2].tolist(), nbr)) +
                                [0] * (max_num_nbr - len(nbr)))
            nbr_fea.append(list(map(lambda x: x[0].coords.tolist(), nbr)) +
                   [[coords[i][0]+radius,coords[i][1],coords[i][2]]] * (max_num_nbr -len(nbr)))
        else:
            nbr_fea_idx.append(list(map(lambda x: x[2].tolist(),
                                        nbr[:max_num_nbr])))
            nbr_fea.append(list(map(lambda x: x[0].coords.tolist(),
                                    nbr[:max_num_nbr])))

    atom_fea=atom_fea.tolist()

    nbr_subtract=[]
    nbr_distance=[]
    atom_levels = None
    g = None
    for i in range(len(nbr_fea)):
        if nbr_fea[i] != []:
            x=nbr_fea[i]-coords[:,np.newaxis,:][i]
            nbr_subtract.append(x)
            nbr_distance.append(np.linalg.norm(x, axis=1).tolist())
        else:
            nbr_subtract.append(np.array([]))
            nbr_distance.append(np.array([]))
    if compute_levels == 1:
        g = build_graph(nbr_fea_idx, nbr_distance)
        if (nx.is_connected(g)):
            #starting_nodes = list(nx.barycenter(g))
            starting_nodes = list(nx.center(g))
            levels = compute_BFS_levels(g, starting_nodes)
        else:
            print('NOT CONNECTED')
            levels = [0] * g.number_of_nodes()
        atom_levels = np.array(levels)
        nlevel = max(levels)
        print(data_path, '#levels', nlevel)
    nbr_fea_idx = np.array(nbr_fea_idx) 
    return material_id,lattice,atom_fea,nbr_fea_idx,nbr_distance,nbr_subtract,volume, atom_levels, g
##########

def main(data_dir, output_path,name_database ,cutoff,max_num_nbr,compress_ratio,id_prop,compute_levels,keep_networx, chunk_size=10000):
    if not os.path.isdir(data_dir):
        print('Not found the data directory: {}'.format(data_dir))
        exit(1)
    #config_path=os.path.join(output_path, 'oqmd_config_onehot.json')
    config_path='./database/mp_config_onehot.json' # gcong
    if name_database=='MP':
        config_path='./database/mp_config_onehot.json'
    elif name_database=='OQMD':
        config_path='./database/oqmd_config_onehot.json'

    if os.path.isfile(config_path):
        print('config exists')
        with open(config_path) as f:
            config = json.load(f)
    else:
        print('buiding config')
        config=build_config(data_dir,config_path)

    #data_files = sorted(glob.glob(os.path.join(data_dir, '*.cif')))
    column_names = ["id"] 
    #df = pd.read_csv("./BW/targets_MP_test1.csv", names=column_names)
    df = pd.read_csv(options['id_prop'])
    data_files= df.id.to_list()
    data_files=[data_dir+"/"+file+".cif" for file in data_files]
    print('data_files type', type(data_files))
    '''
    for i in range(100):
        print(data_files[i])
    print(xxxxx)
    if name_database=='MP':
        data_files = sorted(glob.glob(os.path.join(data_dir, 'mp-*.cif')))
    elif name_database=='OQMD':
        data_files = sorted(glob.glob(os.path.join(data_dir, 'oqmd-*.cif')))       
    ''' 
    for n, chunk in enumerate(tqdm(py_.chunk(data_files[:int(compress_ratio*len(data_files))], chunk_size))):
        graph_names = []
        graph_lattice=[]
        graph_nodes= []
        graph_edges = []
        graph_volume=[]
        graph_levels = []
        graphs=dict()
        gs = []
        for file in chunk:
            material_id,lattice,atom_fea,nbr_fea_idx,nbr_distance,nbr_subtract,volume , atom_levels, g = process(config,file,cutoff,max_num_nbr, compute_levels, keep_networx)
            #print(type(lattice))
            print(material_id[len(data_dir)+1:])
            #print('atom_fea', type(atom_fea) )
            #print('lattice', type(lattice))
            #print(lattice)
            graph_lattice.append(lattice)
            graph_names.append(material_id[len(data_dir)+1:])
            graph_nodes.append(atom_fea)
            graph_edges.append((nbr_fea_idx,nbr_distance,nbr_subtract))
            graph_volume.append(volume)
            if compute_levels == 1:
                graph_levels.append(atom_levels)
            if keep_networx == 1:
                gs.append(g)
        if compute_levels == 1:
            if keep_networx == 1:
                for name, lattice,nodes,neighbors,volume, levels, g in tqdm(zip(graph_names,graph_lattice,graph_nodes, graph_edges,graph_volume, graph_levels, gs)):
                    graphs[name] = (lattice,nodes, neighbors,volume, levels, g)
            else:
                for name, lattice,nodes,neighbors,volume, levels in tqdm(zip(graph_names,graph_lattice,graph_nodes, graph_edges,graph_volume, graph_levels)):
                    graphs[name] = (lattice,nodes, neighbors,volume, levels)
        else:
            for name, lattice,nodes,neighbors,volume in tqdm(zip(graph_names,graph_lattice,graph_nodes, graph_edges,graph_volume) ):
                graphs[name] = (lattice,nodes, neighbors,volume)
        np.savez_compressed(os.path.join(output_path,"my_graph_data_{}_{}_{}_{}_{:03d}.npz".format(name_database,int(cutoff),max_num_nbr,int(compress_ratio*100),n)), graph_dict=graphs)

if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description='Crystal Graph Coordinator.')
    parser.add_argument('--data_dir', metavar='PATH', type=str, default='database/cif',
                        help='The path to a data directory (default: database/cif)')      
    parser.add_argument('--output_path', metavar='PATH', type=str, default='database/npz',
                        help='The output path (default: database/npz)')               
    parser.add_argument('--name_database', metavar='N', type=str, default='OQMD',
                        help='name of database, MP or OQMD (default:OQMD)')
    parser.add_argument('--cutoff', metavar='N', type=float, default=8,
                        help='cutoff distance of neighbors (default : 8A)')
    parser.add_argument('--max_num_nbr', metavar='N', type=int, default=12,
                        help='max neighbors of each node (default : 12)')
    parser.add_argument('--compress_ratio', metavar='N', type=float, default=1,
                        help='compress_ratio (default : 1)')
    parser.add_argument('--id_prop', metavar='PATH', type=str, default='./database/targets_MP_test1.csv',
                        help='The path to a id prop file(default: database/cif)')      
    parser.add_argument('--compute_levels', type=int, default=0,
                        help='whether to find barry centers of the graph and compute levels')
    parser.add_argument('--keep_networx', type=int, default=0,
                        help='whether to save networx graph constructed')
    options = vars(parser.parse_args())

    main(**options)

    options = vars(parser.parse_args())

    main(**options)

