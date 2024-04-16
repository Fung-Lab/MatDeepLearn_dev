"""Langevin dynamics class."""

import math
from typing import List
from time import time

import yaml
from ase.md.md import MolecularDynamics
from ase import units

import numpy as np
import torch
from torch_geometric.data import Data
from torch_geometric.data.batch import Batch
from torch_geometric.loader import DataLoader

from matdeeplearn.common.iupac_2016_masses import atomic_masses_iupac2016
from matdeeplearn.common.ase_utils import MDLCalculator
from matdeeplearn.preprocessor.helpers import generate_node_features



class Langevin():

    def get_masses_of_structure(self, structure: Batch):
        masses = atomic_masses_iupac2016.to(self.device)
        return masses[structure.z].to(self.device)
    
    def setup_model(self, config_str: str, rank='cuda:0'):
        with open(config_str, "r") as yaml_file:
            config = yaml.safe_load(yaml_file)
        gradient = config["model"].get("gradient", False)
        otf_edge_index = config["model"].get("otf_edge_index", False)
        otf_edge_attr = config["model"].get("otf_edge_attr", False)
        self.otf_node_attr = config["model"].get("otf_node_attr", False)
        assert otf_edge_index and otf_edge_attr and gradient, "To use this calculator to calculate forces and stress, you should set otf_edge_index, oft_edge_attr and gradient to True."
        
        self.device = rank if torch.cuda.is_available() else 'cpu'
        self.models = MDLCalculator._load_model(config, self.device)
        self.n_neighbors = config['dataset']['preprocess_params'].get('n_neighbors', 250)
        

    def __init__(self,
                 timestep: float,
                 config_str: str,
                 temperature_K: float=None,
                 friction: float=None,
                 fixcm=True,
                 rank='cuda:0'):

        if friction is None:
            raise TypeError("Missing 'friction' argument.")
        self.fr = friction
        self.temp = units.kB * temperature_K
        self.temperature_K = temperature_K
        self.fix_com = fixcm
        self.setup_model(config_str, rank)
        self.dt = timestep
        
    def set_structure(self, structure: Data):
        self.prepare_structure(structure)
        
    def prepare_structure(self, structure: Data, first_prepare=False, prev_momenta=None):
        self.structure = structure.to(self.device)
        self.masses = self.get_masses_of_structure(structure)
        if not self.otf_node_attr:
            generate_node_features(structure, self.n_neighbors, device=self.device)
            structure.x = structure.x.to(torch.float32)
        if first_prepare:
            self.momenta = Langevin.initialize_maxwell_boltzmann_distribution(self.temperature_K, self.masses)
        else:
            assert prev_momenta is not None, "prev_momenta must be provided if not first prepare."
            self.momenta = prev_momenta.clone()
        self.updatevars()
        
    @staticmethod
    def initialize_maxwell_boltzmann_distribution(temperature_K, masses, device='cuda:0'):
        temp = units.kB * temperature_K
        # np.random.seed(42)
        xi = torch.from_numpy(np.random.standard_normal((len(masses), 3))).to(device).to(torch.float32)
        # xi = torch.from_numpy(np.random.standard_normal((len(masses) // 8, 3))).to(device).to(torch.float32)
        # xi = torch.concatenate([xi for _ in range(8)], dim=0)
        
        momenta = xi * torch.sqrt(masses * temp)[:, None]
        return momenta
        
    def run_simulation(self, structures: List[Data], num_steps: int, batch_size: int=1):
        max_energies, min_energies = torch.full((len(structures),), float('-inf'), device=self.device),\
            torch.full((len(structures),), float('inf'), device=self.device)
        
        start = time()
        for i in range(num_steps):
            if i != 0 and i % 200 == 0:
                print(max_energies, min_energies)
                print(f"{i}: {time() - start:.2f}s")
                start = time()
            loader = DataLoader(structures, batch_size=batch_size)
            for batch_idx, batch in enumerate(loader):
                if i == 0:
                    self.prepare_structure(batch, first_prepare=True)
                else:
                    self.prepare_structure(batch, prev_momenta=batch.momenta)
                
                _, energies, _ = self.step()
                
                max_energies[batch_idx*batch_size:(batch_idx+1)*batch_size] =\
                    torch.max(max_energies[batch_idx*batch_size:(batch_idx+1)*batch_size], energies.flatten())
                min_energies[batch_idx*batch_size:(batch_idx+1)*batch_size] =\
                    torch.min(min_energies[batch_idx*batch_size:(batch_idx+1)*batch_size], energies.flatten())
                
                # print("Structure pos:", self.structure.pos[:5, :])
                for structure_idx in range(batch.batch.max().item() + 1):
                    structures[batch_idx * batch_size + structure_idx].pos = self.structure.pos[batch.batch == structure_idx]
                    structures[batch_idx * batch_size + structure_idx].momenta = self.momenta[batch.batch == structure_idx]
                    
        return max_energies, min_energies
    
    def get_energy_forces(self, batch: Batch):  
        out_list = []
        for model in self.models:
            with torch.no_grad():   
                out_list.append(model(batch.to(self.device)))
        energy = torch.stack([entry["output"] for entry in out_list]).mean(dim=0)
        forces = torch.stack([entry["pos_grad"] for entry in out_list]).mean(dim=0)
        # return list of length num_batches
        return energy, forces

    def todict(self):
        return {'type': 'NVT-Langevin',
            'timestep': self.dt,
            'temperature_K': self.temp / units.kB,
            'friction': self.fr,
            'fixcm': self.fix_com}
        
    @staticmethod
    def get_velocities(momenta, masses):
        return momenta / masses[:, None]
    
    @staticmethod
    def get_center_of_mass(masses, pos):
        indices = slice(None)
        masses = masses[indices]
        com = masses @ pos[indices] / masses.sum()
        return com
    
    @staticmethod
    def set_center_of_mass(masses, structure, com):
        old_com = Langevin.get_center_of_mass(masses, structure.pos)
        difference = old_com - com
        structure.pos = structure.pos + difference
        
    @staticmethod
    def _get_com_velocity(masses, velocity):
        return masses.flatten() @ velocity / masses.sum()

    def updatevars(self):
        dt = self.dt
        T = self.temp
        fr = self.fr
        masses = self.masses
        sigma = torch.sqrt(2 * T * fr / masses)

        self.c1 = dt / 2. - dt * dt * fr / 8.
        self.c2 = dt * fr / 2 - dt * dt * fr * fr / 8.
        self.c3 = math.sqrt(dt) * sigma / 2. - dt**1.5 * fr * sigma / 8.
        self.c5 = dt**1.5 * sigma / (2 * math.sqrt(3))
        self.c4 = fr / 2. * self.c5

    def step(self, forces=None):
        natoms = self.structure.pos.shape[0]
        # print("natoms:", natoms)
        
        energies, forces = self.get_energy_forces(self.structure)        
        # print(forces[-5:])
        
        # This velocity as well as xi, eta and a few other variables are stored
        # as attributes, so Asap can do its magic when atoms migrate between
        # processors.
        self.v = Langevin.get_velocities(self.momenta, self.masses)

        # np.random.seed(42)
        self.xi = torch.from_numpy(np.random.standard_normal((natoms, 3))).to(self.device).to(torch.float32)
        self.eta = torch.from_numpy(np.random.standard_normal((natoms, 3))).to(self.device).to(torch.float32)
        
        # self.xi = torch.from_numpy(np.random.standard_normal((natoms // 8, 3))).to(self.device).to(torch.float32)
        # self.eta = torch.from_numpy(np.random.standard_normal((natoms // 8, 3))).to(self.device).to(torch.float32)
        
        # self.xi = torch.concatenate([self.xi for _ in range(8)], dim=0)
        # self.eta = torch.concatenate([self.eta for _ in range(8)], dim=0)
        
        # self.xi = torch.randn(natoms, 3, dtype=torch.float, device=self.device)
        # self.eta = torch.randn(natoms, 3, dtype=torch.float, device=self.device)

        # First halfstep in the velocity.
        self.v += (self.c1 * forces / self.masses[:, None] - self.c2 * self.v +
                   self.c3[:, None] * self.xi - self.c4[:, None] * self.eta)
        
        # Full step in positions
        x = self.structure.pos
        if self.fix_com:
            old_com = Langevin.get_center_of_mass(self.masses, x)
        # Step: x^n -> x^(n+1) - this applies constraints if any.
        self.structure.pos = x + self.dt * self.v + self.c5[:, None] * self.eta
        if self.fix_com:
            Langevin.set_center_of_mass(self.masses, self.structure, old_com)
            
        # recalc velocities after RATTLE constraints are applied
        self.v = (self.structure.pos - x -
                  self.c5[:, None] * self.eta) / self.dt
        
        energies, forces = self.get_energy_forces(self.structure)
        
        # Update the velocities
        self.v += (self.c1 * forces / self.masses[:, None] - self.c2 * self.v +
                   self.c3[:, None] * self.xi - self.c4[:, None] * self.eta)

        if self.fix_com:  # subtract center of mass vel
            self.v -= self._get_com_velocity(self.masses, self.v)

        # Second part of RATTLE taken care of here
        self.momenta = self.v * self.masses[:, None]
        return self.structure, energies, forces
