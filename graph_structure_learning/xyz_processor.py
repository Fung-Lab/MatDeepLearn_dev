from rdkit import Chem
from rdkit.Chem import AllChem
from rdkit.Chem import rdDetermineBonds
from rdkit import DataStructs
from ase import Atoms
from ase.optimize import BFGS
from ase.calculators.nwchem import NWChem
from ase.io import write
from ase import symbols
import subprocess
import os
import numpy as np

"""
    Use Atomic Simulation Environment (ASE) to process the molecules
    into individual .xyz files and then delete the files.
        - Implement a 'MoleculeProcessor' object that takes a list of
        JSON molecules on construction and has methods that perform the
        exact same process as in our notebook. I.e., the following methods
        must be implemented:
            - converts the molecules into .xyz files,
            - computes and stores all relevent metrics
            (fingerprints, Tanimoto coeffs, etc.),
            - deletes all .xyz files.
"""


class MoleculeProcessor:
    def __init__(self, molList):
        self.rawMols = molList
        self.processedMols = []
        self.metrics = []

    def getMolObjects(self):
        self.processedMols = []
        for entry in dataset:
            structure_id = entry["structure_id"]
            atomic_numbers = entry["atomic_numbers"]
            positions = entry["positions"]

            # Map atomic numbers to symbols using ASE Atoms/symbols
            atomic_symbols = symbols.Symbols(atomic_numbers)
            atoms = Atoms(symbols=atomic_symbols, positions=positions)

            # Write Atoms object to XYZ file
            xyz_filename = f"{structure_id}.xyz"
            with open(xyz_filename, "w") as f:
                f.write(f"{len(atoms)}\n")
                f.write("XYZ file generated from dataset\n")
                for symbol, pos in zip(atomic_symbols, positions):
                    f.write(f"{symbol} {' '.join(map(str, pos))}\n")

            # Convert XYZ file to Mol oject & store it in list
            raw_mol = Chem.MolFromXYZFile(f"{structure_id}.xyz")
            mol = Chem.Mol(raw_mol)
            rdDetermineBonds.DetermineBonds(mol)
            self.processedMols += [(structure_id, mol)]

            # Delete XYZ file
            os.remove(xyz_filename)

    def computeMetrics(self):
        n = len(self.processedMols)
        ecfp_bitvectors = []
        similarity_map = []
        # compute ECFPs
        for i in range(n):
            ecfp_bitvectors.append(AllChem.GetMorganFingerprintAsBitVect(
                mols[i][1], radius=2, nBits=2048))

        # compute Tanimoto similarities
        for i in range(n):
            i_sims = []
            for j in range(n):
                i_sims.append(DataStructs.TanimotoSimilarity(
                    ecfp_bitvectors[i], ecfp_bitvectors[j]))
            similarity_map.append([i_sims])

        self.metrics = np.array(similarity_map)
