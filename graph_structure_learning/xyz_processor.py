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
import logging

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
    def __init__(self, molList=None):
        """
            A list of raw JSON molecules from the QM9 dataset. These will be processed
            into ASE 'Atom' objects for the individual atoms, which will allow us to
            construct 'XYZ' files. The XYZ files can be converted into 'Mol' objects,
            from which the similarity metrics will be computed.
        """
        self.rawMols = molList

        """
            Will hold the 'Mol' objects that have beeen constructed from the JSON 
            molecules.
        """
        self.processedMols = []

        """
            Holds the Tanimoto coefficient similarity metrics for each molecule;
            constructed as an adjacency matrix, where A(i,j) gives the similarity
            between molecule i and j. 
        """
        self.similarityMap = []

    def processMolObjects(self):
        if self.rawMols != None:
            # self.processedMols = []
            # TODO: make sure the 'structure_id', 'atomic_numbers', and 'positions' are extracted properly
            for entry in self.rawMols:
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
                        f.write(f"{symbol} {' '.join(map(lambda x: '{:.20f}'.format(x), pos))}\n")

                # Convert XYZ file to Mol object & store it in list
                try:
                    raw_mol = Chem.MolFromXYZFile(f"{structure_id}.xyz")
                    mol = Chem.Mol(raw_mol)
                    rdDetermineBonds.DetermineBonds(mol)
                    self.processedMols += [(structure_id, mol)]
                    print(f"Successful: {raw_mol}, {structure_id}")
                # By-pass molecules causing error
                except:
                    print("ERROR", structure_id)
                    self.logError(structure_id)

                # Delete XYZ file
                os.remove(xyz_filename)

    def computeMetrics(self):
        n = len(self.processedMols)
        ecfp_bitvectors = []
        similarity_map = []
        # compute ECFPs
        for i in range(n):
            ecfp_bitvectors.append(AllChem.GetMorganFingerprintAsBitVect(
                self.processedMols[i][1], radius=2, nBits=2048))

        # compute Tanimoto similarities
        for i in range(n):
            i_sims = []
            for j in range(n):
                i_sims.append(DataStructs.TanimotoSimilarity(
                    ecfp_bitvectors[i], ecfp_bitvectors[j]))
            similarity_map.append([i_sims])

        self.similarityMap = np.array(similarity_map)
    
    # Helper function that logs errors to logs.log
    def logError(self, structure_id):
        logging.basicConfig(filename="logs.log", format="%(message)s", filemode="a")
        logger = logging.getLogger()
        logger.setLevel(logging.DEBUG)
        logger.info(f"{structure_id} could not be converted into Mol from the XYZ File")

