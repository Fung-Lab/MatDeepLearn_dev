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
