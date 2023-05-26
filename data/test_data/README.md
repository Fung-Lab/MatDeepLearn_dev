Description:

This is a collection of datasets intended to test basic functionality of the package on several core tasks:

1. Graph level prediction of scalar properties with data_graph_scalar.json. The property being predicted is energy in units of electron volts.
  
2. Graph level prediction of vector quantities with data_graph_vector.json. The property being predicted is average absolute forces in x, y, and z directions.

3. Node level prediction of scalar quantities with data_node_scalar.json. The property being predicted is average absolute force per atom.

4. Node level prediction of vector quantities with data_node_vector.json. The property being predicted is force in x, y, and z directions per atom.

5. Prediction on a dataset with periodic and non-periodic structures with data_graph_scalar_cell_mixed.json. Half the dataset contains the cell parameter and the other half does not.

6. Prediction on a dataset stored as individual .cif structure files and graph level target properties in targets.csv. 

7. Prediction on a dataset with manual training, validation, testing, and prediction splits, stored as separate files in raw_graph_scalar_manual_split

Note: These datasets are too small to be used to evaluate the performance of a GNN model. They are intended to be used for development purposes or testing a successful installation. 

Reference(s):
  Source: Data generated within Fung Group.