#for regular data gen without atom levels in the graph
#python AtomGraph.py --data_dir ./Material/cif --name_database MP_test1 --cutoff 8 --max_num_nbr 12 --compress_ratio 1 --output_path ./Material/npz --id_prop ./Material/targets.csv

#python AtomGraph.py --data_dir ./Material/cif --name_database MP_test1 --cutoff 8 --max_num_nbr 12 --compress_ratio 1 --output_path ./Material/levelnpz --id_prop ./Material/targets_material.csv --compute_levels 1 

#python AtomGraph.py --data_dir ./Material18/cif --name_database MP_test1 --cutoff 8 --max_num_nbr 12 --compress_ratio 1 --output_path ./Material18/npz --id_prop ./Material18/targets_material.csv --compute_levels 0

#python AtomGraph.py --data_dir ./Material18/cif --name_database MP_test1 --cutoff 8 --max_num_nbr 12 --compress_ratio 1 --output_path ./Material18/levelnpz --id_prop ./Material18/targets_material.csv --compute_levels 1 

python AtomGraph.py --data_dir ./2D_data_npj/cif --name_database 2D_data --cutoff 8 --max_num_nbr 12 --compress_ratio 1 --output_path ./2D_data_npj/networxnpz --id_prop ./2D_data_npj/targets_material.csv --compute_levels 1 --keep_networx 1
