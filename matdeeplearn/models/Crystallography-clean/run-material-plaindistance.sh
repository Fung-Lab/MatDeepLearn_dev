node_activation=Softmax
bs=64 #100 200
optim=adam #sgd
lr=0.001 #0.01 
workers=4
#nbr_fea_len=128 #this is hardcoded by data generator. Using both combinesets and planewave
nbr_fea_len=64 #this is hardcoded by data generator. Only use combinesets
#h_fea_len=128
#atom_fea_len=128
h_fea_len=192
#atom_fea_len=192
h_fea_len=128
atom_fea_len=128
orig_atom_fea_len=89
n_h=1 #2
k=3
n_conv=3 # 3 4 
epochs=300
milestones=250
#target=formation_energy
target=band_gap
gpu=0
#here we run with-levels -2 SO THAT WE USE PLAIN DISTANCE
for t_ratio in   0.1 #-8 -7 -1000 -500 -300
do
for decay in 0  #1e-3 2e-3 3e-3 4e-3
do
for seed in 42 #197 145231 49993 
do
for my in  4 #459
do
output=outputmaterialplaindistance18-scalededge-target-$target-my-$my-$node_activation-$seed-bs-$bs-$atom_fea_len-$nbr_fea_len-$k-$h_fea_len-$n_h-$decay-$optim-$lr-$n_conv-$t_ratio-epochs-$epochs-milestones-$milestones.txt
#CUDA_VISIBLE_DEVICES=$gpu python process_geo_CGNN.py   --orig_atom_fea_len $orig_atom_fea_len --atom_fea_len $atom_fea_len --n_h $n_h --nbr_fea_len $nbr_fea_len --h_fea_len $h_fea_len --k $k --n_conv $n_conv  --cutoff 8 --max_nei 12 --n_MLP_LR 3 --num_epochs $epochs  --target_name $target --milestones 250 --gamma 0.1 --test_ratio $t_ratio --datafile_name my_graph_data_MP_test1_8_12_100_ --database material --n_grid_K 4 --n_Gaussian 64 --lr $lr  --dataset_path ./Material --batch_size $bs --node_activation $node_activation --seed $seed --my $my --weight_decay $decay --optim $optim --num_workers $workers | tee $output
CUDA_VISIBLE_DEVICES=$gpu python process_geo_CGNN.py   --orig_atom_fea_len $orig_atom_fea_len --atom_fea_len $atom_fea_len --n_h $n_h --nbr_fea_len $nbr_fea_len --h_fea_len $h_fea_len --k $k --n_conv $n_conv  --cutoff 8 --max_nei 12 --n_MLP_LR 3 --num_epochs $epochs  --target_name $target --milestones $milestones --gamma 0.1 --test_ratio $t_ratio --datafile_name my_graph_data_MP_test1_8_12_100_ --database material --n_grid_K 4 --n_Gaussian 64 --lr $lr  --dataset_path ./Material --batch_size $bs --node_activation $node_activation --seed $seed --my $my --weight_decay $decay --optim $optim --num_workers $workers --with_levels -2| tee $output
done
done
done
done
