node_activation=Softmax
bs=64 
optim=adam 
lr=0.001
workers=4
#nbr_fea_len=128 #this is hardcoded by data generator. Using both combinesets and planewave
nbr_fea_len=64 #this is hardcoded by data generator. Only use combinesets
#h_fea_len=192
#atom_fea_len=192
h_fea_len=128
atom_fea_len=128
orig_atom_fea_len=95
n_h=1 
k=3
n_conv=9 
epochs=300
milestones=250
#target=band_gap
target=formation_energy
gpu=1
my=45

#here we run with-levels -2 SO THAT WE USE PLAIN DISTANCE
for t_ratio in  0.1 
do
for decay in 0 # 5e-4  
do
for seed in 42 
do
for my in  4 
do
output=outputminiplaindistance-loadmodelpooling0-scalededge-target-$target-my-$my-$node_activation-$seed-bs-$bs-$atom_fea_len-$nbr_fea_len-$k-$h_fea_len-$n_h-$decay-$optim-$lr-$n_conv-$t_ratio-epochs-$epochs-milestones-$milestones.txt
#CUDA_VISIBLE_DEVICES=$gpu python process_geo_CGNN.py   --orig_atom_fea_len $orig_atom_fea_len --atom_fea_len $atom_fea_len --n_h $n_h --nbr_fea_len $nbr_fea_len --h_fea_len $h_fea_len --k $k --n_conv $n_conv  --cutoff 8 --max_nei 12 --n_MLP_LR 3 --num_epochs $epochs  --target_name $target --milestones $milestones --gamma 0.1 --test_ratio $t_ratio --datafile_name my_graph_data_MP_test1_8_12_100_ --database material --n_grid_K 4 --n_Gaussian 64 --lr $lr  --dataset_path ./Material18-2 --batch_size $bs --node_activation $node_activation --seed $seed --my $my --weight_decay $decay --optim $optim --num_workers $workers --with_levels -2 --load_model --pre_trained_model_path ../GPG/model/model_my-4-atom_fea_len192-nbr_fe_len64cutoff8-maxnei12nconvs12-pooling1.pth  | tee $output
CUDA_VISIBLE_DEVICES=$gpu python process_geo_CGNN.py   --orig_atom_fea_len $orig_atom_fea_len --atom_fea_len $atom_fea_len --n_h $n_h --nbr_fea_len $nbr_fea_len --h_fea_len $h_fea_len --k $k --n_conv $n_conv  --cutoff 8 --max_nei 12 --n_MLP_LR 3 --num_epochs $epochs  --target_name $target --milestones $milestones --gamma 0.1 --test_ratio $t_ratio --datafile_name my_graph_data_2D_data_8_12_100_000 --database material --n_grid_K 4 --n_Gaussian 64 --lr $lr  --dataset_path ./2D_data_npj --batch_size $bs --node_activation $node_activation --seed $seed --my $my --weight_decay $decay --optim $optim --num_workers $workers --with_levels -2 | tee $output
done
done
done
done
