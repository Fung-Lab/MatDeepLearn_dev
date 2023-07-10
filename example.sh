#!/bin/bash

prefix_path="/nethome/sjia65/Datasets/Matbench/"
# dataset=("data_exfoliation.json" "data_phonons.json" "data_dielectric.json" "data_log_gvrh.json" "data_log_kvrh.json" "data_perovskites.json")
dataset=("data_mp_form.json" "data_mp_gap.json")
# batch_size=(32 32 128 128 128 128)
batch_size=(32 32)

for i in ${!dataset[@]}; do
    # concate the path
    path=$prefix_path${dataset[$i]};
    size=${batch_size[$i]};
    pt_path="placeholder";
    
    echo "Run: $path";

    for j in 1 2 3 4 5; do
        seed=$RANDOM;

        echo "#Run: $j";
        echo "Seed: $seed";

        python scripts/main.py \
        --run_mode=train \
        --config_path=configs/forces/011.yml \
        --gpu_id=0 \
        --batch_size=$size \
        --seed=$seed \
        --pt_path=$pt_path \
        --global_pool="global_mean_pool" \
        --src_dir=$path;

        echo "#Run $j finished";
    done;
done;