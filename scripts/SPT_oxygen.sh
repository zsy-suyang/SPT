#!/bin/bash

tasks=("CN"  "CN2" "PSGO" "NNRS" "NNOSM")

yaml_file="/internfs/zsy/proj/AMFormer/config/run/ours_oxygen.yaml"

for task in "${tasks[@]}"; do
    echo "================ Setting task to $task ================"
    yq eval ".task = \"$task\"" "$yaml_file" -i
    echo "Updated YAML file:"
    cat "$yaml_file"
    
    case $task in
        "CN")
            save_metric="acc"
            ckpt_path="/internfs/zsy/proj/AMFormer/Oxygen_SPT_t1_result/save_model/metal/CN/TEST-train_size=1/ver0/212.pth"
            out=6
            ;;
        "CN2")
            save_metric="acc"
            ckpt_path="/internfs/zsy/proj/AMFormer/Oxygen_SPT_t1_result/save_model/metal/CN2/TEST-train_size=1/ver0/113.pth"
            out=28
            ;;
        "PSGO")
            save_metric="acc"
            ckpt_path="/internfs/zsy/proj/AMFormer/Oxygen_SPT_t1_result/save_model/metal/PSGO/TEST-train_size=1/ver0/899.pth"
            out=62
            ;;
        "NNRS")
            save_metric="mse"
            ckpt_path="/internfs/zsy/proj/AMFormer/Oxygen_SPT_t1_result/save_model/metal/NNRS/TEST-train_size=1/ver0/790.pth"
            out=1
            ;;
        "NNOSM")
            save_metric="mse"
            ckpt_path="/internfs/zsy/proj/AMFormer/Oxygen_SPT_t1_result/save_model/metal/NNOSM/TEST-train_size=1/ver0/857.pth"
            out=1
            ;;
        *)
            echo "Unknown task: $task"
            continue
            ;;
    esac
    
    yq eval ".ckpt = \"$ckpt_path\" | .out = $out | .save_metric = \"$save_metric\"" "$yaml_file" -i
    
    python test.py --config "$yaml_file"
done