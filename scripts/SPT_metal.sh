#!/bin/bash

tasks=("OS" "CN" "OCN" "CN2" "PSGO" "NNRS" "MOOD")

yaml_file="/internfs/zsy/proj/AMFormer/config/run/ours_metal.yaml"

for task in "${tasks[@]}"; do
    echo "================ Setting task to $task ================"
    yq eval ".task = \"$task\"" "$yaml_file" -i
    echo "Updated YAML file:"
    cat "$yaml_file"
    
    case $task in
        "OS")
            aux=8
            save_metric="acc"
            ckpt_path="/internfs/zsy/proj/AMFormer/Metal_SPT_t1_result/save_model/metal/OS/TEST-train_size=1/ver0/806.pth"
            out=5
            ;;
        "CN")
            aux=8
            save_metric="acc"
            ckpt_path="/internfs/zsy/proj/AMFormer/Metal_SPT_t1_result/save_model/metal/CN/TEST-train_size=1/ver0/268.pth"
            out=7
            ;;
        "OCN")
            aux=8
            save_metric="acc"
            ckpt_path="/internfs/zsy/proj/AMFormer/Metal_SPT_t1_result/save_model/metal/OCN/TEST-train_size=1/ver0/86.pth"
            out=6
            ;;
        "CN2")
            aux=8
            save_metric="acc"
            ckpt_path="/internfs/zsy/proj/AMFormer/Metal_SPT_t1_result/save_model/metal/CN2/TEST-train_size=1/ver0/332.pth"
            out=25
            ;;
        "PSGO")
            aux=8
            save_metric="acc"
            ckpt_path="/internfs/zsy/proj/AMFormer/Metal_SPT_t1_result/save_model/metal/PSGO/TEST-train_size=1/ver0/389.pth"
            out=68
            ;;
        "NNRS")
            aux=8
            save_metric="mse"
            ckpt_path="/internfs/zsy/proj/AMFormer/Metal_SPT_t1_result/save_model/metal/NNRS/TEST-train_size=1/ver0/975.pth"
            out=1
            ;;
        "MOOD")
            aux=8
            save_metric="mse"
            ckpt_path="/internfs/zsy/proj/AMFormer/Metal_SPT_t1_result/save_model/metal/MOOD/TEST-train_size=1/ver0/902.pth"
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