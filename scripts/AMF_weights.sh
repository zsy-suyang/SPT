#!/bin/bash

tasks=("OS" "CN" "OCN" "CN2" "PSGO" "NNRS" "MOOD")

yaml_file="/internfs/zsy/proj/AMFormer/config/run/ours_mental.yaml"

for task in "${tasks[@]}"; do
    echo "================ Setting task to $task ================"
    yq eval ".task = \"$task\"" "$yaml_file" -
    echo "Updated YAML file:"
    cat "$yaml_file"

    case $task in
        "OS")
            aux=8
            save_metric="acc"
            ckpt_path="/internfs/zsy/proj/AMFormer/Metal_AMF_result/save_model/mental/OS/TEST-train_size=1/ver0/265.pth"
            out=5
            ;;
        "CN")
            aux=8
            save_metric="acc"
            ckpt_path="/internfs/zsy/proj/AMFormer/Metal_AMF_result/save_model/mental/CN/TEST-train_size=1/ver1/228.pth"
            out=7
            ;;
        "OCN")
            aux=8
            save_metric="acc"
            ckpt_path="/internfs/zsy/proj/AMFormer/result/save_model/mental/OCN/TEST-train_size=1/ver1/227.pth"
            out=6
            ;;
        "CN2")
            aux=8
            save_metric="acc"
            ckpt_path="/internfs/zsy/proj/AMFormer/Metal_AMF_result/save_model/mental/OCN/TEST-train_size=1/ver1/298.pth"
            out=25
            ;;
        "PSGO")
            aux=8
            save_metric="acc"
            ckpt_path="/internfs/zsy/proj/AMFormer/Metal_AMF_result/save_model/mental/PSGO/TEST-train_size=1/ver1/391.pth"
            out=68
            ;;
        "NNRS")
            aux=8
            save_metric="mse"
            ckpt_path="/internfs/zsy/proj/AMFormer/Metal_AMF_result/save_model/mental/NNRS/TEST-train_size=1/ver3/491.pth"
            out=1
            ;;
        "MOOD")
            aux=8
            save_metric="mse"
            ckpt_path="/internfs/zsy/proj/AMFormer/Metal_AMF_result/save_model/mental/MOOD/TEST-train_size=1/ver1/ft_trans_best_loss_0.00019638075900729746.pth"
            ckpt_path="/internfs/zsy/proj/AMFormer/Metal_AMF_result/save_model/mental/MOOD/TEST-train_size=1/ver1/ft_trans_best_loss_0.00019638075900729746.pth"
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