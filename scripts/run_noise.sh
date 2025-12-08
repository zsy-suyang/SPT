#!/bin/bash

noise_values=(0.05 0.10 0.15 0.20 0.25 0.30 0.5)
yaml_file="/internfs/zsy/proj/AMFormer/config/run/ours_oxygen.yaml"
# yaml_file="/internfs/zsy/proj/AMFormer/config/run/ours_metal.yaml"

for noise in "${noise_values[@]}"; do
    echo "================ Setting add_noise to $noise ================"
    python -c "import yaml;f='$yaml_file';data=yaml.safe_load(open(f));data['add_noise']=${noise};yaml.safe_dump(data, open(f,'w'))"
    python test.py --config "$yaml_file"
done
