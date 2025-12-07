import torch
from config.cfg import BaseConfig
import os
import numpy as np
import pandas as pd
from models.SPT import SPT  
import random
import yaml
import copy
import json
import pickle
import pathlib

random.seed(42)
np.random.seed(42)
torch.manual_seed(42)

def X_log_normalize(data, min_log_data, max_log_data, log_base=np.e):
    original_structure = [len(sublist) for sublist in data]
    data = [item for sublist in data for item in sublist]
    log_data = np.log(data) if log_base == np.e else np.log10(data)
    
    epsilon = 1e-10 
    normalized_log_data = (log_data - min_log_data) / (max_log_data - min_log_data + epsilon)

    restored_nested_list = []
    start = 0
    for length in original_structure:
        end = start + length
        restored_nested_list.append(normalized_log_data[start:end])
        start = end
    return restored_nested_list

def Y_log_normalize(data, log_base=None):
    data = [sublist  for sublist in data]
    normalized_data = []
    for sublist in data:
        min_values = np.min(sublist)
        max_values = np.max(sublist)
        epsilon = 1e-10
        tmp = (sublist - min_values) / (max_values - min_values + epsilon)
        normalized_data.append(tmp)
    return normalized_data


def process_batch_data(df, args):
    data_re_list = []
    data_its_list = []
    atom_list = []
    spec_name_list = [] 
    num_points = 256
    
    print(f"1. 正在解析和插值 {len(df)} 条光谱数据至 {num_points} 个点...")
    
    for index, row in df.iterrows():
        spec_str = row['XANES_spec_15']  # 能量截断 !!!!!!!!
        absorption_atom = row['absorption_atom_type']
        spec_name = row['spec_Name']
        
        try:
            lines = [line.strip().replace('[', '').replace(']', '') 
                     for line in spec_str.strip().split('\n') if line.strip()]
            spec_array = np.array([list(map(float, line.split())) for line in lines])
            
            # --- 关键改动 1: 使用第0列(能量)和第1列(强度) ---
            XAS_Relative_energy = spec_array[:, 0]
            XAS_Intensity = spec_array[:, 1]  
            
            # --- 关键改动 2: 执行插值 ---
            if len(XAS_Relative_energy) < 2:
                print(f"!! 警告: 光谱 {spec_name} 点数少于2，无法插值，已跳过。")
                continue
                
            XAS_Relative_energy_new = np.linspace(XAS_Relative_energy.min(), XAS_Relative_energy.max(), num_points)
            XAS_Intensity_new = np.interp(XAS_Relative_energy_new, XAS_Relative_energy, XAS_Intensity)
            
            # 存储插值后的数据
            data_re_list.append(XAS_Relative_energy_new)
            data_its_list.append(XAS_Intensity_new) 
            atom_list.append(absorption_atom)
            spec_name_list.append(spec_name) 
            
        except Exception as e:
            print(f"!! 警告: 解析或插值光谱 {spec_name} 失败，已跳过。错误: {e}")
            continue

    if not data_re_list:
        print("!! 错误: 没有数据被成功处理。")
        return None, None, None 
    

    print("3. 正在全局归一化能量 (X_log_normalize)...")
    # 3. 归一化能量（全局）
    data_re_norm_list = X_log_normalize(data_re_list, args.min_log_data, args.max_log_data, log_base=np.e)
    # if args.add_noise != 'None':
    #     print("!!!!!!!!!!!111")
    #     add_noise = [random.uniform(float(args.add_noise), float(args.add_noise)) for _ in range(256)]
    #     data_re_norm_list = [[x + y for x, y in zip(inner_list, add_noise)] for inner_list in data_re_norm_list]
    
    print("4. 正在逐个归一化强度 (Y_log_normalize)...")
    # 4. 归一化强度（逐个）
    data_its_norm_list = Y_log_normalize(data_its_list)

    print("5. 正在组合特征...")
    # 5. 组合特征
    tot_spec_re = np.array(data_re_norm_list)
    tot_spec_its = np.array(data_its_norm_list)
    
    # 交叉组合 (E1, I1, E2, I2, ...)
    tot_spec_in = np.array([np.ravel(np.column_stack((l1, l2))) for l1, l2 in zip(tot_spec_re, tot_spec_its)])
    
    print(f"   - 组合后的光谱特征维度: {tot_spec_in.shape}")

    # 6. 处理类别特征 (原子类型)
    ele_dict = args.ele_dict
    atom_indices = [ele_dict.get(atom) for atom in atom_list] 
    print(f"6. 正在处理原子类型...")

    # 7. 组合类别和连续特征
    X = np.concatenate((np.array(atom_indices)[:, np.newaxis], tot_spec_in), axis=1)

    # 8. 分离 cate 和 cont
    text_col_num = 1
    cate_data = X[:, :text_col_num]
    cont_data = X[:, text_col_num:]

    print(f"   - 类别特征 (cate) 维度: {cate_data.shape}")
    print(f"   - 连续特征 (cont) 维度: {cont_data.shape}")

    # 9. 转换为 Tensors
    cate_tensor = torch.LongTensor(cate_data)
    cont_tensor = torch.FloatTensor(cont_data)

    return cate_tensor, cont_tensor, spec_name_list 

def testing(args):
    runseed = args.seed
    os.environ['CUDA_VISIBLE_DEVICES'] = str(args.gpu_id)
    device = torch.device("cuda:" + str(args.gpu_id) if torch.cuda.is_available() else "cpu")
    torch.manual_seed(runseed)
    np.random.seed(runseed)
    
    print(f"设备: {device}")
    
    csv_path = args.data_file
    print(f"--- 正在从 {csv_path} 加载数据 ---")
    if csv_path.endswith(".pkl"):
        pathlib.WindowsPath = pathlib.PosixPath
        with open(csv_path, "rb") as f:
            df = pickle.load(f)
    else:
        df = pd.read_csv(csv_path)


    print("--- 开始数据预处理 ---")
    cate, cont, valid_spec_name_list = process_batch_data(df, args)
    
    if cate is None or cont is None:
        print("!!! 错误: 数据预处理失败，没有有效数据。正在退出。 !!!")
        return

    cate, cont = cate.to(device), cont.to(device)
    print("--- 数据预处理完成 ---")

    # --- 3. 加载模型 ---
    print("\n--- 加载模型 ---")
    model = SPT(args, device)
    n_parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"模型: SPT, 总参数量: {n_parameters}")
    model.to(device)
    
    state_dict = torch.load(args.ckpt, map_location=device)['net_state_dict']
    model.load_state_dict(state_dict, strict=False)
    print(f"成功加载检查点: {args.ckpt}")


    print("\n--- 开始推理 ---")
    model.eval()
    with torch.no_grad():
        batch_size = cate.shape[0]
        if args.save_metric == 'mse': 
            dummy_label = torch.zeros(batch_size, device=device).float()
        else: 
            dummy_label = torch.zeros(batch_size, device=device).long()

        pred, loss = model(cate, cont, dummy_label)
        
    print("--- 推理完成 ---")

    print(f"\n--- {len(valid_spec_name_list)} 个样本的预测结果 ---")
    
    pred_cpu = pred.cpu()

    value_map = None
    if args.save_metric != 'mse':
        try:
            yaml_path = './config/datasets/M_descriptors.yaml'
            with open(yaml_path, 'r', encoding='utf-8') as file:
                feature_map = yaml.safe_load(file)
            
            task_info = feature_map[args.task]

            value_map = {label: value for label, value in zip(task_info['label'], task_info['value'])}
            print(f"成功加载任务 '{args.task}' 的标签映射。")
        except Exception as e:
            print(f"(警告: 无法加载或解析 {yaml_path} 进行标签映射: {e})")


    result_rows = []
    for i in range(len(valid_spec_name_list)):
        spec_name = valid_spec_name_list[i]
        row_result = {"spec_Name": spec_name}

        if args.save_metric != 'mse':  
            pred_prob = torch.softmax(pred_cpu[i], dim=0).numpy()
            pred_class_idx = torch.argmax(pred_cpu[i]).item()
            predicted_value = value_map.get(pred_class_idx, "未知标签") if value_map else pred_class_idx

            row_result[f"{args.task}_prob"] = json.dumps(pred_prob.tolist())
            row_result[f"{args.task}_pred"] = predicted_value

        else:  
            predicted_value = pred_cpu[i][0].item()
            row_result[f"{args.task}_pred"] = predicted_value
        
        result_rows.append(row_result)

    result_df = pd.DataFrame(result_rows)

    merged_df = pd.merge(df, result_df, on="spec_Name", how="left")

    save_dir = os.path.dirname(csv_path)
    save_path = os.path.join(save_dir, f"exp_Co_pred.csv")

    merged_df.to_csv(save_path, index=False, encoding="utf-8-sig")
    print(f"\n✅ 已将预测结果保存至: {save_path}")
    print(f"  新增列: {args.task}_prob, {args.task}_pred")
    base, _ = os.path.splitext(save_path)
    pkl_path = base + ".pkl"
    merged_df.to_pickle(pkl_path)



if __name__ == '__main__':
    print('================ {:^30s} ================'.format('加载配置 (Loading Config)'))
    args = BaseConfig()
    args = args.initialize()
    
    print(f"使用模型检查点: {args.ckpt}")
    print(f"使用任务: {args.task}")
    
    testing(args)