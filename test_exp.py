import os
import json
import yaml
import pickle
import random
import pathlib
import logging
import numpy as np
import pandas as pd
import torch
from typing import List, Tuple, Optional, Any

# Custom imports (Assuming these exist in your project structure)
from config.cfg import BaseConfig
from models.SPT import SPT

# Configure basic logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Set global seeds for reproducibility
SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

def x_log_normalize(data: List[List[float]], 
                    min_log_val: float, 
                    max_log_val: float, 
                    log_base: float = np.e) -> List[List[float]]:
    """
    Log-normalizes a nested list of numerical data (X-axis/Energy).
    """
    original_lengths = [len(sublist) for sublist in data]
    flat_data = [item for sublist in data for item in sublist]
    
    # Apply log transformation
    log_data = np.log(flat_data) if log_base == np.e else np.log10(flat_data)
    
    # Min-max normalization
    epsilon = 1e-10
    normalized_log_data = (log_data - min_log_val) / (max_log_val - min_log_val + epsilon)

    # Restore nested structure
    restored_list = []
    start_idx = 0
    for length in original_lengths:
        end_idx = start_idx + length
        restored_list.append(normalized_log_data[start_idx:end_idx])
        start_idx = end_idx
        
    return restored_list

def y_log_normalize(data: List[List[float]]) -> List[List[float]]:
    """
    Performs intra-sample min-max normalization on nested lists (Y-axis/Intensity).
    """
    normalized_data = []
    epsilon = 1e-10
    
    for sublist in data:
        arr = np.array(sublist)
        min_val = np.min(arr)
        max_val = np.max(arr)
        norm_arr = (arr - min_val) / (max_val - min_val + epsilon)
        normalized_data.append(norm_arr)
        
    return normalized_data

def preprocess_spectra(df: pd.DataFrame, args: Any) -> Tuple[Optional[torch.Tensor], Optional[torch.Tensor], List[str]]:
    """
    Parses, interpolates, and normalizes spectral data from the dataframe.
    """
    energies_list = []
    intensities_list = []
    atom_type_list = []
    spec_name_list = []
    num_points = 256

    logger.info(f"Parsing and interpolating {len(df)} spectra to {num_points} points...")

    for _, row in df.iterrows():
        spec_str = row['XANES_spec_15']
        absorption_atom = row['absorption_atom_type']
        spec_name = row['spec_Name']

        try:
            # Parse string representation of array
            lines = [line.strip().replace('[', '').replace(']', '') 
                     for line in spec_str.strip().split('\n') if line.strip()]
            spec_array = np.array([list(map(float, line.split())) for line in lines])

            # Col 0: Energy, Col 1: Intensity
            raw_energy = spec_array[:, 0]
            raw_intensity = spec_array[:, 1]

            if len(raw_energy) < 2:
                logger.warning(f"Spectrum {spec_name} has fewer than 2 points. Skipping.")
                continue

            # Interpolation
            new_energy = np.linspace(raw_energy.min(), raw_energy.max(), num_points)
            new_intensity = np.interp(new_energy, raw_energy, raw_intensity)

            energies_list.append(new_energy)
            intensities_list.append(new_intensity)
            atom_type_list.append(absorption_atom)
            spec_name_list.append(spec_name)

        except Exception as e:
            logger.warning(f"Failed to parse or interpolate spectrum {spec_name}: {e}")
            continue

    if not energies_list:
        logger.error("No data was successfully processed.")
        return None, None, []

    # Normalization
    norm_energies = x_log_normalize(energies_list, args.min_log_data, args.max_log_data)
    norm_intensities = y_log_normalize(intensities_list)

    # Feature Concatenation
    tot_spec_energies = np.array(norm_energies)
    tot_spec_intensities = np.array(norm_intensities)
    
    # Interleave energy and intensity values
    combined_features = np.array([
        np.ravel(np.column_stack((e, i))) 
        for e, i in zip(tot_spec_energies, tot_spec_intensities)
    ])

    logger.info(f"Combined spectral feature shape: {combined_features.shape}")

    # Process Atom Types
    atom_indices = [args.ele_dict.get(atom) for atom in atom_type_list]
    
    # Construct Final Tensor Input
    X = np.concatenate((np.array(atom_indices)[:, np.newaxis], combined_features), axis=1)

    # Split into Categorical (Atom) and Continuous (Spectrum)
    text_col_num = 1
    cate_data = X[:, :text_col_num]
    cont_data = X[:, text_col_num:]

    logger.info(f"Categorical features shape: {cate_data.shape}")
    logger.info(f"Continuous features shape: {cont_data.shape}")

    return torch.LongTensor(cate_data), torch.FloatTensor(cont_data), spec_name_list

def run_inference(args: Any):
    """
    Main inference pipeline: Load data -> Preprocess -> Load Model -> Predict -> Save.
    """
    # Device Setup
    os.environ['CUDA_VISIBLE_DEVICES'] = str(args.gpu_id)
    device = torch.device(f"cuda:{args.gpu_id}" if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: {device}")

    # Load Data
    logger.info(f"Loading data from {args.data_file}...")
    if args.data_file.endswith(".pkl"):
        # Fix for loading pickles created on different OS (Windows vs Linux)
        pathlib.WindowsPath = pathlib.PosixPath
        with open(args.data_file, "rb") as f:
            df = pickle.load(f)
    else:
        df = pd.read_csv(args.data_file)

    # Preprocessing
    logger.info("Starting data preprocessing...")
    cate_tensor, cont_tensor, valid_spec_names = preprocess_spectra(df, args)

    if cate_tensor is None or cont_tensor is None:
        logger.error("Data preprocessing failed. Exiting.")
        return

    cate_tensor, cont_tensor = cate_tensor.to(device), cont_tensor.to(device)
    logger.info("Preprocessing complete.")

    # Load Model
    logger.info("Loading model...")
    model = SPT(args, device)
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info(f"Model: SPT | Total Parameters: {n_params}")
    
    model.to(device)
    checkpoint = torch.load(args.ckpt, map_location=device)
    model.load_state_dict(checkpoint['net_state_dict'], strict=False)
    logger.info(f"Checkpoint loaded: {args.ckpt}")

    # Inference
    logger.info("Starting inference...")
    model.eval()
    with torch.no_grad():
        batch_size = cate_tensor.shape[0]
        dummy_label = torch.zeros(batch_size, device=device)
        if args.save_metric != 'mse':
            dummy_label = dummy_label.long()
        else:
            dummy_label = dummy_label.float()

        pred, _ = model(cate_tensor, cont_tensor, dummy_label)
    
    logger.info("Inference complete.")

    # Process Results
    pred_cpu = pred.cpu()
    value_map = None

    # Load Label Mapping for Classification Tasks
    if args.save_metric != 'mse':
        yaml_path = './config/datasets/M_descriptors.yaml'
        try:
            with open(yaml_path, 'r', encoding='utf-8') as file:
                feature_map = yaml.safe_load(file)
            
            task_info = feature_map[args.task]
            value_map = dict(zip(task_info['label'], task_info['value']))
            logger.info(f"Label mapping loaded for task '{args.task}'.")
        except Exception as e:
            logger.warning(f"Could not load label mapping from {yaml_path}: {e}")

    # Generate Result Rows
    result_rows = []
    for i, spec_name in enumerate(valid_spec_names):
        row_result = {"spec_Name": spec_name}

        if args.save_metric != 'mse':
            # Classification
            pred_prob = torch.softmax(pred_cpu[i], dim=0).numpy()
            pred_class_idx = torch.argmax(pred_cpu[i]).item()
            predicted_value = value_map.get(pred_class_idx, "Unknown") if value_map else pred_class_idx

            row_result[f"{args.task}_prob"] = json.dumps(pred_prob.tolist())
            row_result[f"{args.task}_pred"] = predicted_value
        else:
            # Regression
            predicted_value = pred_cpu[i][0].item()
            row_result[f"{args.task}_pred"] = predicted_value
        
        result_rows.append(row_result)

    # Save Results
    result_df = pd.DataFrame(result_rows)
    merged_df = pd.merge(df, result_df, on="spec_Name", how="left")

    save_dir = os.path.dirname(args.data_file)
    save_path = os.path.join(save_dir, "exp_Co_pred.csv")

    merged_df.to_csv(save_path, index=False, encoding="utf-8-sig")
    
    # Also save as pickle
    base_name, _ = os.path.splitext(save_path)
    merged_df.to_pickle(f"{base_name}.pkl")

    logger.info(f"Results saved to: {save_path}")
    logger.info(f"Added columns: {args.task}_prob, {args.task}_pred")

if __name__ == '__main__':
    print('================ {:^30s} ================'.format('Loading Config'))
    config = BaseConfig()
    config = config.initialize()
    
    print(f"Checkpoint: {config.ckpt}")
    print(f"Task: {config.task}")
    
    run_inference(config)