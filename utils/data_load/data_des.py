import random
from torchvision import transforms
import torch
import torch.utils.data as data_utils
import numpy as np
import os
import copy
import csv
import torch
import pickle
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import roc_auc_score
from utils.utils import timer

def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True

setup_seed(42)

def filter_and_return_indices(labels, min_samples=3):
    if isinstance(labels, list):
        labels_series = pd.Series(labels)
    else:
        labels_series = pd.Series(labels)

    label_counts = labels_series.value_counts()
    valid_labels = label_counts[label_counts >= min_samples].index
    valid_bools = labels_series.isin(valid_labels)

    valid_indices = np.where(valid_bools)[0]
    
    return valid_indices

def X_log_normalize(data, min_log_data, max_log_data, log_base=np.e):
    original_structure = [len(sublist) for sublist in data]

    data = [item for sublist in data for item in sublist]
    log_data = np.log(data) if log_base == np.e else np.log10(data)

    normalized_log_data = (log_data - min_log_data) / (max_log_data - min_log_data)

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

class MyDataset(data_utils.Dataset):
    @timer
    def __init__(self, dataset, args):
        self.data_list = []
        self.args = args
        self.metric_name = 'AUC'
        file = args.data_file 

        data_df = pd.read_csv(file, index_col=[0], comment='#')

        data_re = [ele.replace('[', '').replace(']', '').replace('\n', '').replace(',', '').split(' ') for ele in
                   data_df.loc[:, 'XAS_Relative_energy']]
        data_re = [[float(s) for s in sublist if s] for sublist in data_re]
        data_re = X_log_normalize(data_re, min_log_data=args.min_log_data, max_log_data=args.max_log_data, log_base=np.e)

        data_its = [ele.replace('[', '').replace(']', '').replace('\n', '').replace(',', '').split(' ') for ele in
                    data_df.loc[:, 'XAS_Intensity']]
        data_its = [[float(s) for s in sublist if s] for sublist in data_its]

        if args.add_noise != 'None':
            add_noise = [random.uniform(-float(args.add_noise), float(args.add_noise)) for _ in range(256)]
            data_its = [[x + y for x, y in zip(inner_list, add_noise)] for inner_list in data_its]

        data_its = Y_log_normalize(data_its)

        tot_spec_re = np.array(data_re)
        tot_spec_its = np.array(data_its)
        tot_spec_in = np.array([np.ravel(np.column_stack((l1, l2))) for l1, l2 in zip(tot_spec_re, tot_spec_its)])

        ele_dict = args.ele_dict
        # ClS: 'center_atom_type'
        if 'center_atom_type' in data_df.columns: 
            x_center_atom_type = np.array(data_df.loc[:,  'center_atom_type'])
            x_center_atom_type = [ele_dict.get(v, 0) for v in x_center_atom_type] # default 0 if not found
        else:
            x_center_atom_type = [0] * tot_spec_in.shape[0] 
            
        X_raw = np.concatenate((np.array(x_center_atom_type)[:, np.newaxis], tot_spec_in), axis=1)

        
        n_aux = args.aux  
        tot_aux = np.array(data_df.iloc[:, (-n_aux):])
        
        # 判断是否开启多任务模式
        self.is_multi_task = hasattr(args, 'multi_task') and isinstance(args.multi_task, list) and len(args.multi_task) > 0

        labels_dict = {}

        if self.is_multi_task:
            task_list = args.multi_task
        else:
            current_task_name = args.task if isinstance(args.task, str) else args.task['name']
            task_list = [current_task_name]

        print("task_list: ", task_list)
        for t_name in task_list:
            if not hasattr(args, t_name):
                if not self.is_multi_task and isinstance(args.task, dict):
                     t_conf = args.task
                else:
                    raise ValueError(f"Configuration for task '{t_name}' not found in args.")
            else:
                t_conf = getattr(args, t_name)

            raw_col = tot_aux[:, t_conf['idx']]

            if t_conf['mode'] == 'Cls':
                mp = dict(zip(t_conf['value'], t_conf['label']))
                processed_col = np.array([mp.get(v, 0) for v in raw_col])
            else:
                processed_col = raw_col.astype(float)
            
            labels_dict[t_name] = processed_col
        
        stratify_labels = None
        
        if self.is_multi_task:
            strat_task_name = getattr(args, 'stratify_task', None)
            
            if strat_task_name and strat_task_name in labels_dict:
                t_conf = getattr(args, strat_task_name)
                if t_conf['mode'] == 'Cls':
                    stratify_labels = labels_dict[strat_task_name]
                else:
                    print(f"Warning: stratify_task '{strat_task_name}' is Regression. Falling back to random split.")
            else:
                pass
                
        else:
            t_name = task_list[0]
            if not self.is_multi_task and isinstance(args.task, dict):
                 t_conf = args.task
            else:
                 t_conf = getattr(args, t_name)

            if t_conf['mode'] == 'Cls':
                stratify_labels = labels_dict[t_name]
        
        indices = np.arange(len(X_raw))

        if dataset == 'infer':
            train_idx = []
            test_idx = indices
            
        elif dataset in ['train', 'test']:
            if stratify_labels is not None:
                valid_indices = filter_and_return_indices(stratify_labels, min_samples=3)
                indices = indices[valid_indices] 
                stratify_labels = stratify_labels[valid_indices]
            
            if stratify_labels is not None:
                train_idx, test_idx = train_test_split(
                    indices, 
                    test_size=args.test_size, 
                    stratify=stratify_labels, 
                    random_state=args.seed
                )

            else:
                train_idx, test_idx = train_test_split(
                    indices, 
                    test_size=args.test_size, 
                    random_state=args.seed
                )
                

        final_indices = train_idx if dataset == 'train' else test_idx
        
        if len(final_indices) == 0:
            self.cate = np.array([])
            self.cont = np.array([])
            self.label = []
        else:
            text_col_num = 1
            self.cate = X_raw[final_indices, :text_col_num]
            self.cont = X_raw[final_indices, text_col_num:]
            if self.is_multi_task:
                self.label = {}
                for t_name, t_labels in labels_dict.items():
                    self.label[t_name] = t_labels[final_indices]
            else:
                t_name = task_list[0]
                self.label = labels_dict[t_name][final_indices]


    def __getitem__(self, idx):
        cont = copy.deepcopy(self.cont[idx])
        cate = copy.deepcopy(self.cate[idx])
        
        if self.is_multi_task:
            label_dict = {}
            for t_name, t_data in self.label.items():
                val = t_data[idx]
                if hasattr(self.args, t_name):
                    mode = getattr(self.args, t_name)['mode']
                else:
                    mode = 'Cls' if isinstance(val, (int, np.integer)) else 'Reg'
                
                if mode == 'Cls':
                     label_dict[t_name] = torch.tensor(val, dtype=torch.long)
                else:
                     label_dict[t_name] = torch.tensor(val, dtype=torch.float)
            
            return torch.LongTensor(cate), torch.FloatTensor(cont), label_dict
            
        else:
            label = copy.deepcopy(self.label[idx])
            
            if isinstance(self.args.task, dict):
                 mode = self.args.task['mode']
            else:
                 mode = getattr(self.args, self.args.task)['mode']

            if mode == 'Cls':
                 return torch.LongTensor(cate), torch.FloatTensor(cont), torch.LongTensor([label]).squeeze()
            else:
                 return torch.LongTensor(cate), torch.FloatTensor(cont), torch.FloatTensor([label]).squeeze()
    
    def evaluate(self, label, pred):
        return roc_auc_score(label.item(), pred[:,1].item())

    def __len__(self):
        if self.is_multi_task:
            if not self.label: return 0
            first_key = next(iter(self.label))
            return len(self.label[first_key])
        return len(self.label)


if __name__ == '__main__':
    pass