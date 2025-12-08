import torch
from config.cfg import BaseConfig
import os
from utils import data_load
import numpy as np
import pandas as pd
from torchmetrics import Accuracy, AUROC, MeanSquaredError, MeanAbsoluteError, R2Score
from torchmetrics.classification import Precision, Recall, F1Score
from models.SPT import SPT
import random
import models
import yaml
import json

random.seed(42) 
np.random.seed(42)

results_dir = "/internfs/zsy/proj/AMFormer/datasets/"
csv_path = os.path.join(results_dir, "metal_site_Li_Co_O_over.csv") 

save_csv = "/internfs/zsy/proj/AMFormer/datasets/metal_site_Li_Co_O_over.csv"
df = pd.read_csv(csv_path) 


def _load_model(args, device):
    model = SPT(args, device)
    # n_parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)
    # print(f'Total number of trainable parameters: {n_parameters}')
    model.cuda()
    
    if not os.path.exists(args.ckpt):
        raise FileNotFoundError(f"Checkpoint file not found at: {args.ckpt}")
        
    state_dict = torch.load(args.ckpt)['net_state_dict']
    model.load_state_dict(state_dict, strict=False)
    
    is_multi_task = hasattr(args, 'multi_task') and isinstance(args.multi_task, list) and len(args.multi_task) > 0
    return model, is_multi_task

def _move_to_device(data, device):
    if isinstance(data, torch.Tensor):
        return data.to(device)
    elif isinstance(data, dict):
        return {k: v.to(device) for k, v in data.items()}
    return data

def _evaluate_metrics(args, is_multi_task, all_preds, all_labels):
    results = {}

    metric_device = all_labels[args.multi_task[0]].device if is_multi_task and all_labels else torch.device('cpu') 
    
    def get_metric_instance(MetricClass, **kwargs):
        return MetricClass(**kwargs).to(metric_device) 

    if is_multi_task:
        print('\n--- Multi-Task Evaluation Summary ---')
        
        for t_name in args.multi_task:
            if t_name not in all_preds or all_preds[t_name] is None:
                continue

            t_preds = all_preds[t_name]
            t_labels = all_labels[t_name]

            t_conf = getattr(args, t_name, {'mode': 'Cls', 'num': t_preds.shape[1]})
            mode = t_conf.get('mode', 'Cls')
            num_classes = t_conf.get('num')
            
            if mode == 'Reg' or t_preds.shape[1] == 1:
                t_preds = t_preds.squeeze(-1)
                
                mse = get_metric_instance(MeanSquaredError)(t_preds, t_labels.float())
                r2 = get_metric_instance(R2Score)(t_preds, t_labels.float())
                mae = get_metric_instance(MeanAbsoluteError)(t_preds, t_labels.float())
                
                results[f'{t_name}_MSE'] = mse.item()
                results[f'{t_name}_R2'] = r2.item()
                results[f'{t_name}_MAE'] = mae.item()
                
                print(f'[{t_name} (Reg)] MSE: {mse.item():.4f}, R2: {r2.item():.4f}, MAE: {mae.item():.4f}')
            else:
                num_classes = t_preds.shape[1]
                
                acc = get_metric_instance(Accuracy, task="multiclass", num_classes=num_classes, top_k=1)(t_preds, t_labels)
                auc = get_metric_instance(AUROC, task="multiclass", num_classes=num_classes)(t_preds, t_labels)
                f1 = get_metric_instance(F1Score, task="multiclass", num_classes=num_classes, average='weighted')(t_preds, t_labels)
                
                results[f'{t_name}_Acc'] = acc.item()
                results[f'{t_name}_AUC'] = auc.item()
                results[f'{t_name}_F1'] = f1.item()
                
                print(f'[{t_name} (Cls)] Acc: {acc.item():.4f}, AUC: {auc.item():.4f}, F1: {f1.item():.4f}')
        
        return results
    else:
        total_test_pred = all_preds
        total_test_label = all_labels

        if args.save_metric != 'mse':
            pre = get_metric_instance(Precision, task="multiclass", num_classes=args.out, average='weighted')(total_test_pred, total_test_label)
            rec = get_metric_instance(Recall, task="multiclass", num_classes=args.out, average='weighted')(total_test_pred, total_test_label)
            f1 = get_metric_instance(F1Score, task="multiclass", num_classes=args.out, average='weighted')(total_test_pred, total_test_label)
            acc = get_metric_instance(Accuracy, task="multiclass", num_classes=args.out, top_k=1)(total_test_pred, total_test_label)
            auc = get_metric_instance(AUROC, task="multiclass", num_classes=args.out)(total_test_pred, total_test_label)

            results = {'Precison': pre.item(), 'Rec': rec.item(), 'F1': f1.item(), 'Acc': acc.item(), 'AUC': auc.item()}
            print('\n--- Single-Task Classification Summary ---')
            print('Precison: {:.4f}, Rec: {:.4f}, acc: {:.4f}, F1: {:.4f}, auc: {:.4f}\n'.format(
                pre.item(), rec.item(), acc.item(), f1.item(), auc.item()))
            
        else:
            total_test_pred = total_test_pred.squeeze(1)
            mse = get_metric_instance(MeanSquaredError)(total_test_pred, total_test_label)
            rmse = torch.sqrt(mse)
            mae = get_metric_instance(MeanAbsoluteError)(total_test_pred, total_test_label)
            R2 = get_metric_instance(R2Score)(total_test_pred, total_test_label)  

            results = {'MSE': mse.item(), 'MAE': mae.item(), 'RMSE': rmse.item(), 'R2': R2.item()}
            print('\n--- Single-Task Regression Summary ---')
            print('MSE: {:.4f}, MAE: {:.4f}, RMSE: {:.4f}, R2: {:.4f} \n'.format(
                mse.item(), mae.item(), rmse.item(), R2.item()))
                
        return results

def testing(args):
    runseed = args.seed
    os.environ['CUDA_VISIBLE_DEVICES'] = str(args.gpu_id)
    device = torch.device("cuda:" + str(args.gpu_id) if torch.cuda.is_available() else "cpu")
    torch.manual_seed(runseed)
    np.random.seed(runseed)

    dataset = getattr(data_load, args.data_name.lower())
    test_data = dataset(dataset=args.dataset_mode, args=args)      # train, test, infer
    test_dataloader = torch.utils.data.DataLoader(
        test_data, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers
    )

    model, is_multi_task = _load_model(args, device)
    
    print(f'============Begin {args.dataset_mode}============')
    test_iter = iter(test_dataloader)
    epoch_size = len(test_dataloader)
    model.eval()
    total_loss = 0.
    
    if is_multi_task:
        all_preds = {t: [] for t in args.multi_task}
        all_labels = {t: [] for t in args.multi_task}
    else:
        all_preds = []
        all_labels = []

    with torch.no_grad():
        for i in range(epoch_size):
            cate, cont, label = next(test_iter)
            cate, cont = _move_to_device(cate, device), _move_to_device(cont, device)
            label = _move_to_device(label, device)

            pred_out, loss_val = model(cate, cont, label)
            total_loss += loss_val.item()

            if is_multi_task:
                for t_name in args.multi_task:
                    if t_name in pred_out:
                        all_preds[t_name].append(pred_out[t_name].cpu())
                        all_labels[t_name].append(label[t_name].cpu())
            else:
                all_preds.append(pred_out.cpu())
                all_labels.append(label.cpu())
    
    total_loss /= epoch_size
    print(f'Testing Finished. Average Loss: {total_loss:.4f}')

    if is_multi_task:
        final_preds = {t: torch.cat(all_preds[t], dim=0) if all_preds[t] else None for t in args.multi_task}
        final_labels = {t: torch.cat(all_labels[t], dim=0) if all_labels[t] else None for t in args.multi_task}
    else:
        final_preds = torch.cat(all_preds, dim=0)
        final_labels = torch.cat(all_labels, dim=0)

    metric_results = _evaluate_metrics(args, is_multi_task, final_preds, final_labels)

    if not is_multi_task and args.save_metric != 'mse':

        total_pred_class = torch.argmax(torch.softmax(final_preds, dim=1), dim=1).cpu().numpy()
        
        try:
            with open('./config/datasets/M_descriptors.yaml', 'r', encoding='utf-8') as file:
                feature_map = yaml.safe_load(file)
                value_map = {label: value for label, value in zip(feature_map[args.task]['label'], feature_map[args.task]['value'])}
            total_pred_class = np.vectorize(value_map.get)(total_pred_class)
            
            # 使用全局 df 变量进行保存（如果需要启用，请取消注释）
            # global df
            # df[f'SPT_{args.task}_Pred'] = total_pred_class
            # df[f'SPT_{args.task}_SCORE'] = [ [x for x in row] for row in final_preds.cpu().numpy()]
            # df.to_csv(save_csv, index=False) 
        except Exception as e:
            print(f"Warning: Failed to perform YAML mapping and CSV saving for single-task classification. Error: {e}")
            
    # --- CSV 保存 (单任务回归部分) ---
    elif not is_multi_task and args.save_metric == 'mse':
        # 仅针对单任务回归模式执行原有 CSV 保存逻辑（如果需要启用，请取消注释）
        # global df
        # df[f'SPT_{args.task}_Pred'] = final_preds.squeeze(1).cpu().numpy()
        # df.to_csv(save_csv, index=False) 
        pass


if __name__ == '__main__':
    print('================ {:^30s} ================'.format('Loading Config'))
    args = BaseConfig()
    args = args.initialize()
    
    is_mt = hasattr(args, 'multi_task') and isinstance(args.multi_task, list) and len(args.multi_task) > 0
    mode_str = "Multi-Task" if is_mt else "Single-Task"
    if is_mt:
        print(f"Running Mode: {mode_str} ({', '.join(args.multi_task)})")
    else:
        print(f"Running Mode: {mode_str} (Task: {args.task})")
        
    testing(args)