import torch
from spt.config.cfg import parse_config
import os
from spt.utils import data_load
import numpy as np
import pandas as pd
from torchmetrics import Accuracy, AUROC, MeanSquaredError, MeanAbsoluteError, R2Score
from torchmetrics.classification import Precision, Recall, F1Score
import random
from spt import models
import yaml
import json
import time
from spt.utils.runtime import resolve_device

def _load_model(args, device):
    model = models.build_model(args, device)
    model.to(device)
    
    if not os.path.exists(args.ckpt):
        raise FileNotFoundError(f"Checkpoint file not found at: {args.ckpt}")
        
    state_dict = torch.load(args.ckpt, map_location=device)['net_state_dict']
    model.load_state_dict(
        state_dict, strict=bool(getattr(args, "ckpt_strict", True))
    )
    
    is_multi_task = hasattr(args, 'multi_task') and isinstance(args.multi_task, list) and len(args.multi_task) > 0
    return model, is_multi_task

def _move_to_device(data, device):
    if isinstance(data, torch.Tensor):
        return data.to(device)
    elif isinstance(data, dict):
        return {k: v.to(device) for k, v in data.items()}
    return data

def _single_task_mode(args):
    if isinstance(args.task, dict):
        return args.task['mode']
    return getattr(args, args.task)['mode']

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
            
            if mode == 'Reg' or t_preds.shape[1] == 1:
                t_preds = t_preds.squeeze(-1)
                t_labels = t_labels.float().reshape(-1)
                valid = torch.isfinite(t_labels)
                if valid.sum() < 2:
                    raise ValueError(
                        f"Fewer than two valid evaluation rows for {t_name}"
                    )
                t_preds, t_labels = t_preds[valid], t_labels[valid]
                if not torch.isfinite(t_preds).all():
                    raise FloatingPointError(
                        f"Non-finite evaluation predictions for {t_name}"
                    )
                
                mse = get_metric_instance(MeanSquaredError)(t_preds, t_labels.float())
                r2 = get_metric_instance(R2Score)(t_preds, t_labels.float())
                mae = get_metric_instance(MeanAbsoluteError)(t_preds, t_labels.float())
                
                results[f'{t_name}_MSE'] = mse.item()
                results[f'{t_name}_RMSE'] = torch.sqrt(mse).item()
                results[f'{t_name}_R2'] = r2.item()
                results[f'{t_name}_MAE'] = mae.item()
                results[f'{t_name}_N'] = int(valid.sum().item())
                
                print(f'[{t_name} (Reg)] MSE: {mse.item():.4f}, R2: {r2.item():.4f}, MAE: {mae.item():.4f}')
            else:
                num_classes = t_preds.shape[1]
                t_labels = t_labels.reshape(-1).long()
                valid = (t_labels >= 0) & (t_labels < num_classes)
                if not valid.any():
                    raise ValueError(f"No valid evaluation rows for {t_name}")
                t_preds, t_labels = t_preds[valid], t_labels[valid]
                if not torch.isfinite(t_preds).all():
                    raise FloatingPointError(
                        f"Non-finite evaluation logits for {t_name}"
                    )
                
                acc = get_metric_instance(Accuracy, task="multiclass", num_classes=num_classes, top_k=1)(t_preds, t_labels)
                auc = get_metric_instance(AUROC, task="multiclass", num_classes=num_classes)(t_preds, t_labels)
                pre = get_metric_instance(
                    Precision, task="multiclass", num_classes=num_classes,
                    average=getattr(args, 'f1_average', 'weighted'),
                )(t_preds, t_labels)
                rec = get_metric_instance(
                    Recall, task="multiclass", num_classes=num_classes,
                    average=getattr(args, 'f1_average', 'weighted'),
                )(t_preds, t_labels)
                f1 = get_metric_instance(
                    F1Score,
                    task="multiclass",
                    num_classes=num_classes,
                    average=getattr(args, 'f1_average', 'weighted'),
                )(t_preds, t_labels)
                
                results[f'{t_name}_Acc'] = acc.item()
                results[f'{t_name}_AUC'] = auc.item()
                results[f'{t_name}_Precision'] = pre.item()
                results[f'{t_name}_Recall'] = rec.item()
                results[f'{t_name}_F1'] = f1.item()
                results[f'{t_name}_N'] = int(valid.sum().item())
                
                print(f'[{t_name} (Cls)] Acc: {acc.item():.4f}, AUC: {auc.item():.4f}, F1: {f1.item():.4f}')

        primary = []
        for t_name in args.multi_task:
            t_conf = getattr(args, t_name)
            key = f"{t_name}_{'R2' if t_conf['mode'] == 'Reg' else 'F1'}"
            if key not in results:
                raise KeyError(f"Missing Joint primary metric {key}")
            primary.append(float(results[key]))
        if not primary or not np.isfinite(primary).all():
            raise ValueError("Joint evaluation primary metrics must be finite")
        results['score'] = float(np.mean(primary))
        
        return results
    else:
        total_test_pred = all_preds
        total_test_label = all_labels
        # print("total_test_pred: ", total_test_pred)
        # print("total_test_label: ", total_test_label)

        if _single_task_mode(args) == 'Cls':
            f1_average = getattr(args, 'f1_average', 'weighted')
            pre = get_metric_instance(Precision, task="multiclass", num_classes=args.out, average=f1_average)(total_test_pred, total_test_label)
            rec = get_metric_instance(Recall, task="multiclass", num_classes=args.out, average=f1_average)(total_test_pred, total_test_label)
            f1 = get_metric_instance(F1Score, task="multiclass", num_classes=args.out, average=f1_average)(total_test_pred, total_test_label)
            acc = get_metric_instance(Accuracy, task="multiclass", num_classes=args.out, top_k=1)(total_test_pred, total_test_label)
            auc = get_metric_instance(AUROC, task="multiclass", num_classes=args.out)(total_test_pred, total_test_label)

            results = {'Precision': pre.item(), 'Recall': rec.item(), 'F1': f1.item(), 'Acc': acc.item(), 'AUC': auc.item()}
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
    from types import SimpleNamespace
    from spt.utils.execution import inference_backend

    device = resolve_device(args)
    # General evaluation applies saved arithmetic settings without requiring
    # the publication-reference replay device.
    settings = SimpleNamespace(inference_backend=getattr(args, 'inference_backend', None))
    with inference_backend(settings, device) as backend:
        return _testing(args, device, backend)


def _testing(args, device, backend):
    if getattr(args, 'inference_only', False):
        raise ValueError('Use spt application for this prepared application cohort; it is not a computed benchmark split.')
    runseed = args.seed
    evaluation_batch_size = getattr(args, 'eval_batch_size', args.batch_size)
    torch.manual_seed(runseed)
    np.random.seed(runseed)
    random.seed(runseed)

    # 1. 加载模型
    model, is_multi_task = _load_model(args, device)
    print("is_multi_task: ", is_multi_task)
    model.eval()

    # 定义数据加载器类
    DatasetClass = getattr(data_load, args.data_name.lower())

    # ==========================================
    # === [控制开关] 是否需要推理训练集 ===
    # ==========================================
    # 从 args 中获取 'calc_train_set'，如果没有设置则默认为 False
    need_train_eval = getattr(args, 'calc_train_set', False)

    # 2. 准备 Test DataLoader (必须执行)
    test_data = DatasetClass(dataset=args.dataset_mode, args=args) 
    if not len(test_data):
        raise ValueError(f"No eligible rows in the requested {args.dataset_mode!r} evaluation split")
    test_dataloader = torch.utils.data.DataLoader(
        test_data, batch_size=evaluation_batch_size, shuffle=False, num_workers=args.num_workers
    )

    # 3. 准备 Train DataLoader (按需执行)
    if need_train_eval:
        print("\n[Info] 'calc_train_set' is True. Loading Training Data...")
        train_data = DatasetClass(dataset='train', args=args) 
        train_dataloader = torch.utils.data.DataLoader(
            train_data, batch_size=evaluation_batch_size, shuffle=False, num_workers=args.num_workers
        )
    else:
        print("\n[Info] 'calc_train_set' is False. Skipping Training Data inference.")

    # --- 内部函数：通用推理逻辑 ---
    def run_inference(loader, data_name="Test"):
        print(f'============ Begin Inference on {data_name} Set ============')
        
        if is_multi_task:
            local_preds = {t: [] for t in args.multi_task}
            local_labels = {t: [] for t in args.multi_task}
        else:
            local_preds = []
            local_labels = []

        if device.type == "cuda":
            torch.cuda.synchronize()
        start_time = time.time()
        
        with torch.no_grad():
            for i, (cate, cont, label) in enumerate(loader):
                cate, cont = _move_to_device(cate, device), _move_to_device(cont, device)
                label = _move_to_device(label, device)

                pred_out, _ = model(cate, cont, label)
                
                if is_multi_task:
                    for t_name in args.multi_task:
                        if t_name in pred_out:
                            local_preds[t_name].append(pred_out[t_name].cpu())
                            local_labels[t_name].append(label[t_name].cpu())
                else:
                    local_preds.append(pred_out.cpu())
                    local_labels.append(label.cpu())

        if device.type == "cuda":
            torch.cuda.synchronize()
        end_time = time.time()
        
        print(f"Inference Time ({data_name}): {end_time - start_time:.4f} s")
        
        if is_multi_task:
            final_p = {t: torch.cat(local_preds[t], dim=0) if local_preds[t] else None for t in args.multi_task}
            final_l = {t: torch.cat(local_labels[t], dim=0) if local_labels[t] else None for t in args.multi_task}
        else:
            final_p = torch.cat(local_preds, dim=0)
            final_l = torch.cat(local_labels, dim=0)
            
        return final_p, final_l

    # 4. 执行推理
    # 4.1 Test 集 (始终执行)
    test_preds, test_labels = run_inference(test_dataloader, data_name="Test")

    # 4.2 Train 集 (按需执行)
    train_preds, train_labels = None, None
    if need_train_eval:
        train_preds, train_labels = run_inference(train_dataloader, data_name="Train")

    # 5. 计算指标 (Test 集)
    print("\n>>> Calculating Metrics for Test Set...")
    metric_results = _evaluate_metrics(args, is_multi_task, test_preds, test_labels)
    metrics_output = getattr(args, "metrics_output", None)
    if metrics_output not in [None, "None"]:
        metrics_path = os.path.abspath(metrics_output)
        os.makedirs(os.path.dirname(metrics_path), exist_ok=True)
        metrics_payload = metric_results
        if args.dataset_mode == "lockbox":
            metrics_payload = {
                "split": "lockbox",
                "checkpoint": os.path.abspath(args.ckpt),
                "metrics": metric_results,
            }
        with open(metrics_path, "w", encoding="utf-8") as handle:
            json.dump(metrics_payload, handle, indent=2)
        print(f"Saved metrics: {metrics_path}")
        with open(os.path.splitext(metrics_path)[0] + '.execution.json', 'w', encoding='utf-8') as handle:
            json.dump({'device': str(device), 'batch_size': evaluation_batch_size,
                       'inference_backend': backend}, handle, indent=2)
    prediction_output_dir = getattr(args, "prediction_output_dir", None)
    if prediction_output_dir not in [None, "None"]:
        prediction_dir = os.path.abspath(prediction_output_dir)
        os.makedirs(prediction_dir, exist_ok=True)
        row_indices = np.asarray(test_data.indices)
        task_names = args.multi_task if is_multi_task else [args.task]
        for task_name in task_names:
            logits = test_preds[task_name] if is_multi_task else test_preds
            labels = test_labels[task_name] if is_multi_task else test_labels
            task_config = getattr(args, task_name)
            task_rows = row_indices
            if is_multi_task:
                if task_config["mode"] == "Cls":
                    valid = (
                        (labels.reshape(-1) >= 0)
                        & (labels.reshape(-1) < logits.shape[1])
                    ).numpy()
                else:
                    valid = torch.isfinite(labels.reshape(-1)).numpy()
                task_rows = row_indices[valid]
                logits = logits[valid]
                labels = labels[valid]
                if not torch.isfinite(logits).all():
                    raise FloatingPointError(
                        f"Non-finite evaluation predictions for {task_name}"
                    )
            if task_config["mode"] == "Cls":
                probabilities = torch.softmax(logits, dim=1).numpy()
                prediction = probabilities.argmax(axis=1)
                frame = pd.DataFrame(
                    {
                        "row_index": task_rows,
                        "label": labels.numpy(),
                        "prediction": prediction,
                        "confidence": probabilities.max(axis=1),
                    }
                )
                for class_index in range(probabilities.shape[1]):
                    frame[f"probability_{class_index}"] = probabilities[:, class_index]
            else:
                frame = pd.DataFrame(
                    {
                        "row_index": task_rows,
                        "label": labels.numpy(),
                        "prediction": logits.squeeze(-1).numpy(),
                    }
                )
            output_path = os.path.join(
                prediction_dir, f"{task_name}_predictions.csv"
            )
            frame.to_csv(output_path, index=False)
            print(f"Saved predictions: {output_path}")
    
    # 6. 保存结果 (.dat)
    if getattr(args, 'save_dat', False):
        save_dir_base = getattr(
            args,
            "dat_output_dir",
            os.path.join(getattr(args, "project_root", os.getcwd()), "runs", "evaluation", "arrays"),
        )
        os.makedirs(save_dir_base, exist_ok=True)
        print(f"\n[Info] 'save_dat' is True. Saving inference results to {save_dir_base} ...")

        if is_multi_task:
            for t_name in args.multi_task:
                if test_preds[t_name] is not None:
                    # 获取该任务的训练集结果（如果没跑就是 None）
                    cur_train_p = train_preds[t_name] if (need_train_eval and train_preds) else None
                    cur_train_l = train_labels[t_name] if (need_train_eval and train_labels) else None

                    save_dict = {
                        'total_test_pred': test_preds[t_name].cpu(),
                        'total_test_label': test_labels[t_name].cpu(),
                        'total_train_pred': cur_train_p.cpu() if cur_train_p is not None else None,
                        'total_train_label': cur_train_l.cpu() if cur_train_l is not None else None
                    }
                    save_path = os.path.join(save_dir_base, f"SPT_{t_name}_results.dat")
                    torch.save(save_dict, save_path)
                    print(f" -> Saved: {save_path}")
        else:
            save_dict = {
                'total_test_pred': test_preds.cpu(),
                'total_test_label': test_labels.cpu(),
                # 如果没有跑训练集，这里存入的就是 None，不会报错
                'total_train_pred': train_preds.cpu() if train_preds is not None else None,
                'total_train_label': train_labels.cpu() if train_labels is not None else None,
            }
            save_path = os.path.join(save_dir_base, f"SPT_{args.task}_results.dat")
            torch.save(save_dict, save_path)
            print(f" -> Saved: {save_path}")
    else:
        print("\n[Info] 'save_dat' is False or not set. Skipping .dat file saving.")
    return metric_results

def cli(argv=None):
    args = parse_config(
        argv, evaluation=True,
        description="Evaluate a selected SPT checkpoint (test split by default)",
    )
    return testing(args)


if __name__ == '__main__':
    cli()
