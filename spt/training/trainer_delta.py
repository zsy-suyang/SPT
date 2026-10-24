import os
import torch
import torch.nn as nn
from spt import models
import math
from tensorboardX import SummaryWriter
from spt.config.cfg import arg2str
from torchmetrics import Accuracy, AUROC, MeanSquaredError, MeanAbsoluteError, R2Score
from torchmetrics.classification import Precision, Recall, F1Score
from torch.cuda.amp import autocast, GradScaler
import json
import numpy as np
import pandas as pd

class ModelEMA:
    """Exponential moving average of model weights (timm / Mean Teacher)."""

    def __init__(self, model, decay=0.999):
        self.decay = float(decay)
        self.shadow = {
            key: value.detach().clone()
            for key, value in model.state_dict().items()
        }

    @torch.no_grad()
    def update(self, model):
        for key, value in model.state_dict().items():
            if not torch.is_floating_point(value):
                self.shadow[key].copy_(value)
                continue
            self.shadow[key].mul_(self.decay).add_(
                value.detach(), alpha=1.0 - self.decay
            )

    def state_dict(self):
        return self.shadow

    @torch.no_grad()
    def copy_to(self, model):
        model.load_state_dict(self.shadow, strict=True)


class DefaultTrainer(object):

    def __init__(self, args, device=None):
        self.args = args
        self.device = device or torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        )
        self.task = args.task
        self.batch_size = args.batch_size
        
        self.lr = args.lr
        self.lr_current = args.lr
        
        self.epoch = args.epoch
        self.max_iter = args.max_iter
        self.start_iter = args.start_iter
        self.warmup_steps = args.warmup_steps

        self.model = models.build_model(args, self.device)

        self.init_writer()

        self.flag = 0
        self.model.to(self.device)
        self.scaler = GradScaler(enabled=self.device.type == "cuda")
        ema_decay = float(getattr(args, 'ema_decay', 0.0) or 0.0)
        self.ema = (
            ModelEMA(self.model, decay=ema_decay)
            if ema_decay > 0.0 else None
        )
        self.mixup_alpha = float(getattr(args, 'mixup_alpha', 0.0) or 0.0)
        self.mixup_steps = int(getattr(args, 'mixup_steps', 0) or 0)

        self.is_multi_task = hasattr(args, 'multi_task') and isinstance(args.multi_task, list) and len(args.multi_task) > 0
        self.f1_average = getattr(args, 'f1_average', 'weighted')
        self.best_checkpoint_path = None
        self.best_selection = None
        self.final_checkpoint_path = None
        self.last_selection = None

        self.top_k = 1 
        self.best_records = {
            'loss': [],  
            'f1':   [], 
            'r2':   [],
            'score': []
        }
        
        self.metrics = {
            'acc': ['high', 0], 'auc': ['high', 0], 'f1': ['high', 0], 'r2': ['high', -100],
            'loss': ['low', 10000], 'mse': ['low', 10000], 'mae': ['low', 10000], 'score': ['high', 0]
        }
        
        self.start = 0
        self.wrong = None
        task_prefix = 'MultiTask' if self.is_multi_task else str(self.task)
        self.log_path = os.path.join(self.args.save_folder, task_prefix+'_layers='+str(args.e_layers)+'_topk='
                                     +str(args.top_k)+'_kernels='+str(args.num_kernels)+'_lr='+str(args.lr)
                                     +'_bs='+str(args.batch_size)+'_lr_adjust='+str(args.lr_adjust)
                                     +'_epoch='+str(args.epoch)+'_result.txt')

        if args.ckpt not in [None, 'None']:
            state_dict = torch.load(args.ckpt, map_location=self.device)['net_state_dict']
            self.model.load_state_dict(
                state_dict, strict=bool(getattr(args, "ckpt_strict", True))
            )
            # Fine-tuning EMA must begin at the loaded weights, not random initialization.
            if self.ema is not None:
                self.ema = ModelEMA(self.model, decay=ema_decay)

        # params = []
        # for keys, param_value in self.model.named_parameters():
        #     params += [{'params': [param_value], 'lr': self.lr}]

        # self.optim = torch.optim.Adam(params, lr=self.lr, betas=(0.9, 0.999), eps=1e-08)

        weight_decay = getattr(args, 'weight_decay', 1e-4)
        param_optimizer = list(self.model.named_parameters())
        no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight', 'norm.bias', 'norm.weight']

        def decayed(name, parameter):
            # Decay only weight matrices. Every 1-D tensor here is a bias, a
            # normalization scale, or a calibration parameter (ReZero scales,
            # branch gates, basis strengths, channel allocation); shrinking
            # those toward zero biases the architecture toward "module off"
            # instead of regularizing capacity.
            if parameter.ndim <= 1:
                return False
            return not any(nd in name for nd in no_decay)

        optimizer_grouped_parameters = [
            {
                'params': [
                    p for n, p in param_optimizer if decayed(n, p)
                ],
                'weight_decay': weight_decay
            },
            {
                'params': [
                    p for n, p in param_optimizer if not decayed(n, p)
                ],
                'weight_decay': 0.0
            }
        ]
        self.optim = torch.optim.AdamW(optimizer_grouped_parameters, 
                                       lr=self.lr, 
                                       betas=(0.9, 0.999), 
                                       eps=1e-8)

        # if args.ckpt not in [None, 'None'] and 'optim' in torch.load(args.ckpt) and args.load_optim:
        #     self.optim.load_state_dict(torch.load(args.ckpt)['optim'].state_dict())

        self.grads = []
        self.val_f1 = 0
        self.val_mse = 1000
        self.val_R2 = 0

    def hook_fn(self, grad):
        self.grads.append(grad.abs().max())
        return grad
    
    def _move_to_device(self, data):
        if isinstance(data, dict):
            return {k: v.to(self.device) for k, v in data.items()}
        return data.to(self.device)

    def _single_task_mode(self):
        if isinstance(self.args.task, dict):
            return self.args.task['mode']
        return getattr(self.args, self.args.task)['mode']

    def _selection_metric(self):
        if self.is_multi_task:
            return 'score'
        expected = 'f1' if self._single_task_mode() == 'Cls' else 'r2'
        configured = str(
            getattr(self.args, 'selection_metric', expected)
        ).lower()
        if configured != expected:
            raise ValueError(
                f"{self.args.task} is a {self._single_task_mode()} task and must "
                f"use validation {expected.upper()} for checkpoint selection, "
                f"not {configured!r}"
            )
        return expected

    def _joint_selection_score(self, metric_dict):
        """Equal-weight mean of each head's primary validation metric.

        Classification contributes weighted F1 once and regression contributes
        raw R2 once. Every descriptor therefore has one equal vote; negative R2
        remains visible rather than being hidden during checkpoint selection.
        """
        primary = {}
        normalized = []
        for task_name in self.args.multi_task:
            task_config = getattr(self.args, task_name)
            suffix = 'r2' if task_config['mode'] == 'Reg' else 'f1'
            key = f'{task_name}_{suffix}'
            if key not in metric_dict:
                raise KeyError(
                    f"Joint validation did not produce {key!r}"
                )
            value = float(metric_dict[key])
            if not np.isfinite(value):
                raise ValueError(f"Joint validation metric {key} is not finite")
            primary[task_name] = value
            normalized.append(value)
        if not normalized:
            raise ValueError("Joint SPT requires at least one task")
        return float(np.mean(normalized)), primary

    def train(self, train_dataloader, valid_dataloader=None):
        checkpoint_policy = getattr(self.args, 'checkpoint_policy', 'best')
        if checkpoint_policy == 'last' and valid_dataloader is not None:
            raise ValueError(
                "checkpoint_policy='last' requires valid_dataloader=None"
            )
        train_epoch_size = len(train_dataloader)
        train_iter = iter(train_dataloader)
        val_epoch_size = len(valid_dataloader) if valid_dataloader else 0

        self.val_freq = train_epoch_size if self.args.val_freq == -1 else self.args.val_freq
        self.max_iter = train_epoch_size * self.epoch
        print(f"Total Steps Corrected: {self.max_iter} (Epochs: {self.epoch} * Steps/Epoch: {train_epoch_size})")
        
        for step in range(self.start_iter, self.max_iter):
            if step % train_epoch_size == 0:
                print('Epoch: {} ----- step:{} - train_epoch size:{}'.format(step // train_epoch_size, step, train_epoch_size))
                train_iter = iter(train_dataloader)

            # 调用新的学习率调整策略
            self._adjust_learning_rate_iter(step)
            self.train_iter(step, train_iter)

            if (valid_dataloader is not None) and (step + 1) % train_epoch_size == 0:
                epoch = step // train_epoch_size
                val_iter = iter(valid_dataloader)
                metric_dict = self.validation(epoch, val_iter, val_epoch_size)
                self.save_best(metric_dict, epoch)

        if checkpoint_policy == 'last':
            final_epoch = max(0, self.epoch - 1)
            self.final_checkpoint_path = self.save_model(
                final_epoch, best='final', index=self.epoch, gpus=1
            )
            self.last_selection = {
                'epoch_1based': int(self.epoch),
                'epoch0based': int(final_epoch),
                'selected_by': 'fixed_final_epoch',
                'uses_ema': self.ema is not None,
            }
            return self.final_checkpoint_path

        if valid_dataloader is not None and self.best_checkpoint_path is None:
            raise RuntimeError("Validation completed without selecting a checkpoint")
        if bool(getattr(self.args, 'save_final_checkpoint', False)):
            final_epoch = max(0, self.epoch - 1)
            self.save_model(
                final_epoch, best='final', index=final_epoch, gpus=1
            )
        if self.best_checkpoint_path is None:
            raise RuntimeError(
                "Training requires a validation-selected checkpoint"
            )
        return self.best_checkpoint_path

    def _update_top_k(self, metric_key, current_val, epoch, mode='max'):
        records = self.best_records.get(metric_key, [])
        should_save = False

        # 1. 如果记录数未满，直接保存
        if len(records) < self.top_k:
            should_save = True
        else:
            # 2. 如果记录已满，与最差的记录比较
            if mode == 'max':
                # 保留最大值：找出列表中最小的（最差的）进行比较
                worst_record = min(records, key=lambda x: x['val'])
                if current_val > worst_record['val']:
                    should_save = True
            else:
                # 保留最小值：找出列表中最大的（最差的）进行比较
                worst_record = max(records, key=lambda x: x['val'])
                if current_val < worst_record['val']:
                    should_save = True

        if should_save:
            save_name = f'best_{metric_key}'
            save_path = self.save_model(
                epoch, best=save_name, index=current_val, gpus=1
            )
            
            # 更新列表
            records.append({'val': current_val, 'epoch': epoch})
            
            # 排序：好的排前面
            if mode == 'max':
                records.sort(key=lambda x: x['val'], reverse=True) 
            else:
                records.sort(key=lambda x: x['val']) 
            
            # 如果超出 K 个，删除列表末尾（最差的）
            while len(records) > self.top_k:
                to_remove = records.pop()
                # Rounded metric filenames can collide for two genuine
                # improvements. Never delete the checkpoint just written.
                self.delete_model(
                    best=save_name, index=to_remove['val'],
                    protected_path=save_path,
                )
                # print(f"Removed old model: {save_name} with value {to_remove['val']:.5f}")

            self.best_records[metric_key] = records
            
            # 如果当前值是列表里的第一名（绝对最佳），返回 True 用于打印日志
            if records[0]['val'] == current_val:
                if metric_key == self._selection_metric():
                    self.best_checkpoint_path = save_path
                    self.best_selection = {
                        'metric': metric_key,
                        'value': float(current_val),
                        'epoch': int(epoch),
                    }
                return True
        
        return False

    def save_best(self, metric_dict, epoch):
        print("Current Epoch Metrics:", metric_dict)
        self.log = open(self.log_path, mode='a')
        
        if self.is_multi_task:
            current_score, task_primary = self._joint_selection_score(metric_dict)
            is_best = self._update_top_k('score', current_score, epoch, mode='max')
            if is_best:
                self.best_selection['task_metrics'] = task_primary
                self.log.write(
                    f'New Best Mean Primary Score: {current_score:.4f} '
                    f'at epoch {epoch}; tasks={json.dumps(task_primary)}\n'
                )
                 
        else:
            metric_key = self._selection_metric()
            if metric_key not in metric_dict:
                raise KeyError(
                    f"Validation did not produce required selection metric "
                    f"{metric_key!r}"
                )
            is_best = self._update_top_k(
                metric_key, metric_dict[metric_key], epoch, mode='max'
            )
            if is_best:
                self.log.write(
                    f'New Best {metric_key.upper()}: '
                    f'{metric_dict[metric_key]:.4f} at epoch {epoch}\n'
                )

        log_dict = {k: float(v) if isinstance(v, (np.float32, float)) or torch.is_tensor(v) else v for k, v in metric_dict.items()}
        self.log.write(f'step = {epoch}, Metrics = {json.dumps(log_dict)}\n')
        self.log.close()

    def train_main(self, cate, cont, label, step):
        self.model.train()
        self.model.zero_grad()
        self.optim.zero_grad()

        mixup_active = (
            self.mixup_alpha > 0.0
            and step < self.mixup_steps
            and not self.is_multi_task
            and cont is not None
        )
        mix_index = None
        lam = 1.0
        if mixup_active:
            lam = float(
                np.random.beta(self.mixup_alpha, self.mixup_alpha)
            )
            mix_index = torch.randperm(cont.size(0), device=cont.device)
            mixed = cont.clone()
            # Mix intensity only; keep the shared energy grid intact.
            mixed[:, 1::2] = (
                lam * cont[:, 1::2] + (1.0 - lam) * cont[mix_index, 1::2]
            )
            cont = mixed

        if self.args.use_amp == False:
            if mixup_active:
                pred, _ = self.model(cate, cont, None, step=step)
                loss = self._mixup_loss(pred, label, mix_index, lam)
            else:
                pred, loss = self.model(cate, cont, label, step=step)
            loss.backward()
            self.optim.step()
        else:
            with autocast():
                if mixup_active:
                    pred, _ = self.model(cate, cont, None, step=step)
                    loss = self._mixup_loss(pred, label, mix_index, lam)
                else:
                    pred, loss = self.model(cate, cont, label, step=step)
            self.scaler.scale(loss).backward()
            self.scaler.step(self.optim)
            self.scaler.update()

        if self.ema is not None:
            self.ema.update(self.model)

        if torch.isnan(loss) or torch.isinf(loss):
            flag = 'nan_or_inf'
            if torch.isnan(loss).any(): flag = 'nan'
            elif torch.isinf(loss).any(): flag = 'inf'
            print(f"Error: {flag}")
            self.save_model(step, best=flag, index=1, gpus=1)
            raise ValueError

        return pred, loss

    def _mixup_loss(self, pred, label, mix_index, lam):
        if self._single_task_mode() == 'Cls':
            loss_a = torch.nn.functional.cross_entropy(pred, label.long())
            loss_b = torch.nn.functional.cross_entropy(
                pred, label[mix_index].long()
            )
        else:
            target = label.float()
            loss_a = torch.nn.functional.mse_loss(pred.squeeze(-1), target)
            loss_b = torch.nn.functional.mse_loss(
                pred.squeeze(-1), target[mix_index]
            )
        return lam * loss_a + (1.0 - lam) * loss_b

    def train_sub(self, cate, cont, label):
        if not self.is_multi_task:
            self.model.train()
            self.optim.zero_grad()
            loss = self.model(cate, cont, label, delta=True)
            loss.backward()

    def train_iter(self, step, dataloader):
        cate, cont, label = next(dataloader)
        cate, cont = cate.to(self.device), cont.to(self.device)
        
        label = self._move_to_device(label)

        pred, loss = self.train_main(cate, cont, label, step)
        
        if step < self.args.sub_iter:
            self.train_sub(cate, cont, label)

        if step % self.args.display_freq == 0:
            if self.is_multi_task:
                print('Training - Step: {} - Total Loss: {:.4f} - GPU: {}'.format(step, loss.item(), self.args.gpu_id))
                scalars = [loss.item(), self.lr_current]
                names = ['total_loss', 'lr']
            else:
                if self._single_task_mode() == 'Cls':
                    acc = Accuracy(task="multiclass", num_classes=self.args.out, top_k=1).to(self.device)(pred, label)
                    print('Training - Step: {} - Loss: {:.4f} - Acc: {:.4f} - GPU: {}'.format(step, loss, acc, self.args.gpu_id))
                    scalars = [loss.item(), acc, self.lr_current]
                    names = ['loss', 'acc', 'lr']
                else:
                    batch_rmse = torch.mean(
                        (pred.reshape(-1) - label.float()) ** 2
                    ).sqrt()
                    print(
                        'Training - Step: {} - Loss: {:.4f} - '
                        'Batch RMSE: {:.4f} - GPU: {}'.format(
                            step, loss, batch_rmse, self.args.gpu_id
                        )
                    )
                    scalars = [loss.item(), batch_rmse, self.lr_current]
                    names = ['loss', 'rmse', 'lr']
            
            write_scalars(self.writer, scalars, names, step, 'train')

    def _evaluate_multi_task(self, all_preds, all_labels):
        results = {}
        for t_name in all_preds.keys():
            t_preds = torch.cat(all_preds[t_name], dim=0)
            t_labels = torch.cat(all_labels[t_name], dim=0)
            
            if hasattr(self.args, t_name):
                t_conf = getattr(self.args, t_name)
                mode = t_conf['mode']
            else:
                mode = 'Reg' if t_preds.shape[1] == 1 else 'Cls'
            
            num_classes = t_preds.shape[1]

            if mode == 'Reg':
                values = t_preds.squeeze(-1)
                target = t_labels.float().reshape(-1)
                valid = torch.isfinite(target)
                if valid.sum() < 2:
                    raise ValueError(
                        f"Joint validation has fewer than two valid rows for {t_name}"
                    )
                values, target = values[valid], target[valid]
                if not torch.isfinite(values).all():
                    raise FloatingPointError(
                        f"Joint validation produced non-finite predictions for {t_name}"
                    )
                mse = MeanSquaredError().to(self.device)(values, target)
                mae = MeanAbsoluteError().to(self.device)(values, target)
                r2 = R2Score().to(self.device)(values, target)
                
                results[f'{t_name}_mse'] = mse.item()
                results[f'{t_name}_rmse'] = torch.sqrt(mse).item()
                results[f'{t_name}_mae'] = mae.item()
                results[f'{t_name}_r2'] = r2.item()
                results[f'{t_name}_n'] = int(valid.sum().item())
                
                print(f"Task {t_name} (Reg): MSE={mse.item():.4f}, MAE={mae.item():.4f}, R2={r2.item():.4f}")
            else:
                target = t_labels.reshape(-1).long()
                valid = (target >= 0) & (target < num_classes)
                if not valid.any():
                    raise ValueError(
                        f"Joint validation has no valid rows for {t_name}"
                    )
                values, target = t_preds[valid], target[valid]
                if not torch.isfinite(values).all():
                    raise FloatingPointError(
                        f"Joint validation produced non-finite logits for {t_name}"
                    )
                acc = Accuracy(task="multiclass", num_classes=num_classes, top_k=1).to(self.device)(values, target)
                f1 = F1Score(
                    task="multiclass",
                    num_classes=num_classes,
                    average=self.f1_average,
                ).to(self.device)(values, target)
                precision = Precision(
                    task="multiclass", num_classes=num_classes,
                    average=self.f1_average,
                ).to(self.device)(values, target)
                recall = Recall(
                    task="multiclass", num_classes=num_classes,
                    average=self.f1_average,
                ).to(self.device)(values, target)
                auc = AUROC(
                    task="multiclass", num_classes=num_classes,
                    average="macro",
                ).to(self.device)(values, target)
                
                results[f'{t_name}_acc'] = acc.item()
                results[f'{t_name}_auc'] = auc.item()
                results[f'{t_name}_pre'] = precision.item()
                results[f'{t_name}_rec'] = recall.item()
                results[f'{t_name}_f1'] = f1.item()
                results[f'{t_name}_n'] = int(valid.sum().item())
                
                print(f"Task {t_name} (Cls): Acc={acc.item():.4f}, F1={f1.item():.4f}")
        
        return results

    def validation(
        self, epoch, val_iter, val_epoch_size, dataset='val',
        return_outputs=False, apply_ema=True
    ):
        print('============Begin Validation============:epoch:{}'.format(epoch))
        backup = None
        if self.ema is not None and apply_ema:
            backup = {
                key: value.detach().clone()
                for key, value in self.model.state_dict().items()
            }
            self.ema.copy_to(self.model)
        self.model.eval()
        total_loss = 0.

        all_preds = {} if self.is_multi_task else []
        all_labels = {} if self.is_multi_task else []
        with torch.no_grad():
            for i in range(val_epoch_size):
                cate, cont, label = next(val_iter)
                cate, cont = cate.to(self.device), cont.to(self.device)
                label = self._move_to_device(label)

                pred, loss_val = self.model(cate, cont, label)
                total_loss += loss_val.item()
                if self.is_multi_task:
                    for t_name, t_pred in pred.items():
                        if t_name not in all_preds:
                            all_preds[t_name] = []
                            all_labels[t_name] = []
                        all_preds[t_name].append(t_pred)
                        all_labels[t_name].append(label[t_name])
                else:
                    if i == 0:
                        total_pred = pred
                        total_label = label
                    else:
                        total_pred = torch.cat([total_pred, pred], dim=0)
                        total_label = torch.cat([total_label, label], dim=0)

        total_loss /= val_epoch_size
        print(f'{dataset} - epoch: {epoch}, Total Loss: {total_loss:.4f}')
        
        metric_results = {'loss': total_loss}

        if self.is_multi_task:
            mt_metrics = self._evaluate_multi_task(all_preds, all_labels)
            metric_results.update(mt_metrics)
            metric_results['score'], _ = self._joint_selection_score(
                metric_results
            )
        else:
            if self._single_task_mode() == 'Cls':
                # Classification
                acc = Accuracy(task="multiclass", num_classes=self.args.out, top_k=1).to(self.device)(total_pred, total_label)
                auc = AUROC(task="multiclass", num_classes=self.args.out).to(self.device)(total_pred, total_label)
                pre = Precision(task="multiclass", num_classes=self.args.out, average=self.f1_average).to(self.device)(total_pred, total_label)
                rec = Recall(task="multiclass", num_classes=self.args.out, average=self.f1_average).to(self.device)(total_pred, total_label)
                f1 = F1Score(task="multiclass", num_classes=self.args.out, average=self.f1_average).to(self.device)(total_pred, total_label)
                
                metric_results.update({'acc': acc.item(), 'auc': auc.item(), 'pre': pre.item(), 'rec': rec.item(), 'f1': f1.item()})
                
                print('Epoch: {}, Pre: {:.4f}, Rec: {:.4f}, F1: {:.4f}, acc: {:.4f}, auc: {:.4f}'.format(
                    epoch, pre.item(), rec.item(), f1.item(), acc.item(), auc.item()))
                
            else:
                # Regression
                if self.args.out == 1:
                    mse = MeanSquaredError().to(self.device)(total_pred.squeeze(1), total_label) 
                    rmse = torch.sqrt(mse)
                    mae = MeanAbsoluteError().to(self.device)(total_pred.squeeze(1), total_label) 
                    r2 = R2Score().to(self.device)(total_pred.squeeze(1), total_label) 
                else:
                    mse = MeanSquaredError().to(self.device)((torch.nn.functional.softmax(total_pred, dim=1) * torch.arange(self.args.out, device=self.device)).sum(dim=1), total_label)
                    r2 = torch.tensor(0)
                    mae = torch.tensor(0)
                    rmse = torch.tensor(0)

                metric_results.update({'mse': mse.item(), 'rmse': rmse.item(), 'mae': mae.item(), 'r2': r2.item()})
                print('Epoch: {}, Loss: {:.4f}, MSE: {:.4f}, MAE: {:.4f}, R2: {:.4f}'.format(epoch, total_loss, mse, mae, r2))


        scalars = []
        names = []
        for k, v in metric_results.items():
            names.append(k)
            scalars.append(v)
                
        write_scalars(self.writer, scalars, names, epoch, dataset)
        if backup is not None:
            self.model.load_state_dict(backup, strict=True)
        if return_outputs:
            if self.is_multi_task:
                outputs = {
                    'predictions': {
                        key: torch.cat(values, dim=0).detach().cpu()
                        for key, values in all_preds.items()
                    },
                    'labels': {
                        key: torch.cat(values, dim=0).detach().cpu()
                        for key, values in all_labels.items()
                    },
                }
            else:
                outputs = {
                    'predictions': total_pred.detach().cpu(),
                    'labels': total_label.detach().cpu(),
                }
            return metric_results, outputs
        return metric_results

    def test(self, test_dataloader):
        print('============Begin Testing============')
        test_iter = iter(test_dataloader)
        self.validation(0, test_iter, len(test_dataloader), dataset='test')

    def evaluate_test(
        self, checkpoint_path, test_dataloader, test_dataset, split_name='test'
    ):
        """Evaluate the validation-selected checkpoint once on held-out test data."""
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        self.model.load_state_dict(
            checkpoint['net_state_dict'],
            strict=bool(getattr(self.args, 'ckpt_strict', True)),
        )
        # ``save_model`` already stores the EMA weights when EMA is enabled, so
        # the checkpoint *is* the selected model. Letting validation() re-apply
        # the EMA shadow here would silently discard checkpoint selection and
        # score the final training epoch instead.
        metric_results, outputs = self.validation(
            self.best_selection['epoch'] if self.best_selection else 0,
            iter(test_dataloader),
            len(test_dataloader),
            dataset=split_name,
            return_outputs=True,
            apply_ema=False,
        )

        default_dir = os.path.join(
            self.args.save_folder,
            self.args.exp_name,
            f'{split_name}_results',
        )
        output_dir = getattr(self.args, 'test_results_dir', None)
        if output_dir in [None, 'None']:
            output_dir = default_dir
        os.makedirs(output_dir, exist_ok=True)

        split_file = getattr(self.args, 'split_file', None)
        summary = {
            'checkpoint': os.path.abspath(checkpoint_path),
            'split_file': (
                os.path.abspath(split_file)
                if split_file not in [None, 'None'] else None
            ),
            'selection': self.best_selection,
            'split': split_name,
            'parameter_count': int(sum(
                parameter.numel() for parameter in self.model.parameters()
                if parameter.requires_grad
            )),
            'metrics': metric_results,
        }
        with open(
            os.path.join(output_dir, 'metrics.json'), 'w', encoding='utf-8'
        ) as handle:
            json.dump(summary, handle, indent=2)

        row_indices = np.asarray(test_dataset.indices)
        task_names = self.args.multi_task if self.is_multi_task else [self.args.task]
        for task_name in task_names:
            predictions = (
                outputs['predictions'][task_name]
                if self.is_multi_task else outputs['predictions']
            )
            labels = (
                outputs['labels'][task_name]
                if self.is_multi_task else outputs['labels']
            )
            task_config = (
                getattr(self.args, task_name)
                if not isinstance(task_name, dict) else task_name
            )
            task_rows = row_indices
            if self.is_multi_task:
                if task_config['mode'] == 'Cls':
                    valid = (
                        (labels.reshape(-1) >= 0)
                        & (labels.reshape(-1) < predictions.shape[1])
                    ).numpy()
                else:
                    valid = torch.isfinite(labels.reshape(-1)).numpy()
                task_rows = row_indices[valid]
                predictions = predictions[valid]
                labels = labels[valid]
                if not torch.isfinite(predictions).all():
                    raise FloatingPointError(
                        f"Joint test produced non-finite predictions for {task_name}"
                    )
            if task_config['mode'] == 'Cls':
                probabilities = torch.softmax(predictions, dim=1).numpy()
                frame = pd.DataFrame({
                    'row_index': task_rows,
                    'label': labels.numpy(),
                    'prediction': probabilities.argmax(axis=1),
                    'confidence': probabilities.max(axis=1),
                })
                for class_index in range(probabilities.shape[1]):
                    frame[f'probability_{class_index}'] = probabilities[:, class_index]
            else:
                frame = pd.DataFrame({
                    'row_index': task_rows,
                    'label': labels.numpy(),
                    'prediction': predictions.squeeze(-1).numpy(),
                })
            frame.to_csv(
                os.path.join(output_dir, f'{task_name}_predictions.csv'),
                index=False,
            )
        print(f"Saved held-out test results: {os.path.abspath(output_dir)}")


    def _adjust_learning_rate_iter(self, step):
        if step < self.warmup_steps:
            lr_scale = float(step) / float(max(1, self.warmup_steps))
        else:
            progress = float(step - self.warmup_steps)
            max_progress = float(self.max_iter - self.warmup_steps)
            max_progress = max(1.0, max_progress) 

            if self.args.lr_adjust == 'fix':
                decay_count = sum([1 for v in self.args.stepvalues if step >= v])
                lr_scale = self.args.gamma ** decay_count

            elif self.args.lr_adjust == 'poly':
                lr_scale = (1 - progress / max_progress) ** 0.9

            elif self.args.lr_adjust == 'cosine':
                lr_scale = 0.5 * (1.0 + math.cos(math.pi * progress / max_progress))
            
            else:
                lr_scale = 1.0

        # 计算当前 LR
        self.lr_current = self.args.lr * lr_scale
        
        # 考虑 end_learning_rate (如果有设置)
        if hasattr(self, 'end_learning_rate') and self.end_learning_rate > 0:
             self.lr_current = max(self.lr_current, self.end_learning_rate)
        else:
             self.lr_current = max(self.lr_current, 1e-8)

        # 应用到优化器
        for param_group in self.optim.param_groups:
            param_group['lr'] = self.lr_current

    def init_writer(self):
        if not os.path.exists(self.args.save_folder):
            os.makedirs(self.args.save_folder, exist_ok=True)

        log_path = os.path.join(self.args.save_log, self.args.exp_name)
        log_config_path = os.path.join(log_path, 'configs.log')
        self.writer = SummaryWriter(log_path)
        with open(log_config_path, 'w') as f:
            f.write(arg2str(self.args))

    def delete_model(self, best, index, protected_path=None):
        if index == 0 or index == 1000000:
            return

        # 统一格式化为 5 位小数，确保文件名匹配
        if isinstance(index, (float, np.floating)):
             index_val = f"{index:.5f}"
        else:
             index_val = str(index)
        
        save_fname = '%s_%s_%s.pth' % (self.model.model_name(), best, index_val) 
        save_path = os.path.join(self.args.save_folder, self.args.exp_name, save_fname)
        
        if (
            protected_path is not None
            and os.path.abspath(save_path) == os.path.abspath(protected_path)
        ):
            return
        if os.path.exists(save_path):
            os.remove(save_path)
            print("Deleted old model: " + save_path)
        else:
            # 如果文件没找到，可能是精度问题，尝试不带格式化的查找（可选）
            pass

    def save_model(self, step, best='best_acc', index=None, gpus=1):
        model_save_path = os.path.join(self.args.save_folder, self.args.exp_name)
        if not os.path.exists(model_save_path):
            os.makedirs(model_save_path, exist_ok=True)

        if isinstance(index, (float, np.floating)):
             index_val = f"{index:.5f}"
        else:
             index_val = str(index)

        save_fname = '%s_%s_%s.pth' % (self.model.model_name(), best, index_val)
        save_path = os.path.join(self.args.save_folder, self.args.exp_name, save_fname)

        if gpus == 1:
            if self.ema is not None:
                state_dict = self.ema.state_dict()
            else:
                state_dict = self.model.state_dict()
        else:
            if self.ema is not None:
                state_dict = self.ema.state_dict()
            else:
                state_dict = self.model.module.state_dict()

        save_dict = {
            'net_state_dict': state_dict,
            'exp_name': self.args.exp_name,
            'iter': step,
            'index': index,
        }
        if bool(getattr(self.args, 'save_optimizer_state', False)):
            save_dict['optim_state_dict'] = self.optim.state_dict()
        torch.save(save_dict, save_path)
        print(best + ' Model Saved: ' + save_path)
        return save_path

def write_scalars(writer, scalars, names, n_iter, tag=None):
    for scalar, name in zip(scalars, names):
        if tag is not None:
            name = '/'.join([tag, name])
        writer.add_scalar(name, scalar, n_iter)
