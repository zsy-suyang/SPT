import os
import torch
import torch.nn as nn
import models
from tensorboardX import SummaryWriter
from config.cfg import arg2str
from torchmetrics import Accuracy, AUROC, MeanSquaredError, MeanAbsoluteError, R2Score
from torchmetrics.classification import Precision, Recall, F1Score
from torch.cuda.amp import autocast, GradScaler
import json
import numpy as np
from models.SPT import SPT

class DefaultTrainer(object):

    def __init__(self, args, device=None):
        self.args = args
        self.task = args.task
        self.batch_size = args.batch_size
        self.lr = self.lr_current = args.lr
        self.epoch = args.epoch
        self.start_iter = args.start_iter
        self.max_iter = args.max_iter
        self.warmup_steps = args.warmup_steps

        self.model = SPT(args, device)

        self.init_writer()

        self.flag = 0
        self.model.cuda()
        self.scaler = GradScaler()

        self.is_multi_task = hasattr(args, 'multi_task') and isinstance(args.multi_task, list) and len(args.multi_task) > 0

        self.metrics = {
            'acc': ['high', 0],
            'auc': ['high', 0],
            'f1':  ['high', 0],  
            'r2':  ['high', -100],
            'loss': ['low', 10000],
            'mse': ['low', 10000],
            'mae': ['low', 10000], 
            'score': ['high', 0]  
        }
        
        self.start = 0
        self.wrong = None
        task_prefix = 'MultiTask' if self.is_multi_task else str(self.task)
        self.log_path = os.path.join(self.args.save_folder, task_prefix + '_result.txt')

        print('LR = {}'.format(self.lr))

        if args.ckpt not in [None, 'None']:
            state_dict = torch.load(args.ckpt)['net_state_dict']
            self.model.load_state_dict(state_dict, strict=False)

        params = []
        for keys, param_value in self.model.named_parameters():
            params += [{'params': [param_value], 'lr': self.lr}]

        self.optim = torch.optim.Adam(params, lr=self.lr,
                                      betas=(0.9, 0.999), eps=1e-08)

        if args.ckpt not in [None, 'None'] and 'optim' in torch.load(args.ckpt) and args.load_optim:
            self.optim.load_state_dict(torch.load(args.ckpt)['optim'].state_dict())

        self.grads = []
        self.val_f1 = 0
        self.val_mse = 1000
        self.val_R2 = 0

    def hook_fn(self, grad):
        self.grads.append(grad.abs().max())
        return grad
    
    def _move_to_device(self, data):
        if isinstance(data, dict):
            return {k: v.cuda() for k, v in data.items()}
        return data.cuda()

    def train(self, train_dataloader, valid_dataloader=None):
        
        train_epoch_size = len(train_dataloader)
        train_iter = iter(train_dataloader)
        val_epoch_size = len(valid_dataloader) if valid_dataloader else 0

        self.val_freq = train_epoch_size if self.args.val_freq == -1 else self.args.val_freq
        self.max_iter = train_epoch_size * self.epoch
        
        for step in range(self.start_iter, self.max_iter):

            if step % train_epoch_size == 0:
                print('Epoch: {} ----- step:{} - train_epoch size:{}'.format(step // train_epoch_size, step,
                                                                            train_epoch_size))
                train_iter = iter(train_dataloader)

            self._adjust_learning_rate_iter(step)
            self.train_iter(step, train_iter)

            if (valid_dataloader is not None) and step % train_epoch_size == 0 and step // train_epoch_size > 0:
                epoch = step // train_epoch_size
                val_iter = iter(valid_dataloader)

                metric_dict = self.validation(epoch, val_iter, val_epoch_size)

                self.save_best(metric_dict, epoch)

    def save_best(self, metric_dict, epoch):
        print("Current Epoch Metrics:", metric_dict)
        self.log = open(self.log_path, mode='a')
        
        if self.is_multi_task:
            if 'loss' in metric_dict and metric_dict['loss'] < self.metrics['loss'][1]:
                self.metrics['loss'][1] = metric_dict['loss']
                self.save_model(epoch, best='best_loss', index=self.metrics['loss'][1], gpus=1)

            current_score = 0
            for k, v in metric_dict.items():
                if k.endswith('_acc') or k.endswith('_f1'):
                    current_score += v
                elif k.endswith('_r2'):
                    current_score += max(v, 0) 
            
            if current_score > self.metrics['score'][1]:
                self.metrics['score'][1] = current_score
                self.save_model(epoch, best='best_score', index=self.metrics['score'][1], gpus=1)
                self.log.write(f'New Best Score: {current_score:.4f} at epoch {epoch}\n')

            log_dict = {k: float(v) if isinstance(v, (np.float32, float)) or torch.is_tensor(v) else v for k, v in metric_dict.items()}
            self.log.write(f'step = {epoch}, Metrics = {json.dumps(log_dict)}\n')

        else:
            task_type = 'Cls' if 'f1' in metric_dict else 'Reg'
            
            if 'loss' in metric_dict and metric_dict['loss'] < self.metrics['loss'][1]:
                self.metrics['loss'][1] = metric_dict['loss']
                self.save_model(epoch, best='best_loss', index=self.metrics['loss'][1], gpus=1)

            if task_type == 'Cls':
                if metric_dict['f1'] > self.metrics['f1'][1]:
                    self.metrics['f1'][1] = metric_dict['f1']
                    self.save_model(epoch, best='best_f1', index=self.metrics['f1'][1], gpus=1)
                if metric_dict['acc'] > self.metrics['acc'][1]:
                    self.metrics['acc'][1] = metric_dict['acc']
                    self.save_model(epoch, best='best_acc', index=self.metrics['acc'][1], gpus=1)
            else:
                if metric_dict['r2'] > self.metrics['r2'][1]:
                    self.metrics['r2'][1] = metric_dict['r2']
                    self.save_model(epoch, best='best_r2', index=self.metrics['r2'][1], gpus=1)
                if metric_dict['mse'] < self.metrics['mse'][1]:
                    self.metrics['mse'][1] = metric_dict['mse']
                    self.save_model(epoch, best='best_mse', index=self.metrics['mse'][1], gpus=1)

            self.log.write('step = {}, metric = {}\n'.format(epoch, metric_dict))

        self.log.close()

    def train_main(self, cate, cont, label, step):
        self.model.train()
        self.model.zero_grad()
        self.optim.zero_grad()

        if self.args.use_amp == False:
            pred, loss = self.model(cate, cont, label, step=step)
            loss.backward()
            self.optim.step()
        else:
            with autocast():
                pred, loss = self.model(cate, cont, label, step=step)
            self.scaler.scale(loss).backward()
            self.scaler.step(self.optim)
            self.scaler.update()

        if torch.isnan(loss) or torch.isinf(loss):
            flag = 'nan_or_inf'
            if torch.isnan(loss).any(): flag = 'nan'
            elif torch.isinf(loss).any(): flag = 'inf'
            print(f"Error: {flag}")
            self.save_model(step, best=flag, index=1, gpus=1)
            raise ValueError

        return pred, loss

    def train_sub(self, cate, cont, label):
        if not self.is_multi_task:
            self.model.train()
            self.optim.zero_grad()
            loss = self.model(cate, cont, label, delta=True)
            loss.backward()

    def train_iter(self, step, dataloader):
        cate, cont, label = next(dataloader)
        cate, cont = cate.cuda(), cont.cuda()
        
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
                if self.args.save_metric != 'mse':
                    acc = Accuracy(task="multiclass", num_classes=self.args.out, top_k=1).cuda()(pred, label)
                    auc = 0 
                else:
                    acc = auc = 0
                print('Training - Step: {} - Loss: {:.4f} - Acc: {:.4f} - GPU: {}'.format(step, loss, acc, self.args.gpu_id))
                scalars = [loss.item(), acc, self.lr_current]
                names = ['loss', 'acc', 'lr']
            
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
                mse = MeanSquaredError().cuda()(t_preds.squeeze(-1), t_labels.float())
                mae = MeanAbsoluteError().cuda()(t_preds.squeeze(-1), t_labels.float()) 
                r2 = R2Score().cuda()(t_preds.squeeze(-1), t_labels.float())
                
                results[f'{t_name}_mse'] = mse.item()
                results[f'{t_name}_mae'] = mae.item() 
                results[f'{t_name}_r2'] = r2.item()
                
                print(f"Task {t_name} (Reg): MSE={mse.item():.4f}, MAE={mae.item():.4f}, R2={r2.item():.4f}")
            else:
                acc = Accuracy(task="multiclass", num_classes=num_classes, top_k=1).cuda()(t_preds, t_labels)
                f1 = F1Score(task="multiclass", num_classes=num_classes, average='weighted').cuda()(t_preds, t_labels) 
                
                results[f'{t_name}_acc'] = acc.item()
                results[f'{t_name}_f1'] = f1.item()
                
                print(f"Task {t_name} (Cls): Acc={acc.item():.4f}, F1={f1.item():.4f}")
        
        return results

    def validation(self, epoch, val_iter, val_epoch_size, dataset='val'):
        print('============Begin Validation============:epoch:{}'.format(epoch))
        self.model.eval()
        total_loss = 0.

        all_preds = {} if self.is_multi_task else []
        all_labels = {} if self.is_multi_task else []

        with torch.no_grad():
            for i in range(val_epoch_size):
                cate, cont, label = next(val_iter)
                cate, cont = cate.cuda(), cont.cuda()
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
        else:
            if self.args.save_metric != 'mse':
                # Classification
                acc = Accuracy(task="multiclass", num_classes=self.args.out, top_k=1).cuda()(total_pred, total_label)
                auc = AUROC(task="multiclass", num_classes=self.args.out).cuda()(total_pred, total_label)
                pre = Precision(task="multiclass", num_classes=self.args.out, average='weighted').cuda()(total_pred, total_label)
                rec = Recall(task="multiclass", num_classes=self.args.out, average='weighted').cuda()(total_pred, total_label)
                f1 = F1Score(task="multiclass", num_classes=self.args.out, average='weighted').cuda()(total_pred, total_label)
                
                metric_results.update({'acc': acc.item(), 'auc': auc.item(), 'pre': pre.item(), 'rec': rec.item(), 'f1': f1.item()})
                
                print('Epoch: {}, Pre: {:.4f}, Rec: {:.4f}, F1: {:.4f}, acc: {:.4f}, auc: {:.4f}'.format(
                    epoch, pre.item(), rec.item(), f1.item(), acc.item(), auc.item()))
                
            else:
                # Regression
                if self.args.out == 1:
                    mse = MeanSquaredError().cuda()(total_pred.squeeze(1), total_label) 
                    rmse = torch.sqrt(mse)
                    mae = MeanAbsoluteError().cuda()(total_pred.squeeze(1), total_label) 
                    r2 = R2Score().cuda()(total_pred.squeeze(1), total_label) 
                else:
                    mse = MeanSquaredError().cuda()((torch.nn.functional.softmax(total_pred, dim=1) * torch.arange(self.args.out).cuda()).sum(dim=1), total_label)
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
        return metric_results

    def test(self, test_dataloader):
        print('============Begin Testing============')
        test_iter = iter(test_dataloader)
        self.validation(0, test_iter, len(test_dataloader), dataset='test')

    def _adjust_learning_rate_iter(self, step):
        if step <= self.warmup_steps:
            self.lr_current = self.args.lr * float(step) / float(self.warmup_steps)

        if self.args.lr_adjust == 'fix':
            if step in self.args.stepvalues:
                self.lr_current = self.lr_current * self.args.gamma
        elif self.args.lr_adjust == 'poly':
            self.lr_current = self.args.lr * (1 - step / self.args.max_iter) ** 0.9

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

    def delete_model(self, best, index):
        if index == 0 or index == 1000000:
            return

        idx_str = f"{index:.4f}" if isinstance(index, (float, np.floating)) else str(index)
        
        save_fname = '%s_%s_%s.pth' % (self.model.model_name(), best, index) 
        save_path = os.path.join(self.args.save_folder, self.args.exp_name, save_fname)
        if os.path.exists(save_path):
            os.remove(save_path)

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
            state_dict = self.model.state_dict()
        else:
            state_dict = self.model.module.state_dict()

        save_dict = {
            'net_state_dict': state_dict,
            'exp_name': self.args.exp_name,
            'iter': step,
            'index': index,
            'optim':self.optim
        }
        torch.save(save_dict, save_path)
        print(best + ' Model Saved: ' + save_path)

def write_scalars(writer, scalars, names, n_iter, tag=None):
    for scalar, name in zip(scalars, names):
        if tag is not None:
            name = '/'.join([tag, name])
        writer.add_scalar(name, scalar, n_iter)