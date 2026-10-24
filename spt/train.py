import torch
from spt.config.cfg import parse_config
from spt import training
import os
from spt.utils import data_load
import numpy as np
import random
from spt.utils.runtime import resolve_device

def main(args):
    if getattr(args, 'inference_only', False):
        raise ValueError('This configuration is for a measured/application cohort. Use its application inference entry, not training.')
    runseed = args.seed
    evaluation_batch_size = getattr(args, 'eval_batch_size', args.batch_size)
    device = resolve_device(args)
    torch.manual_seed(runseed)
    np.random.seed(runseed)
    random.seed(runseed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(runseed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


    is_multi_task = hasattr(args, 'multi_task') and isinstance(args.multi_task, list) and len(args.multi_task) > 0

    if is_multi_task:
        task_name_for_dir = 'MultiTask_Joint'+'_lr='+str(args.lr)+'_bs='+str(args.batch_size)+'_epoch='+str(args.epoch)
    else:
        task_name_for_dir = args.task+'_layers='+str(args.e_layers)+'_topk='+str(args.top_k)+'_kernels='+str(args.num_kernels) \
                                +'_lr='+str(args.lr)+'_bs='+str(args.batch_size)+'_lr_adjust='+str(args.lr_adjust) \
                                +'_epoch='+str(args.epoch)

    dest_dir = os.path.join(args.save_folder, task_name_for_dir, args.exp_name)
    if os.path.exists(dest_dir) and bool(
        getattr(args, 'fail_if_run_exists', False)
    ):
        raise FileExistsError(
            f"Run output already exists: {dest_dir}. "
            "Choose a new exp_name or remove the stale run explicitly."
        )
    if not os.path.exists(dest_dir):
        os.makedirs(dest_dir)
        args.exp_name = os.path.join(task_name_for_dir, args.exp_name, 'ver0')
    else:
        existing_versions = [x for x in os.listdir(dest_dir) if x.startswith('ver')]
        if not existing_versions:
            num = 0 
        else:
            version_numbers = [int(x[3:]) for x in existing_versions]
            num = max(version_numbers) + 1
        args.exp_name = os.path.join(task_name_for_dir, args.exp_name, 'ver'+str(num))
    print('================ {:^30s} ================'.format(args.exp_name.split('/')[-1]))

    dataset = getattr(data_load, args.data_name.lower())
    train_data = dataset(
        dataset='train',
        args=args
    )
    train_load = torch.utils.data.DataLoader(train_data, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers, drop_last=bool(getattr(args, 'drop_last', True)))
    print('================ {:^30s} ================'.format('train set loaded'))
    evaluate_during_training = bool(getattr(args, 'evaluate_during_training', True))
    val_load = None
    if evaluate_during_training:
        validation_set = getattr(args, 'validation_set', 'val')
        if validation_set != 'val':
            raise ValueError(
                "validation_set must be 'val'; the held-out test set cannot be "
                "used for checkpoint selection"
            )
        val_data = dataset(
            dataset=validation_set,
            args=args
        )
        if len(val_data) == 0:
            raise ValueError(
                "The validation set is empty. Set val_size > 0 before enabling "
                "in-training checkpoint selection."
            )
        val_load = torch.utils.data.DataLoader(
            val_data,
            batch_size=evaluation_batch_size,
            shuffle=False,
            num_workers=args.num_workers,
        )
        print('================ {:^30s} ================'.format(
            'validation set loaded for checkpoint selection'
        ))
    else:
        print('Training will not perform validation-based checkpoint selection.')

    trainer = getattr(training, args.trainer.lower())(args, device)
    selected_checkpoint = trainer.train(train_load, val_load)

    if bool(getattr(args, 'evaluate_test_after_training', False)):
        if val_load is None:
            raise ValueError(
                "evaluate_test_after_training requires validation-based "
                "checkpoint selection"
            )
        evaluation_split = getattr(
            args, 'evaluate_after_training_split', 'test'
        )
        if evaluation_split == 'lockbox':
            raise ValueError(
                "main.py cannot access the lockbox; use the explicit frozen-model "
                "lockbox evaluation workflow"
            )
        test_data = dataset(dataset=evaluation_split, args=args)
        test_load = torch.utils.data.DataLoader(
            test_data,
            batch_size=evaluation_batch_size,
            shuffle=False,
            num_workers=args.num_workers,
        )
        trainer.evaluate_test(
            selected_checkpoint,
            test_load,
            test_data,
            split_name=evaluation_split,
        )
    return selected_checkpoint

def cli(argv=None):
    args = parse_config(argv, description="Train SPT from a portable YAML configuration")
    return main(args)


if __name__ == '__main__':
    cli()
    


    

