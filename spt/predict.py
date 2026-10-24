"""Label-free inference on prepared, absolute-energy XANES spectra."""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from .config.cfg import load_config
from .evaluate import _load_model
from .registry import get_model, verify_asset


def prepare_spectra(frame, args):
    """Match frozen training transformations, without fitting or resampling."""
    from .utils.data_load.data_des import X_log_normalize_optimized, Y_log_normalize_optimized

    required = {'energy_eV','intensity','absorber'}
    if not required.issubset(frame.columns):
        raise ValueError(f'Input CSV requires columns {sorted(required)}')
    frame = frame.copy()
    if 'spectrum_id' not in frame:
        frame['spectrum_id'] = 'spectrum_1'
    if frame['spectrum_id'].isna().any():
        raise ValueError('spectrum_id cannot be missing')
    identifiers, categorical, energies, intensities = [],[],[],[]
    normalization_dtype = getattr(args, 'normalization_dtype', 'float32')
    if normalization_dtype not in ('float32', 'float64'):
        raise ValueError('normalization_dtype must be float32 or float64')
    input_dtype = np.dtype(normalization_dtype)
    n_points = int(args.num_cont)//2
    for identifier, group in frame.groupby('spectrum_id',sort=False):
        e = group.energy_eV.to_numpy(dtype=input_dtype)
        y = group.intensity.to_numpy(dtype=input_dtype)
        elements = group.absorber.unique()
        if len(e)!=n_points or len(elements)!=1:
            raise ValueError(f'{identifier}: expected {n_points} points and one absorber')
        if not np.isfinite(e).all() or not np.isfinite(y).all() or np.any(e<=0) or np.any(np.diff(e)<=0):
            raise ValueError(f'{identifier}: energies must be positive, increasing and finite; intensities must be finite')
        if elements[0] not in args.ele_dict:
            raise ValueError(f'{identifier}: unsupported absorber {elements[0]!r}')
        if np.ptp(y)==0:
            raise ValueError(f'{identifier}: constant intensity is not an informative spectrum')
        identifiers.append(str(identifier))
        categorical.append([args.ele_dict[elements[0]]])
        energies.append(e)
        intensities.append(y)
    if not identifiers:
        raise ValueError('No spectra in input')
    e_scaled = X_log_normalize_optimized(energies,min_log_data=args.min_log_data,max_log_data=args.max_log_data,log_base=np.e)
    y_scaled = Y_log_normalize_optimized(intensities)
    continuous = np.empty((len(identifiers),int(args.num_cont)),dtype=np.float32)
    continuous[:,0::2] = np.asarray(e_scaled,dtype=np.float32)
    continuous[:,1::2] = np.asarray(y_scaled,dtype=np.float32)
    return identifiers, np.asarray(categorical,dtype=np.int64), continuous


def predict_frame(model, frame, args, device='cpu', batch_size=128):
    identifiers,categorical,continuous = prepare_spectra(frame,args)
    model.eval()
    tasks = args.multi_task if isinstance(getattr(args,'multi_task',None),list) else [args.task]
    records = []
    block_size = int(getattr(args, 'inference_block_size', 0))
    if block_size < 0 or batch_size < 1:
        raise ValueError('Invalid inference block or batch size')
    with torch.inference_mode():
        for start in range(0,len(identifiers),batch_size):
            categories = categorical[start:start+batch_size]
            features = continuous[start:start+batch_size]
            retained = len(features)
            if retained < block_size:
                # Preserve the registered numerical block shape without adding spectra.
                categories = np.concatenate([categories, np.repeat(categories[:1], block_size-retained, axis=0)])
                features = np.concatenate([features, np.repeat(features[:1], block_size-retained, axis=0)])
            outputs,_ = model(torch.as_tensor(categories,device=device),
                              torch.as_tensor(features,device=device),None)
            outputs = outputs if isinstance(outputs,dict) else {args.task:outputs}
            for task in tasks:
                values = outputs[task][:retained]
                definition = getattr(args,task)
                classification = definition['mode']=='Cls'
                if classification:
                    probabilities = values.softmax(-1).cpu().numpy()
                    predicted = probabilities.argmax(-1)
                    classes = definition['value']
                else:
                    predicted = values.reshape(-1).cpu().numpy()
                for offset,index in enumerate(predicted):
                    row = {'spectrum_id':identifiers[start+offset],
                           'descriptor':'SGN' if task=='PSGO' else task,
                           'prediction':classes[int(index)] if classification else float(index)}
                    if classification:
                        row['class_score'] = float(probabilities[offset,int(index)])
                        for k,value in enumerate(classes):
                            row[f'score_{value}'] = float(probabilities[offset,k])
                    records.append(row)
    return pd.DataFrame(records)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--model',required=True)
    parser.add_argument('--project-root')
    parser.add_argument('--checkpoint')
    parser.add_argument('--input',required=True)
    parser.add_argument('--output',required=True)
    parser.add_argument('--device',default='cpu')
    parser.add_argument('--batch-size',type=int,default=128)
    parser.add_argument('--allow-unregistered-checkpoint',action='store_true',
                        help='Allow a user-trained checkpoint with the same architecture; do not label it as the paper model.')
    cli = parser.parse_args(argv)
    if cli.batch_size<=0:
        parser.error('batch-size must be positive')
    root,entry = get_model(cli.model,cli.project_root)
    if entry['architecture']!='spt':
        parser.error('The label-free prediction command supports Expert/Joint SPT; use experiment/evaluate for baselines.')
    checkpoint = Path(cli.checkpoint).expanduser().resolve() if cli.checkpoint else root/entry['checkpoint']
    verify_asset(checkpoint,None if cli.allow_unregistered_checkpoint else entry.get('checkpoint_bytes'))
    output = Path(cli.output)
    if output.exists():
        raise FileExistsError(f'Refusing to overwrite {output}')
    args = load_config(root/entry['config'],project_root=root,overrides={'ckpt':str(checkpoint)})
    model,_ = _load_model(args,torch.device(cli.device))
    result = predict_frame(model,pd.read_csv(cli.input),args,cli.device,cli.batch_size)
    output.parent.mkdir(parents=True,exist_ok=True)
    result.to_csv(output,index=False)
    print(f'Wrote {len(result)} descriptor predictions to {output}')


if __name__=='__main__':
    main()
