"""Cohort-fixed inference on the paper's prepared experimental/computed inputs."""
from __future__ import annotations

import argparse
import json
from contextlib import contextmanager
from pathlib import Path


@contextmanager
def _cpu_threads(execution, device):
    """Use the cohort's CPU thread count without changing the caller's setting."""
    import torch
    configured = execution.get('torch_num_threads')
    if configured is not None and (type(configured) is not int or configured <= 0):
        raise ValueError('torch_num_threads must be a positive integer')
    previous = torch.get_num_threads()
    apply = torch.device(device).type == 'cpu' and configured is not None
    try:
        if apply:
            torch.set_num_threads(configured)
        yield torch.get_num_threads()
    finally:
        if apply:
            torch.set_num_threads(previous)


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--cohort',required=True)
    parser.add_argument('--input',help='Optional replacement for the registered prepared spectra CSV')
    parser.add_argument('--output',required=True)
    parser.add_argument('--project-root')
    parser.add_argument('--tasks',nargs='+',help='Optional descriptor subset (SGN is accepted as an alias for PSGO)')
    parser.add_argument('--device',help='Override the registered cohort device')
    parser.add_argument('--batch-size',type=int,help='Override the registered batch size')
    cli=parser.parse_args(argv)
    from .registry import get_model,project_root,verify_asset
    root=project_root(cli.project_root)
    cohorts=json.loads((root/'configs/applications.json').read_text())['cohorts']
    matches=[c for c in cohorts if c['id']==cli.cohort]
    if len(matches)!=1:parser.error('Unknown cohort. See configs/applications.json')
    selected=matches[0]
    input_path=Path(cli.input) if cli.input else root/selected['prepared_input']['path']
    if not cli.input:
        verify_asset(input_path,selected['prepared_input'].get('bytes'))
    tasks=[('PSGO' if t=='SGN' else t) for t in cli.tasks] if cli.tasks else list(selected['models'])
    if any(t not in selected['models'] for t in tasks):parser.error('Descriptor is not registered for this cohort')
    execution=selected.get('execution',{})
    device=cli.device or execution.get('device','cpu')
    batch_size=cli.batch_size or execution.get('batch_size',128)
    if cli.batch_size is not None and cli.batch_size<=0:parser.error('batch-size must be positive')
    output=Path(cli.output)
    if output.exists() or output.with_suffix('.metadata.json').exists():
        raise FileExistsError(f'Refusing to overwrite existing output: {output}')
    # Validate every asset before doing any inference.
    entries={t:get_model(selected['models'][t],root)[1] for t in tasks}
    for entry in entries.values():verify_asset(root/entry['checkpoint'],entry.get('checkpoint_bytes'))
    import pandas as pd
    import torch
    import platform
    from .config.cfg import load_config
    from .evaluate import _load_model
    from .predict import predict_frame
    from .utils.execution import inference_backend
    from .experiments.common import set_seed
    set_seed(execution.get('seed',42))
    frame=pd.read_csv(input_path)
    rows=[]
    backends={}
    with _cpu_threads(execution,device) as num_threads:
        for task,entry in entries.items():
            args=load_config(root/entry['config'],project_root=root,overrides={'ckpt':str(root/entry['checkpoint'])})
            args.normalization_dtype=execution.get('normalization_dtype','float32')
            args.inference_block_size=execution.get('inference_block_size',0)
            args.inference_backend=execution.get('inference_backend',{})
            args.reference_device=None
            if torch.device(device).type=='cuda' and not torch.cuda.is_available():
                raise RuntimeError('This cohort uses CUDA inference. See docs/applications.md for its execution settings.')
            with inference_backend(args,device) as backend:
                model,_=_load_model(args,torch.device(device))
                predicted=predict_frame(model,frame,args,device,batch_size)
                backends[task]=backend
            predicted['model_id']=entry['id']
            rows.append(predicted)
            del model
    output.parent.mkdir(parents=True,exist_ok=True)
    pd.concat(rows,ignore_index=True).to_csv(output,index=False)
    metadata={'cohort':cli.cohort,'spectrum_type':selected['spectrum_type'],
              'input':str(cli.input or selected['prepared_input']['path']),
              'models':{t:selected['models'][t] for t in tasks},
              'device':device,'batch_size':batch_size,
              'normalization_dtype':execution.get('normalization_dtype','float32'),
              'inference_block_size':execution.get('inference_block_size',0),
              'inference_backend':backends,
              'dtype':'float32','python_version':platform.python_version(),
              'torch_version':torch.__version__,
              'torch_num_threads':num_threads,
              'model_assets':{t:{'model_id':entry['id'],
                                  'checkpoint':entry['checkpoint'],
                                  'config':entry['config']}
                              for t,entry in entries.items()},
              'preprocessing':'Input is already prepared and energy-referenced; no registration or interpolation is performed by this command.'}
    output.with_suffix('.metadata.json').write_text(json.dumps(metadata,indent=2)+'\n')
    print(f'Wrote {output}')


if __name__=='__main__':main()
