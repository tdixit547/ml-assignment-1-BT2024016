"""Rerun the recorded development searches. No holdout score enters selection."""
import argparse,hashlib,json,os,subprocess,sys
from pathlib import Path

def main():
    root=Path(__file__).resolve().parents[1]
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data-dir',default=str(root/'data'))
    parser.add_argument('--out',default=str(root/'search_results'))
    parser.add_argument('--include-expensive',action='store_true',help='Also rerun the rejected degree-six ARD fits')
    args=parser.parse_args()
    data=Path(args.data_dir).resolve(); out=Path(args.out).resolve()
    if out.exists() and any(out.iterdir()):
        raise FileExistsError('Choose a new output directory for the search.')
    out.mkdir(parents=True,exist_ok=True)
    env=dict(os.environ,ASSIGNMENT_DATA_DIR=str(data),ASSIGNMENT_SEARCH_DIR=str(out),OPENBLAS_NUM_THREADS='1')
    for name in ['expanded_ridge_search.py','sparse_search.py','ridge_basis_search.py',
                 'fine_sparse_search.py','ard_search.py','repeated_cv.py']:
        command=[sys.executable,'-u',str(root/'experiments'/name)]
        if name=='ard_search.py' and args.include_expensive:
            command.append('--include-expensive')
        subprocess.run(command,env=env,check=True)
    selection=json.loads((out/'frozen_selection.json').read_text())
    config={'selection_rule':'Minimum mean MSE across 3 x 5 development folds among 11 frozen finalists per problem.',
            'input_sha256':{p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(data.glob('BT2024016_*.csv'))},
            'selection':selection}
    (out/'selected_config.json').write_text(json.dumps(config,indent=2))
    print('Search complete. To fit and evaluate, pass --config',out/'selected_config.json','to assignment.py train.')

if __name__=='__main__':
    main()
