import os
# The experiment uses a single BLAS thread for reproducibility.
import sys, json, math, time
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge
from sklearn.model_selection import KFold, cross_validate, train_test_split
from threadpoolctl import threadpool_limits
sys.path.insert(0, str(Path(__file__).parent))
from assignment_baseline import make_model, FEATURES

ROOT=Path(__file__).resolve().parents[1]
DATA_DIR=Path(os.environ.get('ASSIGNMENT_DATA_DIR',str(ROOT/'data')))
OUT=Path(os.environ.get('ASSIGNMENT_SEARCH_DIR',str(ROOT/'search_results')))
OUT.mkdir(parents=True,exist_ok=True)
ALPHAS = [0., 1e-6, 1e-5, 1e-4, 1e-3, .01, .1, 1., 10., 100., 1000.]
manifest = {'seed': 42, 'folds': 5, 'holdout_fraction': .2,
            'all_feature_degrees': {'1': [1,6], '2': [1,12]},
            'alphas': ALPHAS,
            'selection': 'fewest terms within one SE of minimum CV MSE, then strongest eligible alpha',
            'holdout_used_in_search': False}
(OUT/'search_plan.json').write_text(json.dumps(manifest, indent=2))
start = time.monotonic()
with threadpool_limits(limits=1):
    for problem, max_degree in [(1,6),(2,12)]:
        data = pd.read_csv(DATA_DIR / f'BT2024016_train_var{problem}.csv')
        development, _ = train_test_split(data, test_size=.2, random_state=42)
        cv = KFold(5, shuffle=True, random_state=42)
        specs = [(FEATURES[problem], d, a) for d in range(1,max_degree+1) for a in ALPHAS]
        rows = []
        for features, degree, alpha in specs:
            terms = math.comb(len(features)+degree,degree)-1
            row = {'features': ','.join(features), 'degree':degree,'alpha':alpha,
                   'terms':terms, 'status':'ok'}
            if alpha == 0 and terms+1 >= 640:
                row['status']='skipped: underdetermined OLS'
            else:
                model = make_model(degree)
                if alpha > 0:
                    model.set_params(regression=Ridge(alpha=alpha, solver='svd'))
                result = cross_validate(model, development[features], development.y, cv=cv,
                                        scoring='neg_mean_squared_error',return_train_score=True,
                                        error_score='raise',n_jobs=1)
                losses = -result['test_score']
                row.update(cv_mse=float(losses.mean()), cv_mse_se=float(losses.std(ddof=1)/np.sqrt(5)),
                           train_mse=float(-result['train_score'].mean()))
                row.update({f'fold_{i+1}_mse':float(v) for i,v in enumerate(losses)})
            rows.append(row)
            pd.DataFrame(rows).to_csv(OUT/f'var{problem}_search.csv',index=False)
            if alpha == ALPHAS[-1] or len(features)<len(FEATURES[problem]):
                valid = [r for r in rows if r['status']=='ok' and r['degree']==degree and r['features']==row['features']]
                best = min(valid,key=lambda r:r['cv_mse'])
                print(f"var{problem} p={len(features)} d={degree}: best alpha={best['alpha']}, CV={best['cv_mse']:.9g} SE={best['cv_mse_se']:.6g}; elapsed {time.monotonic()-start:.1f}s",flush=True)
        valid = [r for r in rows if r['status']=='ok']
        best = min(valid,key=lambda r:r['cv_mse'])
        threshold=best['cv_mse']+best['cv_mse_se']+1e-12*development.y.var(ddof=0)
        eligible=[r for r in valid if r['cv_mse']<=threshold]
        chosen=min(eligible,key=lambda r:(r['terms'],-r['alpha'],r['cv_mse'],r['features']))
        (OUT/f'var{problem}_selection.json').write_text(json.dumps({'best':best,'chosen':chosen,'threshold':threshold},indent=2))
        print('SELECTED', problem, json.dumps(chosen),flush=True)
