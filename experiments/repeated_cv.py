"""Freeze a shortlist, compare it on three five-fold development partitions."""
import os
os.environ['OPENBLAS_NUM_THREADS']='1'
import sys,time,json
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split,KFold
from threadpoolctl import threadpool_limits
ROOT=Path(__file__).resolve().parents[1]
DATA_DIR=Path(os.environ.get('ASSIGNMENT_DATA_DIR',str(ROOT/'data')))
OUT=Path(os.environ.get('ASSIGNMENT_SEARCH_DIR',str(ROOT/'search_results')))
OUT.mkdir(parents=True,exist_ok=True)
sys.path.insert(0,str(ROOT))
from polynomial_model import fit_model,predict_model

SEEDS=[42,137,2026]
def lasso(d,a,b='monomial',relax=0.):
    return dict(method='lasso',basis=b,degree=d,alpha=a,relax=relax)
def ridge(d,a,b='legendre',p=2.):
    return dict(method='ridge',basis=b,degree=d,alpha=a,penalty=p)

SPECS={1:{
 'lasso5_005':lasso(5,.005), 'lasso5_008':lasso(5,.008),
 'lasso5_010':lasso(5,.01), 'relaxed5_015':lasso(5,.015,relax=.5),
 'relaxed5_020':lasso(5,.02,relax=.5), 'refit5_030':lasso(5,.03,relax=1.),
 'lasso6_010':lasso(6,.01), 'ard5':dict(method='ard',basis='monomial',degree=5),
 'ridge5_legendre':ridge(5,.1),
},2:{
 'ridge12_legendre':ridge(12,.01), 'ridge14_legendre':ridge(14,.01),
 'ridge11_legendre':ridge(11,.01), 'ridge8_legendre':ridge(8,float(10**-.5),p=1.),
 'relaxed9_001':lasso(9,.001,relax=.5), 'refit12_003':lasso(12,.003,relax=1.),
 'refit8_legendre':lasso(8,.02,b='legendre',relax=1.),
 'refit9_legendre':lasso(9,.015,b='legendre',relax=1.),
 'lasso9_0005':lasso(9,.0005),
}}
BLENDS={1:{'average_lasso_ard':['lasso5_008','ard5'],
           'average_lasso_relaxed':['lasso5_008','relaxed5_015']},
        2:{'average_ridge_lasso':['ridge12_legendre','relaxed9_001'],
           'average_three':['ridge12_legendre','relaxed9_001','refit8_legendre']}}

def main():
    # This manifest is written before any repeated-fold or holdout score is seen.
    (OUT/'finalist_specs.json').write_text(json.dumps({'seeds':SEEDS,'specs':SPECS,'blends':BLENDS},indent=2))
    rows=[]; chosen={}; start=time.monotonic()
    with threadpool_limits(limits=1):
        for problem in (1,2):
            d=pd.read_csv(DATA_DIR/f'BT2024016_train_var{problem}.csv')
            dev,_=train_test_split(d,test_size=.2,random_state=42)
            features=d.columns.drop('y').tolist(); x=dev[features].to_numpy(); y=dev.y.to_numpy()
            splitlist=[(r,i,j) for r,seed in enumerate(SEEDS) for i,j in KFold(5,shuffle=True,random_state=seed).split(x)]
            preds={}; all_specs=dict(SPECS[problem])
            for name,spec in SPECS[problem].items():
                pred=np.empty((len(SEEDS),len(y))); counts=[]
                for r,i,j in splitlist:
                    model=fit_model(x[i],y[i],features,spec)
                    pred[r,j]=predict_model(model,x[j]); counts.append(model['active_terms'])
                preds[name]=pred
                losses=[float(np.mean((pred[r,j]-y[j])**2)) for r,i,j in splitlist]
                row={'problem':problem,'candidate':name,'degree':spec['degree'],
                     'cv_mse':float(np.mean(losses)),'fold_mse_sd':float(np.std(losses,ddof=1)),
                     'mean_active_terms':float(np.mean(counts)),'components':1,'spec':json.dumps(spec)}
                row.update({f'fold_{i+1}_mse':v for i,v in enumerate(losses)})
                rows.append(row)
                pd.DataFrame(rows).to_csv(OUT/'repeated_cv.csv',index=False)
                print(f"var{problem} {name}: repeated CV={row['cv_mse']:.8f} elapsed={time.monotonic()-start:.1f}s",flush=True)
            for name,members in BLENDS[problem].items():
                pred=np.mean([preds[s] for s in members],axis=0); preds[name]=pred
                spec={'components':[SPECS[problem][s] for s in members],'weights':[1.]*len(members)}
                all_specs[name]=spec
                losses=[float(np.mean((pred[r,j]-y[j])**2)) for r,i,j in splitlist]
                row={'problem':problem,'candidate':name,'degree':max(s['degree'] for s in spec['components']),
                     'cv_mse':float(np.mean(losses)),'fold_mse_sd':float(np.std(losses,ddof=1)),
                     'mean_active_terms':None,'components':len(members),'spec':json.dumps(spec)}
                row.update({f'fold_{i+1}_mse':v for i,v in enumerate(losses)})
                rows.append(row)
                print(f"var{problem} {name}: repeated CV={row['cv_mse']:.8f}",flush=True)
            selected=min((r for r in rows if r['problem']==problem),key=lambda r:(r['cv_mse'],r['degree'],r['components']))
            chosen[str(problem)]={'name':selected['candidate'],'spec':all_specs[selected['candidate']],
                                  'repeated_cv_mse':selected['cv_mse'],'fold_mse_sd':selected['fold_mse_sd'],
                                  'cv_seeds':SEEDS,'cv_folds':5,'development_rows':len(dev)}
            np.savez_compressed(OUT/f'var{problem}_finalist_oof.npz',y=y,rows=dev.index.to_numpy(),**preds)
            pd.DataFrame(rows).to_csv(OUT/'repeated_cv.csv',index=False)
            (OUT/'frozen_selection.json').write_text(json.dumps(chosen,indent=2))
            print('FROZEN',problem,json.dumps(chosen[str(problem)]),flush=True)

if __name__=='__main__':
    main()
