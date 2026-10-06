"""Development-only exploration of sparse polynomial regression."""
import os
os.environ['OPENBLAS_NUM_THREADS'] = '1'
os.environ['OMP_NUM_THREADS'] = '1'
import time, json, warnings
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.linear_model import lars_path, Ridge
from sklearn.preprocessing import PolynomialFeatures, StandardScaler
from sklearn.model_selection import KFold, train_test_split
from sklearn.metrics import mean_squared_error
from threadpoolctl import threadpool_limits

ROOT=Path(__file__).resolve().parents[1]
DATA_DIR=Path(os.environ.get('ASSIGNMENT_DATA_DIR',str(ROOT/'data')))
OUT=Path(os.environ.get('ASSIGNMENT_SEARCH_DIR',str(ROOT/'search_results')))
OUT.mkdir(parents=True,exist_ok=True)
ALPHAS = np.array([1., .3, .1, .06, .03, .01, .003])
DEGREES = {1: [4, 5, 6, 8, 10], 2: [6, 8, 10, 12, 16, 20]}

def expand(x, degree, basis):
    poly = PolynomialFeatures(degree, include_bias=False).fit(x)
    if basis == 'monomial':
        return poly.transform(x)
    powers = poly.powers_
    vanders = [np.polynomial.legendre.legvander(x[:,j], degree) for j in range(x.shape[1])]
    terms = np.ones((len(x),len(powers)))
    for j in range(x.shape[1]):
        terms *= vanders[j][:, powers[:,j]]
    return terms

def main():
    all_rows=[]
    start=time.monotonic()
    with threadpool_limits(limits=1):
        for problem in (1,2):
            data=pd.read_csv(DATA_DIR / f'BT2024016_train_var{problem}.csv')
            dev,_=train_test_split(data,test_size=.2,random_state=42)
            x=dev.drop(columns='y').to_numpy(); y=dev.y.to_numpy()
            splits=list(KFold(5,shuffle=True,random_state=42).split(x))
            for basis in ('monomial','legendre'):
                for degree in DEGREES[problem]:
                    z=expand(x,degree,basis)
                    losses=np.zeros((len(ALPHAS),3,5)); sizes=np.zeros((len(ALPHAS),5))
                    for fold,(i,j) in enumerate(splits):
                        scaler=StandardScaler().fit(z[i]); a=np.asfortranarray(scaler.transform(z[i])); b=scaler.transform(z[j])
                        ym=y[i].mean(); yc=y[i]-ym
                        with warnings.catch_warnings(record=True) as ws:
                            aa,active,cc=lars_path(a,yc,method='lasso',alpha_min=ALPHAS.min(),max_iter=400,verbose=False)
                        for k,alpha in enumerate(ALPHAS):
                            if alpha < aa[-1]*(1-1e-8):
                                losses[k,:,fold]=np.nan
                                continue
                            coef=np.array([np.interp(alpha,aa[::-1],v[::-1]) for v in cc])
                            support=np.abs(coef)>1e-9
                            sizes[k,fold]=support.sum()
                            prediction=b@coef+ym
                            losses[k,0,fold]=mean_squared_error(y[j],prediction)
                            if support.sum():
                                rr=Ridge(alpha=.01).fit(a[:,support],yc)
                                refit=rr.predict(b[:,support])+ym
                                losses[k,1,fold]=mean_squared_error(y[j],(prediction+refit)/2)
                                losses[k,2,fold]=mean_squared_error(y[j],refit)
                            else:
                                losses[k,1:,fold]=losses[k,0,fold]
                        print(f'var{problem} {basis} d={degree} fold={fold+1} path={len(aa)} warnings={len(ws)} elapsed={time.monotonic()-start:.1f}s',flush=True)
                    for k,alpha in enumerate(ALPHAS):
                        for l,relax in enumerate((0.,.5,1.)):
                            loss=losses[k,l]
                            row={'problem':problem,'basis':basis,'degree':degree,'alpha':float(alpha),'relax':relax,
                                 'cv_mse':float(loss.mean()),'cv_se':float(loss.std(ddof=1)/np.sqrt(5)),
                                 'mean_selected_terms':float(sizes[k].mean()), 'candidate_terms':int(z.shape[1])}
                            row.update({f'fold_{f+1}_mse':float(v) for f,v in enumerate(loss)})
                            all_rows.append(row)
                    pd.DataFrame(all_rows).to_csv(OUT/'sparse_search.csv',index=False)
                    rr=[r for r in all_rows if r['problem']==problem and np.isfinite(r['cv_mse'])]
                    print('CURRENT BEST',json.dumps(min(rr,key=lambda r:r['cv_mse'])),flush=True)

if __name__ == '__main__':
    main()
