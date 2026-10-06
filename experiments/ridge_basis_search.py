"""Compare polynomial bases and degree penalties using development folds only."""
import os
os.environ['OPENBLAS_NUM_THREADS']='1'
import time
import numpy as np
import pandas as pd
from scipy.linalg import svd
from sklearn.model_selection import KFold,train_test_split
from sklearn.preprocessing import PolynomialFeatures,StandardScaler
from threadpoolctl import threadpool_limits
from sparse_search import ROOT, OUT, DATA_DIR, expand

ALPHAS=np.logspace(-5,3,17)
DEGREES={1:[4,5,6,7,8],2:[6,7,8,9,10,11,12,14,16,18,20]}
PENALTIES=[0.,1.,2.]

def main():
    rows=[]; start=time.monotonic()
    with threadpool_limits(limits=1):
        for problem in (1,2):
            data=pd.read_csv(DATA_DIR/f'BT2024016_train_var{problem}.csv')
            dev,_=train_test_split(data,test_size=.2,random_state=42)
            x=dev.drop(columns='y').to_numpy(); y=dev.y.to_numpy()
            splits=list(KFold(5,shuffle=True,random_state=42).split(x))
            for basis in ['monomial','legendre']:
                for degree in DEGREES[problem]:
                    z=expand(x,degree,basis)
                    total_degrees=PolynomialFeatures(degree,include_bias=False).fit(x).powers_.sum(1)
                    losses=np.zeros((len(PENALTIES),len(ALPHAS),5))
                    for f,(i,j) in enumerate(splits):
                        ss=StandardScaler().fit(z[i]); a=ss.transform(z[i]); b=ss.transform(z[j]); ym=y[i].mean()
                        for k,penalty in enumerate(PENALTIES):
                            weights=total_degrees**(-penalty)
                            u,s,v=svd(a*weights,full_matrices=False,check_finite=False)
                            prediction=((b*weights)@v.T)@((s[:,None]/(s[:,None]**2+ALPHAS))*(u.T@(y[i]-ym))[:,None])+ym
                            losses[k,:,f]=((y[j,None]-prediction)**2).mean(0)
                    for k,penalty in enumerate(PENALTIES):
                        for l,alpha in enumerate(ALPHAS):
                            loss=losses[k,l]
                            row={'problem':problem,'basis':basis,'degree':degree,'penalty':penalty,'alpha':alpha,
                                 'cv_mse':float(loss.mean()),'cv_se':float(loss.std(ddof=1)/np.sqrt(5)),'terms':z.shape[1]}
                            row.update({f'fold_{f+1}_mse':float(v) for f,v in enumerate(loss)})
                            rows.append(row)
                    pd.DataFrame(rows).to_csv(OUT/'ridge_basis_search.csv',index=False)
                    best=min([r for r in rows if r['problem']==problem],key=lambda r:r['cv_mse'])
                    print(f"var{problem} {basis} d={degree}, best={best['cv_mse']:.7g} {best['basis']} d={best['degree']} alpha={best['alpha']:.5g} penalty={best['penalty']} elapsed={time.monotonic()-start:.1f}s",flush=True)

if __name__=='__main__':
    main()
