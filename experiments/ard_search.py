import os
os.environ['OPENBLAS_NUM_THREADS']='1'
import time,warnings,sys
import numpy as np
import pandas as pd
from sklearn.linear_model import ARDRegression
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import KFold,train_test_split
from threadpoolctl import threadpool_limits
from sparse_search import ROOT,OUT,DATA_DIR,expand

rows=pd.read_csv(OUT/'ard_search.csv').to_dict('records') if (OUT/'ard_search.csv').exists() else []
start=time.monotonic()
with threadpool_limits(limits=1):
    for problem,degrees in [(1,[5,6] if '--include-expensive' in sys.argv else [5]),(2,[8,9,10])]:
        df=pd.read_csv(DATA_DIR/f'BT2024016_train_var{problem}.csv')
        dev,_=train_test_split(df,test_size=.2,random_state=42)
        x=dev.drop(columns='y').to_numpy(); y=dev.y.to_numpy()
        for basis in ['monomial','legendre']:
            for degree in degrees:
                if any(r['problem']==problem and r['basis']==basis and r['degree']==degree for r in rows):
                    continue
                z=expand(x,degree,basis)
                loss=[]; count=[]; iterations=[]
                for i,j in KFold(5,shuffle=True,random_state=42).split(x):
                    s=StandardScaler().fit(z[i])
                    m=ARDRegression(max_iter=500,tol=1e-4).fit(s.transform(z[i]),y[i])
                    loss.append(np.mean((m.predict(s.transform(z[j]))-y[j])**2))
                    count.append(int(np.count_nonzero(m.coef_))); iterations.append(m.n_iter_)
                row={'problem':problem,'basis':basis,'degree':degree,'cv_mse':np.mean(loss),
                     'cv_se':np.std(loss,ddof=1)/np.sqrt(5),'mean_selected_terms':np.mean(count),'max_iter':max(iterations)}
                row.update({f'fold_{i+1}_mse':v for i,v in enumerate(loss)})
                rows.append(row); pd.DataFrame(rows).to_csv(OUT/'ard_search.csv',index=False)
                print(row,'elapsed',time.monotonic()-start,flush=True)
