"""Refine sparse penalties near the degrees identified in the broad search."""
import numpy as np
import sparse_search as s
if __name__=='__main__':
    s.OUT=s.OUT/'fine_sparse'
    s.OUT.mkdir(parents=True,exist_ok=True)
    s.ALPHAS=np.array([.05,.03,.02,.015,.01,.008,.005,.003,.002,.001,.0005])
    s.DEGREES={1:[5,6],2:[8,9,10,11,12]}
    s.main()
