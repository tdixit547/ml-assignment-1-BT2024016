"""Verify degree, portable inference, row order and bad-input handling."""
import json
from pathlib import Path
import subprocess,sys,tempfile,unittest
import numpy as np
import pandas as pd
from threadpoolctl import threadpool_limits
PROJECT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(PROJECT))
from polynomial_model import fit_model,predict_model,save_model,load_model
from assignment import read_dataset

class WorkflowTests(unittest.TestCase):
    def test_known_polynomial_portable_inference_and_row_order(self):
        rng=np.random.default_rng(40)
        x=rng.uniform(-1,1,(240,3)); t=rng.uniform(-1,1,(39,3)); t[8]=t[2]
        def target(z):
            return 2+3*z[:,0]**3-2*z[:,1]*z[:,2]+z[:,2]**2
        spec={'components':[{'method':'ridge','basis':'legendre','degree':3,'alpha':1e-8},
                            {'method':'ridge','basis':'monomial','degree':3,'alpha':1e-8}],
              'weights':[1.,1.]}
        with threadpool_limits(limits=1):
            model=fit_model(x,target(x),['x1','x2','x3'],spec)
        np.testing.assert_allclose(predict_model(model,t),target(t),rtol=1e-7,atol=1e-7)
        for m in model['components']:
            self.assertEqual(m['fit_rows'],240)
            self.assertTrue((np.array(m['powers']).sum(1)<=3).all())
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp); save_model(model,root/'model.json')
            test=pd.DataFrame(t,columns=['x1','x2','x3']); test['y']=0; test['id']=np.arange(39)[::-1]
            test[['id','x3','x2','x1','y']].to_csv(root/'test.csv',index=False)
            subprocess.run([sys.executable,str(PROJECT/'assignment.py'),'predict',
                            '--model',str(root/'model.json'),'--test',str(root/'test.csv'),
                            '--output',str(root/'pred.csv')],check=True,capture_output=True,text=True)
            result=pd.read_csv(root/'pred.csv')
            self.assertEqual(result.columns.tolist(),['y']); self.assertEqual(len(result),39)
            np.testing.assert_allclose(result.y,target(t),rtol=1e-7,atol=1e-7)
            self.assertEqual(result.y[2],result.y[8])
            np.testing.assert_allclose(predict_model(load_model(root/'model.json'),t[::-1]),result.y[::-1],rtol=1e-12,atol=1e-12)

    def test_sparse_and_relaxed_models_remain_polynomials(self):
        rng=np.random.default_rng(9); x=rng.uniform(-1,1,(160,2)); y=1+x[:,0]**3-2*x[:,1]
        specs=[{'method':'lasso','degree':3,'alpha':.001,'relax':.5},{'method':'ard','degree':3}]
        with threadpool_limits(limits=1):
            for spec in specs:
                m=fit_model(x,y,['x1','x2'],spec)
                self.assertLess(np.mean((predict_model(m,x)-y)**2),1e-4)
                self.assertTrue((np.array(m['powers']).sum(1)<=3).all())
                before=json.dumps(m,sort_keys=True); predict_model(m,x[:3]*1.5)
                self.assertEqual(json.dumps(m,sort_keys=True),before)

    def test_invalid_csvs_are_rejected(self):
        with tempfile.TemporaryDirectory() as temp:
            p=Path(temp)/'bad.csv'; p.write_text('x1,x1,y\n1,2,3\n')
            with self.assertRaisesRegex(ValueError,'duplicate'):
                read_dataset(p,['x1'],training=True)
            p.write_text('x1,y\n,3\n')
            with self.assertRaisesRegex(ValueError,'missing or infinite'):
                read_dataset(p,['x1'],training=True)
            p.write_text('x1,y\n1,3\n')
            with self.assertRaisesRegex(ValueError,'missing columns'):
                read_dataset(p,['x1','x2'],training=True)

if __name__=='__main__':
    unittest.main()
