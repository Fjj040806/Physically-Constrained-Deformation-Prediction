import sys,unittest
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
import numpy as np
from scipy.optimize._numdiff import approx_derivative
from mechanics import energy_gradient,solve,centerline
from network import initialize,forward,backward

class Numerics(unittest.TestCase):
    def test_energy_gradient(self):
        t=np.linspace(.01,.4,12)
        numeric=approx_derivative(lambda z:energy_gradient(z,[1.2],[.3])[0],t).ravel()
        np.testing.assert_allclose(energy_gradient(t,[1.2],[.3])[1][0],numeric,atol=1e-7)
    def test_network_gradient(self):
        w=initialize(2); x=np.array([[.3,.2],[-.4,.7]])
        y,c=forward(w,x); g=backward(w,c,2*y)
        for i in [0,23,70,500,len(w)-1]:
            d=np.zeros_like(w);d[i]=1e-5
            fd=((forward(w+d,x)[0]**2).sum()-(forward(w-d,x)[0]**2).sum())/2e-5
            self.assertAlmostEqual(g[i],fd,places=7)
    def test_small_load(self):
        n=128;s=np.linspace(0,1,n+1)[1:];q=1e-3
        np.testing.assert_allclose(solve(q,0,n),q*(s-s*s/2),atol=2e-7)
    def test_symmetry_and_zero(self):
        np.testing.assert_allclose(solve(0,.3,64),0,atol=1e-12)
        np.testing.assert_allclose(solve(-1.2,.2,64),-solve(1.2,.2,64),atol=1e-8)
    def test_arclength(self):
        xy=centerline(solve(2,.3,128))[0]
        length=np.linalg.norm(np.diff(xy,axis=0),axis=1).sum()
        self.assertLess(abs(length-1),1e-4)
if __name__=='__main__':unittest.main()
