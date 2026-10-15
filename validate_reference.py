"""Independent continuous BVP validation of the discrete energy solver."""
import sys,json
from pathlib import Path
import numpy as np
from scipy.integrate import solve_bvp
sys.path.insert(0,str(Path(__file__).parent/'src'))
from mechanics import solve

def main():
 rows=[]
 for q,a in [(1,0),(3,-.4),(-2,.6)]:
  s=np.linspace(0,1,100)
  def fun(s,y):return np.vstack([y[1]/(1+a*s),-q*np.cos(y[0])])
  def bc(l,r):return np.array([l[0],r[1]])
  sol=solve_bvp(fun,bc,s,np.vstack([q*(s-s*s/2)/(1+a/2),q*(1-s)]),tol=1e-9)
  assert sol.success
  t=solve(q,a,256);err=np.max(np.abs(t-sol.sol(np.linspace(0,1,257)[1:])[0]))
  rows.append(dict(q=q,alpha=a,bvp_success=bool(sol.success),max_angle_difference=float(err)))
  assert err<1e-4
 (Path(__file__).parent/'results/bvp_validation.json').write_text(json.dumps(rows,indent=2))
 print('Independent BVP checks passed.')
if __name__=='__main__':main()
