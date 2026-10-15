"""Reproduce the full benchmark: python run_experiments.py."""
import sys,json,time,csv,platform
from pathlib import Path
import numpy as np
import scipy,matplotlib,PIL,reportlab
sys.path.insert(0,str(Path(__file__).parent/'src'))
from mechanics import solve,centerline,energy_gradient
from network import train,predict
ROOT=Path(__file__).parent

def reference(p,n=256):
    fine=np.array([solve(q,a,n) for q,a in p])
    return fine[:,np.arange(4,n+1,4)-1] if n==256 else fine

def main():
    for d in ['results','models']: (ROOT/d).mkdir(exist_ok=True)
    rng=np.random.default_rng(20261016)
    labels=rng.uniform([-1.5,-.4],[1.5,.4],(24,2))
    physical=rng.uniform([-1.5,-.4],[1.5,.4],(256,2))
    tests={'interpolation':rng.uniform([-1.5,-.4],[1.5,.4],(80,2)),
           'load_extrapolation':np.column_stack([rng.choice([-1,1],80)*rng.uniform(1.5,3,80),rng.uniform(-.4,.4,80)]),
           'stiffness_extrapolation':np.column_stack([rng.uniform(-1.5,1.5,80),rng.choice([-1,1],80)*rng.uniform(.4,.65,80)])}
    targets=reference(labels); refs={k:reference(v) for k,v in tests.items()}
    np.savez_compressed(ROOT/'results/dataset.npz',labels=labels,targets=targets,physical=physical,**{k:v for k,v in tests.items()},**{k+'_reference':v for k,v in refs.items()})
    rows=[]; logs={}
    for seed in [0,1,2]:
        for kind in ['data','energy','hybrid']:
            start=time.perf_counter(); w,info=train(seed,kind,labels,targets,physical)
            elapsed=time.perf_counter()-start;np.savez(ROOT/f'models/{kind}_seed{seed}.npz',weights=w)
            logs[f'{kind}_{seed}']={**info,'training_seconds':elapsed}
            for split,p in tests.items():
                t=predict(w,p); ref=refs[split];xy=centerline(t);xr=centerline(ref)
                e,_=energy_gradient(t,p[:,0],p[:,1]);er,_=energy_gradient(ref,p[:,0],p[:,1])
                _,g=energy_gradient(t,p[:,0],p[:,1])
                rows.append(dict(seed=seed,method=kind,split=split,angle_relative_l2=float(np.linalg.norm(t-ref)/np.linalg.norm(ref)),centerline_rmse=float(np.sqrt(np.mean((xy-xr)**2))),tip_error_mean=float(np.linalg.norm(xy[:,-1]-xr[:,-1],axis=1).mean()),energy_gap_mean=float(np.mean(e-er)),stationarity_rms=float(np.sqrt(np.mean((g*64)**2))),training_seconds=elapsed))
            print(seed,kind,round(elapsed,2),info['iterations'],flush=True)
    with open(ROOT/'results/metrics.csv','w') as f:
        wr=csv.DictWriter(f,fieldnames=list(rows[0]),lineterminator='\n');wr.writeheader();wr.writerows(rows)
    (ROOT/'results/training.json').write_text(json.dumps(logs,indent=2))
    convergence=[]
    for n in [16,32,64,128,256,512]:
        t=solve(3,-.4,n);xy=centerline(t)[0]
        convergence.append({'segments':n,'tip_x':float(xy[-1,0]),'tip_y':float(xy[-1,1]),'energy':float(energy_gradient(t,[3],[-.4])[0][0])})
    (ROOT/'results/convergence.json').write_text(json.dumps(convergence,indent=2))
    (ROOT/'results/environment.json').write_text(json.dumps({'python':platform.python_version(),'platform':platform.platform(),'numpy':np.__version__,'scipy':scipy.__version__,'matplotlib':matplotlib.__version__,'Pillow':PIL.__version__,'reportlab':reportlab.Version,'seed':20261016,'reference_segments':256,'neural_segments':64,'modes':12,'label_count':24,'physical_count':256,'seeds':[0,1,2]},indent=2))
if __name__=='__main__':main()
