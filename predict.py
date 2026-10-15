"""Example: python predict.py --load 2.5 --taper -0.2 --method hybrid"""
import sys,argparse
from pathlib import Path
import numpy as np
sys.path.insert(0,str(Path(__file__).parent/'src'))
from network import predict
from mechanics import centerline
p=argparse.ArgumentParser();p.add_argument('--load',type=float,default=2.5);p.add_argument('--taper',type=float,default=-.2);p.add_argument('--method',choices=['data','energy','hybrid'],default='hybrid');p.add_argument('--output',default='prediction.csv');a=p.parse_args()
if a.taper<=-1: p.error('EI must stay positive: taper must exceed -1.')
w=np.load(Path(__file__).parent/f'models/{a.method}_seed0.npz')['weights'];t=predict(w,[[a.load,a.taper]])[0];xy=centerline(t)[0]
np.savetxt(a.output,np.column_stack([np.linspace(0,1,len(xy)),xy,np.r_[0,t]]),delimiter=',',header='s,x,y,theta',comments='')
print(f'Saved {a.output}; tip = {xy[-1]}. Parameters outside training range are extrapolation.')
