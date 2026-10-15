"""Generate all figures and GIFs from saved models; no hand-authored trajectories."""
import sys,csv,json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation,PillowWriter
sys.path.insert(0,str(Path(__file__).parent/'src'))
from mechanics import centerline,solve
from network import predict
R=Path(__file__).parent
COL={'reference':'#182638','data':'#cf6546','energy':'#247e9c','hybrid':'#7153a1'}
plt.rcParams.update({'font.family':'DejaVu Sans','font.size':10,'axes.spines.top':False,'axes.spines.right':False,'figure.dpi':140})

def main():
 rows=list(csv.DictReader(open(R/'results/metrics.csv')))
 methods=['data','energy','hybrid'];splits=['interpolation','load_extrapolation','stiffness_extrapolation']
 fig,axs=plt.subplots(1,3,figsize=(11,3.5),layout='constrained')
 for ax,split in zip(axs,splits):
  for i,m in enumerate(methods):
   v=[float(r['tip_error_mean']) for r in rows if r['method']==m and r['split']==split]
   ax.bar(i,np.mean(v),color=COL[m],alpha=.8);ax.errorbar(i,np.mean(v),yerr=np.std(v,ddof=1),fmt='none',color='black',capsize=4)
   ax.scatter(np.full(3,i),v,s=12,color='black',zorder=4)
  ax.set_xticks(range(3),methods);ax.set_title(split.replace('_',' ').title());ax.set_ylabel('Mean tip error / finger length');ax.grid(axis='y',alpha=.2)
 fig.savefig(R/'figures/benchmark.png');plt.close(fig)
 p=np.array([[q,a] for q,a in [(1,0),(2.5,-.2),(-2.5,.3)]])
 weights={m:np.load(R/f'models/{m}_seed0.npz')['weights'] for m in methods}
 fig,axs=plt.subplots(1,3,figsize=(11,3.7),layout='constrained')
 for ax,param in zip(axs,p):
  ref=centerline(solve(*param,256))[0];ax.plot(ref[:,0],ref[:,1],color=COL['reference'],lw=3,label='Reference')
  for m,w in weights.items():
   xy=centerline(predict(w,param[None,:]))[0];ax.plot(xy[:,0],xy[:,1],'--',color=COL[m],label=m.title())
  ax.plot([0,0],[-.12,.12],color='black',lw=4);ax.set_aspect('equal');ax.set_title(f'q = {param[0]:g}, alpha = {param[1]:g}');ax.set_xlabel('x / L');ax.set_ylabel('y / L');ax.grid(alpha=.2)
 axs[0].legend(fontsize=8);fig.savefig(R/'figures/shapes.png');plt.close(fig)
 conv=json.loads((R/'results/convergence.json').read_text()); n=np.array([z['segments'] for z in conv]);xy=np.array([[z['tip_x'],z['tip_y']] for z in conv]);err=np.linalg.norm(xy[:-1]-xy[-1],axis=1)
 fig,ax=plt.subplots(figsize=(5.8,3.5),layout='constrained');ax.loglog(n[:-1],err,'o-',color=COL['energy']);ax.set_xlabel('Reference segments');ax.set_ylabel('Tip difference from 512 segments');ax.grid(which='both',alpha=.2);fig.savefig(R/'figures/convergence.png');plt.close(fig)
 # Quasistatic load sweep: each frame is an equilibrium, not dynamic time.
 qs=2.8*np.sin(np.linspace(0,2*np.pi,65));alpha=-.2
 params=np.column_stack([qs,np.full(len(qs),alpha)])
 refs=np.array([centerline(solve(q,alpha,128))[0] for q in qs])
 preds={m:centerline(predict(w,params)) for m,w in weights.items()}
 fig,axs=plt.subplots(1,2,figsize=(10,4.5),layout='constrained');lines={}
 ax=axs[0]
 ax.axhline(0,color='#cccccc',lw=.8);ax.plot([0,0],[-.12,.12],color='black',lw=5)
 for m in ['reference']+methods:lines[m],=ax.plot([],[],color=COL[m],lw=2.5 if m=='reference' else 1.7,ls='-' if m=='reference' else '--',label=m.title())
 ax.set(xlim=(-.08,1.05),ylim=(-.85,.85),xlabel='x / L',ylabel='y / L',title='Flexible finger equilibria');ax.set_aspect('equal');ax.legend(loc='upper right',fontsize=8);ax.grid(alpha=.15)
 ax2=axs[1]
 for m in methods:ax2.plot(qs,np.linalg.norm(preds[m][:,-1]-refs[:,-1],axis=1),color=COL[m],label=m.title())
 cursor=ax2.axvline(0,color='gray',ls=':');ax2.set(xlabel='Dimensionless transverse tip load q',ylabel='Tip error / L',title='Error across the load sweep');ax2.axvspan(-1.5,1.5,color='gray',alpha=.12,label='Training load range');ax2.legend(fontsize=8);ax2.grid(alpha=.15)
 title=fig.suptitle('')
 def update(i):
  for m,line in lines.items():
   xy=refs[i] if m=='reference' else preds[m][i];line.set_data(xy[:,0],xy[:,1])
  cursor.set_xdata([qs[i],qs[i]]);title.set_text(f'Quasistatic load sweep | q = {qs[i]:+.2f} | alpha = {alpha:+.1f} | seed 0')
  return list(lines.values())+[cursor,title]
 ani=FuncAnimation(fig,update,frames=len(qs),interval=80);ani.save(R/'animations/load_sweep.gif',writer=PillowWriter(fps=12));update(10);fig.savefig(R/'figures/animation_still.png');plt.close(fig)
 # Stiffness sweep at fixed load.
 alphas=np.linspace(-.65,.65,41);params=np.column_stack([np.full(41,1.3),alphas]);refs=np.array([centerline(solve(1.3,a,128))[0] for a in alphas]);pred=centerline(predict(weights['hybrid'],params))
 fig,ax=plt.subplots(figsize=(6.2,4.6),layout='constrained');l1,=ax.plot([],[],lw=3,color=COL['reference'],label='Reference');l2,=ax.plot([],[],'--',lw=2,color=COL['hybrid'],label='Hybrid NN');ax.plot([0,0],[-.1,.1],lw=5,color='black');ax.set(xlim=(-.05,1.05),ylim=(-.12,.8),xlabel='x / L',ylabel='y / L');ax.set_aspect('equal');ax.legend();ax.grid(alpha=.2)
 def upd(i):
  l1.set_data(refs[i,:,0],refs[i,:,1]);l2.set_data(pred[i,:,0],pred[i,:,1]);ax.set_title(f'Stiffness sweep: EI(s) = 1 + alpha s\nq = 1.3, alpha = {alphas[i]:+.2f}');return l1,l2
 ani=FuncAnimation(fig,upd,frames=list(range(41))+list(range(39,0,-1)),interval=90);ani.save(R/'animations/stiffness_sweep.gif',writer=PillowWriter(fps=12));plt.close(fig)
if __name__=='__main__':main()
