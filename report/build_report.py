"""Generate result tables from recorded experiments and compile the academic paper.
Requires a TeX installation providing pdflatex (e.g. TeX Live).
"""
from pathlib import Path
import csv,json,subprocess,shutil
import numpy as np
ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'report'
rows=list(csv.DictReader(open(ROOT/'results/metrics.csv')))
SPLITS=['interpolation','load_extrapolation','stiffness_extrapolation']
METHODS=['data','energy','hybrid']
NAMES={'data':'Supervised','energy':'Energy only','hybrid':'Hybrid'}

def values(split,method,metric):
 return np.array([float(r[metric]) for r in rows if r['split']==split and r['method']==method])

lines=[r'\begin{table}[htbp]',r'\centering\small',r'\caption{Tip error in units of $10^{-3}$ finger lengths: mean $\pm$ sample standard deviation across three initialization seeds. Each seed is evaluated on the same 80 test cases per split.}\label{tab:tip}',r'\begin{tabular}{lccc}\toprule',r'Test regime & Supervised & Energy only & Hybrid\\\midrule']
for sp in SPLITS:
 cells=[sp.replace('_',' ').capitalize()]
 for m in METHODS:
  v=values(sp,m,'tip_error_mean')*1000;cells.append(f'${v.mean():.3f} \\pm {v.std(ddof=1):.3f}$')
 lines.append(' & '.join(cells)+r'\\')
lines += [r'\bottomrule\end{tabular}',r'\end{table}']
(OUT/'tip_table.tex').write_text('\n'.join(lines)+'\n')
lines=[r'\begin{table}[htbp]',r'\centering\small',r'\caption{Mean relative angle error $E_\theta$ and coordinate-wise centerline RMSE $E_r$, averaged across initialization seeds. Both columns are expressed in units of $10^{-3}$.}\label{tab:secondary}',r'\begin{tabular}{llrr}\toprule',r'Test regime & Method & $10^3 E_\theta$ & $10^3 E_r$\\\midrule']
for sp in SPLITS:
 for i,m in enumerate(METHODS):
  lines.append(f"{sp.replace('_',' ').capitalize() if i==0 else ''} & {NAMES[m]} & {1000*values(sp,m,'angle_relative_l2').mean():.3f} & {1000*values(sp,m,'centerline_rmse').mean():.3f}"+r'\\')
 if sp!=SPLITS[-1]:lines.append(r'\midrule')
lines += [r'\bottomrule\end{tabular}',r'\end{table}']
(OUT/'secondary_table.tex').write_text('\n'.join(lines)+'\n')
checks=json.loads((ROOT/'results/bvp_validation.json').read_text())
lines=[r'\begin{table}[htbp]',r'\centering\small',r'\caption{Maximum nodal angle difference between the 256-segment reference and the independent continuous boundary-value solver.}\label{tab:bvp}',r'\begin{tabular}{rrc}\toprule',r'$q$ & $\alpha$ & Maximum difference (rad)\\\midrule']
for r in checks:
 lines.append(f"{r['q']:g} & {r['alpha']:g} & ${r['max_angle_difference']:.3e}$".replace('e-06',r'\times 10^{-6}').replace('e-07',r'\times 10^{-7}')+r'\\')
lines += [r'\bottomrule\end{tabular}',r'\end{table}']
(OUT/'verification_table.tex').write_text('\n'.join(lines)+'\n')
if not shutil.which('pdflatex'):raise SystemExit('Install TeX Live with pdflatex to compile the manuscript.')
for _ in range(2):
 result=subprocess.run(['pdflatex','-interaction=nonstopmode','-halt-on-error','paper.tex'],cwd=OUT,stdout=subprocess.PIPE,stderr=subprocess.STDOUT,text=True)
 if result.returncode:print(result.stdout[-6000:]);raise SystemExit(result.returncode)
print('Built report/paper.pdf from paper.tex and recorded metrics.')
