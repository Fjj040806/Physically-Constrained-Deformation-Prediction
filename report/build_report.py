"""Build the report PDF and Markdown from saved experiment outputs."""
from pathlib import Path
import csv,json,html
import numpy as np
from reportlab.platypus import SimpleDocTemplate,Paragraph,Spacer,Image,Table,TableStyle,PageBreak
from reportlab.lib import colors
from reportlab.lib.styles import getSampleStyleSheet,ParagraphStyle
from reportlab.lib.enums import TA_CENTER
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
import matplotlib
font_root=Path(matplotlib.get_data_path())/'fonts'/'ttf'
for name,file in [('Body','DejaVuSans.ttf'),('Body-Bold','DejaVuSans-Bold.ttf'),('Body-Oblique','DejaVuSans-Oblique.ttf'),('Body-BoldOblique','DejaVuSans-BoldOblique.ttf'),('Mono','DejaVuSansMono.ttf')]:
 pdfmetrics.registerFont(TTFont(name,str(font_root/file)))
pdfmetrics.registerFontFamily('Body',normal='Body',bold='Body-Bold',italic='Body-Oblique',boldItalic='Body-BoldOblique')
R=Path(__file__).resolve().parents[1]
rows=list(csv.DictReader(open(R/'results/metrics.csv')))
styles=getSampleStyleSheet()
for st in styles.byName.values(): st.fontName='Body-Bold' if 'Heading' in st.name or st.name=='Title' else 'Body'
styles.add(ParagraphStyle(name='BodyCustom',parent=styles['BodyText'],fontName='Body',fontSize=10,leading=14,spaceAfter=9));styles.add(ParagraphStyle(name='Eq',fontName='Mono',fontSize=9,leading=13,spaceBefore=7,spaceAfter=10));styles.add(ParagraphStyle(name='CaptionCustom',fontName='Body',fontSize=8.5,leading=11,textColor=colors.HexColor('#425065'),spaceAfter=10))
story=[];md=[]
def title(t):story.append(Paragraph(t,styles['Title']));md.append('# '+t+'\n')
def h(t):story.append(Paragraph(t,styles['Heading1']));md.append('## '+t+'\n')
def p(t):story.append(Paragraph(t,styles['BodyCustom']));md.append(t+'\n')
def eq(t):story.append(Paragraph(html.escape(t),styles['Eq']));md.append('```text\n'+t+'\n```\n')
def fig(name,caption,width=475):
 from PIL import Image as PI
 path=R/'figures'/name;w,ht=PI.open(path).size
 story.append(Image(str(path),width=width,height=width*ht/w));story.append(Paragraph(caption,styles['CaptionCustom']));md.append(f'![{caption}](../figures/{name})\n')
def page():story.append(PageBreak())
def table(data,widths=None):
 t=Table([[Paragraph(str(c),styles['CaptionCustom']) for c in r] for r in data],colWidths=widths,repeatRows=1,hAlign='LEFT');t.setStyle(TableStyle([('BACKGROUND',(0,0),(-1,0),colors.HexColor('#e9eef4')),('LINEBELOW',(0,0),(-1,0),.7,colors.HexColor('#62748b')),('VALIGN',(0,0),(-1,-1),'TOP'),('BOTTOMPADDING',(0,0),(-1,-1),7),('TOPPADDING',(0,0),(-1,-1),7)]));story.append(t);md.append('\n'.join('| '+' | '.join(map(str,r))+' |' for r in [data[0],['---']*len(data[0])]+data[1:])+'\n')

title('Physically Constrained Deformation Prediction for a Planar Flexible Finger')
p('<b>Jiajun Fu</b><br/>Duke Kunshan University<br/>Computational research report | October 2026')
h('Abstract')
p('This study tests whether variational physics constraints improve neural prediction of large-rotation deformation in a flexible-finger benchmark. A planar, inextensible Euler elastica with spatially varying bending stiffness is subjected to a conservative transverse tip load. Three parametric neural surrogates share the same architecture and kinematic representation: supervised, energy-only, and hybrid. Training uses 24 labeled equilibria and, where applicable, 256 unlabeled parameter samples. Evaluation uses 80 held-out cases per split and three initialization seeds. Reference solutions are checked against a continuous boundary-value solver, a small-load analytical solution, and mesh refinement. The energy-only model achieves a mean extrapolation tip error of 0.00254 finger lengths versus 0.00548 for supervised learning across initialization seeds. Reported results also quantify interpolation and stiffness extrapolation. Physics-informed training is evaluated empirically rather than presumed superior. The deliverable is a reproducible numerical benchmark, not experimental validation of a robotic gripper.')
h('1. Research question and scope')
p('Can minimum-potential-energy training improve deformation prediction under sparse supervision, and does any benefit persist outside the training parameter domain? The target is a load-to-shape map for a flexible cantilever motivated by soft robotic fingers. This model captures finite rotations and nonlinear load response while retaining a simple, independently verifiable mechanics problem.')
p('The body is a slender planar rod with negligible extension and shear. There is no contact, pressure chamber, viscoelasticity, inertia, or three-dimensional deformation. Accordingly, this study supports conclusions about reduced-order elastic fingers only. It does not establish performance for a fabricated pneumatic gripper.')
p('The variational-learning premise follows the Deep Ritz method [1] and the deep energy approach to mechanics [2]. The present work implements a small parametric neural coefficient map and a transparent benchmark protocol; it does not claim to introduce energy-based neural mechanics.')
page()
h('2. Mechanics and variational formulation')
p('Let S denote physical arclength, L the finger length, B0 a reference bending stiffness, F the transverse tip force, and s = S/L. Define q = FL^2/B0 and B(s)/B0 = 1 + alpha s. Positive stiffness requires alpha &gt; -1. The rotation field theta(s) determines the dimensionless centerline r(s).')
eq("r'(s) = (cos(theta(s)), sin(theta(s)))\nr(0) = (0,0), theta(0) = 0\nPi[theta;q,alpha] = (1/2) integral_0^1 (1+alpha*s) theta'(s)^2 ds\n                    - q integral_0^1 sin(theta(s)) ds")
p('The first term is bending energy divided by B0/L; the second is the potential of a dead load fixed in the laboratory transverse direction. The nonlinear trigonometric kinematics distinguish this model from small-deflection beam theory. A follower force would produce a different problem and is not used.')
eq("delta Pi = integral_0^1 [(1+alpha*s) theta' eta' - q cos(theta) eta] ds\n((1+alpha*s) theta')' + q cos(theta) = 0\ntheta(0) = 0, ((1+alpha*s) theta')(1) = 0")
p('The test variation eta vanishes at the clamp. Integration by parts yields the nonlinear equilibrium equation and the natural zero-moment condition at the free end. Introducing m = (1 + alpha s)theta-prime gives the independent first-order BVP: theta-prime = m/(1 + alpha s), m-prime = -q cos(theta), with theta(0) = 0 and m(1) = 0.')
p('For constant stiffness and sufficiently small q, sin(theta) is approximately theta in the energy and cos(theta) is approximately one in equilibrium. Therefore theta(s) = q(s - s^2/2) to leading order; the tip deflection is approximately q/3. This supplies an analytical verification case.')
h('2.1 Discrete reference')
eq('Pi_h = sum_i B_(i-1/2) (theta_i-theta_(i-1))^2/(2h)\n       - q sum_i w_i sin(theta_i), theta_0 = 0')
p('The reference minimizes this discrete potential with midpoint stiffness, trapezoidal centerline weights, and 256 segments. L-BFGS-B starts from the small-load rotation profile. This computes a low-energy equilibrium branch; no exhaustive search over all stationary branches or global-minimum proof is attempted.')
page()
h('3. Neural surrogates and experiment design')
eq('theta_NN(s;q,alpha) = sum_(k=0)^11 c_k(q,alpha) phi_k(s)\nphi_k(s) = sin((k+1/2)*pi*s)/(2k+1)^2')
p('A tanh network maps normalized inputs (q/1.5, alpha/0.4) through two hidden layers of 24 units to 12 coefficients. The representation satisfies theta(0) = 0 and theta-prime(1) = 0 analytically. Centerlines reconstructed from rotations are inextensible in the continuum; trapezoidal reconstruction introduces a small quadrature error. The frequency scaling moderates optimization conditioning. All three methods use these same kinematic constraints, so the comparison isolates loss choices rather than comparing constrained versus entirely unconstrained kinematics.')
eq('L_data = mean_(labeled cases,nodes) |theta_NN - theta_ref|^2\nL_energy = mean_(unlabeled parameters) Pi_64[theta_NN]\nL_hybrid = L_energy + 0.1 L_data')
p('Potential energy can be negative; its absolute magnitude is not an error metric. Energy-only training requires no reference labels, although labeled cases are retained in the common dataset for the other methods. Physics losses are computed at 64 segments. The coefficient map is a finite-dimensional parametric neural surrogate, not a discretization-invariant neural operator.')
table([['Item','Specification'],['Labeled training cases','24; uniform q in [-1.5,1.5], alpha in [-0.4,0.4]'],['Unlabeled physics cases','256; independent draws from the same rectangle'],['Interpolation test','80 independent cases in the training rectangle'],['Load extrapolation test','80 cases; |q| in [1.5,3], alpha in [-0.4,0.4]'],['Stiffness extrapolation test','80 cases; q in [-1.5,1.5], |alpha| in [0.4,0.65]'],['Repetitions','Initialization seeds 0, 1, 2; dataset seed 20261016'],['Optimizer','L-BFGS-B; at most 1,800 iterations per model']], [125,350])
p('Labels and test references are generated separately. Test references are never included in the training loss. Hyperparameters were chosen for this compact demonstration; an initial conditioning check informed the frequency scaling and iteration budget. There is no claim of a blinded model-selection study or an exhaustive hyperparameter search.')
page()
h('4. Numerical verification')
bvp=json.loads((R/'results/bvp_validation.json').read_text());maximum=max(z['max_angle_difference'] for z in bvp)
p(f'The continuous BVP cross-check uses adaptive collocation at tolerance 1e-9 for (q, alpha) = (1,0), (3,-0.4), and (-2,0.6). The maximum nodal rotation difference from the 256-segment reference is {maximum:.3e} radians. These independent checks support the reference on the examined cases, without proving accuracy everywhere in the parameter domain.')
p('Five automated tests verify the discrete energy gradient by finite differences, neural reverse-mode derivatives on selected weights, the small-load rotation formula, load-sign symmetry and zero loading, and reconstructed arclength. The separate validate_reference.py script reproduces the continuous BVP comparisons.')
fig('convergence.png','Figure 1. Tip convergence at q = 3, alpha = -0.4. Differences are measured against the 512-segment reference, not against an exact solution.',350)
conv=json.loads((R/'results/convergence.json').read_text());delta=np.linalg.norm(np.array([conv[-2]['tip_x'],conv[-2]['tip_y']])-np.array([conv[-1]['tip_x'],conv[-1]['tip_y']]))
p(f'The 256-to-512 segment tip difference is {delta:.3e} finger lengths at this refinement case. Both the reference and neural centerlines are evaluated with their own quadrature grids. Consequently, reported prediction errors include representation, optimization, generalization, and small discretization differences.')
h('4.1 Evaluation measures')
eq('tip error = ||r_NN(1) - r_ref(1)||_2\nangle relative L2 = ||Theta_NN - Theta_ref||_F / ||Theta_ref||_F\ncenterline RMSE = sqrt(mean((R_NN - R_ref)^2))')
p('The CSV also records discrete energy gaps and the RMS of the energy gradient multiplied by 64. This stationarity diagnostic depends on the chosen discretization and includes endpoint effects; it is not reported as a continuum residual norm. Cross-seed standard deviations describe initialization variability on one fixed dataset, not population confidence intervals.')
page()
h('5. Results')
data=[['Test split','Supervised','Energy only','Hybrid']]
for split in ['interpolation','load_extrapolation','stiffness_extrapolation']:
 row=[split.replace('_',' ').title()]
 for method in ['data','energy','hybrid']:
  v=[float(z['tip_error_mean']) for z in rows if z['split']==split and z['method']==method]
  row.append(f'{np.mean(v):.5f} +/- {np.std(v,ddof=1):.5f}')
 data.append(row)
table(data,[130,115,115,115])
p('Table 1. Mean tip error divided by finger length, averaged across three initialization seeds; +/- denotes sample standard deviation. Each seed uses the same 80 test cases per split. Lower is better.')
fig('benchmark.png','Figure 2. Tip errors by test regime. Bars are seed means, error bars are sample standard deviations, and dots are individual runs.')
means={(s,m):np.mean([float(z['tip_error_mean']) for z in rows if z['split']==s and z['method']==m]) for s in ['interpolation','load_extrapolation','stiffness_extrapolation'] for m in ['data','energy','hybrid']}
for s in ['interpolation','load_extrapolation','stiffness_extrapolation']:
 best=min(['data','energy','hybrid'],key=lambda m:means[s,m]);p(f'For {s.replace("_"," ")}, the lowest mean tip error is obtained by the {best} model ({means[s,best]:.5f} L). This ranking applies to the specified architecture, dataset, training budget, and test distribution.')
logs=json.loads((R/'results/training.json').read_text());nconv=sum(z['converged'] for z in logs.values());p(f'Optimizer termination is recorded for every model: {nconv} of 9 runs satisfy the solver termination criteria; other runs stop at the iteration budget or another recorded condition. These outcomes are part of the benchmark, and an optimizer convergence flag alone does not establish physical correctness. Full histories, gradient norms, timings, and solver messages are included in results/training.json.')
page()
h('6. Deformation fields and scientific animations')
fig('shapes.png','Figure 3. Reference and neural centerlines for seed 0. The first case is in-domain; the other two test larger load magnitudes.')
fig('animation_still.png','Figure 4. Frame from the reproducible load-sweep animation. The shaded region in the error panel is the training load interval.')
p('The load-sweep GIF evaluates equilibria along q = 2.8 sin(tau) at alpha = -0.2 and displays all methods against the reference. Tau is a visualization phase, not physical time. The stiffness-sweep GIF varies alpha from -0.65 to 0.65 at q = 1.3, comparing the hybrid model with the reference. The line trajectories are generated from numerical solutions and saved neural weights; they are not generative illustrations.')
p('These animations show bending kinematics and distribution shift. They do not simulate dynamics, actuation pressure, collision, grasp success, or physical sensor measurements. The unscaled centerline axes permit direct geometric comparison.')
page()
h('7. Discussion and limitations')
p('The experiment separates hard kinematic structure from variational training: every neural model satisfies the same clamp and analytical free-end moment constraints, while only two losses use mechanical energy. Therefore a supervised advantage would not show that physics is useless; it would show that this additional energy objective is not sufficient to outperform an already structured surrogate under the tested conditions. Conversely, an energy-model advantage would not establish general superiority across geometries or constitutive laws.')
p('Several mechanisms can affect performance. Spectral truncation limits local curvature representation. The discrete bending term weights short-wavelength rotation errors strongly, creating conditioning challenges. A finite set of unlabeled parameter samples constrains the learned map inside the sampled domain but does not enforce equilibrium at every unseen parameter. Hybrid behavior also depends on the fixed data-loss coefficient. A broader study should examine label-count curves, independent dataset seeds, loss-weight sensitivity, mode counts, physics quadrature resolution, and more optimizer budgets.')
p('The present scope is deliberately limited to one two-parameter rod family. Material behavior is represented through bending stiffness, not a learned three-dimensional hyperelastic constitutive law. Only a few continuous BVP checks and one mesh-refinement case are supplied. No uncertainty calibration, experimental validation, comprehensive speedup study, contact mechanics, or stability-branch classification is claimed.')
h('8. Conclusion and next research steps')
p('This benchmark finds lower mean tip errors for energy-based training than supervised training on all three fixed test splits. The hybrid model has the lowest mean interpolation and stiffness-extrapolation errors, while the energy-only model has the lowest mean load-extrapolation error. These findings are conditional on the present model and protocol. The project combines nonlinear mechanics, structured neural prediction, numerical verification, and scientific animations. Its contribution is an inspectable computational study and an honest comparison of learning objectives. A natural next step is a two-dimensional hyperelastic finger with finite-element reference solutions, followed by pressure actuation and contact. Those extensions require new mechanics verification and should be evaluated as separate studies.')
h('References')
p('[1] E, W., and Yu, B. The Deep Ritz method: A deep learning-based numerical algorithm for solving variational problems. arXiv:1710.00211 (2017). https://doi.org/10.48550/arXiv.1710.00211')
p('[2] Nguyen-Thanh, V. M., Zhuang, X., and Rabczuk, T. A deep energy method for finite deformation hyperelasticity. European Journal of Mechanics - A/Solids 80, 103874 (2020). https://doi.org/10.1016/j.euromechsol.2019.103874')
p('<b>Reproducibility and assistance.</b> All data are synthetic and generated by the supplied solvers. The repository includes source, saved models, raw metrics, environment metadata, and build scripts. Code, analysis, and report drafting were developed with AI assistance. This document is an unreviewed computational project report; it is not a peer-reviewed publication.')

def footer(c,d):
 c.setFont('Body',7);c.setFillColor(colors.HexColor('#62748b'));c.drawString(48,28,'Physically Constrained Deformation Prediction | Computational research report');c.drawRightString(547,28,str(d.page))
SimpleDocTemplate(str(R/'report/research_report.pdf'),pagesize=(595.28,841.89),rightMargin=48,leftMargin=48,topMargin=42,bottomMargin=48,title='Physically Constrained Deformation Prediction',author='Jiajun Fu').build(story,onFirstPage=footer,onLaterPages=footer)
(R/'report/research_report.md').write_text('\n'.join(md))
print('Built research_report.pdf and research_report.md')
