from pathlib import Path
import json,itertools,math
import numpy as np
S=Path('/tmp/r5_analysis');X=json.loads((S/'analysis.json').read_text());A=X['arms'];P=X['pooled'];D=json.loads((S/'decisions.json').read_text());J=json.loads((S/'journals.json').read_text())
def pair(a,b):
 p=X['pairs'].get(a+'__'+b)
 if p:return p
 p=dict(X['pairs'][b+'__'+a]);p['delta']=-p['delta'];p['srA'],p['srB']=p['srB'],p['srA'];p['wins'],p['losses']=p['losses'],p['wins'];p['ci']=[-p['ci'][1],-p['ci'][0]];p['per_task']=[-x for x in p['per_task']];return p
def pp(x):return f'{(100*x if abs(x)>1e-10 else 0):+.2f}'
def ci(x):return f'[{100*x[0]:+.2f}, {100*x[1]:+.2f}]'
def pv(x):return '<.0001' if x<.0001 else f'{x:.4f}'
def stat(a,b):
 p=pair(a,b);return f"{pp(p['delta'])} {ci(p['ci'])}; {p['wins']}/{p['losses']}; p{'' if p['p']<.0001 else '='}{pv(p['p'])}"
def pt(s,n):return 'tail1uc_sp_500' if (s,n)==('sp',500) else f'r5t_p_{s}_{n}_tail1uc'
def k7(s,n):return {('l10',500):'k7_tail1ug',('l10',50):'k7_tail1ug_50',('sp',500):'k7_sp_tail1ug',('sp',50):'k7_sp_tail50'}[s,n]
if __name__=='__main__':
 for s in ['l10','sp']:
  for n in [50,500]:
   print('\nCELL',s,n)
   for a,b in [(pt(s,n),f'cl2_{s}_{n}'),(f'r5q1_c10_p_{s}_{n}',k7(s,n)),(f'r5q1_c10_p_{s}_{n}',f'inf_{s}_L10'),(f'r5q6_p_{s if s=="l10" else "spatial"}_{n}_tail',k7(s,n)),(f'r5q6_p_{s if s=="l10" else "spatial"}_{n}_tail',f'wrist_{s}_{n}')]:print(a,b,stat(a,b))
 for s in ['l10','sp']:
  for n in [50,500]:
   print('\nG',s,n)
   one=f'r5x_g_{s}_{n}_tail1u';two=f'gb_{s}_{n}_tail2u';g=f'r5q2_g_{s if s=="l10" else "spatial"}_{n}_G10';inf=f'r5q2_g_{s if s=="l10" else "spatial"}_policy_L10'
   for a,b in [(one,f'cl2_g_{s}_{n}'),(one,two),(g,one),(g,two),(g,inf),(one,inf),(two,inf)]:print(a,b,stat(a,b))
 for s in ['l10','sp']:
  for a,b in [(f'r5q4_p_{s}_grow250_refit',f'cl2_{s}_50'),(f'r5q4_p_{s}_grow250_frozen',f'cl2_{s}_50'),(f'r5q4_p_{s}_grow250_refit',f'cl2_{s}_500'),(f'r5q4_p_{s}_grow250_refit',f'r5q4_p_{s}_grow250_frozen'),(f'r5q4d_p_{s}_200_refit_tail1',f'r5q4d_p_{s}_200_frozen50_tail1'),(f'r5q4d_p_{s}_300_refit_tail1',f'r5q4d_p_{s}_200_refit_tail1')]:print(a,b,stat(a,b))
 for a in A:
  if a.startswith('r5b') and a.endswith('solve'):print(a,stat(a,a.replace('solve','hand')))
 for ls in [['g500','g500_repa','g500_repb'],['k7_tail1ug','r5b_p_l10_500_hand','r5x_p_l10_500_k7tail_repa']]:
  print('noise',[(l,A[l]['sr'],A[l]['ir']) for l in ls],np.std([A[l]['sr'] for l in ls],ddof=1))
  for a,b in itertools.combinations(ls,2):print(a,b,stat(a,b))
