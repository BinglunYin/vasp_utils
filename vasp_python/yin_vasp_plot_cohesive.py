#!/home/yin/opt/bin/python3


import numpy as np
from ase.io.vasp import read_vasp
import yin_vasp_func as vf 
import math    
from scipy.interpolate import interp1d


jobn, Etot, Eent, pres = vf.vasp_read_post_data()

ASE_Atoms = read_vasp('../y_full_relax/CONTCAR')
atoms_pos = ASE_Atoms.get_positions()
natoms = atoms_pos.shape[0]

V0 = np.linalg.det( ASE_Atoms.cell ) / natoms

Etot = Etot / natoms


# scale
k = np.array([])
for i in np.arange(len(jobn)):
    k = np.append(k, float(jobn[i]) )

    if np.abs(k[i]-1) < 1e-6:
        Emin = Etot[i]


# check
if Etot.min() != Emin:
    import sys
    sys.exit('Emin is wrong. Abort!')


if  Etot[-5:].std() > 5e-4 :
    print('WARNING: Eatom might be wrong!')

Eatom = Etot[-1]

Ecoh = Eatom - Emin


V = V0 * k**3
Vp =  V[0:-1].copy() + np.diff( V)/2
VB = Vp[0:-1].copy() + np.diff(Vp)/2 

qe = vf.phy_const('qe')
p = -np.diff(Etot) / np.diff(V) * qe*1e21  #[GPa]
B = -np.diff(p) / np.diff(Vp) * VB   #[GPa]

fp = interp1d(Vp, p)
fB = interp1d(VB, B)

p0 = fp(V0)
B0 = fB(V0)
print('==> p0, B0:')
print(p0, B0)



#====================
f = open("y_post_cohesive.txt", "w+")

f.write("# results of cohesive energy: \n" )

f.write("%16s %16s %16s \n" \
%('E_min (eV)', 'E_atom (eV)', 'E_coh (eV)') )

f.write("%16.8f %16.8f %16.8f \n" \
%(Emin, Eatom, Ecoh) )


f.write("\n%16s \n" \
%('V0 (Ang^3)') )

f.write("%16.6f \n" \
%( V0 ) )


f.write("\n%16s %16s \n" \
%('p0 (GPa)', 'B (GPa)') )

f.write("%16.6f %16.8f \n" \
%( p0, B0 ) )

f.close() 



#====================

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

plt.rcParams['font.size']=8
#plt.rcParams['font.family'] = 'Arial'
plt.rcParams['axes.linewidth']=0.5
plt.rcParams['axes.grid']=True
plt.rcParams['grid.linestyle']='--'
plt.rcParams['grid.linewidth']=0.2
plt.rcParams["savefig.transparent"]='True'
plt.rcParams['lines.linewidth']=0.8
plt.rcParams['lines.markersize'] = 4/1.6

fig_w = 3.15
fig_h = 7

fig1, ax1 = plt.subplots(nrows=3, ncols=1, \
sharex=True, figsize=(fig_w, fig_h) )

pos  = np.array([0.20, 0.70, 0.75, 0.27])
dpos = np.array([0, -0.32, 0, 0])
ax1[0].set_position(pos)
ax1[1].set_position(pos+dpos)
ax1[2].set_position(pos+dpos*2)


Elimu=math.ceil( Etot.max() /10) *10 
Elimd=math.floor( Etot.min()/10) *10 

plim=-math.floor( p.min()/10) *10

Blimu=math.ceil( B0*1.2 /50) *50 
Blimd=math.floor( B.min()/50) *50 


ax1[0].plot([1, 1], [Elimd, Elimu], '--k')
ax1[0].plot(k, Etot, '-o')

ax1[1].plot([0, 4], [0, 0], '--k')
ax1[1].plot([1, 1], [-plim, plim], '--k')
ax1[1].plot((Vp/V0)**(1/3), p, '-o')
ax1[1].plot(1, p0, 's')

ax1[2].plot([0, 4], [0, 0], '--k')
ax1[2].plot([1, 1], [Blimd, Blimu], '--k')
ax1[2].plot((VB/V0)**(1/3), B, '-o')
ax1[2].plot(1, B0, 's')


ax1[0].set_ylim([Elimd, Elimu])
ax1[1].set_ylim([-plim, plim])
ax1[2].set_ylim([Blimd, Blimu])


plt.setp(ax1[-1], xlabel='$a/a_0$')
plt.setp(ax1[0],  ylabel='energy (eV/atom)')
plt.setp(ax1[1],  ylabel='pressure (GPa)')
plt.setp(ax1[2],  ylabel='B (GPa)')


ax1[0].text(1.5, Elimd+(Elimu-Elimd)*0.6, \
'$E_\mathrm{min}$ = %.4f eV \n$E_\mathrm{atom}$ = %.4f eV \n\n$E_\mathrm{coh}$ = %.4f eV '
%(Emin, Eatom, Ecoh)  )

ax1[0].text(1.5, Elimd+(Elimu-Elimd)*0.3, \
'$V_0$ = %.4f $\mathrm{\AA}^3$ ' %(V0)  )

ax1[1].text(1.5, plim*0.4, \
'from diff: \n$p_0$ = %.1f GPa ' %(p0)  )

ax1[2].text(1.5, Blimd+(Blimu-Blimd)*0.6, \
'from diff: \n$B_0$ = %.1f GPa ' %(B0)  )

plt.savefig('y_post_cohesive.pdf')



