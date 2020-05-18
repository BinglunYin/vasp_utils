#!/home/yin/opt/bin/python3


import numpy as np
from myvasp import vasp_func as vf 
import sys


def main():
    qe = vf.phy_const('qe')
    
    jobn, Etot, Eent, pres = vf.vasp_read_post_data()
    njobs = len(jobn)  # number of jobs

    if njobs < 1.5:
        sys.exit('==> ABORT! more structures needed. ')

    if jobn[0] != '00':
        sys.exit('==> ABORT! no reference state. ')
        
    latoms = vf.get_list_of_atoms()
    
    Asf = np.linalg.norm( \
        np.cross(latoms[0].cell[0, :], latoms[0].cell[1, :] ) )
   
    dE, da33, dpos_all = check_constraints(Etot, latoms)
   
    gamma_s = dE/2/Asf *qe*1e23   #[mJ/m^2]
    
    #=========================
    write_output(Asf, jobn, gamma_s, da33)
    plot_output(gamma_s, da33, dpos_all, latoms, jobn)








def check_constraints(Etot, latoms):
    njobs = len(latoms)
    natoms = latoms[0].positions.shape[0]
    print('njobs, natoms:', njobs, natoms)

    dE = np.array([])
    da33 = np.array([])
    dpos_all = np.zeros([1, natoms, 3])
    
    for i in np.arange(njobs):
        dE = np.append(dE, Etot[i]-Etot[0])
    
        # check latt 
        dlatt = latoms[i].cell[:] - latoms[0].cell[:]

        temp = dlatt[0:2, :].copy()
        temp2 = np.linalg.norm(temp)
        if temp2 > 1e-10:
            print('dlatt:', dlatt)
            print('\n==> i, norm: {0}'.format([i, temp2]) )
            sys.exit("==> ABORT: in-plane lattices changed. \n" )
    
        temp = dlatt[2, 0:2].copy()
        temp2 = np.linalg.norm(temp)
        if temp2 > 1e-10:
            print('\n==> i, norm: {0}'.format([i, temp2]) )
            print('==> WARNING: a31, a32 changed. \n' )
    
        temp = dlatt[2,2].copy()
        da33 = np.append( da33, temp )
        
        # check pos
        dpos = latoms[i].positions - latoms[0].positions
        dposD = dpos @ np.linalg.inv(latoms[i].cell[:])
        dposD = dposD - np.around(dposD)  
        dpos = dposD @ latoms[i].cell[:]

        temp = dpos.copy()
        for j in np.arange(3):
            temp[:,j] = temp[:,j] - temp[:,j].mean()
        dpos_all = np.vstack([ dpos_all, temp[np.newaxis, :] ])
        
      
    dpos_all = np.delete(dpos_all, 0, 0)
    
    if (dE.shape[0] != njobs)  \
        or (len(da33) != njobs)  \
        or (dpos_all.shape[0] != njobs) \
        or (dpos_all.shape[1] != natoms) \
        or (dpos_all.shape[2] != 3) :
        sys.exit("==> ABORT: wrong dimensions. \n" )
   
    return dE, da33, dpos_all




   

def write_output(Asf, jobn, gamma_s, da33):
    njobs = gamma_s.shape[0]
    print(njobs)
       
    f = open('y_post_decohesion.txt','w+')
    f.write('# VASP surface energy: \n' )
    
    f.write('\n%20s: %16.8f \n' %('Asf (Ang^2)', Asf) )
        
    f.write('\n%5s %16s %12s \n' \
        %('jobn', 'gamma_s (mJ/m^2)', 'da33 (Ang)') )
    
    for i in np.arange(njobs):
        f.write('%5s %16.8f %12.8f \n' \
            %(jobn[i], gamma_s[i], da33[i] ))

    f.write(' \n')
    f.close() 





def plot_output(gamma_s, da33, dpos_all, latoms, jobn):
    njobs = len(gamma_s)
    print(njobs)

    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    fig_wh = [3.15, 5]
    fig_subp = [2, 1]
    fig1, ax1 = vf.my_plot(fig_wh, fig_subp)

    disp = da33.copy()
    xi = da33.copy()

    ax1[0].plot(xi, gamma_s*2, '-o')   
      
    tau = np.diff(gamma_s*2) / np.diff(disp) *1e-2  #[GPa]
    x_tau = xi[0:-1].copy() + np.diff(xi)/2
    ax1[1].plot(x_tau, tau, '-o')
    ax1[1].plot([xi.min(), xi.max()], [0, 0], '--k')

    fig_pos  = np.array([0.23, 0.57, 0.70, 0.40])
    fig_dpos = np.array([0, -0.45, 0, 0])

    ax1[0].set_position(fig_pos)
    ax1[1].set_position(fig_pos + fig_dpos  )

    ax1[-1].set_xlabel('Vacuum layer thickness ($\mathrm{\\AA}$)')
    ax1[0].set_ylabel('Decohesion energy (mJ/m$^2$)')
    ax1[1].set_ylabel('Tensile stress $\\sigma$ (GPa)')

    plt.savefig('y_post_decohesion.pdf')


    #=====================
    fig_wh = [5, 5]
    fig_subp = [1, 1]
    fig2, ax2 = vf.my_plot(fig_wh, fig_subp)

    anyplot=0

    xi = latoms[0].positions[:, 2]
    for i in np.arange(njobs):

        if np.linalg.norm( dpos_all[i, :] ) > 1e-10:
            temp = np.hstack([ dpos_all[i, :], xi[:, np.newaxis] ])
            ind = np.argsort(temp[:, -1])
            temp2 = temp[ind, :]
    
            ax2.plot(temp2[:, -1], temp2[:, 0], '-o', label='%s-$u_1$' %(jobn[i]) )
            ax2.plot(temp2[:, -1], temp2[:, 1], '-s', label='%s-$u_2$' %(jobn[i]) )
            ax2.plot(temp2[:, -1], temp2[:, 2], '-^', label='%s-$u_3$' %(jobn[i]) )

            anyplot = 1

    if anyplot==1:
        ax2.legend(loc='lower center', ncol=3, framealpha=0.4)
    
    ax2.set_xlabel('Atom positions in $x_3$ ($\\mathrm{\\AA}$)')
    ax2.set_ylabel('Displacements $u_i$ ($\\mathrm{\\AA}$)')
    ax2.set_position([0.17, 0.10, 0.78, 0.86])

    plt.savefig('y_post_decohesion.ui.pdf')



main()
