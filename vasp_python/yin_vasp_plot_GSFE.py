#!/home/yin/opt/bin/python3


import numpy as np
# from ase.io.vasp import read_vasp
import yin_vasp_func as vf 
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
    a11 = latoms[0].cell[0, 0]
    a22 = latoms[0].cell[1, 1]
    # natoms = latoms[0].positions.shape[0]
    
    dE, da3, dpos3 = check_constraints(Etot, latoms)
   
    gamma = dE/Asf *qe*1e23   
    sf, usf = find_sf_usf(gamma)
    
    #=========================
    write_output(Asf, a11, a22, sf, usf, jobn, gamma, da3)
    plot_GSFE(jobn, gamma, da3, dpos3, latoms)








def check_constraints(Etot, latoms):
    njobs = len(latoms)
    natoms = latoms[0].positions.shape[0]
    print(njobs, natoms)

    dE = np.array([])
    da3 = np.zeros([1, 3])
    dpos3 = np.zeros([1, natoms])
    
    for i in np.arange(njobs):
        dE = np.append(dE, Etot[i]-Etot[0])
    
        # check latt 
        dlatt = latoms[i].cell[:] - latoms[0].cell[:]
        temp = np.linalg.norm(dlatt[0:2, :])
        if temp > 1e-10:
            print('\n==> i, norm: {0}'.format([i, temp]) )
            sys.exit("==> ABORT: in-plane lattices relaxed. \n" )
    
        temp = dlatt[2,:].copy()
        da3 = np.vstack([ da3, temp[np.newaxis,:] ])
        
        # check pos
        dpos = latoms[i].positions - latoms[0].positions
        dposD = dpos @ np.linalg.inv(latoms[0].cell[:])
        dposD = dposD - np.around(dposD)  
        dpos = dposD @ latoms[0].cell[:]

        temp = np.linalg.norm(dpos[:,0:2])
        if temp > 1e-10:
            print(dpos)
            print('\n==> i, norm: {0}'.format([i, temp]) )
            sys.exit("==> ABORT: atoms show in-plane relaxation. \n" )
    
        temp = dpos[:, 2].copy()
        temp = temp - temp.mean()
        dpos3 = np.vstack([ dpos3, temp[np.newaxis, :] ])
        
    da3 = np.delete(da3, 0, 0)
    dpos3 = np.delete(dpos3, 0, 0)
    
    if (dE.shape[0] != njobs)  \
        or (da3.shape[0] != njobs) or (da3.shape[1] != 3) \
        or (dpos3.shape[0] != njobs) or (dpos3.shape[1] != natoms):
        sys.exit("==> ABORT: wrong dimensions. \n" )

    for i in np.arange(njobs):
        temp = np.abs(da3[i, 1]*da3[1, 0] - da3[i, 0]*da3[1, 1])
        if temp > 1e-10:
            print('\n==> i, norm: {0}'.format([i, temp]) )
            sys.exit("==> ABORT: slip is not along a line. \n" )
    
    return dE, da3, dpos3



def find_sf_usf(gamma):
    njobs = gamma.shape[0]
    print(njobs)
    sf = np.array([])
    usf = np.array([])
    for i in np.arange(1, njobs-1, 1):
        if (gamma[i] < gamma[i-1]) and (gamma[i] < gamma[i+1]):
            sf = np.append(sf, gamma[i])
        if (gamma[i] > gamma[i-1]) and (gamma[i] > gamma[i+1]):
            usf = np.append(usf, gamma[i])
    return sf, usf


   

def write_output(Asf, a11, a22, sf, usf, jobn, gamma, da3):
    njobs = gamma.shape[0]
    print(njobs)
       
    f = open('y_post_GSFE.txt','w+')
    f.write('# VASP GSFE: \n' )
    
    f.write('\n%20s: %16.8f \n' %('Asf (Ang^2)', Asf) )

    if sf.shape[0] > 0.5 :
        f.write('%20s: %16.8f \n' %('local min (mJ/m^2)', sf.min() ) )
    
    if usf.shape[0] > 0.5:
        f.write('%20s: %16.8f \n' %('local max (mJ/m^2)', usf.min() ) )
        
        
    f.write('\n%5s %16s %12s %10s %10s %10s \n' \
        %('jobn', 'gamma (mJ/m^2)', 
        'da33 (Ang)', 'da31/a11', 'da32/a22',
        'slip (Ang)' ) )
    
    for i in np.arange(njobs):
        f.write('%5s %16.8f %12.8f %10.4f %10.4f %10.4f \n' \
            %(jobn[i], gamma[i], 
            da3[i, 2], da3[i, 0]/a11,  da3[i, 1]/a22, 
            np.linalg.norm(da3[i, 0:2])  ))

    f.write(' \n')
    f.close() 





def plot_GSFE(jobn, gamma, da3, dpos3, latoms):
    njobs = gamma.shape[0]
    print(njobs)

    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    fig_wh = [3.15, 7]
    fig_subp = [3, 1]
    fig1, ax1 = vf.my_plot(fig_wh, fig_subp)

    xi = np.array([])
    disp = np.array([])
    for i in np.arange( njobs ):
        xi = np.append(xi, float(jobn[i]) )
        disp = np.append(disp, np.linalg.norm(da3[i, 0:2]))

    xi = xi / np.around( np.max([xi.max(), 10]) , -1)

    ax1[0].plot(xi, gamma, '-o')   
    ax1[1].plot(xi, da3[:,2], '-o')
      
    tau = np.diff(gamma) / np.diff(disp) *1e-2  #[GPa]
    x_tau = xi[0:-1].copy() + np.diff(xi)/2
    ax1[2].plot(x_tau, tau, '-o')
    ax1[2].plot([xi.min(), xi.max()], [0, 0], '--k')

    fig_pos  = np.array([0.22, 0.70, 0.70, 0.28])
    fig_dpos = np.array([0, -0.31, 0, 0])

    ax1[0].set_position(fig_pos)
    ax1[1].set_position(fig_pos + fig_dpos  )
    ax1[2].set_position(fig_pos + fig_dpos*2)

    ax1[-1].set_xlabel('Normalized slip vector')
    ax1[0].set_ylabel('GSFE $\\gamma$ (mJ/m$^2$)')
    ax1[1].set_ylabel('Inelastic normal displacement $\\Delta_n$ ($\\mathrm{\\AA}$)')
    ax1[2].set_ylabel('Shear stress $\\tau$ (GPa)')

    dxi = np.around( xi.max()/6, 1)
    ax1[-1].set_xticks( np.arange(0, xi.max()+dxi, dxi ) )


    plt.savefig('y_post_GSFE.pdf')


    #=====================
    fig_wh = [5, 5]
    fig_subp = [1, 1]
    fig2, ax2 = vf.my_plot(fig_wh, fig_subp)

    xi = latoms[0].positions[:, 2]
    for i in np.arange(njobs):
        temp = np.array([ xi, dpos3[i, :] ])
        ind = np.argsort(temp[0, :])
        temp2 = temp[:,ind]

        ax2.plot(temp2[0, :], temp2[1, :], '-o', label=jobn[i])
    
    ax2.legend(loc='lower center', ncol=5, framealpha=0.4)
    ax2.set_xlabel('Atom positions in $x_3$ ($\\mathrm{\\AA}$)')
    ax2.set_ylabel('Displacement $u_3$ ($\\mathrm{\\AA}$)')
    ax2.set_position([0.17, 0.10, 0.78, 0.86])

    plt.savefig('y_post_GSFE.u3.pdf')



main()



