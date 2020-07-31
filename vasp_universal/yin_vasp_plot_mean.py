#!/home/yin/opt/bin/python3

import numpy as np 
from myvasp import vasp_func as vf 
import sys



def main():
    jobn, Etot, Eent, pres = vf.vasp_read_post_data()
    njobs = len(jobn)
    
    if njobs < 0.5:
        sys.exit('ABORT. no jobs found. ')

    atoms = vf.get_list_of_atoms()

    V = np.array([])
    V0 = np.array([])
    p = np.array([])

    for i in np.arange(njobs):
        V = np.append(V, atoms[i].get_volume() )
        V0 = np.append(V0, V[i] / len(atoms[i].get_positions()) )
        p = np.append(p, np.mean( pres[i,0:3] ) )

    write_output(jobn, V, V0, p)






def write_output(jobn, V, V0, p):
    f = open('y_post_mean.txt', 'w+')
    f.write('# VASP mean of y_dir: \n' )
    
    f.write('%12s %12s %12s %12s \n' \
        %('mean', 'std', 'max', 'min' ) )

    f.write(' V0 (Ang^3/atom):\n')
    f.write('%12.4f %12.4f %12.4f %12.4f \n' \
        %(V0.mean(), V0.std(), V0.max(), V0.min() ) )

    f.write('\n p (kBar):\n')
    f.write('%12.4f %12.4f %12.4f %12.4f \n\n' \
        %(p.mean(), p.std(), p.max(), p.min() ) )


    a_fcc = (V0.mean()*4)**(1/3)
    a_bcc = (V0.mean()*2)**(1/3)

    f.write('%12s %12s \n' \
            %('a_fcc', 'a_bcc' ) )
    f.write('%12.4f %12.4f \n' \
            %(a_fcc, a_bcc ) )


    f.write('\n%16s %12s %12s %12s \n' \
            %('jobn', 'V', 'V0', 'p'  ) )
    for i in np.arange( len(jobn) ):
        f.write('%16s %12.4f %12.4f %12.4f \n' \
            %(jobn[i], V[i], V0[i], p[i]  ) )


    f.close()
    
    

main()






