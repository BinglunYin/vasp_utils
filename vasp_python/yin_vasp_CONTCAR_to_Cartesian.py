#!/home/yin/opt/bin/python3

import numpy as np
from ase.io.vasp import read_vasp, write_vasp
from ase.constraints import FixedLine
import sys, os


filename1 = './CONTCAR'


f = open(filename1,'r')
a0 = float( f.readlines()[1] )
f.close()

ASE_Atoms = read_vasp(filename1)
atoms_pos = ASE_Atoms.get_positions()
natoms = atoms_pos.shape[0]

ASE_Atoms.set_positions( atoms_pos/a0   )
ASE_Atoms.set_cell( ASE_Atoms.cell/a0   )


if len(sys.argv)  == 1:        
    filename2 = 'POSCAR_Cartesian'

elif len(sys.argv)  == 2:
    if sys.argv[1] == '-FFT':
        filename2 = 'POSCAR_Cartesian_FFT'
        constrained_atoms = np.arange(natoms)  # apply constraint to all atoms
        constraint_direction = ASE_Atoms.cell[2,:]          # relax only in the a3 direction
        zonly_constraint = FixedLine(constrained_atoms, constraint_direction)
        ASE_Atoms.set_constraint(zonly_constraint)


write_vasp(filename2, ASE_Atoms,
label='system_name', direct=False)


with open(filename2) as f:
    lines = f.readlines()

lines[1] = ' %.16f \n' % (a0)

with open('file_temp', "w") as f:
    f.writelines(lines)

os.replace('file_temp', filename2)



