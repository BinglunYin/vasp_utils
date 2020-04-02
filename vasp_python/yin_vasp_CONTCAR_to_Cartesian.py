#!/home/yin/opt/bin/python3

import numpy as np
from ase.io.vasp import read_vasp, write_vasp
from ase.constraints import FixedLine

import sys
# Count the arguments
narg = len(sys.argv) - 1


ASE_Atoms = read_vasp('./CONTCAR')
atoms_pos = ASE_Atoms.get_positions()
natoms = atoms_pos.shape[0]


if narg == 1:
    if sys.argv[1] == '-FFT':
        constrained_atoms = np.arange(natoms)  # apply constraint to all atoms
        constraint_direction = [0,0,1]            # relax only in the [0,0,1] direction
        zonly_constraint = FixedLine(constrained_atoms, constraint_direction)
        ASE_Atoms.set_constraint(zonly_constraint)


write_vasp('POSCAR_Cartesian', ASE_Atoms,
label='system_name')


