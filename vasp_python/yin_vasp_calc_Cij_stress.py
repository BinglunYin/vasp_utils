#!/home/yin/opt/bin/python3


import numpy as np
import yin_vasp_func as vf 


jobn, Etot, Eent, pres = vf.vasp_read_post_data()

if Etot.shape[0] < 5.9:
    import sys
    sys.exit("\n==> ABORT: insufficient data \n" )


pres0 = vf.read_pressure('../y_full_relax/OUTCAR')

temp = np.vstack( (pres, pres0) ) 

s=temp.copy()
s[:,3] = temp[:,4]
s[:,4] = temp[:,5]
s[:,5] = temp[:,3]

s = s.T *(-0.1)   # stress data, 6*7, [GPa]


# direct results

s_d = ( s[:,0:6].copy().T - s[:,-1].copy() ).T
Cij_d = s_d /0.002


# fitting

from sympy import * 

Cij = zeros(6)
Stress0 = zeros(6, 7)

c = MatrixSymbol('c', 6, 6)
s0 = MatrixSymbol('s0', 6, 1)

for i in np.arange(0, 6, 1):
    for j in np.arange(i, 6, 1):
        Cij[i,j] = c[i,j]
        Cij[j,i] = c[i,j]

    for k in np.arange(7):
        Stress0[i,k] = s0[i,0]

# print( Cij, Stress0 )


Strain = eye(6)*0.002
Strain = Matrix([Strain, zeros(1, 6)])
Strain = Strain.T
# print(Strain)


Stress = Cij * Strain + Stress0


err=Stress - s
# print(err)

y2=0
for i in np.arange(err.shape[0]):
    for j in np.arange(err.shape[1]):
        y2 = y2 + (err[i,j])**2

# print('==> y2:')
# print(y2)


myeqn = zeros(27, 1)
myvar = zeros(27, 1)
k=-1
for i in np.arange(6):
    for j in np.arange(i, 6, 1):
        k=k+1
        myeqn[k,0] = diff(y2, c[i,j])
        myvar[k,0] = c[i,j]

for i in np.arange(6):
    k=k+1
    myeqn[k,0] = diff(y2, s0[i,0])
    myvar[k,0] = s0[i,0]
    
fitres = solve(myeqn, myvar)


C14=[]
for i in np.arange(3):
    for j in np.arange(3, 6, 1):
        C14.append( fitres[c[i, j]] )
C14 = np.array(C14, dtype=np.float64 )


# for cubic
Cij_cubic = [
[ fitres[c[0, 0]] , fitres[c[1, 1]] , fitres[c[2, 2]] ], 
[ fitres[c[0, 1]] , fitres[c[0, 2]] , fitres[c[1, 2]] ], 
[ fitres[c[3, 3]] , fitres[c[4, 4]] , fitres[c[5, 5]] ], 
]


# for hcp
Cij_hcp = [
[ fitres[c[0, 0]] , fitres[c[1, 1]]  ], 
[ fitres[c[0, 1]]  ], 
[ fitres[c[0, 2]] , fitres[c[1, 2]]  ], 
[ fitres[c[2, 2]]  ], 
[ fitres[c[3, 3]] , fitres[c[4, 4]]  ],
[ fitres[c[5, 5]]  ],
]



#====================
f = open("y_post_Cij.txt","w+")

f.write("# Cij from stress-strain method: \n" )

f.write("\n# stress data (GPa): \n"  )
for i in np.arange(s.shape[0]):
    for j in np.arange(s.shape[1]):
        f.write("%10.6f " %(s[i,j]) )
    f.write(" \n")


f.write("\n# stress_d data (GPa): \n"  )
for i in np.arange(s_d.shape[0]):
    for j in np.arange(s_d.shape[1]):
        f.write("%10.6f " %(s_d[i,j]) )
    f.write(" \n")


f.write("\n# Cij_d, direct results: \n"  )
for i in np.arange(Cij_d.shape[0]):
    for j in np.arange(Cij_d.shape[1]):
        f.write("%10.2f " %(Cij_d[i,j]) )
    f.write(" \n")


f.write("\n# Cij, fitting results: \n"  )
for i in np.arange(6):
    for j in np.arange(6):
        if j<i:
            f.write("%10s " %('*') )
        else:
            f.write("%10.2f " %(fitres[c[i, j]]) )
    f.write(" \n")


f.write("\n# Stress0, fitting results: \n"  )
for i in np.arange(6):
    f.write("%10.6f " %(fitres[s0[i, 0]]) )
    f.write(" \n")


f.write("\n# mean and std: \n"  )
f.write("# C14: \n"  )
f.write("%10.2f %10.2f \n" %( C14.mean(), C14.std()) )


f.write("\n# if cubic, C11, C12, C44: \n"  )
cij_mean=np.array([])
for i in np.arange(len(Cij_cubic)):
    temp=np.array(Cij_cubic[i],  dtype=np.float64 )
    cij_mean = np.append(cij_mean, temp.mean() )
    f.write("%10.2f %10.2f \n" %( temp.mean(), temp.std()) )


f.write("\n# if hcp, C11, C12, C13, C33, C44, (C66): \n"  )
cij_mean=np.array([])
for i in np.arange(len(Cij_hcp)):
    temp=np.array(Cij_hcp[i],  dtype=np.float64 )
    cij_mean = np.append(cij_mean, temp.mean() )
    f.write("%10.2f %10.2f \n" %( temp.mean(), temp.std()) )

f.write("\n# C66-(C11-C12)/2: \n"  )
f.write("%10.2f \n\n" %( cij_mean[5] - ( cij_mean[0] - cij_mean[1] )/2 ))


f.close() 




