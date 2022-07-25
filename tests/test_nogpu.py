
import timeit,os,datetime
from myutils._f2py import mindist_grid
from myutils._f2py.mindist_PBC import mindist_pbc
from myutils._f2py.mindist_PBC_grid import mindist_pbc_grid
print(mindist_grid.__file__)
m_time = os.path.getmtime(mindist_grid.__file__)
dt_m = datetime.datetime.fromtimestamp(m_time)
print(dt_m)

from myutils.mindist import mindist, mindist_fort, mindist_pbc as min_pbc
import numpy as np
import MDAnalysis as mda




u1 = mda.Universe("../rsc/system1.gro")
c1 = u1.select_atoms("protein and not type H")
a1 = u1.select_atoms("resname POPC and not type H")

u2 = mda.Universe("../rsc/system1.gro")
c2 = u2.select_atoms("protein and not type H")
a2 = u2.select_atoms("resname POPC and not type H")

u3 = mda.Universe("../rsc/system1.gro")
c3 = u3.select_atoms("protein and not type H")
a3 = u3.select_atoms("resname POPC and not type H")

a = a1.positions.astype(np.float64) + 20 #np.random.random((50000,3))*200
center = c1.positions.astype(np.float64) #np.random.random((10000,3))*50
binsize=15

print(a.shape,center.shape)

print(a.dtype, center.dtype)
box = u1.trajectory[0].triclinic_dimensions.astype(np.float64)

mgri  = mindist(a,center, use_cuda=False, bruteforce=False, gridsize = binsize)
mpbc  = mindist_pbc.mindist(a,center, box)
mbrut = mindist(a,center, use_cuda=False, bruteforce=True) 
zeros1 = mbrut-mindist_fort(a,center)

invbox = np.linalg.inv(box)

unitpos_a, unitpos_c = mindist_pbc.inv(a,center,box)
ua = a @ invbox %1
uc = center @ invbox %1
print(ua.dtype, uc.dtype)


#print(ua@box-unitpos_a@box)
#print(uc@box-unitpos_c@box)
print("inv diffs")
print(np.max(np.abs(ua@box-unitpos_a@box)))
print(np.max(np.abs(ua@box-unitpos_a@box)))


print("-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.")

print(100*np.sum(zeros1<1e-12)/len(zeros1), "%")

diffs = np.abs(mgri-mbrut)
print("grid diffs")
print(np.max(diffs))
print(mgri[np.argmax(diffs)], mbrut[np.argmax(diffs)])
print(diffs)
print(100*np.sum(diffs==0)/len(diffs), "%")

print()
print(mpbc)
diffs = np.abs(mbrut-mpbc)
print("pbc  diffs")
print(np.max(diffs))
print(diffs)
print(mpbc)
print(mbrut)
print(np.any(mbrut>mpbc))
print(100*np.sum(diffs<1e-12)/len(diffs), "%")

selfdiff = np.abs(mpbc-mindist_pbc.mindist(a+100,center+100, u1.trajectory[0].triclinic_dimensions))
print("pbc  selfdiffs")
print(np.max(selfdiff))
print(selfdiff)
print(100*np.sum(selfdiff<1e-12)/len(selfdiff), "%")


diffs = np.abs(mbrut-mindist_pbc.mindist(a,center, u1.trajectory[0].triclinic_dimensions*10))
print("pbc  bigbox diffs")
print(np.max(diffs))
print(diffs)
print(mpbc)
print(mbrut)
print(100*np.sum(diffs<1e-12)/len(diffs), "%")

mgridpbc = mindist_pbc_grid.mindist(a,center, u1.trajectory[0].triclinic_dimensions, 15) 
diffs = np.abs(mgridpbc-mpbc)
print("grid pbc  diffs")
print(np.max(diffs))
print(diffs)
print(mgridpbc[np.argmax(diffs)], mpbc[np.argmax(diffs)])
print(a[np.argmax(diffs)])
print(u1.trajectory[0].triclinic_dimensions)
for i in range(3):
    print(u1.trajectory[0].triclinic_dimensions[:,i])
print(mgridpbc)
print(mpbc)
print(100*np.sum(diffs<1e-12)/len(diffs), "%")

print()
#quit()


num=10
rep=5

for i, (asel, csel) in enumerate(zip((a1,a2,a3),(c1,c2,c3))):
    box = asel.universe.trajectory[0].triclinic_dimensions
    a = asel.positions
    center = csel.positions

    print(f"-----------------------------------------------\nsystem {i+1}")
    print(f"center {len(csel)} atoms, a {len(asel)} atoms")

    res = np.array(timeit.repeat(lambda: mindist(a,center, use_cuda=False, bruteforce=True), repeat=rep, number=num))/num
    print(f"brute        ", np.min(res), end=", ")
    print("    ", res)
    res = np.array(timeit.repeat(lambda: min_pbc(a,center, box, bruteforce=True, use_cuda=False), repeat=rep, number=num))/num
    print(f"PBC          ", np.min(res), end=", ")
    print("    ", res)

    print(f"------------ normal grid")
    for gd in (10,12.5,15,17.5,20):
        res = np.array(timeit.repeat(lambda: mindist(a,center, use_cuda=False, bruteforce=False, gridsize = gd), repeat=rep, number=num))/num
        print("gridsize=%.1f"%gd, np.min(res), end=", ")
        print("    ", res)

    print(f"------------ PBC grid")
    for gd in (10,12.5,15,17.5,20):
        res = np.array(timeit.repeat(lambda: min_pbc(a,center, box, use_cuda=False, bruteforce=False, gridsize = gd), repeat=rep, number=num))/num
        print("gridsize=%.1f"%gd, np.min(res), end=", ")
        print("    ", res)
