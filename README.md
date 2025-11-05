# mindist

Method for calculating minimum distances between two sets of atoms in MD simulations.

## Requirements

1. Python 3 with NumPy
1. [Numba](https://nvidia.github.io/numba-cuda/user/installation.html) with cuda acceleration (only for `mindist_cuda`)
1. gfortran (might work, but has not been tested with other fortran compilers)


## Licensing

All software, in this repo, except for `lib/stdlib_sort_index_real.f90` (see the subsection below) is covered by the license in `LICENSE`.



### stdlib sort

The mindist_grid function uses a sort function, not written by me. The file `lib/stdlib_sort_index_real.f90` is modified from the [Fortran-lang/stdlib](https://stdlib.fortran-lang.org/) project ([github](https://github.com/fortran-lang/stdlib)), released under MIT license. Parts of it are initially from different projects, and are covered by different license, as indicated in the file itself.

## Installation

Just run 

```
pip install .
```

in the project root. This will take care of the dependencies and compile the `_f2py` library.

This will not install numba or the cudatoolkit, so if you want to use the GPU, you will have ot install them manually. This can also be done after installation. Do remember to check that your cudatoolit version matches your cuda version.

If you want to uninstall the project, run

```
pip uninstall mindist
```



## Working idea

The module includes the functions listed below. You can use them e.g.

```python
import mindist
import MDAnalysis as mda

u = mda.Universe("struct.gro")

lipids = u.select_atoms("resname POPC")
prot   = u.select_atoms("protein")
# Calculate minimum distances from atoms in prot to any atom in lipids
mind = mindist.mindist(prot.positions, lipids.positions)
```

or to take the pbc into account


```python
import mindist
import MDAnalysis as mda

u = mda.Universe("struct.gro")

lipids = u.select_atoms("resname POPC")
prot   = u.select_atoms("protein")
box = u.trajectory[0].triclinic_dimensions
# Calculate minimum distances from atoms in prot to any atom in lipids
mind = mindist.mindist_pbc(prot.positions, lipids.positions, box)
```

### mindist()

Calculate the minimum distance for each of the *N* atom in **a** to any of the *M* atom in **center**

A wrapper function to call any of the three functions below. By default uses `mindist_cuda` if available, `mindist_omp` if not. All should give the same result within machine precision, but which is fastest depends on your simulation system and hardware. In most cases the cuda seems to be fastest, even with relatively small systems, while the grid-based is slowest, except for very large systems, where the omp based was slower. I mainly tested with systems, where the centre group is a relatively centralized protein (i.e. atoms only near the centre of the box). If your two sets of atoms are more or less uniformly distributed in the system, the grid based might be faster. And of course this also largely depends on your hardware. The default order gives a good and quick starting point, but if performance is absolutely essential, the only way to know for sure is to run benchmarks.

#### mindist_grid()

A grid based approach, scales roughly as $\mathcal O(n_cn_a\log(n_c)+Nn_cm)$, where $n_c$ and $n_a$ are the number of gridpoints for just the center, and both groups, respectively. $m$ is the average number of (center group) atoms in the gridcells, excluding empty cells.

#### mindist_omp()

A brute force approach, accelerated only with OpenMP. Scales as $\mathcal O(NM)$.

#### mindist_cuda()

Same as `mindist_omp()`, but using `numba` to accelerate with cuda. Best when $N$ is very large.

If `numba` or `numba.cuda` can not be imported, this function is unreachable. A boolean flag `mindist.can_use_cuda` will be set to False, with the `ImportError` reachable as `mindist.no_cuda_reason`. If the libraries were imported without problem, `mindist.can_use_cuda=True` and `mindist.no_cuda_reason=None`


### mindist_pbc()

Like `mindist()`, but takes the periodic boundary conditions into account. Assumes they are applied in all three directions. Works by doing the calculations in reciprocal space (relative to the box vectors). Should work for any box defined by a 3 by 3 matrix of box vectors (e.g. for cubic box a diagonal matrix with box lengths on the diagonal).

#### mindist_pbc_grid()

Like `mindist_grid()`, but takes the periodic boundary conditions into account.

#### mindist_pbc_omp()

Like `mindist_omp()`, but takes the periodic boundary conditions into account.

#### mindist_pbc_cuda()

Like `mindist_cuda()`, but takes the periodic boundary conditions into account.