from . import _f2py
from . import mindist as _mindist

mindist          = _mindist.mindist
mindist_omp      = _mindist.mindist_omp
mindist_grid     = _mindist.mindist_grid
mindist_pbc      = _mindist.mindist_pbc
mindist_pbc_omp  = _mindist.mindist_pbc_omp
mindist_pbc_grid = _mindist.mindist_pbc_grid

can_use_cuda  = _mindist._can_use_cuda
no_cuda_reson = _mindist.reason
if(can_use_cuda):
    mindist_cuda     = _mindist.mindist_cuda
    mindist_cuda_pbc = _mindist.mindist_cuda_pbc
