# MIT License
# 
# Copyright (c) 2022 Santeri Paajanen
# 
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
# 
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
# 
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

from . import _f2py
import sys

mindist_omp        = _f2py.mindist
__mindist_grid     = _f2py.mindist_grid
mindist_pbc_omp    = _f2py.mindist_PBC
__mindist_pbc_grid = _f2py.mindist_PBC_grid

_can_use_cuda = False
reason = None

try:
    from numba import cuda
    from numba import float64 as numba_float64
    import numpy as np
    import math

    # Code for "clever" choosing of nthreads
    MAX_THREADS_PER_BLOCK = cuda.get_current_device().MAX_THREADS_PER_BLOCK
    # 4 times SM count should give good occupancy... I think
    MIN_BLOCKS = cuda.get_current_device().MULTIPROCESSOR_COUNT*4
    WARP_SIZE = cuda.get_current_device().WARP_SIZE
    # Possible nthreads are multiples of WARP_SIZE up to MAX_THREADS_PER_BLOCK.
    # Should be ordered in descending order
    nthread_pos = np.arange(WARP_SIZE, MAX_THREADS_PER_BLOCK+1, WARP_SIZE)[::-1]

    def get_nthreads(n):
        """
        Parameters:
            n: the total number of threads
        Returns:
            nthreads: The smallest number of threads from nthread_pos, that gices at least MIN_BLOCKS blocks
        """
        global nthread_pos,MIN_BLOCKS
        bigenough = (n-1)//nthread_pos+1>=MIN_BLOCKS
        i = np.argmax(bigenough)
        return nthread_pos[i] if bigenough[i] else nthread_pos[-1]

    @cuda.jit("void(f8[:,:], f8[:,:], f8[:])")
    def __mindist_cuda(a, center, out):
        i = cuda.grid(1)
        if(i>=out.shape[0]): return
        mind=1000000.0
        for j in range(center.shape[0]):
            dist=0.0
            for k in range(a.shape[-1]):
                dist += (a[i, k]-center[j,k])**2
                
            if(dist<mind):
                mind=dist
        out[i]=math.sqrt(mind)


    def mindist_cuda(a, center, nthreads=None):
        """
        The numba cuda function used by mindist

        parameters:
        a: (n,k) array of n k-dimensional coordinates
        center: (m,k) array of m  k-dimensional coordinates
        nthreads: Changes the amount of threads in the cuda kernel.
                  If None, tries to be smart. [default: None].

        returns:
        out: array of (n,) minimum distances
        """

        if(a.dtype!=np.dtype('float64')):
            a = a.astype(np.float64)
        if(center.dtype!=np.dtype('float64')):
            center = center.astype(np.float64)

        d_a = cuda.to_device(a)
        d_c = cuda.to_device(center)
        d_o = cuda.device_array(d_a.shape[0])
        if(nthreads is None):
            nthreads = get_nthreads(a.shape[0])

        __mindist_cuda[(a.shape[0]-1)//nthreads+1, nthreads](d_a, d_c, d_o)
        out = d_o.copy_to_host()
        return out

    @cuda.jit("void(f8[:,:], f8[:,:], f8[:,:], f8[:])")
    def __mindist_cuda_pbc(s_a, s_c, box, out):
        i = cuda.grid(1)
        if(i>=out.shape[0]): return
        mind=1000000.0
        diff  = cuda.local.array(3,numba_float64)
        diff2 = cuda.local.array(3,numba_float64)
        for j in range(s_c.shape[0]):
            dist=0.0
            for k in range(3):
                diff[k] = s_a[i,k]-s_c[j,k]
                if(diff[k]>0.5):
                    diff[k] -= 1.0
                elif(diff[k]<-0.5):
                    diff[k] += 1.0

            for l in range(3):
                diff2[l]=0.0
                for m in range(3):
                    diff2[l] +=  box[m,l]*diff[m]
                
            for k in range(3):
                dist += diff2[k]**2

            if(dist<mind):
                mind=dist
        out[i]=math.sqrt(mind)

    def mindist_cuda_pbc(a, center, box, nthreads=None):
        """
        The numba cuda function used by mindist_pbc

        parameters:
        a:        (n,3) array of n 3-dimensional coordinates
        center:   (m,3) array of m 3-dimensional coordinates
        nthreads: Changes the amount of threads in the cuda kernel.
                  If None, tries to be smart. [default: None]
        returns:
        out: array of (n,) minimum distances
        """

        if(a.dtype!=np.dtype('float64')):
            a = a.astype(np.float64)
        if(center.dtype!=np.dtype('float64')):
            center = center.astype(np.float64)
        if(box.dtype!=np.dtype('float64')):
            box = box.astype(np.float64)
        
        invbox = np.linalg.inv(box)

        d_a = cuda.to_device((a @ invbox) % 1)
        d_c = cuda.to_device((center @ invbox) % 1)
        d_b = cuda.to_device(box)
        d_o = cuda.device_array(d_a.shape[0])
        if(nthreads is None):
            nthreads = get_nthreads(a.shape[0])
        __mindist_cuda_pbc[(a.shape[0]-1)//nthreads+1, nthreads](d_a, d_c, d_b, d_o)
        out = d_o.copy_to_host()
        return out

    _can_use_cuda = True

except ImportError as e:
    reason = e

except Exception  as e:
    print("IMporting numba and numba.cuda successfully, but another exception caught:")
    print("%s: %s"%(e.__class__.__name__,str(e)), file=sys.stderr)

    if(e.__class__ is cuda.cudadrv.driver.LinkerError and "Unsupported .version" in str(e)):
        print("Check that your cuda version is same or newer than cudatoolkit", file=sys.stderr)
        v = cuda.cudadrv.runtime.Runtime().get_version()
        print("cudatoolkit version %s"%(".".join([str(i) for i in v])), file=sys.stderr)


    print("Continuing without cuda", file=sys.stderr)
    
    reason = e
    

def mindist_grid(a, center, gridsize=15.0):
    """
    A grid-based method for calculating the minimum distances. Both atomgroups are put into
    their own grids, and distances between grid cenres are put into a matrix and sorted.
    For each atom in a, the center-grids are iterated starting from the closest, until the next
    bin is far enough to not possibly include any atom closer than the closest so far found.

    The default gridsize should be a good guess for atomic systems, assuming distances are in
    Ångströms. If they are in nm use 1.5 instead. The real best value might of course depend
    on your system, but short of timing your system with different values, this is your best
    bet.

    parameters:
        a:        shape(n,3) array of n 3-dimensional coordinates
        center:   shape(m,3) array of m 3-dimensional coordinates
        gridsize: Side length of cubic gridcells [default: 10.0]

    returns:
        out: array of (n,) minimum distances
    """
    return __mindist_grid(a,center, gridsize)


def mindist_pbc_grid(a, center, box, gridsize=12.5):
    """
    A grid-based method for calculating the minimum distances, taking PBC into account.
    A grid spanning the box is made, and distances between grid centes are put into a matrix and sorted.
    For each atom in a, the gridpoints are iterated starting from the closest, until the next
    bin is far enough to not possibly include any atom closer than the closest so far found.

    NOTE! For some reason this distance is twice what I expected, meaning there might be
        something I have overlooked. For now it seems to work fine.

    The default gridsize should be a good guess for atomic systems, assuming distances are in
    Ångströms. If they are in nm use 1.5 instead. The real best value might of course depend
    on your system, but short of timing your system with different values, this is your best
    bet.

    parameters:
        a:        shape(n,3) array of n 3-dimensional coordinates
        center:   shape(m,3) array of m 3-dimensional coordinates
        box:      shape(3,3) array of box vectors
        gridsize: Side length of cubic gridcells [default: 10.0]

    returns:
        out: array of (n,) minimum distances
    """
    return __mindist_pbc_grid(a,center, box, gridsize)




def mindist_pbc(a, center, box, bruteforce=True, gridsize=12.5, use_cuda=_can_use_cuda, nthreads=None):
    """
    Calculates the minimun distance of each atom in a to any atoms in center, taking into account the PBC.

    A brute force method using either OpenMP or cuda for accelerations, or a grid based one.

    parameters:
        a:        shape(n,3) array of n 3-dimensional coordinates
        center:   shape(m,3) array of m 3-dimensional coordinates
        box:      shape(3,3) array of box vectors
        bruteforce: Whether to use bruteforce calculation or a grid-based method
        gridsize: The gridsize of the grid-based method [default: 10.0]
        use_cuda: For bruteforce calculation, whether to use the cuda GPU-accelerated
                    version or the fortran OMP-accelerated one. By default chooses cuda if available,
                    otherwise OMP. Ignored if bruteforce=False.
        nthreads: Passed down to mindist_numba. Changes the amount of threads in the
                    cuda kernel. If None, tries to be smart. [default: None].

    returns:
        out: array of (n,) minimum distances
    """
    if(bruteforce and use_cuda):
        if(_can_use_cuda):
            return mindist_cuda_pbc(a,center, box,nthreads)
        print("Failed to use cuda JIT with numba", file=sys.stderr)
        print("Reason: %s: %s"%(reason.__class__.__name__,str(reason)), file=sys.stderr)
        print("Using copmiled fortran version", file=sys.stderr)
    if(bruteforce):
        return mindist_pbc_omp(a,center, box)
    return mindist_pbc_grid(a,center, box, gridsize)


def mindist(a, center, bruteforce=True, gridsize=15.0, use_cuda=_can_use_cuda, nthreads=None):
    """
    Calculates the minimun distance of each atom in a to any atoms in center.

    parameters:
        a:        shape(n,k) array of n k-dimensional coordinates
        center:   shape(m,k) array of m k-dimensional coordinates
        bruteforce: Whether to use bruteforce calculation or a grid-based method
        gridsize: The gridsize of the grid-based method [default: 10.0]
        use_cuda: For bruteforce calculation, whether to use the cuda GPU-accelerated
                    version or the fortran OMP-accelerated one. By default chooses cuda if available,
                    otherwise OMP. Ignored if bruteforce=False.
        nthreads: Passed down to mindist_numba. Changes the amount of threads in the
                    cuda kernel. If None, tries to be smart. [default: None].

    returns:
        out: array of (n,) minimum distances

    NOTE! if bruteforce=False, then k must 3
    """
    if(bruteforce and use_cuda):
        if(_can_use_cuda):
            return mindist_cuda(a,center,nthreads)
        print("Failed to use cuda JIT with numba", file=sys.stderr)
        print("Reason: %s: %s"%(reason.__class__.__name__,str(reason)), file=sys.stderr)
        print("Using copmiled fortran version", file=sys.stderr)
    if(bruteforce):
        return mindist_omp(a,center)
    return mindist_grid(a,center, gridsize)
