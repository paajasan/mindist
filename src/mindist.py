# MIT License
#
# Copyright (c) 2025 Santeri Paajanen
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

mindist_omp = _f2py.mindist
__mindist_grid = _f2py.mindist_grid
mindist_omp_traj = _f2py.mindist_trajectory
mindist_omp_pbc = _f2py.mindist_PBC
mindist_omp_pbc_traj = _f2py.mindist_PBC_trajectory
__mindist_pbc_grid = _f2py.mindist_PBC_grid

_can_use_cuda = False
reason = None

try:
    from numba import cuda
    from numba import float64 as numba_float64
    import numpy as np
    import math

    _cuda_device = cuda.select_device(0)
    # Code for "clever" choosing of nthreads
    MAX_THREADS_PER_BLOCK = _cuda_device.MAX_THREADS_PER_BLOCK
    # 4 times SM count should give good occupancy... I think
    MIN_BLOCKS = _cuda_device.MULTIPROCESSOR_COUNT*4
    WARP_SIZE = _cuda_device.WARP_SIZE
    # Possible nthreads are multiples of WARP_SIZE up to MAX_THREADS_PER_BLOCK.
    # Should be ordered in descending order
    nthread_pos = np.arange(WARP_SIZE,
                            MAX_THREADS_PER_BLOCK+1,
                            WARP_SIZE)[::-1]

    def get_nthreads(n):
        """
        Parameters:
            n: the total number of threads
        Returns:
            nthreads: The smallest number of threads from nthread_pos, that gives at least MIN_BLOCKS blocks
        """
        global nthread_pos, MIN_BLOCKS
        bigenough = (n-1)//nthread_pos+1 >= MIN_BLOCKS
        i = np.argmax(bigenough)
        return int(nthread_pos[i]) if bigenough[i] else int(nthread_pos[-1])

    def get_2nthreads(n0, n1):
        nt1 = get_nthreads(n1)
        nt0 = 2**((get_nthreads(n0*n1)//nt1).bit_length()-1)
        return nt0, nt1

    @cuda.jit(["void(f4[:,:], f4[:,:], f4[:])",
               "void(f8[:,:], f8[:,:], f8[:])"])
    def __mindist_cuda(a, center, out):
        i = cuda.grid(1)
        if (i >= out.shape[0]):
            return
        mind = 1000000.0
        for j in range(center.shape[0]):
            dist = 0.0
            for k in range(a.shape[-1]):
                dist += (a[i, k]-center[j, k])**2

            if (dist < mind):
                mind = dist
        out[i] = math.sqrt(mind)

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
        dtype = max(a.dtype, center.dtype)
        dtype = max(dtype, np.dtype('float32'))
        dtype = min(dtype, np.dtype('float64'))

        if (a.dtype != dtype):
            a = a.astype(dtype)
        if (center.dtype != dtype):
            center = center.astype(dtype)

        d_a = cuda.to_device(np.ascontiguousarray(a))
        d_c = cuda.to_device(np.ascontiguousarray(center))
        d_o = cuda.device_array(d_a.shape[0], dtype=dtype)
        if (nthreads is None):
            nthreads = get_nthreads(a.shape[0])

        __mindist_cuda[(a.shape[0]-1)//nthreads+1, nthreads](d_a, d_c, d_o)
        out = d_o.copy_to_host()
        return out

    @cuda.jit(["void(f4[:,:,:], f4[:,:,:], f4[:,:])",
               "void(f8[:,:,:], f8[:,:,:], f8[:,:])"])
    def __mindist_cuda_traj(a, center, out):
        i, j = cuda.grid(2)
        if (i >= out.shape[0] or j >= out.shape[1]):
            return
        mind = 1000000.0
        for k in range(center.shape[1]):
            dist = 0.0
            for l in range(a.shape[-1]):
                dist += (a[i, j, l]-center[i, k, l])**2

            if (dist < mind):
                mind = dist
        out[i, j] = math.sqrt(mind)

    def mindist_cuda_traj(a, center, nthreads=None, batch_size=2048):
        """
        The numba cuda function used by mindist

        parameters:
            a: (n,k) array of n k-dimensional coordinates
            center: (m,k) array of m  k-dimensional coordinates
            nthreads: Changes the amount of threads in the cuda kernel.
                    If None, tries to be smart. [default: None].
            batch_size: Runs in batches of this size over the trajectory.
                        [default: 2048]

        returns:
            out: array of (n,) minimum distances
        """
        dtype = max(a.dtype, center.dtype)
        dtype = max(dtype, np.dtype('float32'))
        dtype = min(dtype, np.dtype('float64'))

        if (a.dtype != dtype):
            a = a.astype(dtype)
        if (center.dtype != dtype):
            center = center.astype(dtype)

        out = np.zeros(a.shape[:2], dtype=dtype)
        for i in range(0, a.shape[0], batch_size):
            a_batch = a[i:i+batch_size]
            c_batch = center[i:i+batch_size]
            d_a = cuda.to_device(np.ascontiguousarray(a_batch))
            d_c = cuda.to_device(np.ascontiguousarray(c_batch))
            d_o = cuda.device_array(d_a.shape[:2], dtype=dtype)
            if (nthreads is None):
                nt0, nt1 = get_2nthreads(*a_batch.shape[:2])
            else:
                if type(nthreads) == tuple:
                    nt0, nt1 = nthreads
                else:
                    nt0 = nt1 = nthreads

            __mindist_cuda_traj[((a_batch.shape[0]-1)//nt0+1, (a.shape[1]-1)//nt1+1),
                                (nt0, nt1)](d_a, d_c, d_o)
            out[i:i+batch_size] = d_o.copy_to_host()
        return out

    @cuda.jit(["void(f4[:,:], f4[:,:], f4[:,:], f4[:])",
               "void(f8[:,:], f8[:,:], f8[:,:], f8[:])"])
    def __mindist_cuda_pbc(s_a, s_c, box, out):
        i = cuda.grid(1)
        if (i >= out.shape[0]):
            return
        mind = 1000000.0
        diff = cuda.local.array(3, s_a.dtype)
        diff2 = cuda.local.array(3, s_a.dtype)
        for j in range(s_c.shape[0]):
            dist = 0.0
            for k in range(3):
                diff[k] = s_a[i, k]-s_c[j, k]
                if (diff[k] > 0.5):
                    diff[k] -= 1.0
                elif (diff[k] < -0.5):
                    diff[k] += 1.0

            for l in range(3):
                diff2[l] = 0.0
                for m in range(3):
                    diff2[l] += box[m, l]*diff[m]

            for k in range(3):
                dist += diff2[k]**2

            if (dist < mind):
                mind = dist
        out[i] = math.sqrt(mind)

    def mindist_cuda_pbc(a, center, box, nthreads=None):
        """
        The numba cuda function used by mindist_pbc

        parameters:
            a:        (n,3) array of n 3-dimensional coordinates
            center:   (m,3) array of m 3-dimensional coordinates
            box:      (n,3,3) array of 3 by 3 box vectors
            nthreads: Changes the amount of threads in the cuda kernel.
                    If None, tries to be smart. [default: None]

        returns:
            out: array of (n,) minimum distances
        """
        dtype = max(a.dtype, center.dtype, box.dtype)
        dtype = max(dtype, np.dtype('float32'))
        dtype = min(dtype, np.dtype('float64'))

        if (a.dtype != dtype):
            a = a.astype(dtype)
        if (center.dtype != dtype):
            center = center.astype(dtype)
        if (box.dtype != dtype):
            box = box.astype(dtype)

        invbox = np.linalg.inv(box)

        d_a = cuda.to_device(np.ascontiguousarray((a @ invbox) % 1))
        d_c = cuda.to_device(np.ascontiguousarray((center @ invbox) % 1))
        d_b = cuda.to_device(np.ascontiguousarray(box))
        d_o = cuda.device_array(d_a.shape[0], dtype=dtype)
        if (nthreads is None):
            nthreads = get_nthreads(a.shape[0])
        __mindist_cuda_pbc[(a.shape[0]-1)//nthreads+1,
                           nthreads](d_a, d_c, d_b, d_o)
        out = d_o.copy_to_host()
        return out

    @cuda.jit(["void(f4[:,:,:], f4[:,:,:], f4[:,:,:], f4[:,:])",
               "void(f8[:,:,:], f8[:,:,:], f8[:,:,:], f8[:,:])"])
    def __mindist_cuda_pbc_traj(s_a, s_c, box, out):
        i, j = cuda.grid(2)
        if (i >= out.shape[0] or j >= out.shape[1]):
            return
        mind = 1000000.0
        diff = cuda.local.array(3, s_a.dtype)
        diff2 = cuda.local.array(3, s_a.dtype)
        for k in range(s_c.shape[1]):
            dist = 0.0
            for l in range(3):
                diff[l] = s_a[i, j, l]-s_c[i, k, l]
                if (diff[l] > 0.5):
                    diff[l] -= 1.0
                elif (diff[l] < -0.5):
                    diff[l] += 1.0

            for l in range(3):
                diff2[l] = 0.0
                for m in range(3):
                    diff2[l] += box[i, m, l]*diff[m]

            for l in range(3):
                dist += diff2[l]**2

            if (dist < mind):
                mind = dist
        out[i, j] = math.sqrt(mind)

    def mindist_cuda_pbc_traj(a, center, box, nthreads=None, batch_size=2048):
        """
        The numba cuda function used by mindist_pbc

        parameters:
            a:          (n,m,3) array of n 3-dimensional coordinates
            center:     (n,k,3) array of m 3-dimensional coordinates
            box:        (n,3,3) array of 3 by 3 box vectors for each frame.
            nthreads:   Changes the amount of threads in the cuda kernel.
                        If None, tries to be smart. [default: None]
            batch_size: Runs in batches of this size over the trajectory.
                        [default: 2048]

        returns:
            out: array of (n,m) minimum distances
        """
        dtype = max(a.dtype, center.dtype, box.dtype)
        dtype = max(dtype, np.dtype('float32'))
        dtype = min(dtype, np.dtype('float64'))

        if (a.dtype != dtype):
            a = a.astype(dtype)
        if (center.dtype != dtype):
            center = center.astype(dtype)
        if (box.dtype != dtype):
            box = box.astype(dtype)

        invbox = np.linalg.inv(box)
        out = np.zeros(a.shape[:2], dtype=dtype)
        for i in range(0, a.shape[0], batch_size):
            a_batch = a[i:i+batch_size]
            c_batch = center[i:i+batch_size]
            ib_batch = invbox[i:i+batch_size]
            b_batch = box[i:i+batch_size]
            d_a = cuda.to_device((a_batch @ ib_batch) % 1)
            d_c = cuda.to_device((c_batch @ b_batch) % 1)
            d_b = cuda.to_device(np.ascontiguousarray(b_batch))
            d_o = cuda.device_array(d_a.shape[:2], dtype=dtype)
            if (nthreads is None):
                nt0, nt1 = get_2nthreads(*a_batch.shape[:2])
            else:
                if type(nthreads) == tuple:
                    nt0, nt1 = nthreads
                else:
                    nt0 = nt1 = nthreads
            __mindist_cuda_pbc_traj[((a_batch.shape[0]-1)//nt0+1, (a.shape[1]-1)//nt1+1),
                                    (nt0, nt1)](d_a, d_c, d_b, d_o)
            out[i:i+batch_size] = d_o.copy_to_host()
        return out

    _can_use_cuda = True

except ImportError as e:
    reason = e

except Exception as e:
    print("Imported numba and numba.cuda successfully, but another exception caught:")
    print("%s: %s" % (e.__class__.__name__, str(e)), file=sys.stderr)

    if (e.__class__ is cuda.cudadrv.driver.LinkerError and "Unsupported .version" in str(e)):
        print("Check that your cuda version is same or newer than cudatoolkit", file=sys.stderr)
        v = cuda.cudadrv.runtime.Runtime().get_version()
        print("cudatoolkit version %s" %
              (".".join([str(i) for i in v])), file=sys.stderr)

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
    return __mindist_grid(a, center, gridsize)


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
    return __mindist_pbc_grid(a, center, box, gridsize)


def mindist_pbc(a, center, box, bruteforce=True, gridsize=12.5, use_cuda=_can_use_cuda, nthreads=None):
    """
    Calculates the minimun distance of each atom in a to any atoms in center, taking into account the PBC.

    A brute force method using either OpenMP or cuda for accelerations, or a grid based one.

    parameters:
        a:        shape(n,3) or shape(t,n,3) array of n 3-dimensional coordinates
        center:   shape(m,3) or shape(t,n,3) array of m 3-dimensional coordinates
        box:      shape(3,3) or shape(t,3,3) array of box vectors
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
    if (bruteforce and use_cuda):
        if (_can_use_cuda):
            if (len(a.shape) == 2):
                return mindist_cuda_pbc(a, center, box, nthreads)
            return mindist_cuda_pbc_traj(a, center, box, nthreads)
        print("Failed to use cuda JIT with numba", file=sys.stderr)
        print("Reason: %s: %s" %
              (reason.__class__.__name__, str(reason)), file=sys.stderr)
        print("Using copmiled fortran version", file=sys.stderr)
    if (bruteforce):
        if (len(a.shape) == 2):
            return mindist_omp_pbc(a, center, box)
        return mindist_omp_pbc_traj(a, center, box)
    if (len(a.shape) == 2):
        return mindist_pbc_grid(a, center, box, gridsize)
    return np.array(mindist_pbc_grid(a[i], center[i], gridsize[i]) for i in range(a.shape[0]))


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
    if (bruteforce and use_cuda):
        if (_can_use_cuda):
            if (len(a.shape) == 2):
                return mindist_cuda(a, center, nthreads)
            return mindist_cuda_traj(a, center, nthreads)
        print("Failed to use cuda JIT with numba", file=sys.stderr)
        print("Reason: %s: %s" %
              (reason.__class__.__name__, str(reason)), file=sys.stderr)
        print("Using copmiled fortran version", file=sys.stderr)
    if (bruteforce):
        if (len(a.shape) == 2):
            return mindist_omp(a, center, nthreads)
        return mindist_omp_traj(a, center, nthreads)
    if (len(a.shape) == 2):
        return mindist_grid(a, center, gridsize)
    return np.array(mindist_grid(a[i], center[i], gridsize[i]) for i in range(a.shape[0]))
