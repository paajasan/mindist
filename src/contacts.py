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
import numpy as np

__contacts_pbc_grid = _f2py.contacts_PBC_grid_trajectory


def contacts_traj(a, center, box, cutoff=6, gridsize=12.5):
    """
    A grid-based method for calculating the contacts, taking PBC into account.

    The default gridsize should be a good guess for atomic systems, assuming distances are in
    Ångströms. If they are in nm use cutoff=.6 and gridsize=1.25 instead. The real best value
    will of course depend on your system, but short of timing your system with different values,
    this is your best bet.

    parameters:
        a:        shape(n,m,3) array of n 3-dimensional coordinates
        center:   shape(n,m,3) array of m 3-dimensional coordinates
        box:      shape(n,3,3) array of box vectors
        gridsize: Side length of cubic gridcells [default: 12.5]

    returns:
        out: boolean array of (n,m) contacts
    """
    if (gridsize < cutoff):
        gridsize = cutoff
        print(f"Gridsize should be at least the size of cutoff. Setting to {cutoff}.",
              file=sys.stderr)
    return np.ascontiguousarray(__contacts_pbc_grid(a, center, box, cutoff, gridsize), dtype=bool)


def contacts(a, center, box, cutoff=6, gridsize=12.5):
    """
    A grid-based method for calculating the contacts, taking PBC into account.

    The default gridsize should be a good guess for atomic systems, assuming distances are in
    Ångströms. If they are in nm use cutoff=.6 and gridsize=1.25 instead. The real best value
    will of course depend on your system, but short of timing your system with different values,
    this is your best bet.

    Parallelization is only done over trajectories.

    parameters:
        a:        shape([n,]m,3) array of (n by) m 3-dimensional coordinates
        center:   shape([n,]k,3) array of (n by) k 3-dimensional coordinates
        box:      shape([n,]3,3) array of (n) box vectors
        gridsize: Side length of cubic gridcells [default: 12.5]

    returns:
        out: boolean array of ([n,]m) contacts
    """
    if (a.shape == 2):
        a = a[None, ...]
        center = center[None, ...]
        box = box[None, ...]
    return contacts_traj(a, center, box, cutoff, gridsize)
