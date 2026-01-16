!! MIT License
!! 
!! Copyright (c) 2022 Santeri Paajanen
!! 
!! Permission is hereby granted, free of charge, to any person obtaining a copy
!! of this software and associated documentation files (the "Software"), to deal
!! in the Software without restriction, including without limitation the rights
!! to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
!! copies of the Software, and to permit persons to whom the Software is
!! furnished to do so, subject to the following conditions:
!! 
!! The above copyright notice and this permission notice shall be included in all
!! copies or substantial portions of the Software.
!! 
!! THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
!! IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
!! FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
!! AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
!! LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
!! OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
!! SOFTWARE.

module contacts_PBC_grid_trajectory
  use mindist_PBC_utils,  only: matinv3, invert
  use contacts_PBC_grid_utils, only: in_contact, atomlist
  implicit none

  contains

subroutine contacts(a, center, n, m, k, box, cutoff, binsize, out)
  implicit none
  integer, parameter:: dp=kind(0.d0)

!     Calculate the minimum distance for each element in a to groups in center
  integer, intent(in) :: n,m,k
  integer             :: t,i,j,l,nbins(3),curbin(3)
  real(dp), intent(in)  :: a(n,m,3), center(n,k,3)
  real(dp), intent(in)  :: binsize, cutoff, box(n,3,3)
  logical*1, intent(out)  :: out(n,m)
  real(dp)              :: binsizes(3), s_a(m,3), s_c(k,3)

  type(atomlist), dimension (:,:,:), allocatable :: atoms_in_bins

!f2py intent(in,hide)  n
!f2py intent(in,hide)  m
!f2py intent(in,hide)  k
!f2py intent(in)   binsizes
!f2py intent(in), depend(n,m)  a
!f2py intent(in), depend(n,k)  center
!f2py intent(out), depend(n,m) out

  
  !$OMP PARALLEL PRIVATE(atoms_in_bins, nbins, curbin, t, i, j, l, binsizes, s_a, s_c)

  !$OMP DO
    do t=1,n
        call invert(a(t,:,:), center(t,:,:), m, k, box(t,:,:), s_a(:,:), s_c(:,:))
        ! invert binsizes
        do i=1,3
            binsizes(i) = binsize/sqrt(sum(box(t,i,:)**2))
        enddo

        ! Calc closest number of bins
        nbins = nint(1.0/binsizes)
        binsizes = 1.0_dp / nbins

        
        ! Allocate grid
        allocate(atoms_in_bins(nbins(1),nbins(2),nbins(3)))
        ! Set empty values in grid
        do i=1,nbins(1)
            do j=1,nbins(2)
                do l=1,nbins(3)
                    atoms_in_bins(i,j,l)%num = 0
                    atoms_in_bins(i,j,l)%next = 1
                enddo
            enddo
        enddo


        ! Calculate number of atoms in gridcells
        do i=1,k
            curbin=floor((s_c(i,:))/binsizes)+1
            l = atoms_in_bins(curbin(1),curbin(2),curbin(3))%num
            atoms_in_bins(curbin(1),curbin(2),curbin(3))%num = l+1
        enddo


        ! Allocate grid lists if needed
        do i=1,nbins(1)
            do j=1,nbins(2)
                do l=1,nbins(3)
                    if(atoms_in_bins(i,j,l)%num>0) then
                    allocate(atoms_in_bins(i,j,l)%pos(atoms_in_bins(i,j,l)%num,3))
                    endif
                enddo
            enddo
        enddo

        ! Add atoms poitions to respective grid lists
        do i=1,k
            curbin=floor((s_c(i,:))/binsizes)+1
            j = atoms_in_bins(curbin(1),curbin(2),curbin(3))%next
            atoms_in_bins(curbin(1),curbin(2),curbin(3))%pos(j,:)=s_c(i,:)
            atoms_in_bins(curbin(1),curbin(2),curbin(3))%next = j+1
        enddo


        ! Calculate contacts
        do i=1,m
            out(t,i) = in_contact(s_a(i,:), atoms_in_bins, box(t,:,:), binsizes, cutoff)
        enddo

        ! Free memory
        do i=1,nbins(1)
            do j=1,nbins(2)
                do l=1,nbins(3)
                    if(allocated(atoms_in_bins(i,j,l)%pos)) then
                        deallocate(atoms_in_bins(i,j,l)%pos)
                    endif
                enddo
            enddo
        enddo

        deallocate(atoms_in_bins)

    enddo
  !$OMP END DO
  !$OMP END PARALLEL
end


end module contacts_PBC_grid_trajectory