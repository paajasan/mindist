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

module mindist_PBC_grid
  use mindist_PBC_utils,  only: matinv3, invert
  use mindist_PBC_grid_utils, only: setup_utils, cleanup_utils, grid_dists, atomlist
  implicit none

  contains

subroutine mindist(a, center, n, m, box, binsize, out)
  implicit none
  integer, parameter:: dp=kind(0.d0)

!     Calculate the minimum distance for each element in a to groups in center
  integer, intent(in) :: n,m
  integer             :: i,j,k,nbins(3),curbin(3)
  real(dp), intent(in)  :: a(n,3), center(m,3)
  real(dp), intent(in)  :: binsize, box(3,3)
  real(dp), intent(out) :: out(n)
  real(dp)              :: binsizes(3), s_a(n,3), s_c(m,3), invbox(3,3)

  type(atomlist), dimension (:,:,:), allocatable :: atoms_in_bins

!f2py intent(in,hide)  n
!f2py intent(in,hide)  m
!f2py intent(in)   binsizes
!f2py intent(in), depend(n)  a
!f2py intent(in), depend(m)  center
!f2py intent(out), depend(n) out

  ! inverse box
  invbox = matinv3(box)


  ! invert coordinates
  
  call invert(a, center, n, m, box, s_a, s_c)

  ! invert binsizes
  !binsizes = binsize
  do i=1,3
    binsizes(i) = binsize/sqrt(sum(box(i,:)**2))
  enddo


  ! Put everything back in box
  s_a = modulo(s_a, 1.0_dp)
  s_c = modulo(s_c, 1.0_dp)

  ! Calc closest number of bins
  nbins = nint(1.0/binsizes)
  binsizes = 1.0_dp / nbins

  
  allocate(atoms_in_bins(nbins(1),nbins(2),nbins(3)))

  do i=1,nbins(1)
    do j=1,nbins(2)
      do k=1,nbins(3)
        atoms_in_bins(i,j,k)%num = 0
        atoms_in_bins(i,j,k)%next = 1
      enddo
    enddo
  enddo

  do i=1,m
    curbin=floor((s_c(i,:))/binsizes)+1
    k = atoms_in_bins(curbin(1),curbin(2),curbin(3))%num
    atoms_in_bins(curbin(1),curbin(2),curbin(3))%num = k+1
  enddo


  do i=1,nbins(1)
    do j=1,nbins(2)
      do k=1,nbins(3)
        if(atoms_in_bins(i,j,k)%num>0) then
          allocate(atoms_in_bins(i,j,k)%pos(atoms_in_bins(i,j,k)%num,3))
        endif
      enddo
    enddo
  enddo



  do i=1,m
    curbin=floor((s_c(i,:))/binsizes)+1
    j = atoms_in_bins(curbin(1),curbin(2),curbin(3))%next
    atoms_in_bins(curbin(1),curbin(2),curbin(3))%pos(j,:)=s_c(i,:)
    atoms_in_bins(curbin(1),curbin(2),curbin(3))%next = j+1
  enddo



  call setup_utils(nbins, binsizes, box, atoms_in_bins)

  
!$OMP PARALLEL DO PRIVATE(curbin, i,j)
  do i=1,n
    call grid_dists(s_a(i,:), out(i))
  enddo
!$OMP END PARALLEL DO



  do i=1,nbins(1)
    do j=1,nbins(2)
      do k=1,nbins(3)
        if(allocated(atoms_in_bins(i,j,k)%pos)) then
          deallocate(atoms_in_bins(i,j,k)%pos)
        endif
      enddo
    enddo
  enddo

  deallocate(atoms_in_bins)

  call cleanup_utils()

end


end module mindist_PBC_grid