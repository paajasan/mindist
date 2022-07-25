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


module mindist_grid
  use mindist_grid_utils, only: setup_utils, cleanup_utils, grid_dists, atomlist
  implicit none

  contains

subroutine mindist(a, center, n, m, binsize, out)
  implicit none
  integer, parameter:: dp=kind(0.d0)

!     Calculate the minimum distance for each element in a to groups in center
  integer, intent(in) :: n,m
  integer             :: i,j,k,nbins(3),nbins_c(3),curbin(3)
  real(dp), intent(in)  :: a(n,3), center(m,3)
  real(dp), intent(in)  :: binsize
  real(dp), intent(out) :: out(n)
  real(dp)              :: mins(3),maxs(3),mins_c(3),maxs_c(3)

  type(atomlist), dimension (:,:,:), allocatable :: atoms_in_bins

!f2py intent(in,hide)  n
!f2py intent(in,hide)  m
!f2py intent(in)   binsize
!f2py intent(in), depend(n)  a
!f2py intent(in), depend(m)  center
!f2py intent(out), depend(n) out


  mins(:)=center(1,:)
  maxs(:)=center(1,:)
  
  !$OMP PARALLEL DO PRIVATE(i,j) reduction(min:mins)  reduction(max:maxs)
  do i=2,m
    do j=1,3
      if(mins(j)>center(i,j)) then
        mins(j)=center(i,j)
      else if(maxs(j)<center(i,j)) then
        maxs(j)=center(i,j)
      endif
    enddo
  enddo
  !$OMP END PARALLEL DO

  mins_c = mins
  maxs_c = maxs


  !$OMP PARALLEL DO PRIVATE(i,j) reduction(min:mins)  reduction(max:maxs)
  do i=1,n
    do j=1,3
      if(mins(j)>a(i,j)) then
        mins(j)=a(i,j)
      else if(maxs(j)<a(i,j)) then
        maxs(j)=a(i,j)
      endif
    enddo
  enddo
  !$OMP END PARALLEL DO

  do i=1,3
    mins(i) = mins(i)-binsize/2
    nbins(i) = floor((binsize/2+maxs(i)-mins(i))/binsize)+1
    mins_c(i) = mins_c(i)-binsize/2
    nbins_c(i) = floor((binsize/2+maxs_c(i)-mins_c(i))/binsize)+1
  enddo

  
  allocate(atoms_in_bins(nbins_c(1),nbins_c(2),nbins_c(3)))

  do i=1,nbins_c(1)
    do j=1,nbins_c(2)
      do k=1,nbins_c(3)
        atoms_in_bins(i,j,k)%num = 0
        atoms_in_bins(i,j,k)%next = 1
      enddo
    enddo
  enddo


  do i=1,m
    do j=1,3
      curbin(j)=floor((center(i,j)-mins_c(j))/binsize)+1
    enddo
    k = atoms_in_bins(curbin(1),curbin(2),curbin(3))%num
    atoms_in_bins(curbin(1),curbin(2),curbin(3))%num = k+1
  enddo


  do i=1,nbins_c(1)
    do j=1,nbins_c(2)
      do k=1,nbins_c(3)
        if(atoms_in_bins(i,j,k)%num>0) then
          allocate(atoms_in_bins(i,j,k)%pos(atoms_in_bins(i,j,k)%num,3))
        endif
      enddo
    enddo
  enddo



  do i=1,m
    do j=1,3
      curbin(j)=floor((center(i,j)-mins_c(j))/binsize)+1
    enddo
    j = atoms_in_bins(curbin(1),curbin(2),curbin(3))%next
    atoms_in_bins(curbin(1),curbin(2),curbin(3))%pos(j,:)=center(i,:)
    atoms_in_bins(curbin(1),curbin(2),curbin(3))%next = j+1
  enddo



  call setup_utils(nbins, mins, nbins_c, mins_c, binsize, atoms_in_bins)

  
!$OMP PARALLEL DO PRIVATE(curbin, i,j)
  do i=1,n
    call grid_dists(a(i,:), out(i))
  enddo
!$OMP END PARALLEL DO


  do i=1,nbins_c(1)
    do j=1,nbins_c(2)
      do k=1,nbins_c(3)
        if(allocated(atoms_in_bins(i,j,k)%pos)) then
          deallocate(atoms_in_bins(i,j,k)%pos)
        endif
      enddo
    enddo
  enddo

  deallocate(atoms_in_bins)

  call cleanup_utils()

end


end module mindist_grid