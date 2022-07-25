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

module mindist_PBC_grid_utils
  use mindist_PBC_utils, only: distance_PBC
  use stdlib_sorting, only: sort_index,int_size
  implicit none
  integer, parameter :: dp=kind(0.d0)
  integer            :: numbins,nbins(3)
  real(dp)           :: binsizes(3), box(3,3), bindiff
  real(dp), allocatable :: bindists(:,:)
  integer(int_size), allocatable :: sorted_index(:,:)

  type :: atomlist
    integer :: num,next
    real(dp), dimension(:,:), allocatable :: pos
  end type atomlist



  type(atomlist),allocatable   :: atoms_in_bins(:,:,:)  
  
contains

subroutine int_to_bin(i,bin_ind)
  integer, intent(in)  :: i
  integer, intent(out) :: bin_ind(3)

  bin_ind(1)=modulo(i-1,nbins(1))+1
  bin_ind(2)=modulo((i-1)/nbins(1),nbins(2))+1
  bin_ind(3)=modulo((i-1)/(nbins(1)*nbins(2)),nbins(3))+1
end

subroutine bin_to_int(bin_ind,i)
  integer, intent(in)  :: bin_ind(3)
  integer, intent(out) :: i

  i=bin_ind(1)+(bin_ind(2)-1)*nbins(1)+(bin_ind(3)-1)*nbins(1)*nbins(2)
end



subroutine setup_utils(nubins, bin_sizes, boxv, atomlists_bins)
  real(dp), intent(in)  :: bin_sizes(3), boxv(3,3)
  integer, intent(in)  :: nubins(3)
  type(atomlist), dimension (:,:,:), intent(in) :: atomlists_bins
  integer               :: i,j,k,l,nb,bin1(3),bin2(3)
  integer               :: bindiff_signs(3)
  real(dp)               :: tmp(3)

  atoms_in_bins = atomlists_bins

  nbins = nubins
  binsizes = bin_sizes
  box = boxv

  numbins=nbins(1)*nbins(2)*nbins(3)

  allocate(bindists(numbins,numbins))
  allocate(sorted_index(numbins,numbins))

  !$OMP PARALLEL DO private(nb,j,bin1,bin2)
  do i=1,numbins
    call int_to_bin(i,bin1)
    nb = atoms_in_bins(bin1(1),bin1(2),bin1(3))%num
    if(nb>0) then
      do j=1,numbins
        call int_to_bin(j,bin2)
        call distance_PBC((bin1*binsizes-0.5*binsizes),(bin2*binsizes-0.5*binsizes), box, bindists(i,j))
      enddo
    else
      bindists(i,:) = 1e10_dp
    endif
  enddo
  !$OMP END PARALLEL DO

  !$OMP PARALLEL DO
  do i=1,numbins
    call sort_index(bindists(:,i),sorted_index(:,i))
    bindists(:,i) = sqrt(bindists(:,i))
  enddo
  !$OMP END PARALLEL DO

  bindiff =  0
  do i=-1,1
    do k=-1,1
      do l=-1,1
        tmp = 0.0
        bindiff_signs = (/i,k,l/)
        do j=1,3
          tmp = tmp+ bindiff_signs(j)*box(j,:)*binsizes
        enddo
        ! TODO: why is the coefficient of 4 needed!
        bindiff = max(2*sqrt(sum(tmp**2)), bindiff)
      enddo
    enddo
  enddo

end

subroutine cleanup_utils()

  deallocate(bindists)
  deallocate(sorted_index)
end


subroutine grid_dists(pos, mind)
  real(dp), intent(in)  :: pos(3)
  real(dp)              :: dist,mind2
  integer               :: i,j,k, bi,curbin(3), bin_ind(3)
  type(atomlist) :: curlist

  real(dp), intent(out) :: mind

  curbin=floor(pos/binsizes)+1

  call bin_to_int(curbin,bi)

  mind  = huge(1._dp)
  mind2 = huge(1._dp)
  do i=1,numbins
    if(mind<bindists(i,bi)-bindiff) then
      exit
    endif
    j = int(sorted_index(i, bi))

    call int_to_bin(j, bin_ind)
    
    curlist = atoms_in_bins(bin_ind(1),bin_ind(2),bin_ind(3)) 
    if(curlist%num==0) then
      exit
    endif

    do k=1,curlist%num
      call distance_PBC(curlist%pos(k,:), pos, box,dist)
      if(dist<mind2) then
        mind  = sqrt(dist)
        mind2 = dist
      endif
    enddo
  enddo

end
  



end module mindist_PBC_grid_utils