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

module mindist_grid_utils
  use stdlib_sorting, only: sort_index,int_size
  implicit none
  integer, parameter :: dp=kind(0.d0)
  integer            :: numbins,nbins(3),numbins_c,nbins_c(3)
  real(dp)           :: binsize, mins(3), mins_c(3)
  real(dp), allocatable :: bindists(:,:)
  integer(int_size), allocatable :: sorted_index(:,:)

  ! atomlist saves the num for numatoms, as well as next, to keep
  ! track how many have been added. pos is the array of positions
  type :: atomlist
    integer :: num,next
    real(dp), dimension(:,:), allocatable :: pos
  end type atomlist


  ! Grid of atomlists
  type(atomlist),allocatable   :: atoms_in_bins(:,:,:)  
  
contains

! Helper subroutine to get the i'th bin indices
subroutine int_to_bin(i,bin_ind)
  integer, intent(in)  :: i
  integer, intent(out) :: bin_ind(3)

  bin_ind(1)=modulo(i-1,nbins(1))+1
  bin_ind(2)=modulo((i-1)/nbins(1),nbins(2))+1
  bin_ind(3)=modulo((i-1)/(nbins(1)*nbins(2)),nbins(3))+1
end

! Helper subroutine to get the number of a bin
subroutine bin_to_int(bin_ind,i)
  integer, intent(in)  :: bin_ind(3)
  integer, intent(out) :: i

  i=bin_ind(1)+(bin_ind(2)-1)*nbins(1)+(bin_ind(3)-1)*nbins(1)*nbins(2)
end

! Helper subroutine to calculate squared distance of two points
subroutine distance(pos1, pos2, dist)
  integer, parameter:: dp=kind(0.d0)
  real(dp), intent(in) :: pos1(3),pos2(3)
  real(dp)              :: diff(3)
  real(dp), intent(out) :: dist

  diff = pos1-pos2 
  dist = sum(diff**2)
end

! Like int_to_bin, but in the center grid
subroutine int_to_bin_c(i,bin_ind)
  integer, intent(in)  :: i
  integer, intent(out) :: bin_ind(3)

  bin_ind(1)=modulo(i-1,nbins_c(1))+1
  bin_ind(2)=modulo((i-1)/nbins_c(1),nbins_c(2))+1
  bin_ind(3)=modulo((i-1)/(nbins_c(1)*nbins_c(2)),nbins_c(3))+1
end

! Like bin_to_int, but in the center grid
subroutine bin_to_int_c(bin_ind,i)
  integer, intent(in)  :: bin_ind(3)
  integer, intent(out) :: i

  i=bin_ind(1)+(bin_ind(2)-1)*nbins_c(1)+(bin_ind(3)-1)*nbins_c(1)*nbins_c(2)
end


subroutine setup_utils(nubins, minims, nubins_c, minims_c, bin_size, atomlists_bins)
  real(dp), intent(in)  :: minims(3), minims_c(3), bin_size
  integer, intent(in)  :: nubins(3), nubins_c(3)
  type(atomlist), dimension (:,:,:), intent(in) :: atomlists_bins
  integer               :: i,j,nb,bin1(3),bin2(3)
  real(dp)              :: dist

  atoms_in_bins = atomlists_bins

  nbins = nubins
  mins  = minims
  nbins_c = nubins_c
  mins_c  = minims_c
  binsize = bin_size

  numbins=nbins(1)*nbins(2)*nbins(3)
  numbins_c=nbins_c(1)*nbins_c(2)*nbins_c(3)

  allocate(bindists(numbins_c,numbins))
  allocate(sorted_index(numbins_c,numbins))

  !$OMP PARALLEL DO private(dist,nb,j,bin1,bin2)
  do i=1,numbins_c
    call int_to_bin_c(i,bin1)
    nb = atoms_in_bins(bin1(1),bin1(2),bin1(3))%num
    do j=1,numbins
      if(nb>0) then
        ! Get indices of bin2 and calculate distance of bin centers
        call int_to_bin(j,bin2)
        call distance(mins_c+bin1*binsize,mins+bin2*binsize, dist)
      else
        ! If there are no atoms in the grdpoint, teh distance is set to a lage value
        dist = 10*sum(nbins**2)*binsize**2
      endif
      bindists(i,j) = dist
    enddo
  enddo
  !$OMP END PARALLEL DO

  !$OMP PARALLEL DO
  do i=1,numbins
    ! sort bindists in ascending order
    call sort_index(bindists(:,i),sorted_index(:,i))
    bindists(:,i) = sqrt(bindists(:,i))
  enddo
  !$OMP END PARALLEL DO



end

subroutine cleanup_utils()

  deallocate(bindists)
  deallocate(sorted_index)
end


subroutine grid_dists(pos, mind)
  real(dp), intent(in)  :: pos(3)
  real(dp)              :: dist,bin_diff,mind2
  integer               :: i,j,k, bi,curbin(3), bin_ind(3)
  type(atomlist) :: curlist

  real(dp), intent(out) :: mind

  do i=1,3
    curbin(i)=floor((pos(i)-mins(i))/binsize)+1
  enddo

  call bin_to_int(curbin,bi)
  ! maximum amount the atom-to-atom distance can vary from bin-to-bin distance
  bin_diff = 1.7321*binsize

  mind  = huge(1._dp)
  mind2 = huge(1._dp)
  ! iterate over bins
  do i=1,numbins_c
    ! if net bin is too far, we can stop iterating
    if(mind<bindists(i,bi)-bin_diff) then
      exit
    endif
    ! get next closest bin
    j = int(sorted_index(i, bi))

    call int_to_bin_c(j, bin_ind)
    ! get atom list of the bin
    curlist = atoms_in_bins(bin_ind(1),bin_ind(2),bin_ind(3)) 
    ! Technically this should not happen
    ! TODO: remove below
    if(curlist%num==0) then
      cycle
    endif

    ! Go through the atoms in the bin
    do k=1,curlist%num
      ! Calulate squared distance and if smaller than pevious smallest, save
      call distance(curlist%pos(k,:), pos,dist)
      if(dist<mind2) then
        mind2 = dist
        mind  = sqrt(dist)
      endif
    enddo
  enddo

end
  



end module mindist_grid_utils