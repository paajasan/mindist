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

module contacts_PBC_grid_utils
  use mindist_PBC_utils, only: distance_PBC
  implicit none
  integer, parameter :: dp=kind(0.d0)

  type :: atomlist
    integer :: num,next
    real(dp), dimension(:,:), allocatable :: pos
  end type atomlist



  
contains


function in_contact(pos, atoms_in_bins, box, binsizes, cutoff) result(has_contact)
  real(dp), intent(in)       :: pos(3), binsizes(3), box(3,3), cutoff
  type(atomlist), intent(in) :: atoms_in_bins(:,:,:)
  logical*1                  :: has_contact
  real(dp)                   :: dist, cutoff2
  integer                    :: i,j,k,l,bin_ind(3),bin_mod(3)
  type(atomlist)             :: curlist

  cutoff2 = cutoff*cutoff

  ! 0 indexed bin index
  bin_ind=modulo(floor(pos/binsizes), shape(atoms_in_bins))

  do i=-3,3
    do j=-3,3
      do k=-3,3
          ! Add offset, wrap indices and add  for 1 indexed indices
          bin_mod = modulo(bin_ind+[k,j,i], shape(atoms_in_bins))+1
          curlist = atoms_in_bins(bin_mod(1),bin_mod(2),bin_mod(3)) 
          if(curlist%num==0) then
              continue
          endif

          do l=1,curlist%num
              call distance_PBC(curlist%pos(l,:), pos, box, dist)
              if(dist<cutoff2) then
                  has_contact = .true.
                  return
              endif
          enddo
      enddo
    enddo
  enddo

  has_contact=.false.
end
  



end module contacts_PBC_grid_utils