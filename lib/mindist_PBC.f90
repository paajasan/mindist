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


module mindist_PBC
  use mindist_PBC_utils, only: matinv3, distance_PBC, invert
  implicit none

contains


subroutine inv(a, center, n, m, box, s_a, s_c)
  implicit none
  integer, parameter:: dp=kind(0.d0)

!     Calculate the minimum distance for each element in a to groups in center, taking PBC into account
  
  integer, intent(in)   :: n,m
  real(dp), intent(in)  :: a(n,3), center(m,3), box(3,3)
  real(dp), intent(out) :: s_a(n,3), s_c(m,3)

!f2py intent(in,hide)  n
!f2py intent(in,hide)  m
  call invert(a, center, n, m, box, s_a, s_c)

end subroutine inv

  subroutine mindist(a, center, n, m, box, out)
    implicit none
    integer, parameter:: dp=kind(0.d0)

  !     Calculate the minimum distance for each element in a to groups in center, taking PBC into account
    integer, intent(in) :: n,m
    integer             :: i,j
    real(dp), intent(in)  :: a(n,3), center(m,3), box(3,3)
    real(dp)              :: s_a(n,3), s_c(m,3)
    real(dp), intent(out) :: out(n)
    real(dp)              :: dist

  !f2py intent(in,hide)  n
  !f2py intent(in,hide)  m
  !f2py intent(in), depend(n)  a
  !f2py intent(in), depend(m)  center
  !f2py intent(in)             box
  !f2py intent(out), depend(n) out

    call invert(a, center, n, m, box, s_a, s_c)


  !$OMP PARALLEL DO private(dist)
    do i=1,n
      out(i) = huge(1._dp)
      do j=1,m
        call distance_PBC(s_a(i,:), s_c(j,:), box, dist)
        if(dist<out(i)) then
          out(i)=dist
        endif
      enddo
    enddo
  !$OMP END PARALLEL DO

    out = sqrt(out)

  end subroutine mindist


end module mindist_PBC