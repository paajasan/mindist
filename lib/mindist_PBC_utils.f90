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


module mindist_PBC_utils
  implicit none
  integer, parameter:: dp=kind(0.d0)

contains


pure function matinv3(A) result(B)
  ! From https://fortranwiki.org/fortran/show/Matrix+inversion, public domain
  !! Performs a direct calculation of the inverse of a 3×3 matrix.
  real(dp), intent(in) :: A(3,3)   !! Matrix
  real(dp)             :: B(3,3)   !! Inverse matrix
  real(dp)             :: detinv

  ! Calculate the inverse determinant of the matrix
  detinv = 1/(A(1,1)*A(2,2)*A(3,3) - A(1,1)*A(2,3)*A(3,2)&
            - A(1,2)*A(2,1)*A(3,3) + A(1,2)*A(2,3)*A(3,1)&
            + A(1,3)*A(2,1)*A(3,2) - A(1,3)*A(2,2)*A(3,1))

  ! Calculate the inverse of the matrix
  B(1,1) = +detinv * (A(2,2)*A(3,3) - A(2,3)*A(3,2))
  B(2,1) = -detinv * (A(2,1)*A(3,3) - A(2,3)*A(3,1))
  B(3,1) = +detinv * (A(2,1)*A(3,2) - A(2,2)*A(3,1))
  B(1,2) = -detinv * (A(1,2)*A(3,3) - A(1,3)*A(3,2))
  B(2,2) = +detinv * (A(1,1)*A(3,3) - A(1,3)*A(3,1))
  B(3,2) = -detinv * (A(1,1)*A(3,2) - A(1,2)*A(3,1))
  B(1,3) = +detinv * (A(1,2)*A(2,3) - A(1,3)*A(2,2))
  B(2,3) = -detinv * (A(1,1)*A(2,3) - A(1,3)*A(2,1))
  B(3,3) = +detinv * (A(1,1)*A(2,2) - A(1,2)*A(2,1))
end function


subroutine invert(a, center, n, m, box, s_a, s_c)
  implicit none
  integer, parameter:: dp=kind(0.d0)

!     Calculate the minimum distance for each element in a to groups in center, taking PBC into account
  
  integer, intent(in)   :: n,m
  real(dp), intent(in)  :: a(n,3), center(m,3), box(3,3)
  real(dp), intent(out) :: s_a(n,3), s_c(m,3)
  real(dp)              :: invbox(3,3)

  invbox = matinv3(box)


  s_a = matmul(a,invbox)
  s_c = matmul(center,invbox)
  !call vecsmul3(invbox, a, n, s_a)
  !call vecsmul3(invbox, center, m, s_c)


  ! Put everything back in box
  s_a = modulo(s_a, 1.0_dp)
  s_c = modulo(s_c, 1.0_dp)

end subroutine invert

subroutine distance_PBC(s_1, s_2, box, dist)
  integer, parameter:: dp=kind(0.d0)
  real(dp), intent(in) :: s_1(3),s_2(3),box(3,3)
  real(dp)              :: diff(3)
  real(dp), intent(out) :: dist
  integer               :: i

  diff = s_1-s_2 
  do i=1,3
    if(abs(diff(i))>0.5) then
      diff(i) = diff(i)-sign(1.0_dp,diff(i))
    endif
  enddo

  diff = matmul(diff, box)

  dist = sum(diff**2)
end


end module mindist_PBC_utils