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


subroutine mindist(a, center, n, m, k, d, out)
  implicit none

!     Calculate the minimum distance for each element in a to groups in center
  integer, parameter:: dp=kind(0.d0)
  integer, intent(in) :: n,m,k,d
  integer             :: i,j,l
  real(dp), intent(in)  :: a(n,m,d), center(n,k,d)
  real(dp), intent(out) :: out(n,m)
  real(dp)              :: dist, x

!f2py intent(in,hide)  n
!f2py intent(in,hide)  m
!f2py intent(in,hide)  k
!f2py intent(in,hide)  d
!f2py intent(in), depend(n,k,d)  a
!f2py intent(in), depend(n,k,d)  center
!f2py intent(out), depend(n,m) out

!$OMP PARALLEL DO private(dist)
  do i=1,n
    do j=1,m
        x = huge(1._dp)
        do l=1,k
            dist = sum((a(i,j,:)-center(i,l,:))**2)
            if(dist<x) then
                x=dist
            endif
        enddo
        out(i,j) = sqrt(x)
    enddo
  enddo
!$OMP END PARALLEL DO

end
