C     ********************************************
C     NUMERICAL GRID GENERATION -ELLIPTICAL SYSTEM
C     GENERATES STRUCTURED GRID AROUND 2D AEROFOIL
C     GRID LINES ARE ATTRACTED TOWARDS THE AF WALL
C     PROGRAM MADE ON 25 04 1993
C     PROGRAM TOUCHED ON 20 05 1993
C     ********************************************
C     MAIN PROGRAM
      implicit double precision (a-h,o-z)
      double precision m, p, t, pi, c1, c2, c3, c4, c5, t1
      double precision theta, normal, shift, xn, yn 
      double precision X(5125,1921),Y(5125,1921)
      double precision xc(5125), yc(5125)
      double precision Xe(10000000),Ye(10000000) 
      integer N

C     Read the input from the file input.dat
      
      OPEN(UNIT=1,FILE='input.in')
      READ(1,*) NIMAX,NJMAX
      READ(1,*) r
      READ(1,*) dc
      READ(1,*) shift
      READ(1,*) m, p, t
      CLOSE(1)

C     Initialize some constants required later      
      
      pi = 4.0d0*datan(1.0d0)
      N = NIMAX+1
      normal=1.00893041136514d0
      c1=0.2969d0
      c2=-0.126d0
      c3=-0.3516d0
      c4=0.2843d0
      c5=-0.1015d0

C     Define the Outer Boundary (Circle)      

      Do i = 1,N
        t1 = 2.0d0*pi*real(i-1)/real(N-1)
        xc(i) = r*dcos(t1)
        if(shift==2) then
           xc(i) = xc(i)+0.5d0
        endif
           yc(i) = r*dsin(t1)
      enddo
      xc(N) = xc(1)
      yc(N) = yc(1)

      open(unit=1,file='farfield.dat')
      if( dc == 1 ) then
        do i = 1,N
          write(1,*)xc(i),yc(i)
        enddo
      else if ( dc == 2 ) then
        do i = N,1,-1
          write(1,*)xc(i),yc(i)
        enddo
      endif
      close(1)

C     Define the points on the Aerofoil Profile

      do i=1,nimax
         theta = 2.00 * pi * (i-1) / n
         Xe(i) = (0.5 * ( cos(theta) + 1.0d0 )) * normal
         xn = Xe(i)
         Ye(i)=5.0d0*t*(c1*xn**0.5+c2*xn+c3*xn**2+c4*xn**3+c5*xn**4)

         if(theta<pi) Ye(i) = -Ye(i)
         if(m==0)then
           yn = 0.00
         else if(xn<=p)then
           yn = m * normal * (2.00 * p * xn - xn**2 )/p**2
         else
           yn = m * normal * (1-2.00*p+2.0*p*xn-xn**2)/(1-p)**2
         endif
         Ye(i) = Ye(i) + yn
      end do

      do i=1,nimax
        Xe(i) = Xe(i) / normal
        Ye(i) = Ye(i) / normal
      end do

C     INITIALISES X,Y VALUES
      IMAX=NIMAX+2
      JMAX=NJMAX
      DO 5 I=1,IMAX
      DO 5 J=1,JMAX
      X(I,J)=0.0
      Y(I,J)=0.0
5     CONTINUE

c       x,y values on the boundary are determined
C       ----------------------------------------------
        NPOB=IMAX-2
      
        open(unit=20,file='farfield.dat')
        do i = 1,NPOB
          read(20,*)x(i,jmax),y(i,jmax) 
        enddo
        close(20)

        x(imax-1,jmax)=x(1,jmax)
        y(imax-1,jmax)=y(1,jmax)
        x(imax,jmax)=x(2,jmax)
        y(imax,jmax)=y(2,jmax)


        NPIB=IMAX-2


c       --------------------------------------------
c----Reading Aerofoil cocordinates---------------
c-----------------------------------------------------
	do i=1,imax-2
           x(i,1)= Xe(i)
           y(i,1)= Ye(i)
  	end do 

c       fixing x,y in overlap region for periodic b.c
        x(imax-1,1)=x(1,1)
        y(imax-1,1)=y(1,1)
        x(imax,1)=x(2,1)
        y(imax,1)=y(2,1)

c-------------------------------------------------------------------------

      IS=2
      JS=2
      IE=IMAX-1
      JE=JMAX-1

      CALL GG1(X,Y,IMAX,JMAX,IS,IE,JS,JE)


80    STOP
      END

C     ****************************************************
C     SUBROUTINE GG1
C     INITIALISES AND FINDS X,Y VALUES FOR BOUNDARY NODES
C     CALLS GG2 FOR INTERPOLATION
C     RETURNS X,Y VALUES TO MAIN PROGRAM
C     ****************************************************

      SUBROUTINE GG1(X,Y,IMAX,JMAX,IS,IE,JS,JE)
      implicit double precision (a-h,o-z)
      DIMENSION X(5125,1921),Y(5125,1921)

      CALL GG2 (X,Y,IMAX,JMAX)
      CALL GG3 (X,Y,IS,IE,JS,JE,IMAX,JMAX)
      OPEN(UNIT=2, file='grid.dat')
      DO 1 J=1,JMAX
      DO 1 I=1,(IMAX-2)
      WRITE(2,45)X(I,J),Y(I,J),I,J
45    FORMAT(1x,2(2x,e15.8),2x,i4,2x,i4)
1     CONTINUE
      CLOSE(2)

      open(3,file='gnu.dat')
      do 445 i = 1,imax-2
      do 445 j = 1,jmax-1
      write(3,446) x(i,j),y(i,j)
      write(3,447) x(i,j+1),y(i,j+1)

      write(3,446) x(i,j),y(i,j)
      write(3,447) x(i+1,j),y(i+1,j)
445   continue 

      open(unit=44,file='temp.dat')
      do j = 1,imax-2
      write(44,*)x(j,jmax-1),y(j,jmax-1)
      enddo
446   format(2f18.6)
447   format(2f18.6/)

      RETURN
      END

C     *****************************************************************
C     SUBROUTINE : GRID GENERATION-2
C     FINDS X,Y VALUES AT ALL NODES
C     USES LAGRANGIAN INTERPOLATION SCHEME
C     *****************************************************************

C     TRANSFINITE LINEAR INTERPOLATION
C     ---------------------------------
      SUBROUTINE GG2(X,Y,IMAX,JMAX)
      implicit double precision (a-h,o-z)
      DIMENSION X(5125,1921),Y(5125,1921),X1(5125,1921),Y1(5125,1921)
c     1          X2(367,128),Y2(367,128)
      DO 30 I=1,IMAX
      DO 30 J=2,JMAX-1
      
      X(I,J)=X(I,1)+(FLOAT(J-1)/FLOAT(JMAX-1))*
     1       (X(I,JMAX)-X(I,1))
      Y(I,J)=Y(I,1)+(FLOAT(J-1)/FLOAT(JMAX-1))*
     1       (Y(I,JMAX)-Y(I,1))
30    CONTINUE
      RETURN
      END

C     ******************************************************
C     SOLVES THE EQUATIONS ;SUB: SOLVER
C     ******************************************************

      SUBROUTINE SOLVER(IS,IE,JS,JE,IMAX,JMAX,
     1                  AP,AE,AW,AN,AS,Z,T)
      implicit double precision (a-h,o-z)
      DIMENSION AP(5125,1921),AE(5125,1921),AW(5125,1921),AN(5125,1921),
     1          AS(5125,1921),Z(5125,1921),T(5125,1921),FI(5125),
     2          A(5125),B(5125),C(5125),D(5125)

C     HORIZONTAL SWEEP
C     ----------------
      DO 30 J=JS,JE

      TW=T(1,J)
      TE=T(IMAX,J)

      DO 10 I=IS,IE
      A(I)=AP(I,J)
      B(I)=AE(I,J)
      C(I)=AW(I,J)
      D(I)=AN(I,J)*T(I,J+1)+AS(I,J)*T(I,J-1)+
     1     Z(I,J)*(T(I+1,J+1)-T(I+1,J-1)-
     2     T(I-1,J+1)+T(I-1,J-1))
10    CONTINUE

      CALL TDMA(A,B,C,D,IS,IE,FI,TW,TE)

      DO 20 I=IS,IE
      T(I,J)=FI(I)
20    CONTINUE

30    CONTINUE

C     VERTICAL SWEEP
C     --------------
      DO 60 I=IS,IE

      TN=T(I,JMAX)
      TS=T(I,1)

      DO 40 J=JS,JE
      A(J)=AP(I,J)
      B(J)=AN(I,J)
      C(J)=AS(I,J)
      D(J)=AE(I,J)*T(I+1,J)+AW(I,J)*T(I-1,J)+
     1     Z(I,J)*(T(I+1,J+1)-T(I+1,J-1)-
     2     T(I-1,J+1)+T(I-1,J-1))
40    CONTINUE

      CALL TDMA(A,B,C,D,JS,JE,FI,TS,TN)

      DO 50 J=JS,JE
      T(I,J)=FI(J)
50    CONTINUE

60    CONTINUE

      RETURN
      END

C     *************************************************
C     TRIDIAGONAL MATRIX ALGORITHM
C     *************************************************

      SUBROUTINE TDMA(A,B,C,D,IS,IE,FI,T1,T2)
      implicit double precision (a-h,o-z)
      DIMENSION A(5125),B(5125),C(5125),D(5125),
     +          AA(5125),BB(5125),FI(5125)          

      AA(IS)=B(IS)/A(IS)
      BB(IS)=(C(IS)*T1+D(IS))/A(IS)

      DO 10 I=IS+1,IE
      DR=A(I)-C(I)*AA(I-1)
      AA(I)=B(I)/DR
      BB(I)=(D(I)+(BB(I-1)*C(I)))/DR
10    CONTINUE

      FI(IE)=AA(IE)*T2+BB(IE)

      DO 20 I=IE-1,IS,-1
      FI(I)=AA(I)*FI(I+1)+BB(I)
20    CONTINUE

      RETURN
      END

C     *********************************************************
C     SUBROUTINE:GG3
C     GENERATES GRIDS USING ELLIPTICAL GRID GENERATION SYSTEM
C     *********************************************************

      SUBROUTINE GG3(X,Y,IS,IE,JS,JE,IMAX,JMAX)
      implicit double precision (a-h,o-z)
      DIMENSION X(5125,1921),Y(5125,1921),APX(5125,1921),
     1          AWX(5125,1921),ANX(5125,1921),ASX(5125,1921),
     2          XOLD(5125,1921),YOLD(5125,1921),P(5125),
     3          AEX(5125,1921),ZX(5125,1921),Q(5125)

      double precision JACO,JACO2

      PRINT*,'GIVE THE VALUES OF THE GRID CONTROL PARAMETERS ABCD'
      READ(*,*)A,B,C,D

C     K HAS BEEN ASSIGNED A DUMMY VALUE
      K=1
      L=1

      DO 2 I=IS,IE
      DO 2 J=JS,JE
      IF (I.EQ.K) THEN
         P(I)=0.0
      ELSE
         P(I)=(-A)*FLOAT((I-K)/ABS(I-K))*EXP((-B)*FLOAT(ABS(I-K)))
      ENDIF
      IF (J.EQ.L) THEN
         Q(J)=0.0
      ELSE
         Q(J)=(-C)*FLOAT((J-L)/ABS(J-L))*EXP((-D)*FLOAT(ABS(J-L)))
      ENDIF
2     CONTINUE

5     DO 7 J=1,JMAX
      X(IMAX,J)=X(2,J)
      Y(IMAX,J)=Y(2,J)
      X(1,J)=X(IMAX-1,J)
      Y(1,J)=Y(IMAX-1,J)
7     CONTINUE

      DO 10 I=IS,IE
      DO 10 J=JS,JE

C     CALCULATES DERIVATIVES

      XZ=(X(I+1,J)-X(I-1,J))/2.
      XE=(X(I,J+1)-X(I,J-1))/2.
      YZ=(Y(I+1,J)-Y(I-1,J))/2.
      YE=(Y(I,J+1)-Y(I,J-1))/2.

      print*, X(I+1,J), X(I-1,J)
      stop

      ALPHA= XE*XE+YE*YE
      GAMA = XZ*XZ+YZ*YZ
      BEETA= XZ*XE+YZ*YE
      JACO=XZ*YE-XE*YZ
      JACO2=JACO*JACO

      APX(I,J)=(2./JACO2)*(ALPHA+GAMA)
      AEX(I,J)=(P(I)/2.)+(ALPHA/JACO2)
      AWX(I,J)=(-P(I)/2.)+(ALPHA/JACO2)
      ANX(I,J)=(Q(J)/2.)+(GAMA/JACO2)
      ASX(I,J)=(-Q(J)/2.)+(GAMA/JACO2)
      ZX(I,J) = (-BEETA)/(2.*JACO2)
10    CONTINUE

C     STORES X,Y AS XOLD, YOLD

      DO 20 I=IS,IE
      DO 20 J=JS,JE
      XOLD(I,J)=X(I,J)
      YOLD(I,J)=Y(I,J)
20    CONTINUE

C     SOLVES X,Y VALUES
      CALL SOLVER(IS,IE,JS,JE,IMAX,JMAX,APX,AEX,AWX,ANX,ASX,ZX,X)
      CALL SOLVER(IS,IE,JS,JE,IMAX,JMAX,APX,AEX,AWX,ANX,ASX,ZX,Y)

C     FINDS EUCLEADIAN NORM

      DIFFX2=0.0
      DIFFY2=0.0
      DO 30 I=IS,IE
      DO 30 J=JS,JE
      DIFFX=X(I,J)-XOLD(I,J)
      DIFFY=Y(I,J)-YOLD(I,J)
      DIFFX2=DIFFX**2+DIFFX2
      DIFFY2=DIFFY**2+DIFFY2
30    CONTINUE
      DIFF2=DIFFX2
      IF (DIFFX2.LE.DIFFY2) DIFF2=DIFFY2
      RMSE=SQRT(DIFF2)
 
      PRINT*,RMSE

      IF (RMSE.LT.0.0001) GOTO 40
      GOTO 5

40    RETURN
      END

