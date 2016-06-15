#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
This Python script was used to obtain the results presented in
'Timoshenko Beam Element with Anisotropic Cross-Sectional Properties'
at the ECCOMAS Congress 2016.

Please refere to the following papers if you use the relevant part of the script:

Alexander R. Stäblein, and Morten H. Hansen.
"Timoshenko beam element with anisotropic cross-sectional properties."
ECCOMAS Congress 2016, VII European Congress on Computational Methods in Applied Sciences and Engineering.
Crete Island, Greece, 5-10 June 2016

Taeseong Kim, Anders M. Hansen, and Kim Branner.
"Development of an anisotropic beam finite element for composite wind turbine blades in multibody system."
Renewable Energy 59 (2013): 172-183.

Jean-Marc Battini, and Costin Pacoste.
"Co-rotational beam elements with warping effects in instability problems."
Computer Methods in Applied Mechanics and Engineering 191.17 (2002): 1755-1789.
"""

import numpy as np

np.set_printoptions(precision=6)
np.set_printoptions(suppress=True)
np.set_printoptions(linewidth=500)

##################################################################
# General Functions
##################################################################

def cs_stiff(cs):
    """ Returns a 6x6 cross-sectional stiffness matrix for a list
    of beam properties. """
    cs = cs_prop[cs]
    K11=cs[ 0]; K12=cs[ 1]; K13=cs[ 2]; K14=cs[ 3]; K15=cs[ 4]; K16=cs[ 5]
    K22=cs[ 6]; K23=cs[ 7]; K24=cs[ 8]; K25=cs[ 9]; K26=cs[10]
    K33=cs[11]; K34=cs[12]; K35=cs[13]; K36=cs[14]
    K44=cs[15]; K45=cs[16]; K46=cs[17]
    K55=cs[18]; K56=cs[19]
    K66=cs[20]
    
    Kcs = np.array([
    [K11,K12,K13,K14,K15,K16],
    [K12,K22,K23,K24,K25,K26],
    [K13,K23,K33,K34,K35,K36],
    [K14,K24,K34,K44,K45,K46],
    [K15,K25,K35,K45,K55,K56],
    [K16,K26,K36,K46,K56,K66]])
    return Kcs

def gauss_pts(ngp,a,b):
    """ Returns the sample points and weights for Gauss integration over
    the interval [a,b]. """
    xi, wi = np.polynomial.legendre.leggauss(ngp)
    # change of integration interval
    xi = [(b-a)/2.*x+(a+b)/2. for x in xi]
    wi = [w*(b-a)/2. for w in wi]
    return xi, wi


##################################################################
# Timoshenko Beam Element by Stäblein and Hansen
##################################################################

def shape_fcn(csK,L,z):
    """ Returns the strain displacement matrix B and shape function matrix N. """
    csK = cs_prop[csK]
    K12=csK[ 1]; K13=csK[ 2]
    K22=csK[ 6]; K23=csK[ 7]; K24=csK[ 8]; K25=csK[ 9]; K26=csK[10]
    K33=csK[11]; K34=csK[12]; K35=csK[13]; K36=csK[14]
    K55=csK[18]; K56=csK[19]
    K66=csK[20]

    A  = np.array([[z,1,0,0,0,0,0,0,0,0,0,0,0,0],
                   [0,0,z**3,z**2,z,1,0,0,0,0,0,0,0,0],
                   [0,0,0,0,0,0,z**3,z**2,z,1,0,0,0,0],
                   [0,0,0,0,0,0,0,0,0,0,0,0,z,1],
                   [0,0,0,0,0,0,-3*z**2,-2*z,-1,0,1,0,0,0],
                   [0,0,3*z**2,2*z,1,0,0,0,0,0,0,-1,0,0]])

    dA = np.array([[1,0,0,0,0,0,0,0,0,0,0,0,0,0],
                   [0,0,3*z**2,2*z,1,0,0,0,0,0,0,0,0,0],
                   [0,0,0,0,0,0,3*z**2,2*z,1,0,0,0,0,0],
                   [0,0,0,0,0,0,0,0,0,0,0,0,1,0],
                   [0,0,0,0,0,0,-6*z,-2,0,0,0,0,0,0],
                   [0,0,6*z,2,0,0,0,0,0,0,0,0,0,0]])

    E  = np.array([[-K13,0,-6*K36*z+6*K56,-2*K36,0,0,6*K35*z-6*K55,2*K35,0,0,-K33,-K23,-K34,0],
                   [K12,0,6*K26*z+6*K66,2*K26,0,0,-6*K25*z-6*K56,-2*K25,0,0,K23,K22,K24,0],
                   [0,1,0,0,0,0,0,0,0,0,0,0,0,0],
                   [0,0,0,0,0,1,0,0,0,0,0,0,0,0],
                   [0,0,0,0,0,0,0,0,0,1,0,0,0,0],
                   [0,0,0,0,0,0,0,0,0,0,0,0,0,1],
                   [0,0,0,0,0,0,0,0,-1,0,1,0,0,0],
                   [0,0,0,0,1,0,0,0,0,0,0,-1,0,0],
                   [L,1,0,0,0,0,0,0,0,0,0,0,0,0],
                   [0,0,L**3,L**2,L,1,0,0,0,0,0,0,0,0],
                   [0,0,0,0,0,0,L**3,L**2,L,1,0,0,0,0],
                   [0,0,0,0,0,0,0,0,0,0,0,0,L,1],
                   [0,0,0,0,0,0,-3*L**2,-2*L,-1,0,1,0,0,0],
                   [0,0,3*L**2,2*L,1,0,0,0,0,0,0,-1,0,0]])
    T = np.zeros((14,12))
    T[2:,:] = np.identity(12)

    T_N = np.zeros((6,6))
    T_N[1,5] = -1
    T_N[2,4] =  1

    N  = np.dot(np.dot( A,np.linalg.inv(E)),T)
    dN = np.dot(np.dot(dA,np.linalg.inv(E)),T)
    B  = dN + np.dot(T_N,N)

    return B, N

def ele_stiff(cs,L):
    """ Returns the Mass and Stiffness Matrix of a beam element. """
    Me  = np.zeros((12,12))
    Ke  = np.zeros((12,12))
    Mcs = cs_stiff(cs+'_M')
    Kcs = cs_stiff(cs+'_K')
    ngp = 4
    xi, wi = gauss_pts(ngp,0,L)
    for i in range(ngp): # Gauss integration
        B, N = shape_fcn(cs+'_K',L,xi[i])
        Me += wi[i]*np.dot(np.dot(N.T,Mcs),N)
        Ke += wi[i]*np.dot(np.dot(B.T,Kcs),B)

    return Me, Ke

##################################################################
# Beam Element by Kim et al.
##################################################################

def tkim_B(x, n):
    """ Returns the strain displacement matrix B. """
    N  = np.zeros((6,n*6))
    dN = np.zeros((6,n*6))
    N[:,:6] = np.identity(6)
    for i in np.arange(1,n):
        N[:,i*6:(i+1)*6]  = np.identity(6)*x**i
        dN[:,i*6:(i+1)*6] = np.identity(6)*i*x**(i-1)
    B = np.zeros((6,6))
    B[1,5]=-1.
    B[2,4]= 1.
    B = np.dot(B,N) + dN
    return B

def tkim_stiff(cs,L):
    """ Returns the Stiffness Matrix of a beam element. """
    n = 6
    D  = np.zeros((6*n,6*n))
    S = cs_stiff(cs+'_K')
    xi, wi = gauss_pts(n,0,L)
    for i in range(n): # Gauss integration
        B = tkim_B(xi[i], n)
        D += wi[i]*np.dot(np.dot(B.T,S),B)
    
    N1 = np.zeros((12,12))
    N1[:6,:6]=N1[6:,:6]=np.identity(6)
    N1[6:,6:]=np.identity(6)*L
    N2 = np.zeros((12,6*(n-2)))
    for i in range(n-2):
        N2[6:,i*6:(i+1)*6] = np.identity(6)*L**(i+2)
    A1 = np.zeros((n*6,12))
    A1[:12,:] = np.identity(12)
    A2 = np.zeros((6*n,(n-2)*6))
    A2[12:,:] = np.identity((n-2)*6)

    Y1 = np.dot(A1,np.linalg.inv(N1))
    Y2 = A2-np.dot(Y1,N2)
    P  =  np.dot(np.dot(Y2.T,D),Y1)
    Q  = -np.dot(np.dot(Y2.T,D),Y2)
    N = Y1+np.dot(np.dot(Y2,np.linalg.inv(Q)),P)
    K = np.dot(np.dot(N.T,D),N)    
    return K

 
##################################################################
# Finite Rotation Functions
##################################################################

def skw_mat(v):
    """ Returns the skew-symmetric matrix of a vector. """
    S = np.zeros((3,3))
    S[1,0] =  v[2]
    S[2,0] = -v[1]
    S[0,1] = -v[2]
    S[2,1] =  v[0]
    S[0,2] =  v[1]
    S[1,2] = -v[0]
    return S
    
def rot_mat(v):
    """ Returns a rotation matrix from a rotation vector. """
    phi = np.linalg.norm(v)
    if phi>1e-9:
        n = v/phi
    else:
        n = np.zeros(3)
    S = skw_mat(n)
    R = np.identity(3)+np.sin(phi)*S+(1-np.cos(phi))*np.dot(S,S)
#        R = np.identity(3)+np.sin(phi)*self.skw_mat(n)+(1-np.cos(phi))*np.dot(self.skw_mat(n),self.skw_mat(n))
    return R

def rot_vec(R):
    """ Returns a rotation vector from a rotation-matrix. """
    a = (np.trace(R)-1.)/2.

    if a>1.0: # cos is only defined between [-1,1]
        a = 1.0
    elif a<-1:
        a = -1.0
    phi = np.arccos(a)

    if phi>1e-9:
        r = phi/(2.*np.sin(phi))*np.array([R[2,1]-R[1,2],R[0,2]-R[2,0],R[1,0]-R[0,1]])
    else:
        r = 0.5*np.array([R[2,1]-R[1,2],R[0,2]-R[2,0],R[1,0]-R[0,1]])
    return r
  
def Ts_matrix(v):
    """ Returns the variations of a rotation vector. """
    phi = np.linalg.norm(v)
    if phi>1e-9:
        n = v/phi
        a = (phi/2.)/np.tan(phi/2.)
    else:
        n = np.zeros(3)
        a = 1.
    Ts = a*np.identity(3)+(1.-a)*np.outer(n,n)-0.5*skw_mat(v)
    return Ts

##################################################################
# Co-rotational Formulation by Battini and Pacoste
##################################################################

def getTangentStiffness(cs, edof, frml):
    """ Returns the stiffness and force vector of an element in global coordinates. """
    l0, l, R_0, R_1g, R_2g, q, R_r = get_rotmats(edof)
    f_l, d_l, K_l = get_fl(cs, l0, l, R_0, R_1g, R_2g, R_r, frml)
    B, r, E, nu, G, P, B_a, B_l = get_B(l, q, R_r, d_l)        
    K_g = get_Kg(l, R_r, f_l, d_l, r, E, nu, G, P, B_a, B_l)
    
    Fint = np.dot(B.T,f_l)
    Kcr  = np.dot(np.dot(B.T,K_l),B)+K_g 
    return Kcr, Fint
    
def get_rotmats(edof):
    """ Returns various transformation matrices of the co-rotational formulation. """
    # Lengths
    l0 = np.linalg.norm(edof[0,6:9]-edof[0,0:3])
    l  = np.linalg.norm(edof[1,6:9]-edof[1,0:3])

    # Reference Frame
    R_0 = np.zeros((3,3))
    R_0[:,0] = (edof[0,6:9]-edof[0,0:3])/l0
    if R_0[2,0]<.9: # check if element is nearly vertical
        R_0[:,2] = np.array([0.,0.,1.])
    else:
        R_0[:,2] = np.array([-1.,0.,0.])
    R_0[:,1] = np.cross(R_0[:,2],R_0[:,0])/np.linalg.norm(np.cross(R_0[:,2],R_0[:,0]))
    R_0[:,2] = np.cross(R_0[:,0],R_0[:,1])
   
    # q Vectors
    R_1g  = rot_mat(edof[1,3:6])
    R_2g  = rot_mat(edof[1,9:12])
    
    q = np.zeros((3,3))
    q[1] = np.dot(R_1g,R_0)[:,1]
    q[2] = np.dot(R_2g,R_0)[:,1]
    q[0] = 0.5*(q[1]+q[2])
    
    # Current Frame
    R_r = np.zeros((3,3))
    R_r[:,0] = (edof[1,6:9]-edof[1,0:3])/l
    R_r[:,2] = np.cross(R_r[:,0],q[0])/np.linalg.norm(np.cross(R_r[:,0],q[0]))
    R_r[:,1] = np.cross(R_r[:,2],R_r[:,0])

    return l0, l, R_0, R_1g, R_2g, q, R_r

def get_fl(cs, l0, l, R_0, R_1g, R_2g, R_r, frml):
    """ Returns force, displacements and stiffness in element coordinate system. """
    # Local Displacements
    d_l = np.zeros(7)
    d_l[0] = l - l0
    d_l[1:4] = rot_vec(np.dot(np.dot(R_r.T,R_1g),R_0))
    d_l[4:7] = rot_vec(np.dot(np.dot(R_r.T,R_2g),R_0))
    
    # Local Stiffness Matrix
    if frml=='alsta':
        M_l, K_l = ele_stiff(cs,l)
    elif frml=='tkim':
        K_l = tkim_stiff(cs,l)        
    T = np.zeros((12,7))
    T[3:6,1:4] = T[9:12,4:7]=np.identity(3)
    T[6,0] = 1.
    K_l = np.dot(np.dot(T.T,K_l),T)

    # Local Force
    f_l = np.dot(K_l, d_l)

    return f_l, d_l, K_l

def get_B(l, q, R_r, d_l):
    """ Returns the strain-displacement matrix of the beam element. """
    # r matrix
    r = np.zeros((1,12))
    r[0,0:3] = -R_r[:,0]
    r[0,6:9] =  R_r[:,0]

    # E matrix
    E = np.zeros((12,12))
    for i in range(4):
        E[i*3:i*3+3,i*3:i*3+3] = R_r

    # G matrix
    for i in range(3):
        q[i] = np.dot(R_r.T,q[i])
    nu   = q[0,0]/q[0,1]
    nu11 = q[1,0]/q[0,1]
    nu12 = q[1,1]/q[0,1]
    nu21 = q[2,0]/q[0,1]
    nu22 = q[2,1]/q[0,1]
    G = np.array([[0.,0.,nu/l,nu12/2.,-nu11/2.,0,0,0,-nu/l,nu22/2.,-nu21/2.,0.],\
               [0.,0.,1./l,0.,0.,0.,0.,0.,-1./l,0.,0.,0.],\
               [0.,-1./l,0.,0.,0.,0.,0.,1./l,0.,0.,0.,0.]])

    # P matrix
    P = np.zeros((6,12))
    P[0:3,3:6]  = np.identity(3)
    P[3:6,9:12] = np.identity(3)
    P[0:3,:] = P[0:3,:]-G
    P[3:6,:] = P[3:6,:]-G
    
    # Ba matrix
    B_a = np.zeros((7,12))
    B_a[0,:] = r
    B_a[1:7,:] = np.dot(P,E.T)

    # Bl matrix
    B_l = np.zeros((7,7))
    B_l[0,0] = 1
    B_l[1:4,1:4] = Ts_matrix(d_l[1:4])
    B_l[4:7,4:7] = Ts_matrix(d_l[4:7])

    # B matrix
    B = np.dot(B_l,B_a)
    
    return B, r, E, nu, G, P, B_a, B_l

def get_Kg(l, R_r, f_l, d_l, r, E, nu, G, P, B_a, B_l):
    """ Returns the geometric stiffness of the element. """
    # Kh matrix
    def Kh(phi,v):
        a = np.linalg.norm(phi)
        if a>1e-9:
            nu = (2.*np.sin(a)-a*(1.+np.cos(a)))/(2.*a**2.*np.sin(a))
            mu = (a*(a+np.sin(a))-8.*np.sin(a/2.)**2.)/(4.*a**4.*np.sin(a/2.)**2.)
        else:
            nu = 1./12.   # limit does not really matter as terms are zero
            mu = 1./360.
        b = np.outer(phi,v)
        c = np.outer(v,phi)
        d = np.inner(phi,v)
        e = skw_mat(phi)
        f = skw_mat(v)
        Kh = nu*(b-2*c+d*np.identity(3))+mu*np.dot(np.dot(e,e),c)-0.5*f
        Kh = np.dot(Kh,Ts_matrix(phi))
        return Kh

    K_h = np.zeros((7,7))
    K_h[1:4,1:4] = Kh(d_l[1:4],f_l[1:4])
    K_h[4:7,4:7] = Kh(d_l[4:7],f_l[4:7])

    # D matrix
    d = 1./l * (np.identity(3)-np.outer(R_r[:,0],R_r[:,0]))
    D = np.zeros((12,12))
    D[0:3,0:3] = D[6:9,6:9] =  d
    D[0:3,6:9] = D[6:9,0:3] = -d

    # f_a vector
    f_a = np.dot(B_l.T, f_l)
    
    # Q matrix
    Ptm = np.dot(P.T,f_a[1:])
    Q = np.zeros((12,3))
    for i in range(4):
        Q[i*3:i*3+3,:] = skw_mat(Ptm[i*3:i*3+3])

    # a matrix
    a = np.zeros((3,1))
    a[1] = nu/l*(f_a[1]+f_a[4])-1./l*(f_a[2]+f_a[5])
    a[2] = 1./l*(f_a[3]+f_a[6])
    
    # Kg matrix
    K_g = np.dot(np.dot(B_a.T,K_h),B_a)+D*f_a[0]-np.dot(np.dot(np.dot(E,Q),G),E.T)+np.dot(np.dot(np.dot(E,G.T),a),r)

    return K_g

##################################################################
# Solver
##################################################################
    
def assemble_stiff_lin(dofs, elements, cs_prop):
    """ Returns the linear global stiffness matrix (without co-rotational formulation).
        WARNING: Element orientation is not considered here
        (i.e. only works for a cantilever)! """
    ndof = len(dofs[0])
    M = np.zeros((ndof,ndof))
    K = np.zeros((ndof,ndof))
    for ele in elements:
        d1 = ele[0][0]*6*np.ones(6,dtype='int')+np.arange(6,dtype='int')
        d2 = ele[0][1]*6*np.ones(6,dtype='int')+np.arange(6,dtype='int')
        dof = np.hstack((d1,d2))
        L = np.linalg.norm(dofs[1,d2[:3]]-dofs[1,d1[:3]])
        cs = ele[2]
        Me, Ke = ele_stiff(cs,L)
        M[np.ix_(dof,dof)] += Me     
        K[np.ix_(dof,dof)] += Ke     
    return M, K

def solve_lin(dofs, elements, cs_prop, Fext):
    """ Linear solver. """
    M, K = assemble_stiff_lin(dofs, elements, cs_prop)
    for i in range(6):
        K[i,:]=K[:,i]=0
        K[i,i]=1
    u = np.linalg.solve(K,Fext)
    return u

def assemble_stiff(dofs, elements, cs_prop, frml):
    """ Returns the global stiffness using the co-rotational formulation. """
    ndof = len(dofs[0])
    K = np.zeros((ndof,ndof))
    F = np.zeros(ndof)
    for ele in elements:
        d1 = ele[0][0]*6*np.ones(6,dtype='int')+np.arange(6,dtype='int')
        d2 = ele[0][1]*6*np.ones(6,dtype='int')+np.arange(6,dtype='int')
        dof = np.hstack((d1,d2))
        edof = dofs[:,dof]
        cs = ele[2]
        Ke, Fi = getTangentStiffness(cs,edof, frml)
        K[np.ix_(dof,dof)] += Ke
        F[dof] += Fi
    return K, F

def solve(dofs, elements, cs_prop, Fext, nstep=1, niter=100, eps=1e-3, frml='alsta', v=True):
    """ Newton-Raphson solver. """
    ndof = len(dofs[0])
    for step in range(1,nstep+1):
        Fe = float(step)/nstep*Fext
        if v:
            print 'Load Iteration %2d' %step
        itr = 1
        cont = True
        while cont:
            K, Fi = assemble_stiff(dofs, elements, cs_prop, frml)
            Res = Fi-Fe
            # boundary conditions
            for i in range(6):
                K[i,:]=K[:,i]=0
                K[i,i]=1
                Res[i]=0
            # solve
            u = np.linalg.solve(K,Res)
            # update displacements
            for i in range(ndof/6):
                n=i*6
                dofs[1,n:n+3] -= u[n:n+3]
                r  = dofs[1,n+3:n+6]
                dr = u[n+3:n+6]
                R  = rot_mat(r)
                dR = rot_mat(dr)
                R = np.dot(dR.T,R)
                dofs[1,n+3:n+6] = rot_vec(R)
            if (itr==niter) or (np.linalg.norm(Res, np.inf)<eps*np.linalg.norm(Fe, np.inf)):
                cont = False
            if v:
                print 'Iter.: %3d   Res.: %.3f' %(itr, np.linalg.norm(Res, np.inf))
            itr += 1
    disp = dofs[1,:]-dofs[0,:]
    return disp

##################################################################
# Validation Cases
##################################################################

# Define CS Properties for all cases
cs_prop = {}
cs_prop['Hodges_K'] = ( 
     5.0576E+06, 0.0000E+00, 0.0000E+00, -1.7196E+04, 0.0000E+00, 0.0000E+00,
     7.7444E+05, 0.0000E+00, 0.0000E+00,  8.3270E+03, 0.0000E+00,
     2.9558E+05, 0.0000E+00, 0.0000E+00,  9.0670E+03,
     1.5041E+02, 0.0000E+00, 0.0000E+00,
     2.4577E+02, 0.0000E+00,
     7.4529E+02)
cs_prop['Hodges_M'] = ( 
     0.13098E+00, 0.0000E+00, 0.0000E+00, 0.0000E+00, 0.0000E+00, 0.0000E+00,
     0.13098E+00, 0.0000E+00, 0.0000E+00, 0.0000E+00, 0.0000E+00,
     0.13098E+00, 0.0000E+00, 0.0000E+00, 0.0000E+00,
     2.5824E-05, 0.0000E+00, 0.0000E+00,
     6.5010E-06, 0.0000E+00,
     1.9323E-05)
cs_prop['Wang_K'] = (
     1368.17E+03,  0.000E+00,  0.000E+00, 0.0000E+00, 0.0000E+00, 0.0000E+00,
       88.56E+03,  0.000E+00,  0.000E+00, 0.0000E+00, 0.0000E+00,
       38.78E+03,  0.000E+00,  0.000E+00, 0.0000E+00,
       16.96E+03, 17.610E+03, -0.351E+03,
       59.12E+03, -0.370E+03,
      141.47E+03)
cs_prop['Wang_M'] = ( 
     1.000E+00, 0.000E+00, 0.000E+00, 0.000E+00, 0.000E+00, 0.000E+00,
     1.000E+00, 0.000E+00, 0.000E+00, 0.000E+00, 0.000E+00,
     1.000E+00, 0.000E+00, 0.000E+00, 0.000E+00,
     1.000E+00, 0.000E+00, 0.000E+00,
     1.000E+00, 0.000E+00,
     1.000E+00)
cs_prop['Chandra_K'] = (
    1.1387E+06, 2.9089E+05, 2.3845E-08, -3.0458E+01, 1.2674E+01, -2.6154E-10,
    4.1889E+05, 1.8074E-08, -1.1932E+01, 8.6893E+00, -2.0877E-10,
    3.1219E+05, -8.4468E-11, 4.8853E-11, 1.2302E+01,
    6.2692E+01, -2.1741E+01, -1.8638E-12,
    3.5146E+01, 7.7549E-13,
    8.0594E+01)
cs_prop['Chandra_M'] = ( 
     1.000E+00, 0.000E+00, 0.000E+00, 0.000E+00, 0.000E+00, 0.000E+00,
     1.000E+00, 0.000E+00, 0.000E+00, 0.000E+00, 0.000E+00,
     1.000E+00, 0.000E+00, 0.000E+00, 0.000E+00,
     1.000E+00, 0.000E+00, 0.000E+00,
     1.000E+00, 0.000E+00,
     1.000E+00)
cs_prop['Bathe_K'] = (
     1.0000E+07, 0.0000E+00, 0.0000E+00, 0.0000E+00, 0.0000E+00, 0.0000E+00,
     0.5000E+07, 0.0000E+00, 0.0000E+00, 0.0000E+00, 0.0000E+00,
     0.5000E+07, 0.0000E+00, 0.0000E+00, 0.0000E+00,
     0.7000E+06, 0.0000E+00, 0.0000E+00,
     8.3333E+05, 0.0000E+00,
     8.3333E+05)
cs_prop['Bathe_M'] = ( 
     1.000E+00, 0.000E+00, 0.000E+00, 0.000E+00, 0.000E+00, 0.000E+00,
     1.000E+00, 0.000E+00, 0.000E+00, 0.000E+00, 0.000E+00,
     1.000E+00, 0.000E+00, 0.000E+00, 0.000E+00,
     1.000E+00, 0.000E+00, 0.000E+00,
     1.000E+00, 0.000E+00,
     1.000E+00)

def hodges():
    ''' Results of section '3.1 Eigenfrequencies of a coupled cantilever' '''
    # Create Elements
    nele = 16
    elements=[]
    for i in range(nele):
        dof = (0+i,1+i)
        ori = np.zeros(3)
        elements.append((dof,ori,'Hodges'))
   
    # Create Nodes
    ndof = (nele+1)*6
    dofs=np.zeros((2,ndof))
    for i in range(nele+1):
        dofs[0,i*6+0]=i*2.54/nele
    dofs[1,:]=dofs[0,:]
    
    # Analyse Model
    M, K = assemble_stiff_lin(dofs, elements, cs_prop)
    M = M[6:,6:]; K = K[6:,6:]
    A = np.dot(np.linalg.inv(M),K)
    w, v = np.linalg.eig(A)
    ms = []
    for i in range(len(w)):
        ms.append((w[i], v[:,i]))
    ms.sort(key=lambda m: np.linalg.norm(m[0]))
    
    # Print Results
    print 'Eigenfrequencies of coupled cantilever:'
    modes = ['1 vert. ','1 horiz.','2 vert. ','2 horiz.','3 vert. ','3 horiz.','1 tors.  ', '2 tors.  ']
    nmode = [0,1,2,3,4,5,9,15]
    for i, j in enumerate(nmode):
        print modes[i] + '  %6.2f' %(np.sqrt(np.linalg.norm(ms[j][0]))/(2*np.pi))
    print '---------------'

hodges()

def wang():
    ''' Results of section '3.2 Tip displacements and rotations of a coupled cantilever' '''
    # Create Elements
    nele = 10
    elements=[]
    for i in range(nele):
        dof = (0+i,1+i)
        ori = np.zeros(3)
        elements.append((dof,ori,'Wang'))
    
    # Create Nodes
    ndof = (nele+1)*6
    dofs=np.zeros((2,ndof))
    for i in range(nele+1):
        dofs[0,i*6+0]=i*10./nele
    dofs[1,:]=dofs[0,:]
    
    # Analyse Model
    Fext = np.zeros(ndof)
    Fext[-4] = 150
    u = solve(dofs, elements, cs_prop, Fext, nstep=1, eps=1e-6, frml='alsta')
    prs = np.array(u[-6:])

    # Transform rotations into Wiener-Milenkovic Parameter
    prs[-3:] = 4*np.tan(np.linalg.norm(prs[-3:])/4)*prs[-3:]/np.linalg.norm(prs[-3:])
    print 'Tip displacements of coupled cantilever:'
    print '%8.5f %8.5f %8.5f %8.5f %8.5f %8.5f' %(prs[0],prs[1],prs[2],prs[3],prs[4],prs[5])
    print '---------------'

wang()

def chandra():
    ''' Results of section '3.3 Curvature and twist of a coupled cantilever' '''
    # Create Elements
    nele = 10
    elements=[]
    for i in range(nele):
        dof = (0+i,1+i)
        ori = np.zeros(3)
        elements.append((dof,ori,'Chandra'))
    
    # Create Nodes
    ndof = (nele+1)*6
    dofs=np.zeros((2,ndof))
    for i in range(nele+1):
        dofs[0,i*6+0]=i*30.*0.0254/nele
    dofs[1,:]=dofs[0,:]

    # Analyse Model
    scl =  1 # scaling factor
    Fext = np.zeros(ndof)
    Fext[-4] = 4.448*scl

    u = solve(dofs, elements, cs_prop, Fext, nstep=1, eps=1e-3, frml='alsta')

    print 'Bending Slope at Tip:'
    print -np.rad2deg(u[-2])
    print 'Twist Angle at Tip:'
    print -np.rad2deg(u[-3])
    print '---------------'

chandra()


def prebend():
    ''' Results of section '3.4 Pre-bend cantilever' '''
    # Create Elements
    nele = 8
    elements=[]
    for i in range(nele):
        dof = (0+i,1+i)
        ori = np.zeros(3)
        sct = 'Bathe'
        elements.append((dof,ori,sct))
    
    # Create Nodes
    ndof = (nele+1)*6
    dofs=np.zeros((2,ndof))
    for i in range(nele+1):
        dofs[0,i*6+0]=np.sin(float(i)/nele*np.pi/4)*100
        dofs[0,i*6+1]=(1-np.cos(float(i)/nele*np.pi/4))*100
    dofs[1,:]=dofs[0,:]

    # Analyse Model
    Fext = np.zeros(ndof)
    Fext[-4] = 300
    # Uncoupled
    ucpl = solve(dofs, elements, cs_prop, Fext, nstep=1)[-6:-3]
    # Coupled
    btc = -0.3
    K45 = btc*np.sqrt(0.7000E+06*8.3333E+05)
    cs_prop['Bathe_K'] = ( # square section: a=1.0
         1.0000E+07, 0.0000E+00, 0.0000E+00, 0.0000E+00, 0.0000E+00, 0.0000E+00,
         0.5000E+07, 0.0000E+00, 0.0000E+00, 0.0000E+00, 0.0000E+00,
         0.5000E+07, 0.0000E+00, 0.0000E+00, 0.0000E+00,
         0.7000E+06, K45, 0.0000E+00,
         8.3333E+05, 0.0000E+00,
         8.3333E+05)
    u1 = solve(dofs, elements, cs_prop, Fext, nstep=1,frml='alsta')[-6:-3]
    u2 = solve(dofs, elements, cs_prop, Fext, nstep=1,frml='tkim')[-6:-3]

    print 'Present uncpld :'+str(ucpl)
    print 'Present cpld   :'+str(u1)
    print 'Kim et al. cpld:'+str(u2)

prebend()
