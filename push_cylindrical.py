import numpy as np
from monkey_boris import push_Boris
# from cyl_plot import  draw_particles_3D
import matplotlib.pyplot as plt

def get_theta(x,y):
    th = np.arctan2(y, x)
    if th < 0.0:
        th += 2 * np.pi
    if th > 2 * np.pi:
        th -= 2 * np.pi
    return th

def cart2pol(x, y):
    theta = get_theta(x,y)
    r     = np.hypot(x, y)
    return r,theta

def pol2cart( r,theta):
    x = r * np.cos(theta)
    y = r * np.sin(theta)
    return x, y

def vector_cart2pol(vx,vy,theta):
    vr  = np.cos(theta)*vx+ np.sin(theta)*vy
    vth = -np.sin(theta) * vx + np.cos(theta) * vy
    return vr,vth


def vector_pol2cart(vr, vth, theta):
    vx  = np.cos(theta) * vr - np.sin(theta) * vth
    vy =  np.sin(theta) * vr + np.cos(theta) * vth
    return vx,vy

def XtoL(x,x0,dh):
    lc = np.divide(x - x0,dh)

    return lc

def interpolate(data,i,j,k,di,dj,f1,f2):
    out_of_bounds = (i < 0 or i >= data.shape[0]-1 or
                     j < 0 or j >= data.shape[1]-1 or
                     k < 0 or k >= data.shape[2] - 1)
    if out_of_bounds:
        return 0.0

    val = (data[i][j][k] * (1 - di) * (1 - dj) * f2 +
           data[i + 1][j][k] * (di) * (1 - dj) * f2 +
           data[i + 1][j + 1][k] * (di) * (dj) * f1 +
           data[i][j + 1][k] * (1 - di) * (dj) * f1)
    return val


def get_polar_field_2D(lc,r0,dr,data):
    i = int(lc[0])
    di = lc[0] - i
    k = int(lc[2])
    dk = lc[2] - k

    j = int(lc[1])

    dj = lc[1] - j
    # compute correction factors
    rj = r0 + j * dr
    f1 = (rj + 0.5 * dj * dr) / (rj + 0.5 * dr)
    f2 = (rj + 0.5 * (dj + 1) * dr) / (rj + 0.5 * dr)

    #gather electric field onto particle position
    val = interpolate(data,i,j,k,di,dj,f1,f2)
    val1 = interpolate(data,i,j,k+1,di,dj,f1,f2)

    t = val*(1-dk)+ val1*dk
    return t


def check_bounds(x,v,rmax,zmax,i):
    r, th = cart2pol(x[0], x[1])
    z = x[2]
    return  not  ( r < 0 or r > rmax or z < 0 or z > zmax)

def capture(x,v,rmax,zmax):
    X = np.concatenate((x, v), axis=0)
    r = np.hypot(x[0,:], x[1,:])

    X = X[:,r < rmax]
    X = X[:,X[2,:] < zmax]
    X = X[:,X[2,:] > 0]
    v = X[3:,:]
    x = X[:3,:]

    return x,v

from polar_trace import draw_polar_trace

def cyl2cart_allcoordinates(rr,theta,zz):
    xx = np.zeros((rr.shape[0],3))
    for i in range(rr.shape[0]):
        r   = rr[i]
        th  = theta[i]
        z   = zz[i]
        x,y = pol2cart(r,th)
        xx[i,0] = x
        xx[i,1] = y
        xx[i,2] = z
    return xx


def cyl2cart_coordinates_fields(rr,theta,zz,vrr,vtheta,vzz,Er_spiral,Etheta_spiral,Ez_spiral,
             Br_spiral,Btheta_spiral,Bz_spiral,
             r_linspace,theta_linspace,z_linspace,dt,qm):

    dr     = r_linspace[1] - r_linspace[0]
    dtheta = theta_linspace[1] - theta_linspace[0]
    dz     = z_linspace[1] - z_linspace[0]
    dh = np.array([dr,dtheta,dz])
    x0 = np.zeros(3)

    rmax = np.max(r_linspace)
    zmax = np.max(z_linspace)

    xx = []
    vv = []
    EE = []
    BB = []
    for i in range(rr.shape[0]):
        r   = rr[i]
        th  = theta[i]
        z   = zz[i]
        vr  = vrr[i]
        vth = vtheta[i]
        vz  = vzz[i]
        xcyl = np.array([r,th,z])
        lc   = XtoL(xcyl,x0,dh)

        er = get_polar_field_2D(lc, r, dr, Er_spiral)
        et = get_polar_field_2D(lc, r, dr, Etheta_spiral)
        ez = get_polar_field_2D(lc, r, dr, Ez_spiral)
        br = get_polar_field_2D(lc, r, dr, Br_spiral)
        bt = get_polar_field_2D(lc, r, dr, Btheta_spiral)
        bz = get_polar_field_2D(lc, r, dr, Bz_spiral)

        x,y = pol2cart(r,th)
        vx,vy = vector_pol2cart(vr, vth,th)
        Ex,Ey = vector_pol2cart(er,et,th)
        Bx,By = vector_pol2cart(br,bt,th)

        x = np.array([x,y,z])
        v = np.array([vx, vy, vz])
        E = np.array([Ex, Ey, ez])
        B = np.array([Bx, By, bz])
        xx.append(x)
        vv.append(v)
        EE.append(E)
        BB.append(B)

    xx = np.array(xx)
    vv = np.array(vv)
    EE = np.array(EE)
    BB = np.array(BB)

    return xx,vv,EE,BB

def cart2cyl_coordinates_velocities(xx1,vv1):

    rr     = np.zeros(xx1.shape[0])
    theta  = np.zeros(xx1.shape[0])
    vrr    = np.zeros(xx1.shape[0])
    vtheta = np.zeros(xx1.shape[0])
    zz     = np.zeros(xx1.shape[0])
    vzz    = np.zeros(xx1.shape[0])

    for i in range(xx1.shape[0]):
        x1 = xx1[i]
        v1 = vv1[i]
        r1, th1 = cart2pol(x1[0], x1[1])
        vr1, vth1 = vector_cart2pol(v1[0], v1[1], th1)
        rr[i] = r1
        theta[i] = th1
        zz[i] = x1[2]
        vrr[i] = vr1
        vtheta[i] = vth1
        vzz[i] = v1[2]

    return rr, theta, zz, vrr, vtheta, vzz


def check_orbit(x,x1,v,E,r_linspace,qm):
    dr = r_linspace[1] - r_linspace[0]
    r,th = cart2pol(x[:,0],x[:,1])
    r1, th1 = cart2pol(x[:, 0], x[:, 1])
    vr1, vth1 = vector_cart2pol(v[:,0], v[:,1], th)
    er1, eth1 = vector_cart2pol(E[:, 0], E[:, 1], th)
    ac = vth1**2/r
    print('radius ',r,r1,np.abs(r-r1),'radial force ',qm*er1,'centripetal acc ',ac,np.abs(ac+qm*er1))
    qq = 0




def push_cyl(x,v,Er_spiral,Etheta_spiral,Ez_spiral,
             Br_spiral,Btheta_spiral,Bz_spiral,
             r_linspace,theta_linspace,z_linspace,dt,qm):

    rmax = np.max(r_linspace)
    zmax = np.max(z_linspace)



    dr = r_linspace[1] - r_linspace[0]
    dz = z_linspace[1] - z_linspace[0]
    # zmax = np.max(z_linspace)

    rr, theta, zz, vrr, vtheta, vzz = cart2cyl_coordinates_velocities(x, v)

    _, _, E, B =     cyl2cart_coordinates_fields(rr, theta, zz,
                                 vrr, vtheta, vzz,
                                 Er_spiral, Etheta_spiral, Ez_spiral,
                                 Br_spiral, Btheta_spiral, Bz_spiral,
                                 r_linspace, theta_linspace, z_linspace, dt, qm)
    r0, th0 = cart2pol(x[:, 0], x[:, 1])
    x1, v1 = push_Boris(x, v, qm, E, B, -dt*0.5)

    check_orbit(x, x1, v, E, r_linspace,qm)

    x2,v2 = capture(x1.T,v.T,rmax-dr, zmax-dz)
    # x1 = cyl2cart_allcoordinates(rr,theta,zz)
    qq = 0

    return x2,v2
