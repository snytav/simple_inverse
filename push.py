import torch
import numpy as np
#/*converts physical position to logical coordinate*/
def XtoL(x,x0,dh):
    lc = torch.divide(x - x0,dh)
    return lc

def gather(lc,data):
    lc_i = lc.int()
    i = lc_i[0].item()
    j = lc_i[1].item()
    k = lc_i[2].item()
    d = lc - lc.int()
    di = d[0].item()
    dj = d[1].item()
    dk = d[2].item()
    val = (data[i][j][k] * (1 - di) * (1 - dj) * (1 - dk) +
      data[i + 1][j][k] * (di) * (1 - dj) * (1 - dk) +
      data[i + 1][j + 1][k] * (di) * (dj) * (1 - dk) +
      data[i][j + 1][k] * (1 - di) * (dj) * (1 - dk) +
      data[i][j][k + 1] * (1 - di) * (1 - dj) * (dk) +
      data[i + 1][j][k + 1] * (di) * (1 - dj) * (dk) +
      data[i + 1][j + 1][k + 1] * (di) * (dj) * (dk) +
      data[i][j + 1][k + 1] * (1 - di) * (dj) * (dk))

    return val

def get_field(xx,x0,dh,F):
    el = torch.zeros_like(xx)
    bl = torch.zeros_like(xx)
    for i,x in enumerate(xx):
        Ex = F[0, : ,:, :]
        Ey = F[1, :, :, :]
        Ez = F[2, :, :, :]
        Bx = F[3, : ,:, :]
        By = F[4, :, :, :]
        Bz = F[5, :, :, :]
        lc = XtoL(x,x0,dh)
        ex = gather(lc,Ex)
        ey = gather(lc,Ey)
        ez = gather(lc,Ez)
        bx = gather(lc,Bx)
        by = gather(lc,By)
        bz = gather(lc,Bz)
        el[i, 0] = ex
        el[i, 1] = ey
        el[i, 2] = ez
        bl[i, 0] = bx
        bl[i, 1] = by
        bl[i, 2] = bz



    return el,bl


def push(x,v,qm,E,B,dt):
    v_minus = v + qm * E * dt / 2
    t = qm * B * dt / 2
 #   t = t.reshape(1, v.shape[1])
   # t = np.repeat(t, v.shape[0], axis=0)
    t = t.double()
    v_prime = v_minus + torch.linalg.cross(v_minus, t)
    cp = torch.sum(torch.multiply(t, t), axis=1)
    cp1 = torch.add(torch.ones_like(cp),cp)
    cp2 = cp1.reshape(1, cp1.shape[0])
    cp3 = cp2.repeat(t.shape[1],1)
    cp3 = cp3.T
    s = 2.0 * torch.divide(t,cp3)

    # s = s.repeat(v_prime.shape[0],1)
    v_plus = v_minus + torch.linalg.cross(v_prime, s)
    v = v_plus + qm * E * dt / 2




    return v

def multi_step_push(NT,pos,x0,dh,vel,qm,dt,F):
    history = torch.zeros((pos.shape[0], NT + 1, 3))
    for n in range(NT + 1):
        E,B = get_field(pos,x0,dh,F)
        vel = push(pos, vel, qm, E, B, dt)
        # X[:,n,:] = pos
        # V[:,n,:] = vel
        pos = torch.add(pos,vel*dt)
        qq = 0
        history[:, n, :] = pos[:, :].clone()
    return history

if __name__ == '__main__':
        x0 = torch.zeros(3)
        x  = torch.ones(3) * 0.05
        dh = torch.ones(3)*0.1
        lc = XtoL(x,x0,dh)
        gather(lc,torch.ones(3,3,3))