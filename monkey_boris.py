import numpy as np # qq
import torch


def push_Boris(x,v,qm,E,B,dt):
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
