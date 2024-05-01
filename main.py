import torch
import numpy as np

import os

os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'

#dimensions
N = 40
# setting field as 4D cartesian tensor (Ex,Ey,Ez,Bx,By,Bz)
F = torch.zeros(6,N,N,N)

# for magnetic field Bz 0.01 Tl
B0 = 0.01
F[5,:,:,:] = B0
F.requires_grad = True

# particle initial settings
q = -1.602176565e-19
m = 9.10938215e-31  # *1836     # Hydrogen ion
c = 3e8  # m/sec- velocity of light
R = 0.06  # characteristic radius
qm = q / m
vt = 1e3
rL = m * vt / (np.abs(q) * np.linalg.norm(B0))
wL = (np.abs(q) * np.linalg.norm(B0)) / (m)  # =eB/mc   Kotelkikov,Cebotaev, p.47
L = 4*rL
dh = L/N*torch.ones(3)
pos = torch.tensor([[rL, 0, 0]])
pos = pos.double()
vel = torch.tensor([[0,  vt, 0]])
vel = vel.double()
#vel = np.array([[0, vt, 0], [0, 3 * vt, 0], [0, 1.5 * vt, 0]])
dt = 5e-13
x0 = torch.zeros(3)
pos.requires_grad = True
vel.requires_grad = True


# pushing with cartesian mesh-based field
from push import multi_step_push
base_history = multi_step_push(10000,pos,x0,dh,vel,qm,dt,F)

# visualization of the track
from cyl_plot import multi_particles_3D
multi_particles_3D(base_history, 'BasicFieldsHistory',['basic'])
qq = 0

#creating the distorted path
d_hist = base_history*2.0
lf = torch.max(torch.abs(torch.subtract(base_history,d_hist)))
lf.backward(retain_graph=True)
dF = F.grad
h = torch.cat((base_history,d_hist),0)
multi_particles_3D(h, 'History_mult_2_N_40',['basic','r*2.0'])
qq = 0
optimizer = torch.optim.Adam([F],lr=0.01)
optimizer.zero_grad()
lf = torch.max(torch.abs(torch.subtract(base_history,d_hist)))
lf.backward(retain_graph=True)
optimizer.step()
d_hist = multi_step_push(10000,pos,x0,dh,vel,qm,dt,F)
lf = torch.max(torch.abs(torch.subtract(base_history,d_hist)))
print(lf.item(),torch.max(F[5,:,:,:]).item())

h = torch.cat((h,d_hist),0)
multi_particles_3D(h, 'History_mult_2',['basic','r*2.0','shift'])
qq = 0