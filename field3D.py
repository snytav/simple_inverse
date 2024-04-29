import numpy as np
from mayavi.mlab import *
import matplotlib.pyplot as plt

def volume_field_plot(s,name):
# u =    np.sin(np.pi*x) * np.cos(np.pi*z)
# v = -2*np.sin(np.pi*y) * np.cos(2*np.pi*z)
# w = np.cos(np.pi*x)*np.sin(np.pi*z) + np.cos(np.pi*y)*np.sin(2*np.pi*z)
    fig = figure()
    v = volume_slice(s, plane_orientation='x_axes', slice_index=int(s.shape[0]/2))
    v = volume_slice(s, plane_orientation='y_axes', slice_index=int(s.shape[1] / 2))
    v = volume_slice(s, plane_orientation='z_axes', slice_index=int(s.shape[2] / 2))
    outline()
    title(name)
    # fig.renwin.isometric_view()
    xlabel('X')
    ylabel('Y')
    zlabel('Z')
    orientation_axes()
    colorbar(v)
    axes()
    savefig(name+'_3D.png')
    plt.show()
    qq = 0

