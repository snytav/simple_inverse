
import matplotlib.pyplot as plt


def draw_polar_trace(r,theta,rmax,name):
    fig, ax = plt.subplots(subplot_kw={'projection': 'polar'})
    ax.plot(theta, r,'o')
    ax.set_rmax(rmax)
    ax.set_rticks([0.5, 1, 1.5, 2])  # Less radial ticks
    ax.set_rlabel_position(-22.5)  # Move radial labels away from plotted line
    ax.grid(True)

    ax.set_title(name, va='bottom')
    plt.show()
