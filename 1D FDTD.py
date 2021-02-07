import numpy as np
import math
import time
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

maxFreq = 1000
c0 = 299792458
dz = (c0/maxFreq)/10
ER = 1
UR = 1
dt = dz/(2*c0)
mEy = []
mHx = []
tau = 0.5/maxFreq
t0 = 6*tau
material_name = 'Cavity'

for i in range(100):
    mEy.append(c0*dt/ER)
    mHx.append(c0*dt/UR)

xvalues = np.linspace(0,100,100)
Timesteps = 1000
ZPoints = 100
Ey = np.zeros((Timesteps,ZPoints))
Hx = np.zeros((Timesteps,ZPoints))

for T in range(Timesteps):
    for a in range(ZPoints-1):
        Hx[T][a] = Hx[T-1][a] + mHx[a]*(Ey[T-1][a+1] - Ey[T-1][a])/dz
        
    for a in range(ZPoints-1):
        Ey[T][a] = Ey[T-1][a] + mEy[a]*(Hx[T][a] - Hx[T][a-1])/dz
    
fig = plt.figure(figsize=(8, 8))
ax = fig.add_axes([0.1, 0.1, 0.8, 0.8])
cols = ["red","blue"]
lineOne, = ax.plot([],[],lw=2, color=cols[0])
lineTwo, = ax.plot([],[],lw=2, color=cols[1])
ax.set_ylim(-1.5,1.5)
ax.set_xlim(0,99)
ax.legend(["Electric Field", "Magnetic Field"])
time_count = ax.text(50,1.25,'Time step '+str(0)+ ' of '+str(Timesteps), ha='center')
plt.gcf().canvas.set_window_title('1D FDTD simulation. Number of cells: '+str(ZPoints)+', Time steps: '+str(Timesteps) + ', Material: '+material_name)
fig.suptitle('1D FDTD simulation. Number of cells: '+str(ZPoints)+', Time steps: '+str(Timesteps) + ', Material: '+material_name)


def animate(T):
    global xvalues
    global Timesteps
    points = []
    points.append(Ey[T])
    points.append(Hx[T])
    time_count.set_text(u'Time step '+str(T)+ ' of '+str(Timesteps))
    lineOne.set_data(xvalues,points[0])
    lineTwo.set_data(xvalues,points[1])
    return lineOne,lineTwo, time_count,

ani = FuncAnimation(fig, animate, frames=Timesteps, interval = 20,blit=True)
ani.save('1D FDTD.mp4')

plt.show()

