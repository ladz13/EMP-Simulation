import numpy as np
import math
import time
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

maxFreq = 1000
c0 = 299792458
dz = (c0/maxFreq)/10
EROne = 1
UROne = 1
ERTwo = 3
URTwo = 5
dt = dz/(2*c0)
mEyOne = []
mHxOne = []
mEyTwo = []
mHxTwo = []
tau = 0.5/maxFreq
t0 = 6*tau
material_name = 'Cavity'

H3 = H2 = H1 = 0

E3 = E2 = E1 = 0

for i in range(50):
    mEyOne.append(c0*dt/EROne)
    mHxOne.append(c0*dt/UROne)
    mEyTwo.append(c0*dt/ERTwo)
    mHxTwo.append(c0*dt/URTwo)

xvalues = np.linspace(0,100,100)
Timesteps = 1000
XPoints = 100
Ey = np.zeros((Timesteps,XPoints))
Hx = np.zeros((Timesteps,XPoints))
KSrc = 30
EySrc = float(0)
HxSrc = float(0)

def GaussianPulse(t):
    global t0
    global tau
    return np.exp(-((t-t0)/tau)**2)


for T in range(Timesteps):
    HxSrc = - GaussianPulse((T + 0.5)*dt)
    for a in range(50):
        Hx[T][a] = Hx[T-1][a] + mHxOne[a]*(Ey[T-1][a+1] - Ey[T-1][a])/dz
    for a in range(49):
        Hx[T][(a+50)] = Hx[T-1][(a+50)] + mHxTwo[a]*(Ey[T-1][(a+50+1)] - Ey[T-1][(a+50)])/dz
    Hx[T][99] = Hx[T-1][99] + mHxTwo[49]*(E3 - Ey[T-1][99])/dz
    H3 = H2
    H2 = H1
    H1 = Hx[T][0]
    Hx[T][KSrc-1] = Hx[T][KSrc-1] - mHxOne[KSrc]*EySrc/dz
    
    Ey[T][0] = Ey[T-1][0] + mEyOne[0]*(Hx[T][0]-H3)/dz
    for a in range(1,50):
        Ey[T][a] = Ey[T-1][a] + mEyOne[a]*(Hx[T][a] - Hx[T][a-1])/dz
    for a in range(50):
        Ey[T][(a+50)] = Ey[T-1][(a+50)] + mEyTwo[a]*(Hx[T][(a+50)] - Hx[T][(a+50-1)])/dz   
    
    E3 = E2
    E2 = E1
    E1 = Ey[T-1][99]
    EySrc = GaussianPulse(T*dt)
    Ey[T][KSrc] = Ey[T][KSrc] - mEyOne[KSrc]*HxSrc/dz 
    
fig = plt.figure(figsize=(8, 8))
ax = fig.add_axes([0.1, 0.1, 0.8, 0.8])
cols = ["red","blue"]
lineOne, = ax.plot([],[],lw=2, color=cols[0])
lineTwo, = ax.plot([],[],lw=2, color=cols[1])
ax.set_ylim(-1.5,1.5)
ax.set_xlim(0,99)
ax.legend(["Electric Field", "Magnetic Field"])
time_count = ax.text(50,1.25,'Time step '+str(0)+ ' of '+str(Timesteps), ha='center')
plt.gcf().canvas.set_window_title('1D FDTD simulation. Number of cells: '+str(XPoints)+', Time steps: '+str(Timesteps) + ', Material: '+material_name)
fig.suptitle('1D FDTD simulation. Number of cells: '+str(XPoints)+', Time steps: '+str(Timesteps) + ', Material: '+material_name)
rectangle = plt.Rectangle((50, -1.5), 50, 3, fc='c')
plt.gca().add_patch(rectangle)

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
ani.save('1D FDTD With Object.mp4')

plt.show()
