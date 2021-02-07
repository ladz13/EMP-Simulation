import numpy as np
import math
import time
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import matplotlib.patches as patches

maxFreq = 5000000000
c0 = 299792458
dz = (c0/maxFreq)/10
EROne = 1
UROne = 1
ERTwo = 3
URTwo = 5
dt = dz/(2*c0)
mEy = []
mHx = []
tau = 0.5/maxFreq
t0 = 6*tau
material_name = 'Cavity'

H3 = H2 = H1 = 0

E3 = E2 = E1 = 0

for i in range(100):
    mEy.append(c0*dt/EROne)
    mHx.append(c0*dt/UROne)

for i in range(50,80):
    mEy[i] = (c0*dt/ERTwo)
    mHx[i] = (c0*dt/URTwo)

xvalues = np.linspace(0,100,100)
Timesteps = 1000
XPoints = 100
Ey = np.zeros((Timesteps,XPoints))
Hx = np.zeros((Timesteps,XPoints))
KSrc = 30
EySrc = float(0)
HxSrc = float(0)

ftwo = 5000000000
NFREQ = 100
FREQ = np.linspace(0,ftwo,NFREQ)
fmax = 0.5/dt
freq = np.linspace(-fmax,fmax,Timesteps)

K = np.zeros((NFREQ),dtype = complex)
EyR = np.zeros((1000,NFREQ),dtype = complex)
EyT = np.zeros((1000,NFREQ),dtype = complex)
SRC = np.zeros((1000,NFREQ),dtype = complex)
Esrc = 0

for i in range(NFREQ):
    K[i] = np.exp(-1j*2*math.pi*dt*FREQ[i])

def GaussianPulse(t):
    global t0
    global tau
    return np.exp(-((t-t0)/tau)**2)

for T in range(Timesteps):
    HxSrc = - GaussianPulse((T + 0.5)*dt)
    for a in range(XPoints-1):
        Hx[T][a] = Hx[T-1][a] + mHx[a]*(Ey[T-1][a+1] - Ey[T-1][a])/dz
    Hx[T][99] = Hx[T-1][99] + mHx[99]*(E3 - Ey[T-1][99])/dz
    H3 = H2
    H2 = H1
    H1 = Hx[T][0]
    if T < 100:
        Hx[T][KSrc-1] = Hx[T][KSrc-1] - mHx[KSrc]*EySrc/dz
        
    Ey[T][0] = Ey[T-1][0] + mEy[0]*(Hx[T][0]-H3)/dz
    for a in range(1,XPoints):
        Ey[T][a] = Ey[T-1][a] + mEy[a]*(Hx[T][a] - Hx[T][a-1])/dz
    E3 = E2
    E2 = E1
    E1 = Ey[T][99]

    EySrc = GaussianPulse(T*dt)
    if True or T < 100:
        Ey[T][KSrc] = Ey[T][KSrc] - mEy[KSrc]*HxSrc/dz

    for nf in range(NFREQ):
        EyR[T][nf] = EyR[T-1][nf] + (K[nf]**T)*Ey[T][0]
        EyT[T][nf] = EyT[T-1][nf] + (K[nf]**T)*Ey[T][99]
        SRC[T][nf] = SRC[T-1][nf] + (K[nf]**T)*EySrc

EyR = EyR*dt
EyT = EyT*dt
SRC = SRC*dt
REF = np.square(np.abs(np.divide(EyR,SRC)))
TRN = np.square(np.abs(np.divide(EyT,SRC)))
CON = np.add(REF,TRN)
SRC = np.abs(SRC)

fig = plt.figure(figsize=(8, 8))

ax1 = fig.add_subplot(212)
cols = ["red","blue"]
lineOne, = ax1.plot([],[],lw=2, color=cols[0])
lineTwo, = ax1.plot([],[],lw=2, color=cols[1])
ax1.set_ylim(-1.5,1.5)
ax1.set_xlim(0,99)
ax1.legend(["Electric Field", "Magnetic Field"])
time_count = ax1.text(50,1.25,'Time step '+str(0)+ ' of '+str(Timesteps), ha='center')
plt.gcf().canvas.set_window_title('1D FDTD simulation. Number of cells: '+str(XPoints)+', Time steps: '+str(Timesteps) + ', Material: '+material_name)
fig.suptitle('1D FDTD simulation. Number of cells: '+str(XPoints)+', Time steps: '+str(Timesteps) + ', Material: '+material_name)
rect = patches.Rectangle((50,-1.5), 30, 3,facecolor='cyan')
ax1.add_patch(rect)


ax2 = fig.add_subplot(231)
line1, = ax2.plot([],[],color="orange")
ax2.set_ylim(0,np.max(REF))
ax2.set_xlim(0,5000000000)
plt.title('Reflectance')

ax3 = fig.add_subplot(232)
line2, = ax3.plot([],[],color="green")
ax3.set_ylim(0,np.max(TRN))
ax3.set_xlim(0,5000000000)
plt.title('Transmittance')

ax4 = fig.add_subplot(233)
line3, = ax4.plot([],[],color="pink")
ax4.set_ylim(0,np.max(CON))
ax4.set_xlim(0,5000000000)
plt.title('REF + TRN')


def animate(T):
    global xvalues
    global Timesteps
    global FREQ
    global REF
    global TRN
    global CON
    points = []
    points.append(Ey[T])
    points.append(Hx[T])
    time_count.set_text(u'Time step '+str(T)+ ' of '+str(Timesteps))
    lineOne.set_data(xvalues,points[0])
    lineTwo.set_data(xvalues,points[1])
    line1.set_data(FREQ,REF[T])
    line2.set_data(FREQ,TRN[T])
    line3.set_data((FREQ,CON[T]))
    return lineOne,lineTwo,time_count,line1,line2,line3,

ani = FuncAnimation(fig, animate, frames=Timesteps, interval = 20,blit=True)

ani.save('1D FDTD With REF and TRN and object.mp4')

plt.show()
plt.show()
