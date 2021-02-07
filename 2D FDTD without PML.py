import numpy as np
import math
import time
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

#Initialising of the constants
c0 = 299792458
dx = 1
dy = 1
dz = 0
dt = np.sqrt(dx**2 + dy**2 + dz**2)/(2*c0)

XPoints = 100
YPoints = 100
XSrc = 30
YSrc = 30
TimeSteps = 200

material_name = 'Cavity'

#Relative electric permitivity and magnetic permeability
ER = np.ones((XPoints,YPoints))
UR = np.ones((XPoints,YPoints))

#Initialising of fields and curl
Ex = np.zeros((TimeSteps,XPoints,YPoints))
Ey = np.zeros((TimeSteps,XPoints,YPoints))
Ez = np.zeros((TimeSteps,XPoints,YPoints))

#Bx = np.zeros((TimeSteps,XPoints,YPoints))
#By = np.zeros((TimeSteps,XPoints,YPoints))
#Bz = np.zeros((TimeSteps,XPoints,YPoints))

Hx = np.zeros((TimeSteps,XPoints,YPoints))
Hy = np.zeros((TimeSteps,XPoints,YPoints))
Hz = np.zeros((TimeSteps,XPoints,YPoints))

#Dx = np.zeros((TimeSteps,XPoints,YPoints))
#Dy = np.zeros((TimeSteps,XPoints,YPoints))
#Dz = np.zeros((TimeSteps,XPoints,YPoints))
#These are the curl arrays
CEx = np.zeros((TimeSteps,XPoints,YPoints))
CEy = np.zeros((TimeSteps,XPoints,YPoints))
CEz = np.zeros((TimeSteps,XPoints,YPoints))

CHx = np.zeros((TimeSteps,XPoints,YPoints))
CHy = np.zeros((TimeSteps,XPoints,YPoints))
CHz = np.zeros((TimeSteps,XPoints,YPoints))


def GaussianPulse(t:float)->float:
    #Morlet wavelet
    return np.exp(-0.5*((t - 5*20)/20)**2)*np.cos(2*np.pi*t/20)
    #Gaussian pulse
    #return np.exp(-0.5*((t - 6*5)/5)**2)

#Each value loops over x and y, i stands for x, j stands for y, and T stands for the timestep
for T in range(1,TimeSteps):
    #Updating the E field curl
    for i in range(XPoints-1):
        for j in range(YPoints-1):
            #This is where y < last point
            CEx[T][i][j] = (Ez[T-1][i][j+1] - Ez[T-1][i][j])/dy
            CEy[T][i][j] = -((Ez[T-1][i+1][j] - Ez[T-1][i][j])/dx)
            CEz[T][i][j] = (Ey[T-1][i+1][j] - Ey[T-1][i][j])/dx - (Ex[T-1][i][j+1] - Ex[T-1][i][j])/dy
        #This is where y = last point
        CEx[T][i][YPoints-1] = (0-Ez[T-1][i][YPoints-1])/dy
        CEy[T][i][YPoints-1] = -((Ez[T-1][i+1][YPoints-1] - Ez[T-1][i][YPoints-1])/dx)
        CEz[T][i][YPoints-1] = (Ey[T-1][i+1][YPoints-1] - Ey[T-1][i][YPoints-1])/dx - (Ex[T-1][i][YPoints-1] - Ex[T-1][i][YPoints-2])/dy
        
    for j in range(YPoints):
        #This is where x = last point
        CEx[T][XPoints-1][j] = (0-Ez[T-1][XPoints-1][j])/dy
        CEy[T][XPoints-1][j] = -((Ez[T-1][XPoints-1][j] - Ez[T-1][XPoints-2][j])/dx)
        CEz[T][XPoints-1][j] = (Ey[T-1][XPoints-1][j] - Ey[T-1][XPoints-2][j])/dx - (Ex[T-1][XPoints-1][j] - Ex[T-1][XPoints-1][j])/dy
        
    for i in range(XPoints):
        for j in range(YPoints):
            #Bx[T][i][j] = Bx[T-1][i][j] - (c0*dt)*CEx[T][i][j]
            #By[T][i][j] = By[T-1][i][j] - (c0*dt)*CEy[T][i][j]
            #Bz[T][i][j] = Bz[T-1][i][j] - (c0*dt)*CEz[T][i][j]
            
            #Hx[T][i][j] = (1/UR[i][j])*Bx[T][i][j]
            #Hy[T][i][j] = (1/UR[i][j])*By[T][i][j]
            #Hz[T][i][j] = (1/UR[i][j])*Bz[T][i][j]

            Hx[T][i][j] = (1/UR[i][j])*((UR[i][j]/1)*Hx[T-1][i][j] - (c0*dt)*CEx[T][i][j])
            Hy[T][i][j] = (1/UR[i][j])*((UR[i][j]/1)*Hy[T-1][i][j] - (c0*dt)*CEy[T][i][j])
            Hz[T][i][j] = (1/UR[i][j])*((UR[i][j]/1)*Hz[T-1][i][j] - (c0*dt)*CEz[T][i][j])
    #Updating the H field curl
            
    #This is at point 0,0
    #CHz[T][0][0] = (Hy[T-1][0][0] - 0)/dx - (H[T-1][0][0] - 0)/dy
    CHx[T][0][0] = (Hz[T][0][0] - Hz[T][0][0])/dy
    CHy[T][0][0] = -((Hz[T][0][0] - Hz[T][0][0])/dx)
    CHz[T][0][0] = (Hy[T][0][0] - Hy[T][0][0])/dx - (Hx[T][0][0] - Hx[T][0][0])/dy
    for i in range(1,XPoints):
        #This is when y = 0
        #CHz[T][i][0] = (Hy[T-1][i][0] - Hy[T-1][i-1][0])/dx - (H[T-1][i][0] - 0)/dy
        CHx[T][i][0] = (Hz[T][i][0] - Hz[T][i][0])/dy
        CHy[T][i][0] = -((Hz[T][i][0] - Hz[T][i-1][0])/dx)
        CHz[T][i][0] = (Hy[T][i][0] - Hy[T][i-1][0])/dx - (Hx[T][i][0] - Hx[T][i][0])/dy
    for j in range(1,YPoints):
        #This is when x = 0
        #CHz[T][0][j] = (Hy[T-1][0][j] - 0)/dx - (H[T-1][0][j] - H[T-1][0][j-1])/dy
        CHx[T][0][j] = (Hz[T][0][j] - Hz[T][0][j-1])/dy
        CHy[T][0][j] = -((Hz[T][0][j] - Hz[T][0][j])/dx)
        CHz[T][0][j] = (Hy[T][0][j] - Hy[T][0][j])/dx - (Hx[T][0][j] - Hx[T][0][j-1])/dy
        for i in range(1,XPoints):
            #This is for where both x and y > 0
            #CHz[T][i][j] = (Hy[T-1][i][j] - Hy[T-1][i-1][j])/dx - (H[T-1][i][j] - H[T-1][i][j-1])/dy
            CHx[T][i][j] = (Hz[T][i][j] - Hz[T][i][j-1])/dy
            CHy[T][i][j] = -((Hz[T][i][j] - Hz[T][i-1][j])/dx)
            CHz[T][i][j] = (Hy[T][i][j] - Hy[T][i-1][j])/dx - (Hx[T][i][j] - Hx[T][i][j-1])/dy
        
    #Updating D from H
    for i in range(XPoints):
        for j in range(YPoints):
            #Dx[T][i][j] = Dx[T-1][i][j] + (c0*dt)*CHx[T][i][j]
            #Dy[T][i][j] = Dy[T-1][i][j] + (c0*dt)*CHy[T][i][j]
            #Dz[T][i][j] = Dz[T-1][i][j] + (c0*dt)*CHz[T][i][j]
            
            #Ex[T][i][j] = (1/ER[i][j])*Dx[T][i][j]
            #Ey[T][i][j] = (1/ER[i][j])*Dy[T][i][j]
            #Ez[T][i][j] = (1/ER[i][j])*Dz[T][i][j]

            Ex[T][i][j] = (1/ER[i][j])*((ER[i][j]/1)*Ex[T-1][i][j] + (c0*dt)*CHx[T][i][j])
            Ey[T][i][j] = (1/ER[i][j])*((ER[i][j]/1)*Ey[T-1][i][j] + (c0*dt)*CHy[T][i][j])
            Ez[T][i][j] = (1/ER[i][j])*((ER[i][j]/1)*Ez[T-1][i][j] + (c0*dt)*CHz[T][i][j])
            
    #Adding in Gaussian pulse soft source
    #Dx[T][XSrc][YSrc] = Dx[T][XSrc][YSrc] + GaussianPulse(T-1)
    #Dy[T][XSrc][YSrc] = Dy[T][XSrc][YSrc] + GaussianPulse(T-1)
    #Dz[T][XSrc][YSrc] = Dz[T][XSrc][YSrc] + GaussianPulse(T-1)
    #Updating E from D
    #Ex[T][XSrc][YSrc] = (1/ER[XSrc][YSrc])*(GaussianPulse(T-1))
    #Ey[T][XSrc][YSrc] = (1/ER[XSrc][YSrc])*(GaussianPulse(T-1))
    Ez[T][XSrc][YSrc] = (1/ER[XSrc][YSrc])*(GaussianPulse(T-1))
            


dpi = 72
fps = 25
width = 12
height = 12
fig = plt.figure(figsize=(width, height), dpi=dpi)
ax = fig.add_axes([0.1, 0.1, 0.8, 0.8], frameon=True, aspect=1)

plt.gcf().canvas.set_window_title('2D FDTD simulation. Number of cells: '+str(XPoints)+' x '+str(YPoints)
                                  + ', Time steps: '+str(TimeSteps)
                                  + ', Material: '+material_name)
fig.suptitle('2D FDTD simulation. Number of cells: '+str(XPoints)+' x '+str(YPoints)
                                  + ', Time steps: '+str(TimeSteps)
                                  + ', Material: '+material_name)

N_E = np.sqrt((Ez[:][:][:]**2).max())
if N_E == 0: N_E = 1

img = plt.imshow(Ez[0]/(N_E), interpolation='bilinear', norm=mpl.colors.SymLogNorm(linthresh=0.003, linscale=0.003, vmin=-1.0, vmax=1.0), cmap = mpl.cm.jet)
time_count = ax.annotate('Time step '+str(T)+ ' of '+str(T), (XPoints//2,YPoints//2), ha='center')

def update(t):
    img.set_data(Ez[t]/(N_E))
    time_count.set_text('Time step '+str(t)+ ' of '+str(T))
    return [img, time_count]

ani = FuncAnimation(fig, update, frames=range(T), interval = 1000//fps,blit=True)
ani.save('2D FDTD without PML.mp4')
plt.show()
