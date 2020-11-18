#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Mar 30 10:49:59 2020

@author: mbeiran
"""
#%%
import numpy as np
import matplotlib.pyplot as plt
import lib_rnns as lr
from matplotlib.gridspec import GridSpec
aa = lr.set_plot()


    
gaussian_norm = (1/np.sqrt(np.pi))
gauss_points, gauss_weights = np.polynomial.hermite.hermgauss(200)
gauss_points = gauss_points*np.sqrt(2)

def Phi (mu, delta0):
    integrand = np.tanh(mu+np.sqrt(delta0)*gauss_points)
    return gaussian_norm * np.dot (integrand,gauss_weights)
def Prime (mu, delta0):
    integrand = 1 - (np.tanh(mu+np.sqrt(delta0)*gauss_points))**2
    return gaussian_norm * np.dot (integrand,gauss_weights)

#%%
N = 200
nbins = 20
s_mn1 = 0.5
s_mn2 = 1.2
m1 = np.random.randn(N)
m1 = m1/np.std(m1)
m2 = np.random.randn(N)
m2 = m2/np.std(m2)
ms = np.linspace(-3, 3)

n1 = s_mn1*m1+0.3*np.random.randn(N)
n2 = s_mn2*m2+0.3*np.random.randn(N)


#%%
# =============================================================================
#           Fig 2
# =============================================================================
ms = np.linspace(-5,5,100)
Sigma = np.zeros((3,3))
Sigma[0,0] = 1.2
Sigma[1,1] = 1.2
Sigma[2,2] = 1.6
Sigma[0,1] = -2.
Sigma[1,0] = 1.
Sigma[0,2] = 1.5

valN = 9

BigSigma = np.zeros((6,6))
BigSigma[0,0] = 1. #Sigma[0,0]
BigSigma[1,1] = 1.#Sigma[1,1]
BigSigma[2,2] = 1. #Sigma[2,2]
BigSigma[3,3] = valN
BigSigma[4,4] = valN
BigSigma[5,5] = valN

Sigma2 = Sigma.T


BigSigma[0,3] = Sigma2[0,0] #m1 - n1
BigSigma[3,0] = Sigma2[0,0]
BigSigma[0,4] = Sigma2[0,1] #m1 - n2
BigSigma[4,0] = Sigma2[0,1]
BigSigma[1,3] = Sigma2[1,0] #m2 - n1
BigSigma[3,1] = Sigma2[1,0]
BigSigma[1,4] = Sigma2[1,1] #m2 - n2
BigSigma[4,1] = Sigma2[1,1]
BigSigma[2,5] = Sigma2[2,2] #m3 - n3
BigSigma[5,2] = Sigma2[2,2]
BigSigma[2,3] = Sigma2[2,0] #m3 - n1
BigSigma[3,2] = Sigma2[2,0]
BigSigma[0,5] = Sigma2[0,2] #m1 - n3
BigSigma[5,0] = Sigma2[0,2]
BigSigma[1,5] = Sigma2[1,2] #m2 - n3
BigSigma[5,1] = Sigma2[1,2]
BigSigma[2,4] = Sigma2[2,1] #m3 - n2
BigSigma[4,2] = Sigma2[2,1]


mean = np.zeros(6)

Dat = np.random.multivariate_normal(mean, BigSigma, size=1000)
M = Dat[:,0:3]
N = Dat[:,3:].T


S=10

fig = plt.figure(figsize=[4.2, 4.2], dpi=450)

gs = GridSpec(7,7)

ax_joint00 = fig.add_subplot(gs[1:3,0:2])
ax_joint01 = fig.add_subplot(gs[1:3,2:4])
ax_joint02 = fig.add_subplot(gs[1:3,4:6])

ax_joint10 = fig.add_subplot(gs[3:5,0:2])
ax_joint11 = fig.add_subplot(gs[3:5,2:4])
ax_joint12 = fig.add_subplot(gs[3:5,4:6])

ax_joint20 = fig.add_subplot(gs[5:7,0:2])
ax_joint21 = fig.add_subplot(gs[5:7,2:4])
ax_joint22 = fig.add_subplot(gs[5:7,4:6])

ax_marg_x0 = fig.add_subplot(gs[0,0:2])
ax_marg_x1 = fig.add_subplot(gs[0,2:4])
ax_marg_x2 = fig.add_subplot(gs[0,4:6])

ax_marg_y0 = fig.add_subplot(gs[1:3,6])
ax_marg_y1 = fig.add_subplot(gs[3:5,6])
ax_marg_y2 = fig.add_subplot(gs[5:7,6])

myl = -12
myl2 = 12

ax_joint00.scatter(M[:,0], N[0,:], s=S, alpha=0.5, label=r'$\sigma_{mn} = 1.2$', rasterized=True)
ax_joint00.plot(ms, Sigma[0,0]*ms, '--', c='k', lw=1)
ax_joint00.set_xlim([-3,3])
ax_joint00.set_xticks([-2., 0, 2.])
ax_joint00.set_xticklabels(['','',''])
ax_joint00.set_ylim([myl,myl2])
ax_joint00.set_yticks([-10, 0, 10])
ax_joint00.set_ylabel(r'$n^{\left(1\right)}_i$')
ax_joint00.spines['top'].set_visible(False)
ax_joint00.spines['right'].set_visible(False)
ax_joint00.yaxis.set_ticks_position('left')
ax_joint00.xaxis.set_ticks_position('bottom')
                                  
ax_joint01.scatter(M[:,1], N[0,:], s=S, alpha=0.5, label=r'$\sigma_{mn} = 1.2$', rasterized=True)
ax_joint01.plot(ms, Sigma[0,1]*ms, '--', c='k', lw=1)
ax_joint01.spines['top'].set_visible(False)
ax_joint01.spines['right'].set_visible(False)
ax_joint01.yaxis.set_ticks_position('left')
ax_joint01.xaxis.set_ticks_position('bottom')
ax_joint01.set_ylim([myl,myl2])
ax_joint01.set_yticks([-10, 0, 10])
ax_joint01.set_yticklabels(['','',''])
ax_joint01.set_xlim([-3,3])
ax_joint01.set_xticks([-2., 0, 2.])
ax_joint01.set_xticklabels(['','',''])

ax_joint02.scatter(M[:,2], N[0,:], s=S, alpha=0.5, label=r'$\sigma_{mn} = 1.2$', rasterized=True)
ax_joint02.plot(ms, Sigma[0,2]*ms, '--', c='k', lw=1)
ax_joint02.spines['top'].set_visible(False)
ax_joint02.spines['right'].set_visible(False)
ax_joint02.yaxis.set_ticks_position('left')
ax_joint02.xaxis.set_ticks_position('bottom')
ax_joint02.set_ylim([myl,myl2])
ax_joint02.set_yticks([-10, 0, 10])
ax_joint02.set_yticklabels(['','',''])
ax_joint02.set_xlim([-3,3])
ax_joint02.set_xticks([-2., 0, 2.])

ax_joint02.set_xticklabels(['','',''])

ax_joint10.scatter(M[:,0], N[1,:], s=S, alpha=0.5, label=r'$\sigma_{mn} = 1.2$', rasterized=True)
ax_joint10.plot(ms, Sigma[1,0]*ms, '--', c='k', lw=1)
ax_joint10.set_xlim([-3,3])
ax_joint10.spines['top'].set_visible(False)
ax_joint10.spines['right'].set_visible(False)
ax_joint10.yaxis.set_ticks_position('left')
ax_joint10.xaxis.set_ticks_position('bottom')
ax_joint10.set_ylim([myl,myl2])
ax_joint10.set_yticks([-10, 0, 10])
ax_joint10.set_xlim([-3,3])
ax_joint10.set_xticks([-2., 0, 2.])
ax_joint01.set_xticklabels(['','',''])
ax_joint10.set_ylabel(r'$n^{\left(2\right)}_i$')

ax_joint11.scatter(M[:,1], N[1,:], s=S, alpha=0.5, label=r'$\sigma_{mn} = 1.2$', rasterized=True)
ax_joint11.plot(ms, Sigma[1,1]*ms, '--', c='k', lw=1)
ax_joint11.set_xlim([-3,3])
ax_joint11.spines['top'].set_visible(False)
ax_joint11.spines['right'].set_visible(False)
ax_joint11.set_ylim([myl,myl2])
ax_joint11.set_yticks([-10, 0, 10])
ax_joint11.set_xticks([-2., 0, 2.])
ax_joint11.set_xlim([-3,3])
ax_joint11.set_yticklabels(['','',''])
ax_joint01.set_xticklabels(['','',''])
ax_joint11.yaxis.set_ticks_position('left')
ax_joint11.xaxis.set_ticks_position('bottom')


ax_joint12.scatter(M[:,2], N[1,:], s=S, alpha=0.5, label=r'$\sigma_{mn} = 1.2$', rasterized=True)
ax_joint12.plot(ms, Sigma[1,2]*ms, '--', c='k', lw=1)
ax_joint12.set_xlim([-3,3])
ax_joint12.spines['top'].set_visible(False)
ax_joint12.spines['right'].set_visible(False)
ax_joint12.set_ylim([myl,myl2])
ax_joint12.set_yticks([-10, 0, 10])
ax_joint12.set_xticks([-2., 0, 2.])
ax_joint12.set_xlim([-3,3])
ax_joint12.set_yticklabels(['','',''])
ax_joint01.set_xticklabels(['','',''])
ax_joint12.yaxis.set_ticks_position('left')
ax_joint12.xaxis.set_ticks_position('bottom')

ax_joint20.scatter(M[:,0], N[2,:], s=S, alpha=0.5, label=r'$\sigma_{mn} = 1.2$', rasterized=True)
ax_joint20.plot(ms, Sigma[2,0]*ms, '--', c='k', lw=1)
ax_joint20.set_xlim([-3,3])
ax_joint20.spines['top'].set_visible(False)
ax_joint20.spines['right'].set_visible(False)
ax_joint20.yaxis.set_ticks_position('left')
ax_joint20.xaxis.set_ticks_position('bottom')
ax_joint20.set_ylim([myl,myl2])
ax_joint20.set_yticks([-10, 0, 10])
ax_joint20.set_xlim([-3,3])
ax_joint20.set_xticks([-2., 0, 2.])
ax_joint20.set_ylabel(r'$n^{\left(3\right)}_i$')
ax_joint20.set_xlabel(r'$m^{\left(1\right)}_i$')

ax_joint21.scatter(M[:,1], N[2,:], s=S, alpha=0.5, label=r'$\sigma_{mn} = 1.2$', rasterized=True)
ax_joint21.plot(ms, Sigma[2,1]*ms, '--', c='k', lw=1)
ax_joint21.set_xlim([-3,3])
ax_joint21.spines['top'].set_visible(False)
ax_joint21.spines['right'].set_visible(False)
ax_joint21.set_ylim([myl,myl2])
ax_joint21.set_yticks([-10, 0, 10])
ax_joint21.set_xticks([-2., 0, 2.])
ax_joint21.set_xlim([-3,3])
ax_joint21.set_yticklabels(['','',''])
ax_joint21.yaxis.set_ticks_position('left')
ax_joint21.xaxis.set_ticks_position('bottom')
ax_joint21.set_xlabel(r'$m^{\left(2\right)}_i$')

ax_joint22.scatter(M[:,2], N[2,:], s=S, alpha=0.5, label=r'$\sigma_{mn} = 1.2$', rasterized=True)
ax_joint22.plot(ms, Sigma[2,2]*ms, '--', c='k', lw=1)
ax_joint22.set_xlim([-3,3])
ax_joint22.spines['top'].set_visible(False)
ax_joint22.spines['right'].set_visible(False)
ax_joint22.set_ylim([myl,myl2])
ax_joint22.set_yticks([-10, 0, 10])
ax_joint22.set_xticks([-2., 0, 2.])
ax_joint22.set_xlim([-3,3])
ax_joint22.set_yticklabels(['','',''])
ax_joint22.yaxis.set_ticks_position('left')
ax_joint22.xaxis.set_ticks_position('bottom')
ax_joint22.set_xlabel(r'$m^{\left(3\right)}_i$')


ax_marg_x0.hist(M[:,0], nbins, alpha=0.5, density=True)
ss = 1.
ax_marg_x0.plot(ms, (1/np.sqrt(2*np.pi*ss**2))*np.exp(-(ms)**2/(2*ss**2)), 'k')

ax_marg_x0.spines['top'].set_visible(False)
ax_marg_x0.spines['right'].set_visible(False)
ax_marg_x0.spines['left'].set_visible(False)
ax_marg_x0.yaxis.set_ticks_position('left')
ax_marg_x0.xaxis.set_ticks_position('bottom')
ax_marg_x0.set_xlim([-3,3])
ax_marg_x0.set_xticks([-2., 0, 2.])
ax_marg_x0.set_ylim([0,0.45])
ax_marg_x0.set_xticklabels(['','',''])
ax_marg_x0.set_yticks([1])

ax_marg_x1.hist(M[:,1], nbins, alpha=0.5, density=True)
ss = 1.
ax_marg_x1.plot(ms, (1/np.sqrt(2*np.pi*ss**2))*np.exp(-(ms)**2/(2*ss**2)), 'k')
ax_marg_x1.spines['top'].set_visible(False)
ax_marg_x1.spines['right'].set_visible(False)
ax_marg_x1.spines['left'].set_visible(False)
ax_marg_x1.yaxis.set_ticks_position('left')
ax_marg_x1.xaxis.set_ticks_position('bottom')
ax_marg_x1.set_xlim([-3,3])
ax_marg_x1.set_ylim([0,0.45])
ax_marg_x1.set_xticks([-2., 0, 2.])
ax_marg_x1.set_xticklabels(['','',''])
ax_marg_x1.set_yticks([1])

ax_marg_x2.hist(M[:,2], nbins, alpha=0.5, density=True)
ss = 1.
ax_marg_x2.plot(ms, (1/np.sqrt(2*np.pi*ss**2))*np.exp(-(ms)**2/(2*ss**2)), 'k')
ax_marg_x2.spines['top'].set_visible(False)
ax_marg_x2.spines['right'].set_visible(False)
ax_marg_x2.spines['left'].set_visible(False)
ax_marg_x2.yaxis.set_ticks_position('left')
ax_marg_x2.xaxis.set_ticks_position('bottom')
ax_marg_x2.set_xlim([-3,3])
ax_marg_x2.set_ylim([0,0.45])
ax_marg_x2.set_xticks([-2., 0, 2.])
ax_marg_x2.set_xticklabels(['','',''])
ax_marg_x2.set_yticks([1])
ss2 = 0
ms = np.linspace(-10,10,100)
ax_marg_y0.hist(N[0,:], nbins, orientation="horizontal", alpha=0.5, density=True)
ss = np.sqrt(valN)
ax_marg_y0.plot((1/np.sqrt(2*np.pi*ss**2))*np.exp(-(ms)**2/(2*ss**2)), ms, 'k')
ax_marg_y0.spines['top'].set_visible(False)
ax_marg_y0.spines['right'].set_visible(False)
ax_marg_y0.spines['bottom'].set_visible(False)
ax_marg_y0.yaxis.set_ticks_position('left')
ax_marg_y0.xaxis.set_ticks_position('bottom')
ax_marg_y0.set_ylim([myl,myl2])
ax_marg_y0.set_xlim([0,0.45])
ax_marg_y0.set_yticks([-10, 0, 10])
ax_marg_y0.set_yticklabels(['','',''])
ax_marg_y0.set_xticks([1])
ax_marg_y0.set_xticklabels([''])

ax_marg_y1.hist(N[1,:], nbins, orientation="horizontal", alpha=0.5, density=True)
ss = np.sqrt(valN)
ax_marg_y1.plot((1/np.sqrt(2*np.pi*ss**2))*np.exp(-(ms)**2/(2*ss**2)), ms, 'k')
ax_marg_y1.spines['top'].set_visible(False)
ax_marg_y1.spines['right'].set_visible(False)
ax_marg_y1.spines['bottom'].set_visible(False)
ax_marg_y1.yaxis.set_ticks_position('left')
ax_marg_y1.xaxis.set_ticks_position('bottom')
ax_marg_y1.set_ylim([myl,myl2])
ax_marg_y1.set_xlim([0,0.45])
ax_marg_y1.set_yticks([-10, 0, 10])
ax_marg_y1.set_yticklabels(['','',''])
ax_marg_y1.set_xticks([1])
ax_marg_y1.set_xticklabels([''])

ax_marg_y2.hist(N[1,:], nbins, orientation="horizontal", alpha=0.5, density=True)
ss = np.sqrt(valN)
ax_marg_y2.plot((1/np.sqrt(2*np.pi*ss**2))*np.exp(-(ms)**2/(2*ss**2)), ms, 'k')
ax_marg_y2.spines['top'].set_visible(False)
ax_marg_y2.spines['right'].set_visible(False)
ax_marg_y2.spines['bottom'].set_visible(False)
ax_marg_y2.yaxis.set_ticks_position('left')
ax_marg_y2.xaxis.set_ticks_position('bottom')
ax_marg_y2.set_ylim([myl,myl2])
ax_marg_y2.set_xlim([0,0.45])
ax_marg_y2.set_yticks([-10, 0, 10])
ax_marg_y2.set_yticklabels(['','',''])
ax_marg_y2.set_xticks([1])
ax_marg_y2.set_xticklabels([''])

plt.savefig('Th_Fig2_4__A.pdf')

#%%
plt.rcParams["axes.grid"] = False
fig = plt.figure(figsize = [3.0, 3.0])
ax = fig.add_subplot(111) 
plt.imshow(Sigma, cmap='coolwarm', vmin = -4, vmax = 4)
Sigma22 = np.zeros_like(Sigma)
Sigma22[np.abs(Sigma)>0]=np.nan
plt.imshow(Sigma22, cmap='OrRd', vmin = 0, vmax = 4)


ax.tick_params(color='white')


for i in range(np.shape(Sigma)[0]):
    for j in range(np.shape(Sigma)[1]):
        ax.text(i, j, str(Sigma[j,i]), va='center', ha='center', fontsize=16)
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.spines['left'].set_visible(False)
ax.spines['bottom'].set_visible(False)

ax.yaxis.set_ticks_position('right')
ax.xaxis.set_ticks_position('top')  
ax.set_xticks([0, 1,2])
ax.set_yticks([0, 1,2])


ax.set_xticklabels([r'$m_i^{\left(1\right)}$', r'$m_i^{\left(2\right)}$', r'$m_i^{\left(3\right)}$' ], fontsize=14)
ax.set_yticklabels([r'$n_i^{\left(1\right)}$', r'$n_i^{\left(2\right)}$', r'$n_i^{\left(3\right)}$' ], fontsize=14)

plt.savefig('Th_Fig2_4__B.pdf')

#%%
fig = plt.figure()
from mpl_toolkits.mplot3d import Axes3D
ax = fig.add_subplot(111, projection='3d',azim=-42, elev=21)


u, vv = np.linalg.eig(Sigma)
vv = np.dot(vv, np.diag(u))
l1 = -0.5
l2 = 2.
cC = np.array((1, 1, 1,))*0.3

from matplotlib.patches import FancyArrowPatch
from mpl_toolkits.mplot3d import proj3d

class Arrow3D(FancyArrowPatch):
    def __init__(self, xs, ys, zs, *args, **kwargs):
        FancyArrowPatch.__init__(self, (0,0), (0,0), *args, **kwargs)
        self._verts3d = xs, ys, zs

    def draw(self, renderer):
        xs3d, ys3d, zs3d = self._verts3d
        xs, ys, zs = proj3d.proj_transform(xs3d, ys3d, zs3d, renderer.M)
        self.set_positions((xs[0],ys[0]),(xs[1],ys[1]))
        FancyArrowPatch.draw(self, renderer)

# mean values
mean_x = 0
mean_y = 0
mean_z = 0

counter = 0
for v in vv[:,1:3].T:
    #ax.plot([mean_x,v[0]], [mean_y,v[1]], [mean_z,v[2]], color='red', alpha=0.8, lw=3)
    #I will replace this line with:
    v1 = np.real(v)
    v1 = v1/np.sqrt(np.sum(v1**2))
    if np.imag(u[counter])==0:
        v1 = v1*np.real(u[counter])
    
    a = Arrow3D([mean_x, v1[0]], [mean_y, v1[1]], 
                [mean_z, v1[2]], mutation_scale=10, 
                lw=2, arrowstyle="-|>", color=cC)
    ax.add_artist(a)
    #ax.plot([0,v1[0]], [0,v1[1]], [0, v1[2]], 'C0')
    #ax.scatter(v1[0], v1[1], v1[2],s=100)
    if np.min(np.abs(v1))>0.:
        ax.plot([0,v1[0]], [0,v1[1]], [0, 0], 'k--', lw=0.5)
        ax.plot([v1[0],v1[0]], [v1[1],v1[1]], [0, v1[2]], 'k--', lw=0.5)
    print(v1)
    
    #ax.plot([v1[0], v1[1], 0], [v1[0], v1[1], v1[2]], '--k')
   
    v2 = np.imag(v)
    if np.sqrt(np.sum(v2**2))>0:
        print('hey')
        v2 = v2/np.sqrt(np.sum(v2**2))
        #v2 = v2*np.real(u[counter])
        print(v2)
        b = Arrow3D([mean_x, v2[0]], [mean_y, v2[1]], 
                [mean_z, (v2[2])], mutation_scale=10, 
                lw=2, arrowstyle="-|>", color=cC)
        ax.add_artist(b)
        #ax.scatter(v2[0], v2[1], v2[2],s=100)
        if np.min(np.abs(v2))>0.:
            ax.plot([0,v2[0]], [0,v2[1]], [0, 0], 'k--')
            ax.plot([v2[0],v2[0]], [v2[1],v2[1]], [0, v2[2]], 'k--')
        #ax.plot([0,0,0], [v2[0], v2[1], v2[2]], 'C1')
        #print(v2)
        v1s = v1
        v2s = v2
    counter+=1
    
xx, yy = np.meshgrid(np.linspace(-1.5, 1.5), np.linspace(-1.5, 1.5))

cvs = np.cross(v1s, v2s)

zz = 0*(1/cvs[2])*(-cvs[0]*xx-cvs[1]*yy)

ax.plot_surface(xx, yy, zz, alpha=0.2)
ax.set_xlabel('x_values')
ax.set_ylabel('y_values')
ax.set_zlabel('z_values')
#ax.set_xlim([-0.5,0.5])
#ax.set_ylim([-0.5,0.5])
#ax.set_zlim([-0.5,1.])
ax.text(1.25, -0.6, 0., r'$ Re(\bf{u}_2)$', fontsize = 12)
ax.text(-3., 0.2, 0.0, r'$Im(\bf{u}_2)$', fontsize = 12)

ax.text(-1.9, 0.3, 0.5, r'$\lambda_1 \bf{u}_1$', fontsize = 12)



ax.axis('off')
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.spines['left'].set_visible(False)
ax.spines['bottom'].set_visible(False)

plt.draw()

plt.savefig('Th_Fig2_4__B1.pdf')
plt.show()

#%%
#
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d',azim=-72, elev=7) 

plt.xlabel('$\kappa_1$')
plt.ylabel('$\kappa_2$')
ax.set_zlabel('$\kappa_3$')




Nn = 1000


Mu= np.zeros((6,1))

inkap1 = np.linspace(-0.8, 0.8, 2)
inkap2 = np.linspace(-0.8, 0.800, 2)
inkap3 = np.linspace(-0.8, 0.800, 2)


dt = 0.2
time = np.arange(0, 80, dt)

ins = np.array(([1,0,0],[1., 0, 0.2],[1., 0, -0.2]))

colors = np.zeros((3,3))

colors[1,:] = 0.6*(0.4+0.1*np.random.rand(3)-0.1)+0.4*np.array((1,0,0))
colors[0,:] = 0.4#0.8*(0.6+0.2*np.random.rand(3)-0.1)+np.array((1,0,0))
colors[2,:] = 0.6*(0.4+0.1*np.random.rand(3)-0.1)+0.4*np.array((0,0,1))

for trials in range(1):
    cC =  0.6+0.2*np.random.rand()-0.1
    
    cC2 =  0.1+0.2*np.random.rand()-0.1
    
    for rr in range(np.shape(ins)[0]):
        sk1 = np.zeros_like(time)
        sk2 = np.zeros_like(time)
        sk3 = np.zeros_like(time)
        
        
        sk1[0] = ins[rr,0]
        sk2[0] = ins[rr,1]
        sk3[0] = ins[rr,2]
        
        x0 = np.array((sk1[0], sk2[0], sk3[0]))
        print(sk3[0])
        for it, ti in enumerate(time[:-1]):
            
            x = x0 + dt*(-x0 + Prime(0, np.sum(x0**2))*np.dot(Sigma, x0))
            sk1[it+1] = x[0]
            sk2[it+1] = x[1]
            sk3[it+1] = x[2]
            
            x0 = x
        
        ax.plot(sk1, sk2, sk3, c=colors[rr,:])
        ax.scatter(sk1[0], sk2[0], sk3[0], s=10, facecolor=colors[rr,:], edgecolor=colors[rr,:], )
        if np.abs(sk3[-1])>0.1:
            ax.scatter(sk1[-1], sk2[-1], sk3[-1], s=50, facecolor=[0.6, 0.6, 0.6], edgecolor='k')
            save = np.array((sk1[-1], sk2[-1], sk3[-1]))

ax.scatter(save[0], save[1], save[2], s=70, facecolor=[0.6, 0.6, 0.6], edgecolor='k')
ax.scatter(-save[0], -save[1], -save[2], s=70, facecolor=[0.6, 0.6, 0.6], edgecolor='k')
                        
ax.set_xticks([-1, 0, 1])
ax.set_yticks([-0.5,  0.5])
ax.set_zticks([-0.5, 0, 0.5])

ax.dist=11
plt.savefig('Th_Fig2_4__C.pdf')    

#%%
#
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d',azim=-72, elev=7) 

plt.xlabel('$\kappa_1$')
plt.ylabel('$\kappa_2$')
ax.set_zlabel('$\kappa_3$')

Nn = 1000


Mu= np.zeros((6,1))

inkap1 = np.linspace(-0.8, 0.8, 2)
inkap2 = np.linspace(-0.8, 0.800, 2)
inkap3 = np.linspace(-0.8, 0.800, 2)


dt = 0.2
time = np.arange(0, 80, dt)

for trials in range(2):
    try_s0 = 100
    for tr in range(2000):
        XX = np.random.multivariate_normal(Mu[:,0], BigSigma, size=Nn)
        try_s = np.sum((np.dot(XX.T,XX)/1000-BigSigma)**2)
        if try_s < try_s0:
            #print(try_s)
            try_s0 = try_s
            XX_s = XX
    M = XX_s[:,0:3]
    N = XX_s[:,3:6]
    
    J = np.dot(M, N.T)/Nn
    
    cC =  0.6+0.2*np.random.rand()-0.1
    
    cC2 =  0.1+0.2*np.random.rand()-0.1
    
    for rr in range(np.shape(ins)[0]):
        sk1 = np.zeros_like(time)
        sk2 = np.zeros_like(time)
        sk3 = np.zeros_like(time)
        
        
        sk1[0] = ins[rr,0]
        sk2[0] = ins[rr,1]
        sk3[0] = ins[rr,2]
            
        
        
        x0 = ins[rr,0]*M[:,0] + ins[rr,1]*M[:,1]+ins[rr,2]*M[:,2]
        sk1[0] = np.mean(M[:,0]*x0)
        sk2[0] = np.mean(M[:,1]*x0)
        sk3[0] = np.mean(M[:,2]*x0)
        
        for it, ti in enumerate(time[:-1]):
            x = x0 + dt*(-x0 + np.dot(J, np.tanh(x0)))
            sk1[it+1] = np.mean(M[:,0]*x)
            sk2[it+1] = np.mean(M[:,1]*x)
            sk3[it+1] = np.mean(M[:,2]*x)
            
            x0 = x
        ax.plot(sk1, sk2, sk3, c=colors[rr,:])
        ax.scatter(sk1[0], sk2[0], sk3[0], s=10, facecolor=colors[rr,:])
        ax.scatter(sk1[-1], sk2[-1], sk3[-1], s=50, facecolor=colors[rr,:])
           
    
ax.scatter(save[0], save[1], save[2], s=70, facecolor=[0.6, 0.6, 0.6], edgecolor='k')
ax.scatter(-save[0], -save[1], -save[2], s=70, facecolor=[0.6, 0.6, 0.6], edgecolor='k')
         
ax.set_xticks([-1, 0, 1])
ax.set_yticks([-0.5,  0.5])
ax.set_zticks([-0.5, 0, 0.5])


ax.dist=11
plt.savefig('Th_Fig2_4__D.pdf')    
#    
