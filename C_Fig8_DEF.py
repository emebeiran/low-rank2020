#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Oct  7 11:01:38 2020

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
def Trip (mu, delta0):
    integrand = 2*(np.cosh(2*(mu+np.sqrt(delta0)*gauss_points))-2 )/np.cosh(mu+np.sqrt(delta0)*gauss_points)**4
    return gaussian_norm * np.dot (integrand,gauss_weights)


#%%

epss = np.linspace(1., 3, 7)#np.array((0, 0.1, 0.2, 0.4, 0.6, 1.0, 2.0))
#epss = np.linspace(0., 2, 7)#np.array((0, 0.1, 0.2, 0.4, 0.6, 1.0, 2.0))

fp1s = np.zeros_like(epss)
fp2s = np.zeros_like(epss)
us = np.zeros((2,len(epss)))*1j
vs = np.zeros((2,2,len(epss)))*1j

clS = np.zeros((3, len(epss)))

cl2 = np.array(( 225/256, 74/256, 51/256))
cl1 = np.array(( 0.5, 0.5, 0.5))

nL = len(epss)-1
for iep, eps in enumerate(epss):
    clS[:,iep] = cl1* (nL-iep)/nL   + cl2* (iep)/nL
#%%
dt = 0.05
T = 40
time = np.arange(0, T, dt)
traj1s = np.zeros((2,len(time), len(epss)))
traj2s = np.zeros((2,len(time), len(epss)))


for iep, eps in enumerate(epss):
    ms = np.linspace(-5,5,100)
    Sigma = np.zeros((2,2))
    Sigma[0,0] = 1.4
    Sigma[1,1] = 1.4
    Sigma[0,1] = 1.*eps
    Sigma[1,0] = -1./eps
    
#    Sigma[0,0] = 2+eps
#    Sigma[1,1] = 2-eps
#    Sigma[0,1] = 1.
#    Sigma[1,0] = -(eps**2+1)
    
    u, v = np.linalg.eig(Sigma)
    us[:,iep] = u
    vs[:,:,iep] = v
    
    l1 = -0.5
    l2 = 2.
    cC = np.array((1, 1, 1,))*0.3

    
    kaps1 = np.linspace(-1.3,1.3, 130)
    kaps2 = np.linspace(-1.3,1.3, 100)
    ksol = np.zeros((len(kaps1), len(kaps2), 2))
    
    K1s, K2s = np.meshgrid(kaps1, kaps2)
    def transf(K):
        return(K*Prime(0, np.dot(K.T, K)))
        
    E = np.zeros((len(kaps1), len(kaps2)))
    for ik1 ,k1 in enumerate(kaps1):
        for ik2, k2 in enumerate(kaps2):
            K = np.array((k1, k2))
            ksol[ik1, ik2, :] = - K+ np.dot(Sigma, transf(K))
            E[ik1, ik2] = np.sqrt(np.sum(ksol[ik1,ik2,:]**2))

    time1 = np.arange(0, 10, dt)
    traj1 = np.zeros((2,len(time)))
    traj1[:,0] = np.array((0.5,0.5))
    
    for it, ti in enumerate(time[:-1]):
        traj1[:,it+1] = traj1[:,it] + dt*(-traj1[:,it] + np.dot(Sigma,transf(traj1[:,it])))
    A = traj1[:,it+1]
    traj1 = np.zeros((2,len(time)))
    traj2 = np.zeros((2,len(time)))
    traj1[:,0] = A
    
    for it, ti in enumerate(time[:-1]):
        traj1[:,it+1] = traj1[:,it] + dt*(-traj1[:,it] + np.dot(Sigma,transf(traj1[:,it])))
    
    traj1s[:,:,iep] = traj1

    
    fig = plt.figure()
    ax = fig.add_subplot(111) 
    im = plt.pcolor(kaps1, kaps2, np.log10(E).T, cmap ='viridis', vmin = -2.,vmax=0, shading= 'auto')
    

    strm = ax.streamplot(kaps1, kaps2, ksol[:,:,0].T, ksol[:,:,1].T, color=[0.4, 0.4, 0.4], linewidth=0.8, cmap='autumn', density=0.6)
    plt.xlabel('$\kappa_1$')
    plt.ylabel('$\kappa_2$')
    plt.scatter([ 0, ], [0], s=50, edgecolor='k', facecolor='w', linewidth=1., zorder=4)   
    plt.plot(traj1[0,:], traj1[1,:], color=clS[:,iep], lw=2.)
    ax.set_xticks([-1, 0, 1])
    ax.set_yticks([-1, 0, 1])
    ax.set_ylim([np.min(kaps2), np.max(kaps2)])
    ax.set_xlim([np.min(kaps1), np.max(kaps1)])
    plt.savefig('Th_FigS3_2_C_'+str(eps)+'_1.pdf')

#%%
fig = plt.figure(figsize = [1.5*2.2 , 1.*2.])
ax = fig.add_subplot(111) 
freq = np.zeros_like(time)
pers = np.zeros_like(epss)
for iep, eps in enumerate(epss):
    #plt.plot(time, np.arctan2(traj1s[0,:,iep],traj1s[1,:,iep]), color=clS[:,iep])
    plt.plot(time, traj1s[1,:,iep], color=clS[:,iep])
    freq = 2*(traj1s[0,:,iep]>0)-1
    time_m1 = time[:-1]
    pers[iep] = 2*np.mean(np.diff(time_m1[np.abs(np.diff(freq))>0]))
print(pers)
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.set_yticks([-1, 0, 1])
ax.set_xticks([0, 10, 20, 30])
ax.set_xlim([0, 32])
ax.yaxis.set_ticks_position('left')
ax.xaxis.set_ticks_position('bottom')  
plt.ylabel('$\kappa_2$')
plt.xlabel(r'time ($\tau$)')
plt.savefig('Th_FigS3_2_C_2.pdf')
#%%

fig = plt.figure()
ax = fig.add_subplot(111) 
for iep, eps in enumerate(epss):
    fp1 = fp1s[iep]
    fp2 = fp2s[iep]
    u = us[:,iep]
    v = vs[:,:,iep]
    
    plt.scatter([ 0, ], [0], s=50, edgecolor='k', facecolor='w', linewidth=1., zorder=4)   
    plt.plot(traj1s[0,:,iep], traj1s[1,:,iep], color=clS[:,iep], lw=1.5)
    plt.plot(-traj1s[0,:,iep], -traj1s[1,:,iep], color=clS[:,iep], lw=1.5)
    
    plt.plot(traj2s[0,:,iep], traj2s[1,:,iep], color=clS[:,iep], lw=1.5)
    plt.plot(-traj2s[0,:,iep], -traj2s[1,:,iep], color=clS[:,iep], lw=1.5)
    
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.set_xticks([-1, 0, 1])
ax.set_yticks([-1, 0, 1])
ax.yaxis.set_ticks_position('left')
ax.xaxis.set_ticks_position('bottom')  
plt.xlabel('$\kappa_1$')
plt.ylabel('$\kappa_2$')
plt.savefig('Th_FigS3_C.pdf')

#%%
fig = plt.figure()
ax = fig.add_subplot(111) 

l1 = -0.5
l2 = 1.2
plt.plot([l1, l2], [0,0], 'k', lw=0.5)
plt.plot( [0,0],[l1, l2], 'k', lw=0.5)

R = 1.13
theta= np.linspace(0,2*np.pi)
plt.plot(R*np.cos(theta), R*np.sin(theta), color='k', lw=0.6)
plt.plot()
epss2 = epss[::-1]
clS2 = clS[:,::-1]
for iep, eps in enumerate(epss):
    Sigma[0,0] = 1.4
    Sigma[1,1] = 1.4
    Sigma[0,1] = 1.*eps
    Sigma[1,0] = -1./eps
        
    u, v = np.linalg.eig(Sigma)
    v1 = np.real(v[:,0])
    fac = np.sqrt(np.sum(v1**2))
    v1 = v1/fac
    print(v1)
    print(v2)
    print('--')
    
    v2 = -np.imag(v[:,1])/fac

    cC = np.array((1, 1, 1,))*0.3

    ax.arrow(0, 0, v1[0], v1[1],   fc=cC, ec='k', alpha =0.8, width=0.04,
                      head_width=0.13, head_length=0.13, zorder=3)
    ax.arrow(0, 0, v2[0], v2[1],   fc=clS[:,iep], ec='k', alpha =0.8,  width=0.04,
                      head_width=0.13, head_length=0.13, zorder=3)
    
#    ax.text(0.8, -0.4, r'Re$(\bf{u})$', fontsize = 12)
#    ax.text(0.2, 1.2, r'Im$(\bf{u})$', fontsize = 12)
    
    ax.text(0.5, -0.2, r'Re$( \bf{u})$', fontsize = 15)
    ax.text(0.1, 0.8, r'Im$( \bf{u})$', fontsize = 15)   
    ax.set_xlim([l1, l2])
    ax.set_ylim([l1, l2])
    
    ax.axis('off')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_visible(False)
    ax.spines['bottom'].set_visible(False)
    
    plt.savefig('Th_FigS3_B.pdf')

    
#%%
Sigma2 = np.zeros_like(Sigma)
Sigma2[:,:] = Sigma
Sigma3 = np.zeros_like(Sigma)*np.nan
Sigma3[1,0] = Sigma[1,0]

Sigma2[1, 0] = np.nan

plt.rcParams["axes.grid"] = False
fig = plt.figure(figsize = [2.0, 2.0])
ax = fig.add_subplot(111) 
plt.imshow(Sigma2, cmap='OrRd', vmin = 0, vmax = 7)
plt.imshow(Sigma3, cmap='PuBu', vmin = -0.7, vmax = 0)

ax.tick_params(color='white')


for i in range(np.shape(Sigma)[0]):
    for j in range(np.shape(Sigma)[1]):
        if i==1 and j==0:
            ax.text(i, j, ' ', va='center', ha='center', fontsize=16)
        elif i==0 and j==1:
            ax.text(i, j, ' ', va='center', ha='center', fontsize=16)
                
        else:
            ax.text(i, j, str(Sigma2[j,i]), va='center', ha='center', fontsize=16)
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.spines['left'].set_visible(False)
ax.spines['bottom'].set_visible(False)

ax.yaxis.set_ticks_position('right')
ax.xaxis.set_ticks_position('top')  
ax.set_xticks([0, 1])
ax.set_yticks([0, 1])


ax.set_xticklabels([r'$m_i^{\left(1\right)}$', r'$m_i^{\left(2\right)}$'], fontsize=14)
ax.set_yticklabels([r'$n_i^{\left(1\right)}$', r'$n_i^{\left(2\right)}$'], fontsize=14)

plt.savefig('Th_FigS3_A1.pdf')
plt.show()


#%%
fig = plt.figure(figsize = [2.0, 1.1])
ax = fig.add_subplot(111) 

clS = np.zeros((3, len(epss)))

cl2 = np.array(( 225/256, 74/256, 51/256))
cl1 = np.array(( 0.5, 0.5, 0.5))

nL = len(epss)-1
for iep, eps in enumerate(epss[:-1]):
    cl11 = cl1* (nL-iep)/nL   + cl2* (iep)/nL
    cl22 = cl1* (nL-iep-1)/nL   + cl2* (iep+1)/nL
    
    int_points = 20
    ps = np.linspace(eps, epss[iep+1], 10)
    nP = len(ps)-1
    plt.scatter(eps, 1.18, s=40, marker='v', color='k', edgecolor=[0.8, 0.8, 0.8], zorder=3)
    plt.scatter(epss[iep+1], 1.18, s=40, marker='v',  color='k', edgecolor=[0.8, 0.8, 0.8], zorder=3)
    
    for ip , p in enumerate(ps[:-1]):
        
        cc = cl11* (nP-ip)/nP   + cl22* (ip)/nP
       
        plt.fill_between([p, ps[ip+1]], [0, 0], [1, 1], color=cc)

ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.spines['left'].set_visible(False)
ax.spines['bottom'].set_visible(False)
    
ax.set_yticks([])
ax.set_xticks([1, 2, 3])
ax.set_xlabel('')

ax.set_xlim([0.9,3.1])
ax.set_ylim([-0.05,1.4])
ax.plot([1,3], [1,1], c='k')
ax.plot([1,1], [0,1], c='k')
ax.plot([3,3], [0,1], c='k')
ax.plot([1,3], [0.0,0.0], c='k', zorder=2)

plt.tight_layout()
plt.savefig('Th_FigS3_A2.pdf')
plt.show()