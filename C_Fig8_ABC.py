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

epss = np.array((0, 0.1, 0.2, 0.4, 0.6, 1.0, 2.0))

fp1s = np.zeros_like(epss)
fp2s = np.zeros_like(epss)
us = np.zeros((2,len(epss)))
vs = np.zeros((2,2,len(epss)))
u3s = np.zeros((2,len(epss)))
v3s = np.zeros((2,2,len(epss)))

dt = 0.1
T = 40
time = np.arange(0, T, dt)
traj1s = np.zeros((2,len(time), len(epss)))
traj2s = np.zeros((2,len(time), len(epss)))

clS = np.zeros((3, len(epss)))

cl2 = np.array(( 225/256, 74/256, 51/256))
cl1 = np.array(( 0.5, 0.5, 0.5))

nL = len(epss)-1
for iep, eps in enumerate(epss):
    clS[:,iep] = cl1* (nL-iep)/nL   + cl2* (iep)/nL

for iep, eps in enumerate(epss):
    ms = np.linspace(-5,5,100)
    Sigma = np.zeros((2,2))
    Sigma[0,0] = 1.2
    Sigma[1,1] = 1.6
    Sigma[0,1] = eps
    
    
    #%%

    
    u, v = np.linalg.eig(Sigma)
    us[:,iep] = u
    vs[:,:,iep] = v
    
    l1 = -0.5
    l2 = 2.
    cC = np.array((1, 1, 1,))*0.3

    
    #%%
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
            
    
    
    search_kap1 = np.linspace(0.2, 1.3, 300)
    E_1 = np.zeros_like(search_kap1)
    for ik1 ,k1 in enumerate(search_kap1):
        K = v[:,0]*k1
        kSS = - K+ np.dot(Sigma, transf(K))
        E_1[ik1] = np.sqrt(np.sum(kSS**2))
    fp1 = search_kap1[np.argmin(E_1)]
    fp1s[iep] = fp1
    
    search_kap2 = np.linspace(0.2, 1.3, 300)
    E_2 = np.zeros_like(search_kap1)
    for ik2 ,k2 in enumerate(search_kap2):
        K = v[:,1]*k2
        kSS = - K+ np.dot(Sigma, transf(K))
        E_2[ik2] = np.sqrt(np.sum(kSS**2))
    fp2 = search_kap2[np.argmin(E_2)]
    fp2s[iep] = fp2
    
    RR = np.array((v[0,0]*fp1, v[1,0]*fp1))
    RR = RR[:,None]
    Triple = Trip(0, fp1**2)
    S2 = -np.eye(2)+Sigma/u[0] + Triple*u[0]*np.dot(RR,RR.T)
    
    u2, v2 = np.linalg.eig(S2)
    print(u2)
    k1 = RR[:,0]

    traj1 = np.zeros((2,len(time)))
    traj2 = np.zeros((2,len(time)))
    traj1[:,0] = RR[:,0] + 0.01 * v2[:,1]
    traj2[:,0] = RR[:,0] - 0.01* v2[:,1]
    
    for it, ti in enumerate(time[:-1]):
        traj1[:,it+1] = traj1[:,it] + dt*(-traj1[:,it] + np.dot(Sigma,transf(traj1[:,it])))
        traj2[:,it+1] = traj2[:,it] + dt*(-traj2[:,it] + np.dot(Sigma,transf(traj2[:,it])))
    
    traj1s[:,:,iep] = traj1
    traj2s[:,:,iep] = traj2
    
    
    RR = np.array((v[0,1]*fp2, v[1,1]*fp2))
    RR = RR[:,None]
    Triple = Trip(0, fp2**2)
    S3 = -np.eye(2)+Sigma/u[1] + Triple*u[1]*np.dot(RR,RR.T)

    u3, v3 = np.linalg.eig(S3)
    print(u3)
    print('--')
    u3s[:,iep] = u3
    v3s[:,:,iep] = v3
    
    
    fig = plt.figure()
    ax = fig.add_subplot(111) 
    im = plt.pcolor(kaps1, kaps2, np.log10(E).T, cmap ='viridis', vmin = -2.,vmax=0, shading= 'auto')
    

    strm = ax.streamplot(kaps1, kaps2, ksol[:,:,0].T, ksol[:,:,1].T, color='w', linewidth=1, cmap='autumn', density=0.6)
    plt.xlabel('$\kappa_1$')
    plt.ylabel('$\kappa_2$')
#    plt.scatter([fp1, 0, -fp1], [0,0,0], s=50, edgecolor='k', facecolor='w', linewidth=1., zorder=4)
#    plt.scatter( [v[0,1]*fp2,-v[0,1]*fp2], [v[1,1]*fp2,-v[1,1]*fp2], s=100, edgecolor='w', facecolor='k', linewidth=1.5, zorder=4)
#    
    plt.scatter([fp1, 0, -fp1], [0,0,0], s=50, edgecolor='k', facecolor='w', linewidth=1., zorder=4)
    plt.scatter( [v[0,1]*fp2,-v[0,1]*fp2], [v[1,1]*fp2,-v[1,1]*fp2], s=80, edgecolor='w', facecolor=clS[:,iep], linewidth=1.5, zorder=4)
    
    plt.plot(traj1s[0,:,iep], traj1s[1,:,iep], color=clS[:,iep], lw=1.5)
    plt.plot(-traj1s[0,:,iep], -traj1s[1,:,iep], color=clS[:,iep], lw=1.5)
    
    plt.plot(traj2s[0,:,iep], traj2s[1,:,iep], color=clS[:,iep], lw=1.5)
    plt.plot(-traj2s[0,:,iep], -traj2s[1,:,iep], color=clS[:,iep], lw=1.5)
    
    ax.set_xticks([-1, 0, 1])
    ax.set_yticks([-1, 0, 1])
    ax.set_ylim([np.min(kaps2), np.max(kaps2)])
    ax.set_xlim([np.min(kaps1), np.max(kaps1)])
    plt.savefig('Th_FigS2_2_C1_'+str(eps)+'.pdf')
    
#%%
    

fig = plt.figure()
ax = fig.add_subplot(111) 
for iep, eps in enumerate(epss):
    fp1 = fp1s[iep]
    fp2 = fp2s[iep]
    u = us[:,iep]
    v = vs[:,:,iep]
    
    plt.scatter([fp1, 0, -fp1], [0,0,0], s=50, edgecolor='k', facecolor='w', linewidth=1., zorder=4)
    plt.scatter( [v[0,1]*fp2,-v[0,1]*fp2], [v[1,1]*fp2,-v[1,1]*fp2], s=80, edgecolor='w', facecolor=clS[:,iep], linewidth=1.5, zorder=4)
    
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
plt.savefig('Th_FigS2_C.pdf')

#%%
fig = plt.figure()
ax = fig.add_subplot(111) 

plt.plot([l1, l2], [0,0], 'k', lw=0.5)
plt.plot( [0,0],[l1, l2], 'k', lw=0.5)
for iep, eps in enumerate(epss):
    Sigma = np.zeros((2,2))
    Sigma[0,0] = 1.2
    Sigma[1,1] = 1.6
    Sigma[0,1] = eps
    
    
    u, v = np.linalg.eig(Sigma)
    l1 = -1.
    l2 = 2.
    cC = np.array((1, 1, 1,))*0.3

    ax.arrow(0, 0, u[0]*v[0,0], u[0]*v[1,0],   fc=cC, ec=cC, alpha =0.8, width=0.06,
                      head_width=0.2, head_length=0.2, zorder=3)
    ax.arrow(0, 0, u[1]*v[0,1], u[1]*v[1,1],   fc=clS[:,iep], ec='k', alpha =0.8,  width=0.06,
                      head_width=0.2, head_length=0.2, zorder=3)
    
    ax.text(0.8, -0.4, r'$\lambda_1 \bf{u}_1$', fontsize = 15)
    ax.text(1., 1.5, r'$\lambda_2 \bf{u}_2$', fontsize = 15)
    
    
    ax.set_xlim([l1, l2])
    ax.set_ylim([l1, l2])
    
    ax.axis('off')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_visible(False)
    ax.spines['bottom'].set_visible(False)
    
    plt.savefig('Th_FigS2_B.pdf')
#%%
Sigma2 = np.zeros_like(Sigma)
Sigma2[:,:] = Sigma


plt.rcParams["axes.grid"] = False
fig = plt.figure(figsize = [2.0, 2.0])
ax = fig.add_subplot(111) 
plt.imshow(Sigma2, cmap='OrRd', vmin = 0, vmax = 4)
ax.tick_params(color='white')


for i in range(np.shape(Sigma)[0]):
    for j in range(np.shape(Sigma)[1]):
        if i==1 and j==0:
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

plt.savefig('Th_FigS2_A1.pdf')
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
ax.set_xticks([0, 1, 2])
ax.set_xlabel('')

ax.set_xlim([-0.1,2.1])
ax.set_ylim([-0.05,1.4])
ax.plot([0,2], [1,1], c='k')
ax.plot([0,0], [0,1], c='k')
ax.plot([2,2], [0,1], c='k')
ax.plot([0,2], [0.0,0.0], c='k', zorder=2)

plt.tight_layout()
plt.savefig('Th_FigS2_A2.pdf')
plt.show()