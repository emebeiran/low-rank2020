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
Sigma = np.zeros((2,2))
Sigma[0,0] = 1.6
Sigma[1,1] = 1.6
Sigma[0,1] = 0.
N = 1000
S=10
M = np.random.randn(N,2)
M = M/np.std(M,0)
ss2 = 0.3

N = np.dot(Sigma, M.T)+ss2*np.random.randn(2,N)

fig = plt.figure(figsize=[3.2, 3.2], dpi=450)
gs = GridSpec(5,5)

ax_joint00 = fig.add_subplot(gs[1:3,0:2])
ax_joint01 = fig.add_subplot(gs[1:3,2:4])
ax_joint10 = fig.add_subplot(gs[3:5,0:2])
ax_joint11 = fig.add_subplot(gs[3:5,2:4])

ax_marg_x0 = fig.add_subplot(gs[0,0:2])
ax_marg_x1 = fig.add_subplot(gs[0,2:4])

ax_marg_y0 = fig.add_subplot(gs[1:3,4])
ax_marg_y1 = fig.add_subplot(gs[3:5,4])

ax_joint00.scatter(M[:,0], N[0,:], s=S, alpha=0.5, label=r'$\sigma_{mn} = 1.2$', rasterized=True)
ax_joint00.plot(ms, Sigma[0,0]*ms, '--', c='k', lw=1)
ax_joint00.set_xlim([-3,3])
ax_joint00.set_xticks([-2., 0, 2.])
ax_joint00.set_xticklabels(['','',''])
ax_joint00.set_ylim([-5.5,5.5])
ax_joint00.set_yticks([-5, 0, 5])
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
ax_joint01.set_ylim([-5.5,5.5])
ax_joint01.set_yticks([-5, 0, 5])
ax_joint01.set_yticklabels(['','',''])
ax_joint01.set_xlim([-3,3])
ax_joint01.set_xticks([-2., 0, 2.])
ax_joint01.set_xticklabels(['','',''])

ax_joint10.scatter(M[:,0], N[1,:], s=S, alpha=0.5, label=r'$\sigma_{mn} = 1.2$', rasterized=True)
ax_joint10.plot(ms, Sigma[1,0]*ms, '--', c='k', lw=1)
ax_joint10.set_xlim([-3,3])
ax_joint10.spines['top'].set_visible(False)
ax_joint10.spines['right'].set_visible(False)
ax_joint10.yaxis.set_ticks_position('left')
ax_joint10.xaxis.set_ticks_position('bottom')
ax_joint10.set_ylim([-5.5,5.5])
ax_joint10.set_yticks([-5, 0, 5])
ax_joint10.set_xlim([-3,3])
ax_joint10.set_xticks([-2., 0, 2.])
ax_joint10.set_ylabel(r'$n^{\left(2\right)}_i$')
ax_joint10.set_xlabel(r'$m^{\left(1\right)}_i$')

ax_joint11.scatter(M[:,1], N[1,:], s=S, alpha=0.5, label=r'$\sigma_{mn} = 1.2$', rasterized=True)
ax_joint11.plot(ms, Sigma[1,1]*ms, '--', c='k', lw=1)
ax_joint11.set_xlim([-3,3])
ax_joint11.spines['top'].set_visible(False)
ax_joint11.spines['right'].set_visible(False)
ax_joint11.set_ylim([-5.5,5.5])
ax_joint11.set_yticks([-5, 0, 5])
ax_joint11.set_xticks([-2., 0, 2.])
ax_joint11.set_xlim([-3,3])
ax_joint11.set_yticklabels(['','',''])
ax_joint11.yaxis.set_ticks_position('left')
ax_joint11.xaxis.set_ticks_position('bottom')
ax_joint11.set_xlabel(r'$m^{\left(2\right)}_i$')

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

ax_marg_y0.hist(N[0,:], nbins, orientation="horizontal", alpha=0.5, density=True)
ss = np.sqrt(Sigma[0,0]**2+ss2**2)
ax_marg_y0.plot((1/np.sqrt(2*np.pi*ss**2))*np.exp(-(ms)**2/(2*ss**2)), ms, 'k')
ax_marg_y0.spines['top'].set_visible(False)
ax_marg_y0.spines['right'].set_visible(False)
ax_marg_y0.spines['bottom'].set_visible(False)
ax_marg_y0.yaxis.set_ticks_position('left')
ax_marg_y0.xaxis.set_ticks_position('bottom')
ax_marg_y0.set_ylim([-5.5,5.5])
ax_marg_y0.set_xlim([0,0.45])
ax_marg_y0.set_yticks([-5., 0, 5.])
ax_marg_y0.set_yticklabels(['','',''])
ax_marg_y0.set_xticks([1])
ax_marg_y0.set_xticklabels([''])

ax_marg_y1.hist(N[1,:], nbins, orientation="horizontal", alpha=0.5, density=True)
ss = np.sqrt(Sigma[1,1]**2+ss2**2)
ax_marg_y1.plot((1/np.sqrt(2*np.pi*ss**2))*np.exp(-(ms)**2/(2*ss**2)), ms, 'k')
ax_marg_y1.spines['top'].set_visible(False)
ax_marg_y1.spines['right'].set_visible(False)
ax_marg_y1.spines['bottom'].set_visible(False)
ax_marg_y1.yaxis.set_ticks_position('left')
ax_marg_y1.xaxis.set_ticks_position('bottom')
ax_marg_y1.set_ylim([-5.5,5.5])
ax_marg_y1.set_xlim([0,0.45])
ax_marg_y1.set_yticks([-5., 0, 5.])
ax_marg_y1.set_yticklabels(['','',''])
ax_marg_y1.set_xticks([1])
ax_marg_y1.set_xticklabels([''])

plt.savefig('Th_Fig22_3_A.pdf')

#%%
plt.rcParams["axes.grid"] = False
fig = plt.figure(figsize = [2.0, 2.0])
ax = fig.add_subplot(111) 
plt.imshow(Sigma, cmap='OrRd', vmin = 0, vmax = 4)
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
ax.set_xticks([0, 1])
ax.set_yticks([0, 1])


ax.set_xticklabels([r'$m_i^{\left(1\right)}$', r'$m_i^{\left(2\right)}$'], fontsize=14)
ax.set_yticklabels([r'$n_i^{\left(1\right)}$', r'$n_i^{\left(2\right)}$'], fontsize=14)

plt.savefig('Th_Fig22_3_B.pdf')


#%%
fig = plt.figure()
ax = fig.add_subplot(111) 

u, v = np.linalg.eig(Sigma)
l1 = -0.5
l2 = 2.
cC = np.array((1, 1, 1,))*0.3
plt.plot([l1, l2], [0,0], 'k', lw=0.5)
plt.plot( [0,0],[l1, l2], 'k', lw=0.5)
ax.arrow(0, 0, u[0]*v[0,0], u[0]*v[1,0],   fc=cC, ec=cC, alpha =0.8, width=0.06,
                  head_width=0.2, head_length=0.2)
ax.arrow(0, 0, u[1]*v[0,1], u[1]*v[1,1],   fc=cC, ec=cC, alpha =0.8,  width=0.06,
                  head_width=0.2, head_length=0.2)

ax.text(0.8, 0.2, r'$\lambda_1 \bf{u}_1$', fontsize = 18)
ax.text(0.55, 1., r'$\lambda_2 \bf{u}_2$', fontsize = 18)


ax.set_xlim([l1, l2])
ax.set_ylim([l1, l2])


ax.axis('off')
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.spines['left'].set_visible(False)
ax.spines['bottom'].set_visible(False)

plt.savefig('Th_Fig22_3_B1.pdf')

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

search_kap2 = np.linspace(0.2, 1.3, 300)
E_2 = np.zeros_like(search_kap1)
for ik2 ,k2 in enumerate(search_kap2):
    K = v[:,1]*k2
    kSS = - K+ np.dot(Sigma, transf(K))
    E_2[ik2] = np.sqrt(np.sum(kSS**2))
fp2 = search_kap2[np.argmin(E_2)]

fig = plt.figure()
ax = fig.add_subplot(111) 
im = plt.pcolor(kaps1, kaps2, np.log10(E).T, cmap ='viridis', vmin = -2.,vmax=0, shading='auto')

#cbar = ax.figure.colorbar(im, ax=ax)
#cbar.set_ticks([-2, -1, 0])
strm = ax.streamplot(kaps1, kaps2, ksol[:,:,0].T, ksol[:,:,1].T, color='w', linewidth=1, cmap='autumn', density=0.6)
#cbar.set_label(r'$\log_{10}$ speed', rotation=270, labelpad=18)
plt.xlabel('$\kappa_1$')
plt.ylabel('$\kappa_2$')
plt.scatter([ 0, ], [0,], s=50, edgecolor='k', facecolor='w', linewidth=1., zorder=4)
#plt.scatter( [v[0,1]*fp2,-v[0,1]*fp2], [v[1,1]*fp2,-v[1,1]*fp2], s=100, edgecolor='w', facecolor='k', linewidth=1.5, zorder=4)
ths = np.linspace(0, np.pi*2)

plt.plot(fp1*np.cos(ths), fp1*np.sin(ths), c='w', lw=3)
plt.plot(fp1*np.cos(ths), fp1*np.sin(ths), c='k', lw=1)

ax.set_xticks([-1, 0, 1])
ax.set_yticks([-1, 0, 1])
ax.set_ylim([np.min(kaps2), np.max(kaps2)])
ax.set_xlim([np.min(kaps1), np.max(kaps1)])
plt.savefig('Th_Fig22_3_C1.pdf')



#%%
np.random.seed(66779)
cC = np.array((1, 1, 1,))*0.3

fig = plt.figure()
ax = fig.add_subplot(111) 

plt.xlabel('$\kappa_1$')
plt.ylabel('$\kappa_2$')
plt.scatter([ 0,], [0,], s=50, edgecolor='k', facecolor='w', linewidth=1., zorder=4)
#plt.scatter( [v[0,1]*fp2,-v[0,1]*fp2], [v[1,1]*fp2,-v[1,1]*fp2], s=100, edgecolor='w', facecolor='k', linewidth=1.5, zorder=4)
#plt.plot(fp1*np.cos(ths), fp1*np.sin(ths), c='w', lw=2)
plt.plot(fp1*np.cos(ths), fp1*np.sin(ths), '--k', lw=1)
ax.set_xticks([-1, 0, 1])
ax.set_yticks([-1, 0, 1])
ax.set_ylim([np.min(kaps2), np.max(kaps2)])
ax.set_xlim([np.min(kaps1), np.max(kaps1)])

Nn = 1000
SigmaTot = np.eye(4)
SigmaTot[2,2] = 4
SigmaTot[3,3] = 4

SigmaTot[0,2] = Sigma[0,0]
SigmaTot[0,3] = Sigma[1,0]
SigmaTot[1,2] = Sigma[0,1]
SigmaTot[1,3] = Sigma[1,1]

SigmaTot[2,0] = SigmaTot[0,2]
SigmaTot[3,0] = SigmaTot[0,3]
SigmaTot[2,1] = SigmaTot[1,2]
SigmaTot[3,1] = SigmaTot[1,3]

Mu= np.zeros((4,1))

inkap1 = np.linspace(-1, 1, 4)
inkap2 = np.linspace(-1.1, 1.1001, 5)

dt = 0.8
time = np.arange(0, 280, dt)

for trials in range(2):
    try_s0 = 100
    for tr in range(20):
        XX = np.random.multivariate_normal(Mu[:,0], SigmaTot, size=Nn)
        try_s = np.sum((np.dot(XX.T,XX)/1000-SigmaTot)**2)
        if try_s < try_s0:
            #print(try_s)
            try_s0 = try_s
            XX_s = XX
    M = XX_s[:,0:2]
    N = XX_s[:,2:4]
    
    J = np.dot(M, N.T)/Nn
    
    cC =  0.6+0.2*np.random.rand()-0.1
    
    for ik1, ink1 in enumerate(inkap1):
        for ik2, ink2 in enumerate(inkap2):
            sk1 = np.zeros_like(time)
            sk2 = np.zeros_like(time)
            
            x0 = ink1*M[:,0] + ink2*M[:,1]
            sk1[0] = np.mean(M[:,0]*x0)
            sk2[0] = np.mean(M[:,1]*x0)
            
            for it, ti in enumerate(time[:-1]):
                x = x0 + dt*(-x0 + np.dot(J, np.tanh(x0)))
                sk1[it+1] = np.mean(M[:,0]*x)
                sk2[it+1] = np.mean(M[:,1]*x)
                x0 = x
            plt.plot(sk1, sk2, c=[cC, cC, cC])
            plt.scatter(sk1[0], sk2[0], s=15, facecolor=[cC, cC, cC])
            if trials==0:
                cC = 0.3
                plt.plot(sk1, sk2, c=[cC, cC, cC])
                plt.scatter(sk1[0], sk2[0], s=15, facecolor=[cC, cC, cC])
                plt.scatter(sk1[-1], sk2[-1], s=40, facecolor=[cC, cC, cC], edgecolor='k', linewidth=1., zorder=5)
            elif trials==1:
                cC = 0.6
                plt.plot(sk1, sk2, c=[cC, cC, cC])
                plt.scatter(sk1[0], sk2[0], s=15, facecolor=[cC, cC, cC])
                plt.scatter(sk1[-1], sk2[-1], s=40, facecolor=[cC, cC, cC], edgecolor='k', linewidth=1., zorder=5)
            
            
ax.set_xticks([-1, 0, 1])
ax.set_yticks([-1, 0, 1])
ax.set_ylim([np.min(kaps2), np.max(kaps2)])
ax.set_xlim([np.min(kaps1), np.max(kaps1)])
        
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.yaxis.set_ticks_position('left')
ax.xaxis.set_ticks_position('bottom')  

plt.savefig('Th_Fig22_3_D.pdf')    
    