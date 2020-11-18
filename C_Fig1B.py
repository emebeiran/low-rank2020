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
# =============================================================================
#           Fig 2
# =============================================================================
ms = np.linspace(-8,8,300)

MuMs = np.zeros((2,2))
MuNs = np.zeros((2,2))

Sigma = np.zeros((2,2,2))+1e-10
Sigma[0,0,0] = 1.2
Sigma[1,1,0] = -1#1.6
Sigma[1,0,0] = 0.5
Sigma[0,1,0] = -0.5


MuMs[0,0] = 0.5#1.
MuMs[1,0] = -0.5#1.
MuNs[0,0] = 1.5
MuNs[1,0] = 3.


Sigma[0,0,1] = -0.3
Sigma[1,1,1] = 1.
Sigma[0,1,1] = 1.2
Sigma[1,1,1] = 0.8

MuMs[0,1] = -0.5#0.5
MuMs[1,1] = 0.5#-0.8
MuNs[0,1] = -1.5#1.
MuNs[1,1] = -3#-1.
#
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

pops = 2
Ns = 100
M = np.random.randn(Ns*pops,2)
N = np.random.randn(2,Ns*pops)
pops=2
BigSigma = np.zeros((4,4, pops))
for ip in range(pops):
    
    S=10
    
    M = M/np.std(M,0)
    ss2 = 0.3
    
    BigSigma[0,0,ip] = 1.-np.mean(MuMs[0, :]**2)
    BigSigma[1,1,ip] = 1.-np.mean(MuMs[1, :]**2)
    
    BigSigma[0,2,ip] = Sigma[0,0,ip]
    BigSigma[0,3,ip] = Sigma[0,1,ip]
    BigSigma[2,0,ip] = Sigma[0,0,ip]
    BigSigma[3,0,ip] = Sigma[0,1,ip]
    
    BigSigma[1,2,ip] = Sigma[1,0, ip]
    BigSigma[1,3,ip] = Sigma[1,1, ip]
    BigSigma[2,1,ip] = Sigma[1,0, ip]
    BigSigma[3,1,ip] = Sigma[1,1, ip]
    
    BigSigma[2,2,ip] = 1.
    BigSigma[3,3,ip] = 1.
    
    sol = np.min(np.real(np.linalg.eigvalsh(BigSigma[:,:,ip])))
    
    
    while sol<0:
        BigSigma[2,2] *= 1.01
        BigSigma[3,3] *= 1.01
        sol = np.min(np.real(np.linalg.eigvals(BigSigma[:,:,ip])))
    BigMean = 0*np.hstack((MuMs[:,ip], MuNs[:,ip]))
    
    err0 = 100.
    Dat = np.random.multivariate_normal(BigMean, BigSigma[:,:,ip], Ns)
    DatS = Dat
    for trials in range(200):
        Dat = np.random.multivariate_normal(BigMean, BigSigma[:,:,ip], Ns)
        err1 = np.std(np.cov(Dat.T)-BigSigma[:,:,ip])
        if err1<err0:
            print(err1)
            err0 = err1
        DatS = Dat
    
    Dat = DatS
    BigMean = np.hstack((MuMs[:,ip], MuNs[:,ip]))
    for i in range(4):
        Dat[:,i] = Dat[:,i]+BigMean[i]-np.mean(Dat[:,i])
    
    
    M[ip*Ns:(ip+1)*Ns,0] = Dat[:,0]
    M[ip*Ns:(ip+1)*Ns,1] = Dat[:,1]
    
    N[0,ip*Ns:(ip+1)*Ns] = Dat[:,2]
    N[1,ip*Ns:(ip+1)*Ns] = Dat[:,3]


ssiz = 30

color1 = np.array((31, 127, 17))/256
color2 = np.array((129, 34, 141))/256
clrs = np.zeros((3,2))
clrs[:,0] = color1
clrs[:,1] = color2
pops=2

nbins=20
for ip in range(pops):
    ax_joint00.scatter(M[ip*Ns:(ip+1)*Ns,0], N[0,ip*Ns:(ip+1)*Ns], s=S, color = clrs[:,ip], alpha=0.5, rasterized=True)
    ax_joint00.plot(ms, np.sqrt(BigSigma[2,2,ip]/BigSigma[0,0,ip])*Sigma[0,0,ip]*(ms- MuMs[0,ip]) +MuNs[0,ip], '--', c='k', lw=0.5)
    ax_joint00.scatter( MuMs[0,ip], +MuNs[0,ip], s=ssiz, edgecolor='k', facecolor= 'w',zorder=5)
    
    ax_joint00.set_xlim([-4.5,4.5])
    ax_joint00.set_xticks([-4., 0, 4.])
    ax_joint00.set_xticklabels(['','',''])
    ax_joint00.set_ylim([-6.5,6.5])
    ax_joint00.set_yticks([-6, 0, 6])
    ax_joint00.set_ylabel(r'$n^{\left(1\right)}_i$')
    ax_joint00.spines['top'].set_visible(False)
    ax_joint00.spines['right'].set_visible(False)
    ax_joint00.yaxis.set_ticks_position('left')
    ax_joint00.xaxis.set_ticks_position('bottom')
               
    ax_joint01.scatter(M[ip*Ns:(ip+1)*Ns,1], N[0,ip*Ns:(ip+1)*Ns], s=S, color = clrs[:,ip],alpha=0.5, rasterized=True)
    ax_joint01.plot(ms, np.sqrt(BigSigma[2,2,ip]/BigSigma[1,1,ip])*Sigma[1,0,ip]*(ms- MuMs[1,ip]) +MuNs[0,ip], '--', c='k', lw=0.5)
    ax_joint01.scatter( MuMs[1,ip], +MuNs[0,ip], s=ssiz, edgecolor='k', facecolor= 'w',zorder=5)
    ax_joint01.spines['top'].set_visible(False)
    ax_joint01.spines['right'].set_visible(False)
    ax_joint01.yaxis.set_ticks_position('left')
    ax_joint01.xaxis.set_ticks_position('bottom')
    ax_joint01.set_ylim([-6.5,6.5])
    ax_joint01.set_yticks([-6, 0, 6])
    ax_joint01.set_yticklabels(['','',''])
    ax_joint01.set_xlim([-4.5,4.5])
    ax_joint01.set_xticks([-4., 0, 4.])
    ax_joint01.set_xticklabels(['','',''])
    
    ax_joint10.scatter(M[ip*Ns:(ip+1)*Ns,0], N[1,ip*Ns:(ip+1)*Ns], s=S, color = clrs[:,ip],alpha=0.5, rasterized=True)
    ax_joint10.plot(ms, np.sqrt(BigSigma[3,3,ip]/BigSigma[0,0,ip])*Sigma[0,1,ip]*(ms- MuMs[0,ip]) +MuNs[1,ip], '--', c='k', lw=0.5)
    ax_joint10.scatter( MuMs[0,ip], +MuNs[1,ip], s=ssiz, edgecolor='k', facecolor= 'w',zorder=5)
    ax_joint10.spines['top'].set_visible(False)
    ax_joint10.spines['right'].set_visible(False)
    ax_joint10.yaxis.set_ticks_position('left')
    ax_joint10.xaxis.set_ticks_position('bottom')
    ax_joint10.set_ylim([-6.5,6.5])
    ax_joint10.set_yticks([-6, 0, 6])
    ax_joint10.set_xlim([-4.5,4.5])
    ax_joint10.set_xticks([-4., 0, 4.])

    ax_joint10.set_ylabel(r'$n^{\left(2\right)}_i$')
    ax_joint10.set_xlabel(r'$m^{\left(1\right)}_i$')
    
    ax_joint11.scatter(M[ip*Ns:(ip+1)*Ns,1], N[1,ip*Ns:(ip+1)*Ns], s=S, color = clrs[:,ip],alpha=0.5, rasterized=True)
    ax_joint11.plot(ms, np.sqrt(BigSigma[3,3,ip]/BigSigma[1,1,ip])*Sigma[1,1,ip]*(ms- MuMs[1,ip]) +MuNs[1,ip], '--', c='k', lw=0.5)
    ax_joint11.scatter( MuMs[1,ip], +MuNs[1,ip], s=ssiz, edgecolor='k', facecolor= 'w',zorder=5)
    ax_joint11.spines['top'].set_visible(False)
    ax_joint11.spines['right'].set_visible(False)
    ax_joint11.set_ylim([-6.5,6.5])
    ax_joint11.set_yticks([-6, 0, 6])
    ax_joint11.set_xlim([-4.5,4.5])
    ax_joint11.set_xticks([-4., 0, 4.])
    
    ax_joint11.set_yticklabels(['','',''])
    ax_joint11.yaxis.set_ticks_position('left')
    ax_joint11.xaxis.set_ticks_position('bottom')
    ax_joint11.set_xlabel(r'$m^{\left(2\right)}_i$')
    
    ax_marg_x0.hist(M[ip*Ns:(ip+1)*Ns,0], nbins, color = clrs[:,ip], alpha=0.5, density=True)
    ss = BigSigma[0,0,ip]
    ax_marg_x0.plot(ms, (1/np.sqrt(2*np.pi*ss))*np.exp(-(ms- MuMs[0,ip])**2/(2*ss)), color = clrs[:,ip],lw=0.5)
    
    ax_marg_x0.spines['top'].set_visible(False)
    ax_marg_x0.spines['right'].set_visible(False)
    ax_marg_x0.spines['left'].set_visible(False)
    ax_marg_x0.yaxis.set_ticks_position('left')
    ax_marg_x0.xaxis.set_ticks_position('bottom')
    ax_marg_x0.set_xlim([-4.5,4.5])
    ax_marg_x0.set_ylim([0,0.6])
    ax_marg_x0.set_xticks([-4., 0, 4.])
    ax_marg_x0.set_xticklabels(['','',''])
    ax_marg_x0.set_yticks([1])


    ax_marg_x1.hist(M[ip*Ns:(ip+1)*Ns,1], nbins,color = clrs[:,ip], alpha=0.5, density=True)
    ss = BigSigma[1,1,ip]
    ax_marg_x1.plot(ms, (1/np.sqrt(2*np.pi*ss))*np.exp(-(ms- MuMs[1,ip])**2/(2*ss)),  color = clrs[:,ip],lw=0.5)
    ax_marg_x1.spines['top'].set_visible(False)
    ax_marg_x1.spines['right'].set_visible(False)
    ax_marg_x1.spines['left'].set_visible(False)
    ax_marg_x1.yaxis.set_ticks_position('left')
    ax_marg_x1.xaxis.set_ticks_position('bottom')
    ax_marg_x1.set_xlim([-4.5,4.5])
    ax_marg_x1.set_ylim([0,0.6])
    ax_marg_x1.set_xticks([-4., 0, 4.])
    ax_marg_x1.set_xticklabels(['','',''])
    ax_marg_x1.set_yticks([1])


    ax_marg_y0.hist(N[0,ip*Ns:(ip+1)*Ns], nbins, orientation="horizontal", color = clrs[:,ip], alpha=0.5, density=True)
    ss = BigSigma[2,2,ip]
    ax_marg_y0.plot((1/np.sqrt(2*np.pi*ss))*np.exp(-(ms-MuNs[0,ip])**2/(2*ss)), ms,  color = clrs[:,ip],lw=0.5)
    ax_marg_y0.spines['top'].set_visible(False)
    ax_marg_y0.spines['right'].set_visible(False)
    ax_marg_y0.spines['bottom'].set_visible(False)
    ax_marg_y0.yaxis.set_ticks_position('left')
    ax_marg_y0.xaxis.set_ticks_position('bottom')
    ax_marg_y0.set_ylim([-6.5,6.5])
    ax_marg_y0.set_xlim([0,0.45])
    ax_marg_y0.set_yticks([-6., 0, 6.])
    ax_marg_y0.set_yticklabels(['','',''])
    ax_marg_y0.set_xticks([1])
    ax_marg_y0.set_xticklabels([''])

    ax_marg_y1.hist(N[1,ip*Ns:(ip+1)*Ns], nbins, orientation="horizontal", color = clrs[:,ip], alpha=0.5, density=True)
    ss = BigSigma[3,3,ip]
    ax_marg_y1.plot((1/np.sqrt(2*np.pi*ss))*np.exp(-(ms-MuNs[1,ip])**2/(2*ss)), ms, color = clrs[:,ip],lw=0.5)
    ax_marg_y1.spines['top'].set_visible(False)
    ax_marg_y1.spines['right'].set_visible(False)
    ax_marg_y1.spines['bottom'].set_visible(False)
    ax_marg_y1.yaxis.set_ticks_position('left')
    ax_marg_y1.xaxis.set_ticks_position('bottom')
    ax_marg_y1.set_ylim([-6.5,6.5])
    ax_marg_y1.set_xlim([0,0.45])
    ax_marg_y1.set_yticks([-6., 0, 6.])
    ax_marg_y1.set_yticklabels(['','',''])
    ax_marg_y1.set_xticks([1])
    ax_marg_y1.set_xticklabels([''])

plt.savefig('Th_Fig0_A.pdf')

#%%
Nneurs = 6

Madap = M[0:Nneurs*2,:]
Madap[Nneurs:,:] = M[-Nneurs:,:]

Nadap = N[:,0:Nneurs*2]
Nadap[:,Nneurs:] = N[:, -Nneurs:]
J = np.dot(Madap, Nadap)/(Ns*pops)
plt.rcParams["axes.grid"] = False
fig = plt.figure(figsize = [2.0, 2.0])
ax = fig.add_subplot(111) 

plt.imshow(J, cmap='coolwarm', vmin = -0.12, vmax = 0.12)
J22 = np.zeros_like(J)
J22[np.abs(J)>0]=np.nan
plt.imshow(J22, cmap='OrRd', vmin = 0, vmax = 0.12)

ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.spines['left'].set_visible(False)
ax.spines['bottom'].set_visible(False)

ax.yaxis.set_ticks_position('right')
ax.xaxis.set_ticks_position('top')  
ax.set_xticks([])
ax.set_yticks([])

plt.savefig('Th_Fig0_B_1.pdf')
#%%
for ip in range(pops):
    fig = plt.figure(figsize = [0.2, 2.0])
    ax = fig.add_subplot(111) 
    Madap1 = np.copy(Madap[:,ip,None])
    Madap2 = np.copy(Madap[:,ip,None])
    Madap1[Nneurs:,:] = np.nan
    Madap2[0:Nneurs,:] = np.nan
    Nadap1 = np.copy(Nadap[None,ip,:])
    Nadap2 = np.copy(Nadap[None,ip,:])
    Nadap1[:,Nneurs:] = np.nan
    Nadap2[:,0:Nneurs] = np.nan
    plt.imshow(Madap1, cmap='Greens')
    plt.imshow(Madap2, cmap='Purples')
    
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_visible(False)
    ax.spines['bottom'].set_visible(False)
    ax.yaxis.set_ticks_position('right')
    ax.xaxis.set_ticks_position('top')  
    ax.set_xticks([])
    ax.set_yticks([])
    plt.savefig('Th_Fig0_B_m'+str(ip+1)+'.pdf')
    
    fig = plt.figure(figsize = [2.0, 0.2])
    ax = fig.add_subplot(111) 
    plt.imshow(Nadap1, cmap='Greens')
    plt.imshow(Nadap2, cmap='Purples')
    
   
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_visible(False)
    ax.spines['bottom'].set_visible(False)
    ax.yaxis.set_ticks_position('right')
    ax.xaxis.set_ticks_position('top')  
    ax.set_xticks([])
    ax.set_yticks([])
    plt.savefig('Th_Fig0_B_n'+str(ip+1)+'.pdf')

#%%

#%%

fig = plt.figure(figsize = [ 2.0, 0.2])
ax = fig.add_subplot(111) 
Madap1 = np.copy(Madap[:,:])
Madap2 = np.copy(Madap[:,:])

Madap1[Nneurs:,:] = np.nan
Madap2[0:Nneurs,:] = np.nan
Nadap1 = np.copy(Nadap)
Nadap2 = np.copy(Nadap)
Nadap1[:,Nneurs:] = np.nan
Nadap2[:,0:Nneurs] = np.nan
plt.imshow(Madap1.T, cmap='Greens')
plt.imshow(Madap2.T, cmap='Purples')

ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.spines['left'].set_visible(False)
ax.spines['bottom'].set_visible(False)
ax.yaxis.set_ticks_position('right')
ax.xaxis.set_ticks_position('top')  
ax.set_xticks([])
ax.set_yticks([])
plt.savefig('Th_Fig0_C_m.pdf')

fig = plt.figure(figsize = [0.2, 2.0])
ax = fig.add_subplot(111) 
plt.imshow(Nadap1.T, cmap='Greens')
plt.imshow(Nadap2.T, cmap='Purples')

   
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.spines['left'].set_visible(False)
ax.spines['bottom'].set_visible(False)
ax.yaxis.set_ticks_position('right')
ax.xaxis.set_ticks_position('top')  
ax.set_xticks([])
ax.set_yticks([])
plt.savefig('Th_Fig0_C_n.pdf')


#%%

fig = plt.figure(figsize = [ 2.0, 0.2])
ax = fig.add_subplot(111) 
Madap1 = np.copy(Madap[:,:])
Madap2 = np.copy(Madap[:,:])

Madap1[Nneurs:,:] = np.nan
Madap2[0:Nneurs,:] = np.nan
Nadap1 = np.copy(Nadap)
Nadap2 = np.copy(Nadap)
Nadap1[:,Nneurs:] = np.nan
Nadap2[:,0:Nneurs] = np.nan
plt.imshow(Madap1.T, cmap='Greens')
plt.imshow(Madap2.T, cmap='Purples')

ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.spines['left'].set_visible(False)
ax.spines['bottom'].set_visible(False)
ax.yaxis.set_ticks_position('right')
ax.xaxis.set_ticks_position('top')  
ax.set_xticks([])
ax.set_yticks([])
plt.savefig('Th_Fig0_C_m.pdf')
#%%
fig = plt.figure(figsize = [1., 2.0])
ax = fig.add_subplot(111) 
Madap1 = MuMs[:,0,None]

plt.imshow(Madap1, cmap='Greens', vmin = -4, vmax=4)
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.spines['left'].set_visible(False)
ax.spines['bottom'].set_visible(False)
ax.yaxis.set_ticks_position('right')
ax.xaxis.set_ticks_position('top')  
ax.set_xticks([])
ax.set_yticks([])
plt.savefig('Th_Fig0_D_pop1_am.pdf')

fig = plt.figure(figsize = [1,2.])
ax = fig.add_subplot(111) 
Madap1 = MuMs[:,1,None]

plt.imshow(Madap1, cmap='Purples', vmin = -4, vmax=4)
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.spines['left'].set_visible(False)
ax.spines['bottom'].set_visible(False)
ax.yaxis.set_ticks_position('right')
ax.xaxis.set_ticks_position('top')  
ax.set_xticks([])
ax.set_yticks([])
plt.savefig('Th_Fig0_D_pop2_am.pdf')

fig = plt.figure(figsize = [2.0,1.])
ax = fig.add_subplot(111) 
Madap1 = MuNs[:,0,None]
plt.imshow(Madap1.T, cmap='Greens', vmin = -4, vmax=4)
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.spines['left'].set_visible(False)
ax.spines['bottom'].set_visible(False)
ax.yaxis.set_ticks_position('right')
ax.xaxis.set_ticks_position('top')  
ax.set_xticks([])
ax.set_yticks([])
plt.savefig('Th_Fig0_D_pop1_an.pdf')

fig = plt.figure(figsize = [ 2.0, 1.])
ax = fig.add_subplot(111) 
Madap1 = MuNs[:,1,None]

plt.imshow(Madap1.T, cmap='Purples', vmin = -4, vmax=4)
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.spines['left'].set_visible(False)
ax.spines['bottom'].set_visible(False)
ax.yaxis.set_ticks_position('right')
ax.xaxis.set_ticks_position('top')  
ax.set_xticks([])
ax.set_yticks([])
plt.savefig('Th_Fig0_D_pop2_an.pdf')

#%%
fig = plt.figure(figsize = [2., 2.0])
ax = fig.add_subplot(111) 
Madap1 = Sigma[:,:,0]

plt.imshow(Madap1, cmap='Greens', vmin = -4, vmax=4)
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.spines['left'].set_visible(False)
ax.spines['bottom'].set_visible(False)
ax.yaxis.set_ticks_position('right')
ax.xaxis.set_ticks_position('top')  
ax.set_xticks([])
ax.set_yticks([])
plt.savefig('Th_Fig0_D_pop1_sigmas.pdf')

#%%
fig = plt.figure(figsize = [2., 2.0])
ax = fig.add_subplot(111) 
Madap1 = Sigma[:,:,1]

plt.imshow(Madap1, cmap='Purples', vmin = -4, vmax=4)
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.spines['left'].set_visible(False)
ax.spines['bottom'].set_visible(False)
ax.yaxis.set_ticks_position('right')
ax.xaxis.set_ticks_position('top')  
ax.set_xticks([])
ax.set_yticks([])
plt.savefig('Th_Fig0_D_pop2_sigmas.pdf')


