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
N = 400
nbins = 20


s_mn1 = 0*np.array((0.5, 0.5, 0.5, 0.5))
s_mn2 = 0*np.array((0.5, 0.5, 0.5, 0.5))

val = 0.2#np.sqrt(1-0.5**2)

pops = 1
olap = 3.

a_m1 = np.sqrt(2-val**2)*np.array((np.cos(np.pi*0), np.cos(np.pi/3), np.cos(2*np.pi/3), np.cos(np.pi), np.cos(4*np.pi/3), np.cos(5*np.pi/3)))
a_m2 = np.sqrt(2-val**2)*np.array((np.sin(np.pi*0), np.sin(np.pi/3), np.sin(2*np.pi/3), np.sin(np.pi), np.sin(4*np.pi/3), np.sin(5*np.pi/3)))
a_n1 = olap*(1/np.sqrt(2-val**2))*np.array((np.cos(np.pi*0), np.cos(np.pi/3), np.cos(2*np.pi/3), np.cos(np.pi), np.cos(4*np.pi/3), np.cos(5*np.pi/3)))
a_n2 = olap*(1/np.sqrt(2-val**2))*pops*np.array((np.sin(np.pi*0), np.sin(np.pi/3), np.sin(2*np.pi/3), np.sin(np.pi), np.sin(4*np.pi/3), np.sin(5*np.pi/3)))



m1 = np.random.randn(N)
n1 = np.random.randn(N)
m2 = np.random.randn(N)
n2 = np.random.randn(N)

sels = 1000
err0 = 50

pops = len(a_m1)
for t in range(sels):
    V = np.random.randn(N//pops, 80)
    CC = np.dot(V, V.T)
    for po in range(pops):
        CC[po,po] = 0.
    
    err = np.std(CC)
    if err<err0:
        Vs = V
        err0 = err
        
ix = 0

for po in range(pops):
    m1[po*(N//pops):(po+1)*(N//pops)] = a_m1[po]+val*V[:, ix] 
    ix += 1
    n1[po*(N//pops):(po+1)*(N//pops)] = a_n1[po]+val*V[:, ix]#+s_mn1[po]*V[:, ix]/val
    ix += 1 
    m2[po*(N//pops):(po+1)*(N//pops)] = a_m2[po]+val*V[:, ix]
    ix += 1
    n2[po*(N//pops):(po+1)*(N//pops)] = a_n2[po]+val*V[:, ix]#+s_mn2[po]*V[:, ix]/val
    ix += 1

#%%
# =============================================================================
#           Fig 2
# =============================================================================
ms = np.linspace(-5,5,100)
Sigma = np.zeros((2,2))
Sigma[0,0] = 1.3
Sigma[1,1] = 1.8
Sigma[0,1] = -0.5
Sigma[1,0] = 0.8


N = 1000
S=10
M = np.vstack((m1, m2)).T
ss2 = 0.3

NNN = np.vstack((n1, n2))

fig = plt.figure(figsize=[3.2, 3.2])#, dpi=600
gs = GridSpec(5,5)

ax_joint00 = fig.add_subplot(gs[1:3,0:2])
ax_joint01 = fig.add_subplot(gs[1:3,2:4])
ax_joint10 = fig.add_subplot(gs[3:5,0:2])
ax_joint11 = fig.add_subplot(gs[3:5,2:4])

ax_marg_x0 = fig.add_subplot(gs[0,0:2])
ax_marg_x1 = fig.add_subplot(gs[0,2:4])

ax_marg_y0 = fig.add_subplot(gs[1:3,4])
ax_marg_y1 = fig.add_subplot(gs[3:5,4])

yl = 4.
ylt = 3.
xl = 2.5
xlt = 2.
ax_joint00.scatter(M[:,0], NNN[0,:], s=S, alpha=0.5, label=r'$\sigma_{mn} = 1.2$', rasterized=True)
ax_joint00.scatter(a_m1, a_n1,
                   s=2*S, edgecolor='k', facecolor='w')
ax_joint00.set_xlim([-xl, xl])
ax_joint00.set_xticks([-xlt, 0, xlt])
ax_joint00.set_xticklabels(['','',''])
ax_joint00.set_ylim([-yl,yl])
ax_joint00.set_yticks([-ylt, 0, ylt])
ax_joint00.set_ylabel(r'$n^{\left(1\right)}_i$')
ax_joint00.spines['top'].set_visible(False)
ax_joint00.spines['right'].set_visible(False)
ax_joint00.yaxis.set_ticks_position('left')
ax_joint00.xaxis.set_ticks_position('bottom')
                                  
ax_joint01.scatter(M[:,1], NNN[0,:], s=S, alpha=0.5, label=r'$\sigma_{mn} = 1.2$', rasterized=True)
ax_joint01.scatter(a_m2, a_n1,
                   s=2*S, edgecolor='k', facecolor='w')
ax_joint01.spines['top'].set_visible(False)
ax_joint01.spines['right'].set_visible(False)
ax_joint01.yaxis.set_ticks_position('left')
ax_joint01.xaxis.set_ticks_position('bottom')
ax_joint01.set_ylim([-yl,yl])
ax_joint01.set_yticks([-ylt, 0, ylt])
ax_joint01.set_yticklabels(['','',''])
ax_joint01.set_xlim([-xl, xl])
ax_joint01.set_xticks([-xlt, 0, xlt])
ax_joint01.set_xticklabels(['','',''])

ax_joint10.scatter(M[:,0], NNN[1,:], s=S, alpha=0.5, label=r'$\sigma_{mn} = 1.2$', rasterized=True)
ax_joint10.scatter(a_m1, a_n2,
                   s=2*S, edgecolor='k', facecolor='w')
ax_joint10.set_xlim([-3,3])
ax_joint10.spines['top'].set_visible(False)
ax_joint10.spines['right'].set_visible(False)
ax_joint10.yaxis.set_ticks_position('left')
ax_joint10.xaxis.set_ticks_position('bottom')
ax_joint10.set_ylim([-yl,yl])
ax_joint10.set_yticks([-ylt, 0, ylt])
ax_joint10.set_xlim([-xl, xl])
ax_joint10.set_xticks([-xlt, 0, xlt])
ax_joint10.set_ylabel(r'$n^{\left(2\right)}_i$')
ax_joint10.set_xlabel(r'$m^{\left(1\right)}_i$')

ax_joint11.scatter(M[:,1], NNN[1,:], s=S, alpha=0.5, label=r'$\sigma_{mn} = 1.2$', rasterized=True)
ax_joint11.scatter(a_m2, a_n2,
                   s=2*S, edgecolor='k', facecolor='w')
ax_joint11.set_xlim([-3,3])
ax_joint11.spines['top'].set_visible(False)
ax_joint11.spines['right'].set_visible(False)
ax_joint11.set_ylim([-yl, yl])
ax_joint11.set_yticks([-ylt, 0, ylt])
ax_joint11.set_xlim([-xl, xl])
ax_joint11.set_xticks([-xlt, 0, xlt])
ax_joint11.set_yticklabels(['','',''])
ax_joint11.yaxis.set_ticks_position('left')
ax_joint11.xaxis.set_ticks_position('bottom')
ax_joint11.set_xlabel(r'$m^{\left(2\right)}_i$')

ax_marg_x0.hist(M[:,0], nbins, alpha=0.5, density=True)
ss = val
m_ms = 0
for ip in range(pops):
    m_ms += (1/pops)*(1/np.sqrt(2*np.pi*ss**2))*(np.exp(-(ms+a_m1[ip])**2/(2*ss**2)))
ax_marg_x0.plot(ms, m_ms, 'k')
ax_marg_x0.spines['top'].set_visible(False)
ax_marg_x0.spines['right'].set_visible(False)
ax_marg_x0.spines['left'].set_visible(False)
ax_marg_x0.yaxis.set_ticks_position('left')
ax_marg_x0.xaxis.set_ticks_position('bottom')
ax_marg_x0.set_xlim([-3,3])
ax_marg_x0.set_xticks([-2., 0, 2.])
ax_marg_x0.set_ylim([0,.8])
ax_marg_x0.set_xticklabels(['','',''])
ax_marg_x0.set_yticks([1])

ax_marg_x1.hist(M[:,1], nbins, alpha=0.5, density=True)
m_ms = 0
for ip in range(pops):
    m_ms += (1/pops)*(1/np.sqrt(2*np.pi*ss**2))*(np.exp(-(ms+a_m2[ip])**2/(2*ss**2)))
ax_marg_x1.plot(ms, m_ms, 'k')
ax_marg_x1.spines['top'].set_visible(False)
ax_marg_x1.spines['right'].set_visible(False)
ax_marg_x1.spines['left'].set_visible(False)
ax_marg_x1.yaxis.set_ticks_position('left')
ax_marg_x1.xaxis.set_ticks_position('bottom')
ax_marg_x1.set_xlim([-3,3])
ax_marg_x1.set_ylim([0,0.8])
ax_marg_x1.set_xticks([-2., 0, 2.])
ax_marg_x1.set_xticklabels(['','',''])
ax_marg_x1.set_yticks([1])

ax_marg_y0.hist(NNN[0,:], nbins, orientation="horizontal", alpha=0.5, density=True)
ss= val
Mu = olap*(1/np.sqrt(1-val**2))
m_ms = 0
for ip in range(pops):
    m_ms += (1/pops)*(1/np.sqrt(2*np.pi*ss**2))*(np.exp(-(ms+a_n1[ip])**2/(2*ss**2)))
ax_marg_y0.plot( m_ms,ms, 'k')
#ax_marg_y0.plot((0.5/np.sqrt(2*np.pi*ss**2))*(np.exp(-(ms+Mu)**2/(2*ss**2)) + np.exp(-(ms-Mu)**2/(2*ss**2))),ms,  'k')
ax_marg_y0.spines['top'].set_visible(False)
ax_marg_y0.spines['right'].set_visible(False)
ax_marg_y0.spines['bottom'].set_visible(False)
ax_marg_y0.yaxis.set_ticks_position('left')
ax_marg_y0.xaxis.set_ticks_position('bottom')
ax_marg_y0.set_ylim([-yl,yl])
ax_marg_y0.set_xlim([0,0.8])
ax_marg_y0.set_yticks([-ylt, 0, ylt])
ax_marg_y0.set_yticklabels(['','',''])
ax_marg_y0.set_xticks([1])
ax_marg_y0.set_xticklabels([''])

ax_marg_y1.hist(NNN[1,:], nbins, orientation="horizontal", alpha=0.5, density=True)
m_ms = 0
for ip in range(pops):
    m_ms += (1/pops)*(1/np.sqrt(2*np.pi*ss**2))*(np.exp(-(ms+a_n2[ip])**2/(2*ss**2)))
ax_marg_y1.plot( m_ms,ms, 'k')
ax_marg_y1.spines['top'].set_visible(False)
ax_marg_y1.spines['right'].set_visible(False)
ax_marg_y1.spines['bottom'].set_visible(False)
ax_marg_y1.yaxis.set_ticks_position('left')
ax_marg_y1.xaxis.set_ticks_position('bottom')
ax_marg_y1.set_ylim([-yl,yl])
ax_marg_y1.set_xlim([0,0.8])
ax_marg_y1.set_yticks([-ylt, 0, ylt])
ax_marg_y1.set_yticklabels(['','',''])
ax_marg_y1.set_xticks([1])
ax_marg_y1.set_xticklabels([''])

plt.savefig('Th_Fig5B_1_A.pdf')

#%%
kaps1 = np.linspace(-1.5,1.5, 150)
kaps2 = np.linspace(-1.5,1.5, 140)
ksol = np.zeros((len(kaps1), len(kaps2), 2))

K1s, K2s = np.meshgrid(kaps1, kaps2)
def transf(K):
    return(K*Prime(0, np.dot(K.T, K)))
    
E = np.zeros((len(kaps1), len(kaps2)))
for ik1 ,k1 in enumerate(kaps1):
    for ik2, k2 in enumerate(kaps2):
        K = np.array((k1, k2))
        ksol[ik1, ik2, :] = - K
        for ip in range(pops):
            ksol[ik1, ik2, 0] += (1/pops)*a_n1[ip]*Phi(a_m1[ip]*k1+a_m2[ip]*k2, val**2*k1**2+val**2*k2**2)
            ksol[ik1, ik2, 1] += (1/pops)*a_n2[ip]*Phi(a_m1[ip]*k1+a_m2[ip]*k2, val**2*k1**2+val**2*k2**2)
        
        E[ik1, ik2] = np.sqrt(np.sum(ksol[ik1,ik2,:]**2))
        
search_kap1 = np.linspace(0.2, 1.5, 300)
E_1 = np.zeros_like(search_kap1)
v = np.array((1,0))
for ik1 ,k1 in enumerate(search_kap1):
    K = v*k1
    kSS = - K[0]
    for ip in range(pops):
        kSS += (1/pops)*a_n1[ip]*Phi(a_m1[ip]*K[0], val**2*K[0]**2)
        
    E_1[ik1] = np.abs(kSS)
fp1 = search_kap1[np.argmin(E_1)]

search_kap1 = np.linspace(0.2, 2., 500)
E_1 = np.zeros_like(search_kap1)
v = np.array((0,1))
E11 = np.zeros_like(E_1)
E12 = np.zeros_like(E_1)

for ik1 ,k1 in enumerate(search_kap1):
    K = v*k1
    kSS = - K
    for ip in range(pops):
        kSS[0] += (1/pops)*a_n1[ip]*Phi(a_m1[ip]*K[0]+a_m2[ip]*K[1], val**2*K[0]**2+val**2*K[1]**2)
        kSS[1] += (1/pops)*a_n2[ip]*Phi(a_m1[ip]*K[0]+a_m2[ip]*K[1], val**2*K[0]**2+val**2*K[1]**2)
    E11[ik1] =kSS[0]
    E12[ik1] =kSS[1]
    
    E_1[ik1] = np.sum(kSS**2)
fp11 = search_kap1[np.argmin(E_1)]

search_kap2 = np.linspace(0.2, 2.0, 300)
E_2 = np.zeros_like(search_kap2)
E21 = np.zeros_like(E_2)
E22 = np.zeros_like(E_2)
An = np.vstack((a_n1, a_n2))
Am = np.vstack((a_m1, a_m2))

for ik2 ,k2 in enumerate(search_kap2):
    W = np.array((np.cos(np.pi/6), np.sin(np.pi/6)))
    K = W*k2
    kSS = - K
    for ip in range(pops):
        kSS[0] += (1/pops)*a_n1[ip]*Phi(a_m1[ip]*K[0]+a_m2[ip]*K[1], val**2*K[0]**2+val**2*K[1]**2)
        kSS[1] += (1/pops)*a_n2[ip]*Phi(a_m1[ip]*K[0]+a_m2[ip]*K[1], val**2*K[0]**2+val**2*K[1]**2)
    E21[ik2] =kSS[0]
    E22[ik2] =kSS[1]        
    E_2[ik2] = np.abs(np.sum(kSS**2))
fp2 = search_kap2[np.argmin(E_2)]

fig = plt.figure()
ax = fig.add_subplot(111) 
im = plt.pcolor(kaps1, kaps2, np.log10(E).T, cmap ='viridis', vmin = -2.,vmax=0, shading='auto')


strm = ax.streamplot(kaps1, kaps2, ksol[:,:,0].T, ksol[:,:,1].T, color='w', linewidth=1, cmap='autumn', density=0.6)

plt.xlabel('$\kappa_1$')
plt.ylabel('$\kappa_2$')

for ii in range(pops):
    plt.scatter(fp2*np.cos(2*np.pi*ii/pops+np.pi/6), fp2*np.sin(2*np.pi*ii/pops+np.pi/6), s=50, edgecolor='k', facecolor='w', linewidth=1., zorder=4 )
    plt.scatter(fp1*np.cos(2*np.pi*ii/pops), fp1*np.sin(2*np.pi*ii/pops), s=50, edgecolor='w', facecolor='k', linewidth=1., zorder=4 )
th = np.linspace(0, 2*np.pi)

ax.set_xticks([-1, 0, 1])
ax.set_yticks([-1, 0, 1])
ax.set_ylim([np.min(kaps2), np.max(kaps2)])
ax.set_xlim([np.min(kaps1), np.max(kaps1)])
plt.savefig('Th_Fig5B_1_C1.pdf')

#%%
fig = plt.figure()
ax = fig.add_subplot(111) 

plt.xlabel('$\kappa_1$')
plt.ylabel('$\kappa_2$')
for ii in range(pops):
    plt.scatter(fp2*np.cos(2*np.pi*ii/pops+np.pi/6), fp2*np.sin(2*np.pi*ii/pops+np.pi/6), s=50, edgecolor='k', facecolor='w', linewidth=1., zorder=4 )
    plt.scatter(fp1*np.cos(2*np.pi*ii/pops), fp1*np.sin(2*np.pi*ii/pops), s=50, edgecolor='w', facecolor='k', linewidth=1., zorder=4 )

Nn = 1200

Mu= np.zeros((4,1))

inkap1 = np.linspace(-1.2, 1.2, 6)
inkap2 = np.linspace(-1.2, 1.2, 6)

dt = 0.1
time = np.arange(0, 120, dt)

for trials in range(2):

    m1 = np.random.randn(Nn)
    n1 = np.random.randn(Nn)
    m2 = np.random.randn(Nn)
    n2 = np.random.randn(Nn)
    
    sels = 100
    err0 = 50
    
    pops = len(a_m1)
    for t in range(sels):
        V = np.random.randn(Nn//pops, pops*4)
        CC = np.dot(V, V.T)
        for po in range(pops):
            CC[po,po] = 0.
        
        err = np.std(CC)
        if err<err0:
            Vs = V
            err0 = err
            
    ix = 0
    
    for po in range(pops):
        m1[po*(Nn//pops):(po+1)*(Nn//pops)] = a_m1[po]+val*V[:, ix] 
        ix += 1
        n1[po*(Nn//pops):(po+1)*(Nn//pops)] = a_n1[po]+val*V[:, ix]#+s_mn1[po]*V[:, ix]/val
        ix += 1 
        m2[po*(Nn//pops):(po+1)*(Nn//pops)] = a_m2[po]+val*V[:, ix]
        ix += 1
        n2[po*(Nn//pops):(po+1)*(Nn//pops)] = a_n2[po]+val*V[:, ix]#+s_mn2[po]*V[:, ix]/val
        ix += 1
    M = np.vstack((m1, m2)).T
    N = np.vstack((n1, n2))

    J = np.dot(M, N)/Nn
    
    cC =  np.ones(3)*0.6
    
    for ik1, ink1 in enumerate(inkap1):
        for ik2, ink2 in enumerate(inkap2):
            sk1 = np.zeros_like(time)
            sk2 = np.zeros_like(time)
            
            x0 = ink1*M[:,0] + ink2*M[:,1]
            sk1[0] = np.mean(M[:,0]*x0)
            sk2[0] = np.mean(M[:,1]*x0)
            
            for it, ti in enumerate(time[:-1]):
                x = x0 + dt*(-x0 + np.dot(M/Nn, N.dot(np.tanh(x0))))
                sk1[it+1] = np.mean(M[:,0]*x)
                sk2[it+1] = np.mean(M[:,1]*x)
                x0 = x
            plt.plot(sk1, sk2, c=cC)
            plt.scatter(sk1[0], sk2[0], s=10, facecolor=cC)
            plt.scatter(sk1[-1], sk2[-1], s=25, facecolor=cC, edgecolor='k', zorder=3)

ax.set_xticks([-1, 0, 1])
ax.set_yticks([-1, 0, 1])
ax.set_ylim([1.1*np.min(kaps2), 1.1*np.max(kaps2)])
ax.set_xlim([1.1*np.min(kaps1), 1.1*np.max(kaps1)])
        
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.yaxis.set_ticks_position('left')
ax.xaxis.set_ticks_position('bottom')  
plt.savefig('Th_Fig5B_1_D.pdf')    