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

np.random.seed(20)
    
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
val = 0.1
olap = 10
a_m1 = np.sqrt((1-val**2))           *np.array((1, 1, 1, 1, -1, -1, -1, -1))
#np.array((0., np.sqrt(8./9.), -np.sqrt(2./9.), -np.sqrt(2./9.), 0., -np.sqrt(8./9.), np.sqrt(2./9.), np.sqrt(2./9.)))
a_m2 = np.sqrt((1-val**2))            *np.array((1, -1, 1, -1, 1, -1, 1, -1))
#*np.array((0., 0., np.sqrt(2./3.), -np.sqrt(2./3.), 0., 0., -np.sqrt(2./3.), np.sqrt(2./3.), ))
a_m3 = np.sqrt((1-val**2))           *np.array((1, -1, -1, 1, 1, -1, -1, 1))     
# *np.array((1., -1./3., -1./3., -1./3., -1., 1./3., 1./3., 1./3.))

a_n1 = olap/np.sqrt((1-val**2))  *np.array((1, 1, 1, 1, -1, -1, -1, -1))   
#*np.array((0., np.sqrt(8./9.), -np.sqrt(2./9.), -np.sqrt(2./9.), 0., -np.sqrt(8./9.), np.sqrt(2./9.), np.sqrt(2./9.)))
a_n2 = olap/np.sqrt((1-val**2))    *np.array((1, -1, 1, -1, 1, -1, 1, -1)) 
#*np.array((0., 0., np.sqrt(2./3.), -np.sqrt(2./3.), 0., 0., -np.sqrt(2./3.), np.sqrt(2./3.), ))
a_n3 = olap/np.sqrt((1-val**2))     *np.array((1, -1, -1, 1, 1, -1, -1, 1))   
#*np.array((1., -1./3., -1./3., -1./3., -1., 1./3., 1./3., 1./3.))

def give_vecs(val=0.1, olap=1.5, N = 400):
    pops = 8
    a_m1 = np.sqrt((1-val**2))           *np.array((1, 1, 1, 1, -1, -1, -1, -1))
    #np.array((0., np.sqrt(8./9.), -np.sqrt(2./9.), -np.sqrt(2./9.), 0., -np.sqrt(8./9.), np.sqrt(2./9.), np.sqrt(2./9.)))
    a_m2 = np.sqrt((1-val**2))            *np.array((1, -1, 1, -1, 1, -1, 1, -1))
    #*np.array((0., 0., np.sqrt(2./3.), -np.sqrt(2./3.), 0., 0., -np.sqrt(2./3.), np.sqrt(2./3.), ))
    a_m3 = np.sqrt((1-val**2))           *np.array((1, -1, -1, 1, 1, -1, -1, 1))     
    # *np.array((1., -1./3., -1./3., -1./3., -1., 1./3., 1./3., 1./3.))
    
    a_n1 = olap/np.sqrt((1-val**2))  *np.array((1, 1, 1, 1, -1, -1, -1, -1))   
    #*np.array((0., np.sqrt(8./9.), -np.sqrt(2./9.), -np.sqrt(2./9.), 0., -np.sqrt(8./9.), np.sqrt(2./9.), np.sqrt(2./9.)))
    a_n2 = olap/np.sqrt((1-val**2))    *np.array((1, -1, 1, -1, 1, -1, 1, -1)) 
    #*np.array((0., 0., np.sqrt(2./3.), -np.sqrt(2./3.), 0., 0., -np.sqrt(2./3.), np.sqrt(2./3.), ))
    a_n3 = olap/np.sqrt((1-val**2))     *np.array((1, -1, -1, 1, 1, -1, -1, 1))   
    #*np.array((1., -1./3., -1./3., -1./3., -1., 1./3., 1./3., 1./3.))
    m1 = np.random.randn(N)
    n1 = np.random.randn(N)
    
    m2 = np.random.randn(N)
    n2 = np.random.randn(N)
    
    m3 = np.random.randn(N)
    n3 = np.random.randn(N)
    
    sels = 1000
    err0 = 50
    
    pops = len(a_m1)
    for t in range(sels):
        V = np.random.randn(N//pops, 50)
        CC = np.dot(V, V.T)
        for po in range(pops):
            CC[po,po] = 0.
        
        err = np.std(CC)
        if err<err0:
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
        m3[po*(N//pops):(po+1)*(N//pops)] = a_m3[po]+val*V[:, ix]
        ix += 1
        n3[po*(N//pops):(po+1)*(N//pops)] = a_n3[po]+val*V[:, ix]#+s_mn2[po]*V[:, ix]/val
        ix += 1
    return(m1, m2, m3, n1, n2, n3)
pops = 8  
m1, m2, m3, n1, n2, n3 = give_vecs(olap=olap)
#%%
# =============================================================================
#           Fig 2
# =============================================================================
ms = np.linspace(-10.5,10.5,3000)


N = 1000
S=10
M = np.vstack((m1, m2, m3)).T
ss2 = 0.3

NNN = np.vstack((n1, n2, n3))

fig = plt.figure(figsize=[4.2, 4.2])#, dpi=600
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

yl = 12.5
ylt = 10.
xl = 2.5
xlt = 2.
ax_joint00.scatter(M[:,0], NNN[0,:], s=S, alpha=0.5, label=r'$\sigma_{mn} = 1.2$', rasterized=True)

for ip in range(pops):
    ax_joint00.scatter(a_m1[ip], a_n1[ip], s=0.5*S, edgecolor='k', facecolor='w')
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
for ip in range(pops):
    ax_joint01.scatter(a_m2[ip], a_n1[ip], s=0.5*S, edgecolor='k', facecolor='w')
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

ax_joint02.scatter(M[:,2], NNN[0,:], s=S, alpha=0.5, label=r'$\sigma_{mn} = 1.2$', rasterized=True)
for ip in range(pops):
    ax_joint02.scatter(a_m3[ip], a_n1[ip], s=0.5*S, edgecolor='k', facecolor='w')
ax_joint02.spines['top'].set_visible(False)
ax_joint02.spines['right'].set_visible(False)
ax_joint02.yaxis.set_ticks_position('left')
ax_joint02.xaxis.set_ticks_position('bottom')
ax_joint02.set_ylim([-yl,yl])
ax_joint02.set_yticks([-ylt, 0, ylt])
ax_joint02.set_yticklabels(['','',''])
ax_joint02.set_xlim([-xl, xl])
ax_joint02.set_xticks([-xlt, 0, xlt])
ax_joint02.set_xticklabels(['','',''])


ax_joint10.scatter(M[:,0], NNN[1,:], s=S, alpha=0.5, label=r'$\sigma_{mn} = 1.2$', rasterized=True)
for ip in range(pops):
    ax_joint10.scatter(a_m1[ip], a_n2[ip], s=0.5*S, edgecolor='k', facecolor='w')
ax_joint10.set_xlim([-3,3])
ax_joint10.spines['top'].set_visible(False)
ax_joint10.spines['right'].set_visible(False)
ax_joint10.yaxis.set_ticks_position('left')
ax_joint10.xaxis.set_ticks_position('bottom')
ax_joint10.set_ylim([-yl,yl])
ax_joint10.set_yticks([-ylt, 0, ylt])
ax_joint10.set_xlim([-xl, xl])
ax_joint10.set_xticks([-xlt, 0, xlt])
ax_joint10.set_xticklabels(['','',''])
ax_joint10.set_ylabel(r'$n^{\left(2\right)}_i$')
#ax_joint10.set_xlabel(r'$m^{\left(1\right)}_i$')

ax_joint11.scatter(M[:,1], NNN[1,:], s=S, alpha=0.5, label=r'$\sigma_{mn} = 1.2$', rasterized=True)
for ip in range(pops):
    ax_joint11.scatter(a_m2[ip], a_n2[ip], s=0.5*S, edgecolor='k', facecolor='w')
ax_joint11.spines['top'].set_visible(False)
ax_joint11.spines['right'].set_visible(False)
ax_joint11.set_ylim([-yl, yl])
ax_joint11.set_yticks([-ylt, 0, ylt])
ax_joint11.set_xlim([-xl, xl])
ax_joint11.set_xticks([-xlt, 0, xlt])
ax_joint11.set_xticklabels(['','',''])
ax_joint11.set_yticklabels(['','',''])
ax_joint11.yaxis.set_ticks_position('left')
ax_joint11.xaxis.set_ticks_position('bottom')
#ax_joint11.set_xlabel(r'$m^{\left(2\right)}_i$')

ax_joint12.scatter(M[:,2], NNN[1,:], s=S, alpha=0.5, label=r'$\sigma_{mn} = 1.2$', rasterized=True)
for ip in range(pops):
    ax_joint12.scatter(a_m3[ip], a_n2[ip], s=0.5*S, edgecolor='k', facecolor='w')
ax_joint12.set_xlim([-3,3])
ax_joint12.spines['top'].set_visible(False)
ax_joint12.spines['right'].set_visible(False)
ax_joint12.set_ylim([-yl, yl])
ax_joint12.set_yticks([-ylt, 0, ylt])
ax_joint12.set_xlim([-xl, xl])
ax_joint12.set_xticks([-xlt, 0, xlt])
ax_joint12.set_xticklabels(['','',''])
ax_joint12.set_yticklabels(['','',''])
ax_joint12.yaxis.set_ticks_position('left')
ax_joint12.xaxis.set_ticks_position('bottom')
#ax_joint12.set_xlabel(r'$m^{\left(3\right)}_i$')

ax_joint20.scatter(M[:,0], NNN[2,:], s=S, alpha=0.5, label=r'$\sigma_{mn} = 1.2$', rasterized=True)
for ip in range(pops):
    ax_joint20.scatter(a_m1[ip], a_n3[ip], s=0.5*S, edgecolor='k', facecolor='w')
ax_joint20.set_xlim([-3,3])
ax_joint20.spines['top'].set_visible(False)
ax_joint20.spines['right'].set_visible(False)
ax_joint20.yaxis.set_ticks_position('left')
ax_joint20.xaxis.set_ticks_position('bottom')
ax_joint20.set_ylim([-yl,yl])
ax_joint20.set_yticks([-ylt, 0, ylt])
ax_joint20.set_xlim([-xl, xl])
ax_joint20.set_xticks([-xlt, 0, xlt])
ax_joint20.set_ylabel(r'$n^{\left(3\right)}_i$')
ax_joint20.set_xlabel(r'$m^{\left(1\right)}_i$')

ax_joint21.scatter(M[:,1], NNN[2,:], s=S, alpha=0.5, label=r'$\sigma_{mn} = 1.2$', rasterized=True)
for ip in range(pops):
    ax_joint21.scatter(a_m2[ip], a_n3[ip], s=0.5*S, edgecolor='k', facecolor='w')
ax_joint21.set_xlim([-3,3])
ax_joint21.spines['top'].set_visible(False)
ax_joint21.spines['right'].set_visible(False)
ax_joint21.set_ylim([-yl, yl])
ax_joint21.set_yticks([-ylt, 0, ylt])
ax_joint21.set_xlim([-xl, xl])
ax_joint21.set_xticks([-xlt, 0, xlt])
ax_joint21.set_yticklabels(['','',''])
ax_joint21.yaxis.set_ticks_position('left')
ax_joint21.xaxis.set_ticks_position('bottom')
ax_joint21.set_xlabel(r'$m^{\left(2\right)}_i$')

ax_joint22.scatter(M[:,2], NNN[2,:], s=S, alpha=0.5, label=r'$\sigma_{mn} = 1.2$', rasterized=True)
for ip in range(pops):
    ax_joint22.scatter(a_m3[ip], a_n3[ip], s=0.5*S, edgecolor='k', facecolor='w')
ax_joint22.set_xlim([-3,3])
ax_joint22.spines['top'].set_visible(False)
ax_joint22.spines['right'].set_visible(False)
ax_joint22.set_ylim([-yl, yl])
ax_joint22.set_yticks([-ylt, 0, ylt])
ax_joint22.set_xlim([-xl, xl])
ax_joint22.set_xticks([-xlt, 0, xlt])
ax_joint22.set_yticklabels(['','',''])
ax_joint22.yaxis.set_ticks_position('left')
ax_joint22.xaxis.set_ticks_position('bottom')
ax_joint22.set_xlabel(r'$m^{\left(3\right)}_i$')

ax_marg_x0.hist(M[:,0], nbins, alpha=0.5, density=True)
ss = val
#ax_marg_x0.plot(ms, (0.5/np.sqrt(2*np.pi*ss**2))*(np.exp(-(ms+np.sqrt(1-val**2))**2/(2*ss**2)) + np.exp(-(ms-np.sqrt(1-val**2))**2/(2*ss**2))), 'k')
sol_n = np.zeros_like(ms)
for ip in range(pops):
    sol_n += (1/pops)*(1/np.sqrt(2*np.pi*ss**2))*np.exp(-(ms+a_m1[ip])**2/(2*ss**2))
ax_marg_x0.plot( ms,  sol_n,'k', lw=0.5)
ax_marg_x0.spines['top'].set_visible(False)
ax_marg_x0.spines['right'].set_visible(False)
ax_marg_x0.spines['left'].set_visible(False)
ax_marg_x0.yaxis.set_ticks_position('left')
ax_marg_x0.xaxis.set_ticks_position('bottom')
ax_marg_x0.set_xlim([-3,3])
ax_marg_x0.set_xticks([-2., 0, 2.])
ax_marg_x0.set_ylim([0,1.4])
ax_marg_x0.set_xticklabels(['','',''])
ax_marg_x0.set_yticks([2])

ax_marg_x1.hist(M[:,1], nbins, alpha=0.5, density=True)
sol_n = np.zeros_like(ms)
for ip in range(pops):
    sol_n += (1/pops)*(1/np.sqrt(2*np.pi*ss**2))*np.exp(-(ms+a_m2[ip])**2/(2*ss**2))
ax_marg_x1.plot( ms,  sol_n,'k', lw=0.5)
#ax_marg_x1.plot(ms, (0.5/np.sqrt(2*np.pi*ss**2))*(np.exp(-(ms+np.sqrt(1-val**2))**2/(2*ss**2)) + np.exp(-(ms-np.sqrt(1-val**2))**2/(2*ss**2))), 'k')
ax_marg_x1.spines['top'].set_visible(False)
ax_marg_x1.spines['right'].set_visible(False)
ax_marg_x1.spines['left'].set_visible(False)
ax_marg_x1.yaxis.set_ticks_position('left')
ax_marg_x1.xaxis.set_ticks_position('bottom')
ax_marg_x1.set_xlim([-3,3])
ax_marg_x1.set_ylim([0,1.4])
ax_marg_x1.set_xticks([-2., 0, 2.])
ax_marg_x1.set_xticklabels(['','',''])
ax_marg_x1.set_yticks([2])

ax_marg_x2.hist(M[:,2], nbins, alpha=0.5, density=True)
sol_n = np.zeros_like(ms)
for ip in range(pops):
    sol_n += (1/pops)*(1/np.sqrt(2*np.pi*ss**2))*np.exp(-(ms+a_m3[ip])**2/(2*ss**2))
ax_marg_x2.plot( ms,  sol_n,'k', lw=0.5)
#ax_marg_x2.plot(ms, (0.5/np.sqrt(2*np.pi*ss**2))*(np.exp(-(ms+np.sqrt(1-val**2))**2/(2*ss**2)) + np.exp(-(ms-np.sqrt(1-val**2))**2/(2*ss**2))), 'k')
ax_marg_x2.spines['top'].set_visible(False)
ax_marg_x2.spines['right'].set_visible(False)
ax_marg_x2.spines['left'].set_visible(False)
ax_marg_x2.yaxis.set_ticks_position('left')
ax_marg_x2.xaxis.set_ticks_position('bottom')
ax_marg_x2.set_xlim([-3,3])
ax_marg_x2.set_ylim([0,1.4])
ax_marg_x2.set_xticks([-2., 0, 2.])
ax_marg_x2.set_xticklabels(['','',''])
ax_marg_x2.set_yticks([2])

ax_marg_y0.hist(NNN[0,:], nbins, orientation="horizontal", alpha=0.5, density=True)
ss= val
Mu = olap*(1/np.sqrt(1-val**2))
sol_n = np.zeros_like(ms)
for ip in range(pops):
    sol_n += (1/pops)*(1/np.sqrt(2*np.pi*ss**2))*np.exp(-(ms+a_n1[ip])**2/(2*ss**2))
ax_marg_y0.plot(sol_n, ms,  'k', lw=0.5)
ax_marg_y0.spines['top'].set_visible(False)
ax_marg_y0.spines['right'].set_visible(False)
ax_marg_y0.spines['bottom'].set_visible(False)
ax_marg_y0.yaxis.set_ticks_position('left')
ax_marg_y0.xaxis.set_ticks_position('bottom')
ax_marg_y0.set_ylim([-yl,yl])
ax_marg_y0.set_xlim([0,1.5])
ax_marg_y0.set_yticks([-ylt, 0, ylt])
ax_marg_y0.set_yticklabels(['','',''])
ax_marg_y0.set_xticks([2])
ax_marg_y0.set_xticklabels([''])

ax_marg_y1.hist(NNN[1,:], nbins, orientation="horizontal", alpha=0.5, density=True)
sol_n = np.zeros_like(ms)
for ip in range(pops):
    sol_n += (1/pops)*(1/np.sqrt(2*np.pi*ss**2))*np.exp(-(ms+a_n2[ip])**2/(2*ss**2))
ax_marg_y1.plot(sol_n, ms,  'k', lw=0.5)
#ax_marg_y1.plot((0.5/np.sqrt(2*np.pi*ss**2))*(np.exp(-(ms+Mu)**2/(2*ss**2)) + np.exp(-(ms-Mu)**2/(2*ss**2))),ms,  'k')
ax_marg_y1.spines['top'].set_visible(False)
ax_marg_y1.spines['right'].set_visible(False)
ax_marg_y1.spines['bottom'].set_visible(False)
ax_marg_y1.yaxis.set_ticks_position('left')
ax_marg_y1.xaxis.set_ticks_position('bottom')
ax_marg_y1.set_ylim([-yl,yl])
ax_marg_y1.set_xlim([0,1.5])
ax_marg_y1.set_yticks([-ylt, 0, ylt])
ax_marg_y1.set_yticklabels(['','',''])
ax_marg_y1.set_xticks([2])
ax_marg_y1.set_xticklabels([''])

ax_marg_y2.hist(NNN[2,:], nbins, orientation="horizontal", alpha=0.5, density=True)
#ax_marg_y2.plot((0.5/np.sqrt(2*np.pi*ss**2))*(np.exp(-(ms+Mu)**2/(2*ss**2)) + np.exp(-(ms-Mu)**2/(2*ss**2))),ms,  'k')
sol_n = np.zeros_like(ms)
for ip in range(pops):
    sol_n += (1/pops)*(1/np.sqrt(2*np.pi*ss**2))*np.exp(-(ms+a_n3[ip])**2/(2*ss**2))
ax_marg_y2.plot(sol_n, ms,  'k', lw=0.5)
ax_marg_y2.spines['top'].set_visible(False)
ax_marg_y2.spines['right'].set_visible(False)
ax_marg_y2.spines['bottom'].set_visible(False)
ax_marg_y2.yaxis.set_ticks_position('left')
ax_marg_y2.xaxis.set_ticks_position('bottom')
ax_marg_y2.set_ylim([-yl,yl])
ax_marg_y2.set_xlim([0,1.5])
ax_marg_y2.set_yticks([-ylt, 0, ylt])
ax_marg_y2.set_yticklabels(['','',''])
ax_marg_y2.set_xticks([2])
ax_marg_y2.set_xticklabels([''])
plt.savefig('Th_Fig6B_1_A_bgoverlap.pdf')



#%%
#kaps1 = np.linspace(-2.5,2.5, 50)
#kaps2 = np.linspace(-2.5,2.5, 40)
#kaps3 = np.linspace(-2.5,2.5, 40)
#
#ksol = np.zeros((len(kaps1), len(kaps2), len(kaps3), 3))
#
#K1s, K2s, K3s = np.meshgrid(kaps1, kaps2,kaps3)
#def transf(K):
#    return(K*Prime(0, np.dot(K.T, K)))
#    
#E = np.zeros((len(kaps1), len(kaps2), len(kaps3)))

a_m1 = np.sqrt((1-val**2))           *np.array((1, 1, 1, 1, -1, -1, -1, -1))
#np.array((0., np.sqrt(8./9.), -np.sqrt(2./9.), -np.sqrt(2./9.), 0., -np.sqrt(8./9.), np.sqrt(2./9.), np.sqrt(2./9.)))
a_m2 = np.sqrt((1-val**2))            *np.array((1, -1, 1, -1, 1, -1, 1, -1))
#*np.array((0., 0., np.sqrt(2./3.), -np.sqrt(2./3.), 0., 0., -np.sqrt(2./3.), np.sqrt(2./3.), ))
a_m3 = np.sqrt((1-val**2))           *np.array((1, -1, -1, 1, 1, -1, -1, 1))     
# *np.array((1., -1./3., -1./3., -1./3., -1., 1./3., 1./3., 1./3.))

a_n1 = olap/np.sqrt((1-val**2))  *np.array((1, 1, 1, 1, -1, -1, -1, -1))   
#*np.array((0., np.sqrt(8./9.), -np.sqrt(2./9.), -np.sqrt(2./9.), 0., -np.sqrt(8./9.), np.sqrt(2./9.), np.sqrt(2./9.)))
a_n2 = olap/np.sqrt((1-val**2))    *np.array((1, -1, 1, -1, 1, -1, 1, -1)) 
#*np.array((0., 0., np.sqrt(2./3.), -np.sqrt(2./3.), 0., 0., -np.sqrt(2./3.), np.sqrt(2./3.), ))
a_n3 = olap/np.sqrt((1-val**2))     *np.array((1, -1, -1, 1, 1, -1, -1, 1))  
fps = np.zeros((3, 100))
allfps = np.zeros((3,100))
eigvals = np.zeros((3,100))

iX = 0

def give_f(K, a_m1, a_m2, a_m3, a_n1, a_n2, a_n3, val):
    ksol = - K
    for ip in range(pops):
        ksol[0] += (1/pops)*a_n1[ip]*Phi(a_m1[ip]*K[0]+a_m2[ip]*K[1]+a_m3[ip]*K[2], val**2*np.sum(K**2))
        ksol[1] += (1/pops)*a_n2[ip]*Phi(a_m1[ip]*K[0]+a_m2[ip]*K[1]+a_m3[ip]*K[2], val**2*np.sum(K**2))
        ksol[2] += (1/pops)*a_n3[ip]*Phi(a_m1[ip]*K[0]+a_m2[ip]*K[1]+a_m3[ip]*K[2], val**2*np.sum(K**2))
    norm = np.sqrt(np.sum(ksol**2))
    return(ksol, norm)
pops = 8
    
for tr in range(200):
    #print(tr)
    k1 = 4*np.random.rand()-2
    k2 = 4*np.random.rand()-2
    k3 = 4*np.random.rand()-2
    norm = 10
    K = np.array((k1, k2, k3))
    while norm > 0.001:
        ksol = - K
        for ip in range(pops):
            ksol[0] += (1/pops)*a_n1[ip]*Phi(a_m1[ip]*K[0]+a_m2[ip]*K[1]+a_m3[ip]*K[2], val**2*np.sum(K**2))
            ksol[1] += (1/pops)*a_n2[ip]*Phi(a_m1[ip]*K[0]+a_m2[ip]*K[1]+a_m3[ip]*K[2], val**2*np.sum(K**2))
            ksol[2] += (1/pops)*a_n3[ip]*Phi(a_m1[ip]*K[0]+a_m2[ip]*K[1]+a_m3[ip]*K[2], val**2*np.sum(K**2))
        norm = np.sqrt(np.sum(ksol**2))
        K = K+0.1*ksol
    cand1 = K
    norm = 10
    K = K + 0.1*np.random.randn(3)
    while norm > 0.001:
        ksol = - K
        for ip in range(pops):
            ksol[0] += (1/pops)*a_n1[ip]*Phi(a_m1[ip]*K[0]+a_m2[ip]*K[1]+a_m3[ip]*K[2], val**2*np.sum(K**2))
            ksol[1] += (1/pops)*a_n2[ip]*Phi(a_m1[ip]*K[0]+a_m2[ip]*K[1]+a_m3[ip]*K[2], val**2*np.sum(K**2))
            ksol[2] += (1/pops)*a_n3[ip]*Phi(a_m1[ip]*K[0]+a_m2[ip]*K[1]+a_m3[ip]*K[2], val**2*np.sum(K**2))
        norm = np.sqrt(np.sum(ksol**2))
        K = K+0.1*ksol
    cand = K
    
    if np.sum((cand-cand1)**2)<0.02:
        cand = 0.5*(cand+cand1)
        if np.min(np.sum((fps.T-cand)**2,1))>0.01:
            fps[:,iX]= cand
            iX +=1
            fps[:,iX]= -cand
            iX +=1
            Jac = np.zeros((3,3))
            per = 0.005
            for ipp in range(3):
                pert = np.zeros(3)
                pert[ipp] = per
                nex, ccc = give_f(K+pert, a_m1, a_m2, a_m3, a_n1, a_n2, a_n3, val) 
                fir, ccc = give_f(K-pert, a_m1, a_m2, a_m3, a_n1, a_n2, a_n3, val) 
                
                Jac[ipp,:] = (nex-fir)/(2*per)
                eigvals[:,iX-2] = np.linalg.eigvals(Jac)
                eigvals[:,iX-1] = eigvals[:,iX-2]
                
fps = fps[:,0:iX]

eigvals = eigvals[:,0:iX]

Rs = np.sqrt(np.sum(fps**2,0))
print(Rs)



appart = Rs>np.mean(Rs)
fp1s  = fps[:,appart]
fp2s  = fps[:,~appart]
eigvals1 = eigvals[:,appart]
eigvals2 = eigvals[:,~appart]

iX = 0
allfps = np.zeros((3,100))
for tr in range(100):
    #print(tr)
    k1 = 4*np.random.rand()-2
    k2 = 4*np.random.rand()-2
    k3 = 4*np.random.rand()-2
    norm = 10
    max_iter = 150
    ite = 0
    dx = 0.1
    K = np.array((k1, k2, k3))
        
    while norm>0.001 and ite<max_iter:
        
        Kold = K
        ksol, norm = give_f(K, a_m1, a_m2, a_m3, a_n1, a_n2, a_n3, val)
        

        K1 = K+dx*np.array((1,0,0))
        ksol1, norm1 = give_f(K1, a_m1, a_m2, a_m3, a_n1, a_n2, a_n3, val)
        K1m = K+dx*np.array((-1,0,0))
        ksol1m, norm1m = give_f(K1m, a_m1, a_m2, a_m3, a_n1, a_n2, a_n3, val)
        K2 = K+dx*np.array((0,1,0))
        ksol2, norm2 = give_f(K2, a_m1, a_m2, a_m3, a_n1, a_n2, a_n3, val)
        K2m = K+dx*np.array((0,-1,0))
        ksol2m, norm2m = give_f(K2m, a_m1, a_m2, a_m3, a_n1, a_n2, a_n3, val)
        K3 = K+dx*np.array((0,0,1))
        ksol3, norm3 = give_f(K3, a_m1, a_m2, a_m3, a_n1, a_n2, a_n3, val)
        K3m = K+dx*np.array((0,0,-1))
        ksol3m, norm3m = give_f(K3m, a_m1, a_m2, a_m3, a_n1, a_n2, a_n3, val)
                
        norms = np.array((norm1, norm1m, norm2, norm2m, norm3, norm3m))
        if np.min(norms)<norm:   
            if norm1==np.min(norms):
                K = K1
                norm = norm1
            elif norm2 == np.min(norms):
                K = K2
                norm = norm2   
            elif norm3 == np.min(norms):
                K = K3
                norm = norm3
            elif norm1m == np.min(norms):
                K = K1m
                norm = norm1m
            elif norm2m == np.min(norms):
                K = K2m
                norm = norm2m
            elif norm3m == np.min(norms):
                K = K3m
                norm = norm3m
        else:
            if dx>0.0001:
                dx = dx*0.5
        ite +=1
    
    if norm<0.001:  
        if np.min(np.sum((allfps.T-K)**2,1))>0.03:
            if np.min(np.sum((fps.T-K)**2,1))>0.03: 
                allfps[:,iX]= K
                iX +=1
                allfps[:,iX]= -K
                iX +=1
allfps = allfps[:,0:iX]                

#%%
from mpl_toolkits.mplot3d import Axes3D
import mpl_toolkits.mplot3d as a3
import matplotlib.colors as colors
import scipy as sp
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d', azim=-59, elev=10)

verts = fp1s.T
#

top = 3
bottom = 2
no = 0
ne = 4
so = 5
se = 1
faces = np.array([ 
   [no, top, ne], [so, top, se], [no, so, top], [ne, se, top],     
   [no, bottom,  ne], [so, bottom, se], [no,so, bottom], [ne,se,bottom]
   ])


    

fne = 1
fse = 3
fno = 7
fso = 4
bne = 5
bse = 6
bno = 2
bso = 0
verts2 = fp2s.T
faces2 = np.array([ 
   [fne,  fno, fso,fse], [bne, bse, bso, bno], [fne, fno, bno, bne], [fse, fso, bso, bse],     
   [fne, fse, bse, bne], [fno, fso, bso, bno]
   ])

for i in np.arange(len(faces)):
    square=[ verts[faces[i,0]], verts[faces[i,1]], verts[faces[i, 2]]]
    face = a3.art3d.Poly3DCollection([square], alpha=0.5)
    face.set_color(colors.rgb2hex(0.6*np.array((1., 0., 0.))+0.4*np.random.rand(3)))
    face.set_edgecolor('k')
    face.set_linewidth(1.)
    face.set_alpha(0.5)
    ax.add_collection3d(face)
for i in np.arange(len(faces2)):
    square2=[ verts2[faces2[i,0]], verts2[faces2[i,1]], verts2[faces2[i, 2]], verts2[faces2[i, 3]]]
    face = a3.art3d.Poly3DCollection([square2], alpha=0.5)
    face.set_color(colors.rgb2hex(0.6*np.array((0., 0., 1.))+0.4*np.random.rand(3)))
    face.set_edgecolor('k')
    face.set_linewidth(1.)
    face.set_alpha(0.5)
    ax.add_collection3d(face)
#ax.scatter(fp2s[0,iXX], fp2s[1,iXX], fp2s[2,iXX], s=100, c='w', edgecolor='k')
if np.sum(allfps**2)>0:
    ax.scatter(allfps[0,:], allfps[1,:], allfps[2,:], s=10, c='w', edgecolor='k')
if np.std(Rs)>0.01:

    for ip in range(np.shape(fp1s)[1]):
        ax.scatter(fp1s[0,ip], fp1s[1,ip], fp1s[2,ip], s=30, c='C0', edgecolor='k')
        #plt.plot([0,fp1s[0,ip]], [0,fp1s[1,ip]], [0,fp1s[2,ip]], c='C0')
        for ip2 in range(np.shape(fp1s)[1]):
            dist = np.sqrt(np.sum((fp1s[:,ip]-fp1s[:,ip2])**2))
            #if dist<10.:
            #    plt.plot([fp1s[0,ip], fp1s[0,ip2]], [fp1s[1,ip], fp1s[1,ip2]], [fp1s[2,ip], fp1s[2,ip2]], c='C0')
    #    
    ##%%
    #fig = plt.figure()
    #ax = fig.add_subplot(111, projection='3d')       
    for ip in range(np.shape(fp2s)[1]):
        ax.scatter(fp2s[0,ip], fp2s[1,ip], fp2s[2,ip], s=30, c='C1', edgecolor='k')
        #plt.plot([0,fp2s[0,ip]], [0,fp2s[1,ip]], [0,fp2s[2,ip]], c='C1')
        for ip2 in range(np.shape(fp2s)[1]):
            dist = np.sqrt(np.sum((fp2s[:,ip]-fp2s[:,ip2])**2))
#            if dist<10.:
#                plt.plot([fp2s[0,ip], fp2s[0,ip2]], [fp2s[1,ip], fp2s[1,ip2]], [fp2s[2,ip], fp2s[2,ip2]], c='C1')
else:
    for ip in range(np.shape(fps)[1]):
        if np.sum(fps[:,ip]**2)>0.2:
            ax.scatter(fps[0,ip], fps[1,ip], fps[2,ip], s=30, c='k')
            #plt.plot([0,fp1s[0,ip]], [0,fp1s[1,ip]], [0,fp1s[2,ip]], c='C0')
            for ip2 in range(np.shape(fps)[1]):
                dist = np.sqrt(np.sum((fps[:,ip]-fps[:,ip2])**2))
                
                #if dist<1.:
                #    plt.plot([fps[0,ip], fps[0,ip2]], [fps[1,ip], fps[1,ip2]], [fps[2,ip], fps[2,ip2]], c='C0')
ax.set_xlabel(r'$\kappa_1$')
ax.set_ylabel(r'$\kappa_2$')
ax.set_zlabel(r'$\kappa_3$')
ax.dist=11
#ax.w_xaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))
#ax.w_yaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))
#ax.w_zaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))
ax.set_xticks([-5,  5])
ax.set_yticks([-5,  5])
ax.set_zticks([-5,  5])
#

plt.savefig('Th_Fig6B_1_B.pdf')
#ip = 2
##ax.scatter(fp2s[0,ip], fp2s[1,ip], fp2s[2,ip], s=150, c='k')
#print(eigvals2[:,ip])

#%%
K = fps[:,0]
ksol = - K
steps = 100
for st in range(steps):
    ksol = -K
    for ip in range(pops):
        ksol[0] += (1/pops)*a_n1[ip]*Phi(a_m1[ip]*K[0]+a_m2[ip]*K[1]+a_m3[ip]*K[2], val**2*np.sum(K**2))
        ksol[1] += (1/pops)*a_n2[ip]*Phi(a_m1[ip]*K[0]+a_m2[ip]*K[1]+a_m3[ip]*K[2], val**2*np.sum(K**2))
        ksol[2] += (1/pops)*a_n3[ip]*Phi(a_m1[ip]*K[0]+a_m2[ip]*K[1]+a_m3[ip]*K[2], val**2*np.sum(K**2))
    norm = np.sqrt(np.sum(ksol**2))
    #print(norm)
    K = K+0.05*ksol
#%%

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d', azim=-59, elev=10) 

plt.xlabel('$\kappa_1$')
plt.ylabel('$\kappa_2$')
ax.set_zlabel('$\kappa_3$')

#plt.scatter([ 0, fp2/np.sqrt(2), fp2/np.sqrt(2), -fp2/np.sqrt(2), -fp2/np.sqrt(2)], \
#            [0, fp2/np.sqrt(2), -fp2/np.sqrt(2), fp2/np.sqrt(2), -fp2/np.sqrt(2)], s=50, edgecolor='k', facecolor='w', linewidth=1., zorder=4)
#plt.scatter([ 0, 0, fp1, -fp1], [fp11, -fp11, 0, 0], s=70, edgecolor='w', facecolor='k', linewidth=1., zorder=5)

Nn = 1200

inkap1 = np.linspace(-10.2, 10.2, 4)
inkap3 = np.linspace(-10.2, 10.2, 4)
inkap2 = np.linspace(-10.2, 10.2, 4)


dt = 0.15
time = np.arange(0, 34, dt)

for i in np.arange(len(faces)):
    square=[ verts[faces[i,0]], verts[faces[i,1]], verts[faces[i, 2]]]
    face = a3.art3d.Poly3DCollection([square], alpha=0.2)
    face.set_color(colors.rgb2hex(0.6*np.array((1., 0., 0.))+0.4*np.random.rand(3)))
    face.set_edgecolor('k')
    face.set_linewidth(1.)
    face.set_alpha(0.5)
    ax.add_collection3d(face)

for i in np.arange(len(faces2)):
    square2=[ verts2[faces2[i,0]], verts2[faces2[i,1]], verts2[faces2[i, 2]], verts2[faces2[i, 3]]]
    face = a3.art3d.Poly3DCollection([square2], alpha=0.25)
    face.set_color(colors.rgb2hex(0.6*np.array((0., 0., 1.))+0.4*np.random.rand(3)))
    face.set_edgecolor('k')
    face.set_linewidth(1.)
    face.set_alpha(0.5)
    ax.add_collection3d(face)
    

for trials in range(1):
    m1, m2, m3, n1, n2, n3 = give_vecs(N = Nn, olap=olap, val=val)    

    M = np.vstack((m1, m2, m3)).T
    N = np.vstack((n1, n2, n3))

    J = np.dot(M, N)/Nn
    
    cC =  np.ones(3)*0.6
    
    fps = np.zeros((3,len(inkap1)*len(inkap2)*len(inkap3)))
    iX = 0
    for ik1, ink1 in enumerate(inkap1):
        #print(ik1)
        for ik2, ink2 in enumerate(inkap2):
            for ik3, ink3 in enumerate(inkap3):
                sk1 = np.zeros_like(time)
                sk2 = np.zeros_like(time)
                sk3 = np.zeros_like(time)
                
                
                x0 = ink1*M[:,0] + ink2*M[:,1] +ink3*M[:,2]
                sk1[0] = np.mean(M[:,0]*x0)
                sk2[0] = np.mean(M[:,1]*x0)
                sk3[0] = np.mean(M[:,2]*x0)
                
                for it, ti in enumerate(time[:-1]):
                    x = x0 + dt*(-x0 + np.dot(J, np.tanh(x0)))+np.sqrt(dt)*0.1*np.random.randn(len(x0))
                    sk1[it+1] = np.mean(M[:,0]*x)/np.mean(M[:,0]**2)
                    sk2[it+1] = np.mean(M[:,1]*x)/np.mean(M[:,1]**2)
                    sk3[it+1] = np.mean(M[:,2]*x)/np.mean(M[:,2]**2)
                    
                    x0 = x
                ax.plot(sk1, sk2,sk3, c=cC)
                ax.scatter(sk1[0], sk2[0], sk3[0], s=10, facecolor=cC)
                ax.scatter(sk1[-1], sk2[-1], sk3[-1], s=25, facecolor=cC, edgecolor='k', zorder=3)
                cand  = np.array((sk1[-1], sk2[-1], sk3[-1]))
                if np.min(np.mean((fps.T-cand)**2,1))>0.1:
                    fps[:,iX] = cand
                    iX+=1
    fps = fps[:,0:iX+1]
                
for ip in range(np.shape(fps)[1]):
    #ax.scatter(fps[0,ip], fps[1,ip], fps[2,ip], s=30, c='k')
    #plt.plot([0,fp1s[0,ip]], [0,fp1s[1,ip]], [0,fp1s[2,ip]], c='C0')
    for ip2 in range(np.shape(fps)[1]):
        dist = np.sqrt(np.sum((fps[:,ip]-fps[:,ip2])**2))
        #if dist<10.:
        #    plt.plot([fps[0,ip], fps[0,ip2]], [fps[1,ip], fps[1,ip2]], [fps[2,ip], fps[2,ip2]], c='k', lw=0.5)
ax.set_xticks([-5, 0, 5])
ax.set_zticks([-5, 0, 5])
ax.set_yticks([-5, 0, 5])

#ax.set_ylim([-8, 8])
#ax.set_xlim([-8, 8])
#ax.set_zlim([-8, 8])        
ax.dist=11
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
#ax.yaxis.set_ticks_position('left')
#ax.xaxis.set_ticks_position('bottom')  
plt.savefig('Th_Fig6B_1_D.pdf')    

##%%
#m1, m2, m3, n1, n2, n3 = give_vecs(N = Nn, olap=4., val=0.)    
#
#M = np.vstack((m1, m2, m3)).T
#N = np.vstack((n1, n2, n3))
#
#J = np.dot(M, N)/Nn
#Av, Vv = np.linalg.eig(J)
#
#plt.scatter(np.real(Av), np.imag(Av))

#%%
Am = np.array((a_m1, a_m2, a_m3))
An = np.array((a_n1, a_n2, a_n3))
Rn = np.sum(An**2,0)

#%%
#fig = plt.figure()
from mpl_toolkits.mplot3d import Axes3D
#ax = fig.add_subplot(111, projection='3d') 
#ax.scatter(fps[0,:], fps[1,:F], fps[2,:])
#%%
m1, m2, m3, n1, n2, n3 = give_vecs(N = Nn, olap=olap, val=val) 
Pops = 8
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d', azim=-83, elev = 31)
ax.scatter(n1, n2, n3, c='r', rasterized=True, alpha=0.5) 

for ip in range(Pops):
    for ip2 in range(Pops):
        dist = np.sqrt(np.sum((An[:,ip]-An[:,ip2])**2))
        if dist<28.:
            ax.plot([An[0,ip], An[0,ip2]], [An[1,ip], An[1,ip2]],[An[2,ip], An[2,ip2]],c='k')
ax.scatter(An[0,0:Pops], An[1,0:Pops], An[2,0:Pops], s=40, edgecolor='g', facecolor='k', zorder=4)
            
ax.set_xlabel(r'$n_i^{(1)}$')
ax.set_ylabel(r'$n_i^{(2)}$')
ax.set_zlabel(r'$n_i^{(3)}$')
ax.dist=11
#ax.w_xaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))
#ax.w_yaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))
#sax.w_zaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))
ax.set_xticks([-8,  8])
ax.set_yticks([-8,  8])
ax.set_zticks([-8,  8])

plt.savefig('Th_Fig6B_1_C.pdf')

