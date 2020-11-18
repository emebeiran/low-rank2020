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


np.random.seed(2)
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
#s_mn1 = 0.5
s_mnI = np.array((-10., 4.5))
s_m2 = np.array((1.98, 0.02))
s_mn = s_mnI/s_m2

color1 = np.array((31, 127, 17))/256
color2 = np.array((129, 34, 141))/256
clrs = np.zeros((3,2))
clrs[:,0] = color1
clrs[:,1] = color2

m1 = np.random.randn(N)
m1[0:N//2] = np.sqrt(s_m2[0])*m1[0:N//2]/np.std(m1[0:N//2])
m1[N//2:] = np.sqrt(s_m2[1])*m1[N//2:]/np.std(m1[N//2:])
n1 = np.zeros_like(m1)
ms = np.linspace(-80.5, 80.5, 4200)

n1[0:N//2] = s_mn[0]*m1[0:N//2]  +3*np.random.randn(N//2)
n1[N//2:] = s_mn[1]*m1[N//2:]    +3*np.random.randn(N//2)

fig = plt.figure(figsize=[3.6, 3.6])
gs = GridSpec(4,4)

ax_joint = fig.add_subplot(gs[1:4,0:3])
ax_marg_x = fig.add_subplot(gs[0,0:3])
ax_marg_y = fig.add_subplot(gs[1:4,3])

#ax_joint.scatter(m2, n2, s=20, alpha=0.5, label=r'$\sigma_{mn} = 1.2$')
ax_joint.scatter(m1[0:N//2], n1[0:N//2], s=20, alpha=0.5, label=r'$\sigma_{mn} =$'+str(s_mn[0]), color =color1)
ax_joint.scatter(m1[N//2:], n1[N//2:], s=20, alpha=0.5, label=r'$\sigma_{mn} =$'+str(s_mn[1]), color =color2)

ax_joint.plot(ms, s_mn[0]*ms, '--', color ='k', lw=1)
ax_joint.plot(ms, s_mn[1]*ms, '--', color ='k', lw=1)

#ax_joint.legend(locolor =6,frameon=False, handletextpad=-0.1)
ax_joint.spines['top'].set_visible(False)
ax_joint.spines['right'].set_visible(False)
ax_joint.yaxis.set_ticks_position('left')
ax_joint.xaxis.set_ticks_position('bottom')

yl = 70.9
ax_joint.set_ylim([-yl, yl])
ax_joint.set_xlim([-4, 4])

ax_joint.set_xlabel(r'$m_i$')
ax_joint.set_ylabel(r'$n_i$')

ax_marg_x.hist(m1[0:N//2], nbins, alpha=0.5, density=True, color=color1)
ax_marg_x.hist(m1[N//2:], nbins, alpha=0.5, density=True, color=color2)

#ax_marg_x.hist(m2, nbins, alpha=0.5, density=True)

ax_marg_x.spines['top'].set_visible(False)
ax_marg_x.spines['right'].set_visible(False)
ax_marg_x.yaxis.set_ticks_position('left')
ax_marg_x.xaxis.set_ticks_position('bottom')

a1 = np.exp(-ms**2/(2*s_m2[0]))/np.sqrt(2*np.pi*s_m2[0])
a2 = np.exp(-ms**2/(2*s_m2[1]))/np.sqrt(2*np.pi*s_m2[1])

#ax_marg_x.plot(ms, 0.5*(a1+a2),  lw=2, color ='k')
ax_marg_x.plot(ms, a1,  lw=2, color =color1)
ax_marg_x.plot(ms, a2,  lw=2, color =color2)
ax_marg_x.set_xlim([-4, 4])

ax_marg_y.hist(n1[0:N//2], nbins, orientation="horizontal", alpha=0.5, density=True, color=color1)
ax_marg_y.hist(n1[N//2:], nbins, orientation="horizontal", alpha=0.5, density=True, color=color2)

ss = s_mn[0]**2*np.var(m1[0:N//2])+3**2 #np.sqrt(s_m2[0])*m1[0:N//2]/np.std(m1[0:N//2])
a1 = np.exp(-ms**2/(2*ss))/np.sqrt(2*np.pi*ss)

ss = s_mn[1]**2*np.var(m1[N//2:])+3**2
a2 = np.exp(-ms**2/(2*ss))/np.sqrt(2*np.pi*ss)
#ax_marg_y.plot( 0.5*np.exp(-ms**2/(2*s_mn[0]**2))/np.sqrt(2*np.pi*s_mn[0]**2)+ 0.5*np.exp(-ms**2/(2*s_mn[1]**2))/np.sqrt(2*np.pi*s_mn[1]**2),ms, lw=2, color ='k')
ax_marg_y.plot( a1,ms, lw=2,  color =color1)
ax_marg_y.plot( a2,ms, lw=2,  color =color2)


ax_marg_y.spines['top'].set_visible(False)
ax_marg_y.spines['right'].set_visible(False)
ax_marg_y.yaxis.set_ticks_position('left')
ax_marg_y.xaxis.set_ticks_position('bottom')
ax_marg_y.set_ylim([-yl, yl])

ax_marg_y.set_xticklabels(['0', '0.5'])

## Turn off tick labels on marginals
plt.setp(ax_marg_x.get_xticklabels(), visible=False)
plt.setp(ax_marg_y.get_yticklabels(), visible=False)

## Set labels on marginals
ax_marg_y.set_xlabel(r'$P(n_i)$')
ax_marg_x.set_ylabel(r'$P(m_i)$')

plt.savefig('Th_Fig5_A.pdf')
plt.show()

#%%
s_mnI = np.array((-10., 4.5))
s_m2 = np.array((1.98, 0.02))
s_mn = s_mnI/s_m2


kappas = np.linspace(-10, 10, 500)
Fk1 = np.zeros_like(kappas)
Fk2 = np.zeros_like(kappas)
Fk1MC = np.zeros_like(kappas)
Fk2MC = np.zeros_like(kappas)
NN = 100000
for ik, ka in enumerate(kappas):
    Fk1[ik]= s_mnI[0]*ka*Prime(0, s_m2[0]*ka**2) 
    Fk2[ik]= s_mnI[1]*ka*Prime(0, s_m2[1]*ka**2)
    
    m1 = np.random.randn(NN)
    m1[0:NN//2] = np.sqrt(s_m2[0])*m1[0:NN//2]/np.std(m1[0:NN//2])
    m1[NN//2:] = np.sqrt(s_m2[1])*m1[NN//2:]/np.std(m1[NN//2:])
    n1 = np.zeros_like(m1)
    ms = np.linspace(-80.5, 80.5, 1200)
    
    n1[0:NN//2] = s_mn[0]*m1[0:NN//2]  +0.3*np.random.randn(NN//2)
    n1[NN//2:] = s_mn[1]*m1[NN//2:]    +0.3*np.random.randn(NN//2)
    Fk1MC[ik] = np.mean(n1[0:NN//2]*np.tanh(m1[0:NN//2]*ka)) 
    Fk2MC[ik] = np.mean(n1[NN//2:]*np.tanh(m1[NN//2:]*ka))
    
fig = plt.figure()
ax = fig.add_subplot(111)
#plt.plot(kappas, Fk2, lw=2)
plt.plot(kappas, -kappas+0.5*(Fk1MC+Fk2MC), color ='C3', lw=2)
#plt.plot(kappas, -kappas+0.5*(Fk1MC+Fk2MC), '--',  color ='C3', lw=2)


Fk = -kappas+0.5*(Fk1MC+Fk2MC)
Fk[Fk>0] = 10
Fk[Fk<0] = -10
kappasm1 = kappas[:-1]
sl = kappasm1[np.diff(Fk)>1]+0.5*(kappas[1]-kappas[0])
sl2 = kappasm1[np.diff(Fk)<-1]+0.5*(kappas[1]-kappas[0])

plt.plot(kappas, kappas*0, '--k')
plt.scatter(sl,np.zeros_like(sl), edgecolor='C3', color='w', s=60, lw=2, zorder=4)

k0s = kappas[np.argmin(np.abs(Fk[kappas<0.1]))]
plt.scatter(sl2,np.zeros_like(sl2), edgecolor='k', color='C3', s=60, lw=1, zorder=4)

plt.ylim([-1.5, 1.5])
plt.ylabel(r'dynamics $d\kappa / dt$')
plt.xlabel(r'latent variable $\kappa$')
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.yaxis.set_ticks_position('left')
ax.xaxis.set_ticks_position('bottom')  
plt.savefig('Th_Fig5_B.pdf')
plt.show()


#%%
s_mnI = np.array((-10., 4.5))
s_m2 = np.array((1.98, 0.02))
s_mn = s_mnI/s_m2


kappas = np.linspace(-10, 10, 500)
Fk1 = np.zeros_like(kappas)
Fk2 = np.zeros_like(kappas)
Fk1MC = np.zeros_like(kappas)
Fk2MC = np.zeros_like(kappas)
Fk1MC2 = np.zeros_like(kappas)
Fk2MC2 = np.zeros_like(kappas)
NN = 100000
for ik, ka in enumerate(kappas):
    Fk1[ik]= s_mnI[0]*ka*Prime(0, s_m2[0]*ka**2) 
    Fk2[ik]= s_mnI[1]*ka*Prime(0, s_m2[1]*ka**2)
    
    m1 = np.random.randn(NN)
    m1[0:NN//2] = np.sqrt(s_m2[0])*m1[0:NN//2]/np.std(m1[0:NN//2])
    m1[NN//2:] = np.sqrt(s_m2[1])*m1[NN//2:]/np.std(m1[NN//2:])
    n1 = np.zeros_like(m1)
    ms = np.linspace(-80.5, 80.5, 1200)
    
    n1[0:NN//2] = s_mn[0]*m1[0:NN//2]  +0.3*np.random.randn(NN//2)
    n1[NN//2:] = s_mn[1]*m1[NN//2:]    +0.3*np.random.randn(NN//2)
    Fk1MC[ik] = np.mean((1./np.cosh(m1[0:NN//2]*ka))**2)  
    Fk1MC2[ik] = np.mean(n1[0:NN//2]*np.tanh(m1[0:NN//2]*ka)) 
    Fk2MC[ik] = np.mean((1./np.cosh(m1[NN//2:]*ka))**2)   #np.mean(n1[NN//2:]*np.tanh(m1[NN//2:]*ka))
    Fk2MC2[ik] = np.mean(n1[NN//2:]*np.tanh(m1[NN//2:]*ka))
    
fig = plt.figure()
ax = fig.add_subplot(111)
#plt.plot(kappas, -kappas, color ='C3', lw=2)
plt.plot(kappas, Fk1MC, color =color1, lw=2)
plt.plot(kappas, Fk2MC, color =color2, lw=2)

Fk = -kappas+0.5*(Fk1MC2+Fk2MC2)
Fk[Fk>0] = 10
Fk[Fk<0] = -10
#%%
kappasm1 = kappas[:-1]
sl = kappasm1[np.diff(Fk)>1]+0.5*(kappas[1]-kappas[0])
sl2 = kappasm1[np.diff(Fk)<-1]+0.5*(kappas[1]-kappas[0])

plt.plot(kappas, kappas*0, '--k')
#plt.scatter(sl,np.zeros_like(sl), edgecolor='C3', color='w', s=60, lw=2, zorder=4)

k0s = kappas[np.argmin(np.abs(Fk[kappas<0.1]))]
#plt.scatter(sl2,np.zeros_like(sl2), edgecolor='k', color='C3', s=60, lw=1, zorder=4)

plt.ylim([0, 1.05])
plt.yticks([0, 0.5, 1.])
plt.ylabel(r'gain $\left\langle \phi^\prime \right\rangle$')
plt.xlabel(r'latent variable $\kappa$')
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.yaxis.set_ticks_position('left')
ax.xaxis.set_ticks_position('bottom')  
plt.savefig('Th_Fig5_C2.pdf')
plt.show()



#%%
NN = 400
    
m1 = np.random.randn(NN)
m1[0:NN//2] = np.sqrt(s_m2[0])*m1[0:NN//2]/np.std(m1[0:NN//2])
m1[NN//2:] = np.sqrt(s_m2[1])*m1[NN//2:]/np.std(m1[NN//2:])
n1 = np.zeros_like(m1)
    
#
J1 = np.dot(m1[:,None], n1[:,None].T)
#J2 = np.dot(m2[:,None], n2[:,None].T)

time = np.linspace(0, 15, 200)
dt = time[1]-time[0]

nS = 10
xs1 = np.zeros((len(time), nS))
xs2 = np.zeros((len(time), nS))

x0 = 0.5*np.random.randn(N)+10*np.random.randn()*m1
#x02 = 0.5*np.random.randn(N)
k1 = np.zeros(len(time))
#k2 = np.zeros(len(time))
for it, t in enumerate(time[:-1]):    
    xs1[it] = x0[0:nS]
    #xs2[it] = x02[0:nS]
    k1[it] = np.mean(m1*x0)
    #k2[it] = np.mean(m2*x02)
    
    x = x0 + dt*(-x0+np.dot(J1, np.tanh(x0)/N))
    #x2 = x02 + dt*(-x02+np.dot(J2, np.tanh(x02)/N))
    
    x0=x
    #x02=x2
    
    xs1[it+1] = x0[0:nS]
    #xs2[it+1] = x02[0:nS]
    k1[it+1] = np.mean(m1*x0)
    #k2[it+1] = np.mean(m2*x02)
#fig = plt.figure()
#ax = fig.add_subplot(111)
##plt.plot(time, xs2, color ='C0')
#plt.plot(time, xs1, color ='C1')
#
#ax.spines['top'].set_visible(False)
#ax.spines['right'].set_visible(False)
#ax.yaxis.set_ticks_position('left')
#ax.xaxis.set_ticks_position('bottom')  
#plt.ylabel(r'activation $x_i\left(t\right)$')
#plt.xlabel(r'time')
#plt.savefig('Th_Fig5_C2.pdf')

#%%

fig = plt.figure()
ax = fig.add_subplot(111)

trials = 20
k0s = np.linspace(-10, 10, trials)
for rep in range(2):

    
    N = 2000
    
    try_Ms = 100
    
    s_mnI = np.array((-10., 4.5))
    s_m2 = np.array((1.98, 0.02))
    s_mn = s_mnI/s_m2
    
    targ0 = 10
    for try_M in range(try_Ms):
        m1 = np.random.randn(N)
        m1[0:N//2] = np.sqrt(s_m2[0])*m1[0:N//2]/np.std(m1[0:N//2])
        m1[N//2:] = np.sqrt(s_m2[1])*m1[N//2:]/np.std(m1[N//2:])
        n1 = np.zeros_like(m1)
        ms = np.linspace(-80.5, 80.5, 1200)
        
        n1[0:N//2] = s_mn[0]*m1[0:N//2]  +0.3*np.random.randn(N//2)
        n1[N//2:] = s_mn[1]*m1[N//2:]    +0.3*np.random.randn(N//2)
    
        targ = (np.mean(m1[0:N//2]*n1[0:N//2]) - s_mnI[0])**2+(np.mean(m1[N//2:]*n1[N//2:]) \
                - s_mnI[1])**2+ (np.mean(m1[0:N//2]*n1[N//2:])) **2+ (np.mean(m1[N//2:]*n1[0:N//2]))**2
        if targ<targ0:
            m1S = m1
            n1S = n1
    print(targ)
    print('---')
    
    
    m1 = m1S
    n1 = n1S
        
    for tr in range(trials):
    
        m1 = m1S
        n1 = n1S
        
    
        J1 = np.dot(m1[:,None], n1[:,None].T)
        J1s1 = J1[0:N, 0:N]
        J1s2 = J1[N:, N:]
    #    J2 = np.dot(m2[:,None], n2[:,None].T)
        key = (16*np.random.rand()-8)
        x0 = 0.5*np.random.randn(N)+m1*k0s[tr]
    #    x02 = 0.5*np.random.randn(N)+0.05*m2*np.random.randn()
        k1 = np.zeros(len(time))
    #    k2 = np.zeros(len(time))
        for it, t in enumerate(time[:-1]):    
            xs1[it] = x0[0:nS]
    #        xs2[it] = x02[0:nS]
            k1[it] = np.mean(m1*x0)
    #        k2[it] = np.mean(m2*x02)
            
            x = x0 + dt*(-x0+np.dot(J1, np.tanh(x0)/N))
    #        x2 = x02 + dt*(-x02+np.dot(J2, np.tanh(x02)/N))
            
            x0=x
    #        x02=x2
            
            k1[it+1] = np.mean(m1*x0)
    #        k2[it+1] = np.mean(m2*x02)
        plt.plot(time, k1, color =[0.4,0.4,0.4])
    #    plt.plot(time, k2, color ='C0')
        
    plt.plot(time, sl2[0]*np.ones_like(time), lw=4, alpha=0.2, color ='C3')
    plt.plot(time, sl2[1]*np.ones_like(time), lw=4, alpha=0.2, color ='C3')
    plt.plot(time, sl2[2]*np.ones_like(time), lw=4, alpha=0.2, color ='C3')
    
    plt.plot(time, sl[0]*np.ones_like(time), '--', lw=4, alpha=0.2, color ='C3')
    plt.plot(time, sl[1]*np.ones_like(time), '--', lw=4, alpha=0.2, color ='C3')

ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.yaxis.set_ticks_position('left')
ax.xaxis.set_ticks_position('bottom')  
plt.ylabel(r'latent variable $\kappa$')
plt.xlabel(r'time')
plt.savefig('Th_Fig5_D.pdf')
