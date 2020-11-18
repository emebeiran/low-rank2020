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

ss1 = np.sqrt(s_mn1+0.3**2)
ss2 = np.sqrt(s_mn2+0.3**2)

fig = plt.figure(figsize=[3.6, 3.6])

gs = GridSpec(4,4)

ax_joint = fig.add_subplot(gs[1:4,0:3])
ax_marg_x = fig.add_subplot(gs[0,0:3])
ax_marg_y = fig.add_subplot(gs[1:4,3])



ax_joint.scatter(m1, n1, s=20,  alpha=0.5, label=r'$\sigma_{mn} = 0.5$')

ax_joint.plot(ms, s_mn1*ms, '--', c='k', lw=1)
ax_joint.plot(ms, ms,  c='k', lw=1)


ax_joint.legend(loc=2,frameon=False, handletextpad=-0.1)
ax_joint.spines['top'].set_visible(False)
ax_joint.spines['right'].set_visible(False)
ax_joint.yaxis.set_ticks_position('left')
ax_joint.xaxis.set_ticks_position('bottom')

ax_joint.set_ylim([-4.2, 4.2])
ax_joint.set_xlabel(r'$m_i$')
ax_joint.set_ylabel(r'$n_i$')

ax_marg_x.hist(m2, nbins, alpha=0.5, density=True)

ax_marg_x.spines['top'].set_visible(False)
ax_marg_x.spines['right'].set_visible(False)
ax_marg_x.yaxis.set_ticks_position('left')
ax_marg_x.xaxis.set_ticks_position('bottom')
ax_marg_x.plot( ms, np.exp(-ms**2/(2*1**2))/np.sqrt(2*np.pi*1**2), c='C0', lw=2)

ax_marg_y.hist(n1, nbins, orientation="horizontal", alpha=0.5, density=True)

ax_marg_y.plot( np.exp(-ms**2/(2*ss1**2))/np.sqrt(2*np.pi*ss1**2),ms, lw=2, c='C0',)


ax_marg_y.spines['top'].set_visible(False)
ax_marg_y.spines['right'].set_visible(False)
ax_marg_y.yaxis.set_ticks_position('left')
ax_marg_y.xaxis.set_ticks_position('bottom')
ax_marg_y.set_ylim([-4.2, 4.2])

ax_marg_y.set_xticklabels(['0', '0.5'])

## Turn off tick labels on marginals
plt.setp(ax_marg_x.get_xticklabels(), visible=False)
plt.setp(ax_marg_y.get_yticklabels(), visible=False)

## Set labels on marginals
ax_marg_y.set_xlabel(r'$P(n_i)$')
ax_marg_x.set_ylabel(r'$P(m_i)$')

plt.savefig('Th_Fig1_A_11.pdf')

#%%
fig = plt.figure(figsize=[3.6, 3.6])

gs = GridSpec(4,4)

ax_joint = fig.add_subplot(gs[1:4,0:3])
ax_marg_x = fig.add_subplot(gs[0,0:3])
ax_marg_y = fig.add_subplot(gs[1:4,3])



ax_joint.scatter(m2, n2, s=20,  alpha=0.5, label=r'$\sigma_{mn} = 1.2$')

ax_joint.plot(ms, s_mn2*ms, '--', c='k', lw=1)
ax_joint.plot(ms, ms,  c='k', lw=1)


ax_joint.legend(loc=2,frameon=False, handletextpad=-0.1)
ax_joint.spines['top'].set_visible(False)
ax_joint.spines['right'].set_visible(False)
ax_joint.yaxis.set_ticks_position('left')
ax_joint.xaxis.set_ticks_position('bottom')

ax_joint.set_ylim([-4.2, 4.2])
ax_joint.set_xlabel(r'$m_i$')
ax_joint.set_ylabel(r'$n_i$')

ax_marg_x.hist(m2, nbins, alpha=0.5, density=True)

ax_marg_x.spines['top'].set_visible(False)
ax_marg_x.spines['right'].set_visible(False)
ax_marg_x.yaxis.set_ticks_position('left')
ax_marg_x.xaxis.set_ticks_position('bottom')
ax_marg_x.plot( ms, np.exp(-ms**2/(2*1**2))/np.sqrt(2*np.pi*1**2), c='C0', lw=2)

ax_marg_y.hist(n2, nbins, orientation="horizontal", alpha=0.5, density=True)

ax_marg_y.plot( np.exp(-ms**2/(2*ss2**2))/np.sqrt(2*np.pi*ss2**2),ms, lw=2, c='C0',)

ax_marg_y.spines['top'].set_visible(False)
ax_marg_y.spines['right'].set_visible(False)
ax_marg_y.yaxis.set_ticks_position('left')
ax_marg_y.xaxis.set_ticks_position('bottom')
ax_marg_y.set_ylim([-4.2, 4.2])

ax_marg_y.set_xticklabels(['0', '0.5'])

## Turn off tick labels on marginals
plt.setp(ax_marg_x.get_xticklabels(), visible=False)
plt.setp(ax_marg_y.get_yticklabels(), visible=False)

## Set labels on marginals
ax_marg_y.set_xlabel(r'$P(n_i)$')
ax_marg_x.set_ylabel(r'$P(m_i)$')

plt.savefig('Th_Fig1_A_12.pdf')
#%%
kappas = np.linspace(-1, 1, 500)
Fk1 = np.zeros_like(kappas)
Fk2 = np.zeros_like(kappas)

for ik, ka in enumerate(kappas):
    Fk1[ik]= -ka + s_mn1*ka*Prime(0, ka**2)
    Fk2[ik]= -ka + s_mn2*ka*Prime(0, ka**2)
fig = plt.figure()
ax = fig.add_subplot(111)
plt.plot(kappas, Fk2, lw=2, c='C3')
#plt.plot(kappas, Fk1, lw=2)

plt.plot(kappas, kappas*0, '--k')
plt.scatter(0,0, edgecolor='C3', color='w', s=60, lw=2, zorder=4)

k0s = kappas[np.argmin(np.abs(Fk2[kappas<0.1]))]
plt.scatter(k0s,0, edgecolor='k', color='C3', s=60, lw=1, zorder=4)
plt.scatter(-k0s,0, edgecolor='k', color='C3', s=60, lw=1, zorder=4)

plt.ylim([-0.5, 0.5])
plt.ylabel(r'dynamics $d\kappa / dt$')
plt.xlabel(r'latent variable $\kappa$')
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.yaxis.set_ticks_position('left')
ax.xaxis.set_ticks_position('bottom')  
plt.savefig('Th_Fig1_C_2.pdf')

#%%
kappas = np.linspace(-1, 1, 500)
Fk1 = np.zeros_like(kappas)
Fk2 = np.zeros_like(kappas)

for ik, ka in enumerate(kappas):
    Fk1[ik]= -ka + s_mn1*ka*Prime(0, ka**2)
    Fk2[ik]= -ka + s_mn2*ka*Prime(0, ka**2)
fig = plt.figure()
ax = fig.add_subplot(111)
#plt.plot(kappas, Fk2, lw=2)
plt.plot(kappas, Fk1, lw=2, c='C3')

plt.plot(kappas, kappas*0, '--k')
#plt.scatter(0,0, edgecolor='C3', color='w', s=60, lw=2, zorder=4)

k0s = kappas[np.argmin(np.abs(Fk2[kappas<0.1]))]
plt.scatter(0,0, edgecolor='k', color='C3', s=60, lw=1, zorder=4)
#plt.scatter(-k0s,0, edgecolor='k', color='C3', s=60, lw=1, zorder=4)

plt.ylim([-0.5, 0.5])
plt.ylabel(r'dynamics $d\kappa / dt$')
plt.xlabel(r'latent variable $\kappa$')
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.yaxis.set_ticks_position('left')
ax.xaxis.set_ticks_position('bottom')  
plt.savefig('Th_Fig1_C_1.pdf')

#%%
J1 = np.dot(m1[:,None], n1[:,None].T)
J2 = np.dot(m2[:,None], n2[:,None].T)

time = np.linspace(0, 40, 1000)
dt = time[1]-time[0]

nS = 10
xs1 = np.zeros((len(time), nS))
xs2 = np.zeros((len(time), nS))

x0 = 0.5*np.random.randn(N)
x02 = 0.5*np.random.randn(N)
k1 = np.zeros(len(time))
k2 = np.zeros(len(time))
for it, t in enumerate(time[:-1]):    
    xs1[it] = x0[0:nS]
    xs2[it] = x02[0:nS]
    k1[it] = np.mean(m1*x0)
    k2[it] = np.mean(m2*x02)
    
    x = x0 + dt*(-x0+np.dot(J1, np.tanh(x0)/N))
    x2 = x02 + dt*(-x02+np.dot(J2, np.tanh(x02)/N))
    
    x0=x
    x02=x2
    
    xs1[it+1] = x0[0:nS]
    xs2[it+1] = x02[0:nS]
    k1[it+1] = np.mean(m1*x0)
    k2[it+1] = np.mean(m2*x02)
fig = plt.figure()
ax = fig.add_subplot(111)
plt.plot(time, xs2, c='C0')
#plt.plot(time, xs1, c='C1')

ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.yaxis.set_ticks_position('left')
ax.xaxis.set_ticks_position('bottom')  
plt.ylabel(r'activation $x_i\left(t\right)$')
plt.xlabel(r'time')
plt.savefig('Th_Fig1_B_2.pdf')

#%%
J1 = np.dot(m1[:,None], n1[:,None].T)
J2 = np.dot(m2[:,None], n2[:,None].T)

time = np.linspace(0, 40, 1000)
dt = time[1]-time[0]

nS = 10
xs1 = np.zeros((len(time), nS))
xs2 = np.zeros((len(time), nS))

x0 = 0.5*np.random.randn(N)
x02 = 0.5*np.random.randn(N)
k1 = np.zeros(len(time))
k2 = np.zeros(len(time))
for it, t in enumerate(time[:-1]):    
    xs1[it] = x0[0:nS]
    xs2[it] = x02[0:nS]
    k1[it] = np.mean(m1*x0)
    k2[it] = np.mean(m2*x02)
    
    x = x0 + dt*(-x0+np.dot(J1, np.tanh(x0)/N))
    x2 = x02 + dt*(-x02+np.dot(J2, np.tanh(x02)/N))
    
    x0=x
    x02=x2
    
    xs1[it+1] = x0[0:nS]
    xs2[it+1] = x02[0:nS]
    k1[it+1] = np.mean(m1*x0)
    k2[it+1] = np.mean(m2*x02)
fig = plt.figure()
ax = fig.add_subplot(111)
#plt.plot(time, xs2, c='C0')
plt.plot(time, xs1, c='C0')

ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.yaxis.set_ticks_position('left')
ax.xaxis.set_ticks_position('bottom')  
plt.ylabel(r'activation $x_i\left(t\right)$')
plt.xlabel(r'time')
plt.savefig('Th_Fig1_B_1.pdf')
#%%
fig = plt.figure()
ax = fig.add_subplot(111)
trials = 10

N = 1000
for tr in range(trials):
    m1 = np.random.randn(N)
    m1 = m1/np.std(m1)
    m2 = np.random.randn(N)
    m2 = m2/np.std(m2)
    ms = np.linspace(-3, 3)
    
    n1 = s_mn1*m1+0.3*np.random.randn(N)
    n2 = s_mn2*m2+0.3*np.random.randn(N)

    J1 = np.dot(m1[:,None], n1[:,None].T)
    J2 = np.dot(m2[:,None], n2[:,None].T)
    x0 = 0.5*np.random.randn(N)+0.05*m1*np.random.randn()
    x02 = 0.5*np.random.randn(N)+0.05*m2*np.random.randn()
    k1 = np.zeros(len(time))
    k2 = np.zeros(len(time))
    for it, t in enumerate(time[:-1]):    
        xs1[it] = x0[0:nS]
        xs2[it] = x02[0:nS]
        k1[it] = np.mean(m1*x0)
        k2[it] = np.mean(m2*x02)
        
        x = x0 + dt*(-x0+np.dot(J1, np.tanh(x0)/N))
        x2 = x02 + dt*(-x02+np.dot(J2, np.tanh(x02)/N))
        
        x0=x
        x02=x2
        
        k1[it+1] = np.mean(m1*x0)
        k2[it+1] = np.mean(m2*x02)
    plt.plot(time, k1, c='C3')
    #plt.plot(time, k2, c='C0')
    
plt.plot(time, 0*time, lw=4, alpha=0.2, c='C3')
#plt.plot(time, k0s*np.ones_like(time), lw=4, alpha=0.2, c='C3')
#plt.plot(time, -k0s*np.ones_like(time), lw=4, alpha=0.2, c='C3')

ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.yaxis.set_ticks_position('left')
ax.xaxis.set_ticks_position('bottom')  
plt.ylabel(r'latent variable $\kappa$')
plt.xlabel(r'time')
plt.savefig('Th_Fig1_D_1.pdf')

#%%
fig = plt.figure()
ax = fig.add_subplot(111)
trials = 10

N = 1000
for tr in range(trials):
    m1 = np.random.randn(N)
    m1 = m1/np.std(m1)
    m2 = np.random.randn(N)
    m2 = m2/np.std(m2)
    ms = np.linspace(-3, 3)
    
    n1 = s_mn1*m1+0.3*np.random.randn(N)
    n2 = s_mn2*m2+0.3*np.random.randn(N)

    J1 = np.dot(m1[:,None], n1[:,None].T)
    J2 = np.dot(m2[:,None], n2[:,None].T)
    x0 = 0.5*np.random.randn(N)+0.05*m1*np.random.randn()
    x02 = 0.5*np.random.randn(N)+0.05*m2*np.random.randn()
    k1 = np.zeros(len(time))
    k2 = np.zeros(len(time))
    for it, t in enumerate(time[:-1]):    
        xs1[it] = x0[0:nS]
        xs2[it] = x02[0:nS]
        k1[it] = np.mean(m1*x0)
        k2[it] = np.mean(m2*x02)
        
        x = x0 + dt*(-x0+np.dot(J1, np.tanh(x0)/N))
        x2 = x02 + dt*(-x02+np.dot(J2, np.tanh(x02)/N))
        
        x0=x
        x02=x2
        
        k1[it+1] = np.mean(m1*x0)
        k2[it+1] = np.mean(m2*x02)
    #plt.plot(time, k1, c='C3')
    plt.plot(time, k2, c='C3')
    
#plt.plot(time, 0*time, lw=4, alpha=0.2, c='C3')
plt.plot(time, k0s*np.ones_like(time), lw=4, alpha=0.2, c='C3')
plt.plot(time, -k0s*np.ones_like(time), lw=4, alpha=0.2, c='C3')

ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.yaxis.set_ticks_position('left')
ax.xaxis.set_ticks_position('bottom')  
plt.ylabel(r'latent variable $\kappa$')
plt.xlabel(r'time')
plt.savefig('Th_Fig1_D_2.pdf')

