#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Apr  9 11:12:47 2020

@author: mbeiran
"""

import numpy as np
import matplotlib.pyplot as plt
import lib_rnns as lr
import matplotlib.patches as patches
from scipy.linalg import sqrtm
#from matplotlib.gridspec import GridSpec
aa = lr.set_plot()

verbose = False
    
gaussian_norm = (1/np.sqrt(np.pi))
gauss_points, gauss_weights = np.polynomial.hermite.hermgauss(200)
gauss_points = gauss_points*np.sqrt(2)

def Phi (mu, delta0):
    integrand = np.tanh(mu+np.sqrt(delta0)*gauss_points)
    return gaussian_norm * np.dot (integrand,gauss_weights)
def Prime (mu, delta0):
    integrand = 1 - (np.tanh(mu+np.sqrt(delta0)*gauss_points))**2
    return gaussian_norm * np.dot (integrand,gauss_weights)

def func(sol, mu=1.):
    s = np.zeros_like(sol)
    s[0] = sol[1]
    s[1] = -sol[0]+mu*(1-sol[0]**2)*sol[1]
    return(s)
    
def funcG(sol, Sigma, sMx, sMy, sI, muMx, muMy, muN, muI):
    s = np.zeros_like(sol)
    S = np.shape(Sigma)[2]
    for iS in range(S):
        Mu = muI[iS] + sol[0]*muMx[iS] + sol[1]*muMy[iS]
        Delta = sI[iS] + sMx[iS]*sol[0]**2 + sMy[iS]*sol[1]**2
        
        P =  Prime(Mu, Delta)
        P1 = muN[:,iS]*Phi(Mu, Delta) 
        
        s += P1 + np.dot(Sigma[:,:,iS], np.array((sol[0],sol[1])))*P
    s[0] = s[0]/S-sol[0]
    s[1] = s[1]/S-sol[1]
    return(s)
            
def VP_field( xs, ys, mu=1.):
    X, Y = np.meshgrid(xs,ys)
    U = np.zeros_like(X)
    V = np.zeros_like(X)
    
    for ix, x in enumerate(xs):
        for iy, y in enumerate(ys):
            U[ix, iy] = y
            V[ix, iy] = -x+mu*(1-x**2)*y
            
    E = np.sqrt(U**2+V**2)
    return(X, Y, E, U, V)

def VP_func( x, y, mu=1.):

    U = y
    V = -x+mu*(1-x**2)*y
            
    return( U, V)


def VP_field2( xs, ys, mu=1.):
    X, Y = np.meshgrid(xs,ys)
    U = np.zeros_like(X)
    V = np.zeros_like(X)
    
    for ix, x in enumerate(xs):
        for iy, y in enumerate(ys):
            delta = x**2+y**2
            U[ix, iy] = -x+(2*x-0*y)*Prime(0,delta)
            V[ix, iy] = -y+(0*x+1.2*y)*Prime(0,delta)
            
    E = np.sqrt(U**2+V**2)
    return(X, Y, E, U, V)
    
def VP_func2( x, y, mu=1.):

    delta = x**2+y**2
    U = -x+(2*x-0*y)*Prime(0,delta)
    V = -y+(0*x+1.2*y)*Prime(0,delta)
            
    return(U, V)
    
    
def VP_approx( xs, ys, Sigma, sMx, sMy, sI, muMx, muMy, muN, muI):
    X, Y = np.meshgrid(xs,ys)
    U = np.zeros_like(X)
    V = np.zeros_like(X)
    
    S = np.shape(Sigma)[2]
    for ix, x in enumerate(xs):
        for iy, y in enumerate(ys):
            sol = 0
            for iS in range(S):
                Mu = muI[iS] + x*muMx[iS] + y*muMy[iS]
                Delta = sI[iS] + sMx[iS]*x**2 + sMy[iS]*y**2
                
                P =  Prime(Mu, Delta)
                P1 = muN[:,iS]*Phi(Mu, Delta) 
                
                sol += P1 + np.dot(Sigma[:,:,iS], np.array((x,y)))*P
            U[ix, iy] = sol[0]/S-x
            V[ix, iy] = sol[1]/S-y
            
    E = np.sqrt(U**2+V**2)
    return(X, Y, E, U, V)
    
def VP_approxnum( xs, ys, Mv, Nv, Iv):
    X, Y = np.meshgrid(xs,ys)
    U = np.zeros_like(X)
    V = np.zeros_like(X)
    
    for ix, x in enumerate(xs):
        for iy, y in enumerate(ys):
            U[ix, iy] = -x + np.mean(Nv[:,0]*np.tanh(Iv+x*Mv[:,0]+y*Mv[:,1]))
            V[ix, iy] = -y + np.mean(Nv[:,1]*np.tanh(Iv+x*Mv[:,0]+y*Mv[:,1]))
            
    E = np.sqrt(U**2+V**2)
    return(X, Y, E, U, V)

def VP_funcnum( x, y, Mv, Nv, Iv):
    U = -x + np.mean(Nv[:,0]*np.tanh(Iv+x*Mv[:,0]+y*Mv[:,1]))
    V = -y + np.mean(Nv[:,1]*np.tanh(Iv+x*Mv[:,0]+y*Mv[:,1]))
            
    return(U, V)
def get_network(Sigma, sMx, sMy, sI, muMx, muMy, MuN, muI, S, NperP):
    
# =============================================================================
#     Initialize matrices
# =============================================================================
    N = S*NperP
    rank = 2
    Mv = np.zeros((N, rank))
    Nv = np.zeros((N, rank))
    Iv  =     np.zeros((N))
    sMxP = np.zeros_like(sMx)
    sMyP = np.zeros_like(sMy)
    sIP = np.zeros_like(sI)
    muMxP = np.zeros_like(muMx)
    muMyP = np.zeros_like(muMy)
    MuNP = np.zeros_like(MuN)
    muIP = np.zeros_like(muI)
    SigmaP = np.zeros_like(Sigma)
# =============================================================================
#     Go through populations
# =============================================================================
    for iS in range(S):   
        val = -1
        SS = Sigma[:,:,iS]
        rSS = np.zeros((5,5)) #BigSigma
        rSS[0,0] = sMx[iS] #m1^2
        rSS[1,1] = sMy[iS] #m2^2
        
        rSS[0,2] = SS[0,0]
        rSS[2,0] = rSS[0,2]
        rSS[0,3] = SS[1,0]
        rSS[3,0] = rSS[0,3]
        
        rSS[1,2] = SS[0,1]
        rSS[2,1] = rSS[1,2]
        rSS[1,3] = SS[1,1]
        rSS[3,1] = rSS[1,3]
        
        rSS[2,2] = 1.1
        rSS[3,3] = 1.1
        
        rSS[4,4] = sI[iS]
        val = np.min(np.linalg.eigvalsh(rSS))
        
        #Make BigSigma positive definite
        #mVal = np.max(np.abs(SS))
        cnt =0
        vals = []
        diag1 = []
        diag2 = []
        while val<1e-7:
            if cnt<200:
                rSS[2,2] = 1.2*rSS[2,2]
                rSS[3,3] = 1.2*rSS[3,3]
            else:
                rSS[2,2] = 1.02*rSS[2,2]
                rSS[3,3] = 1.02*rSS[3,3]
            val = np.min(np.linalg.eigvalsh(rSS))
            diag1.append(rSS[2,2])
            diag2.append(rSS[3,3])
            
            vals.append(val)
            cnt +=1
        
        Mean = np.array((muMx[iS], muMy[iS],MuN[0,iS], MuN[1,iS], muI[iS] ))
        
        error = 1e8
        counter =0
        
        #Take minimal finite-size out of 500 trials
        while error>0.1 and counter<500:
            counter+=1
            Sol = np.random.multivariate_normal(Mean, rSS, NperP)  
            MeanP = np.mean(Sol,0)
            rSSP  = np.cov(Sol.T)
            rSS_corr = np.zeros_like(rSS)
            rSS_corr[:,:] = rSS
            rSS_corr[np.abs(rSS_corr)<1e-10] = 1e-10
            sol      = (rSSP-rSS_corr)/rSS_corr
            sol[np.abs(sol)>1e8] = 0.
            sol[2,2] = 0.
            sol[3,3] = 0.
            error2 = np.std(sol)+np.std(Mean-MeanP)
            if error2<error:
                error=error2
                Solsav = Sol

        Sol = Solsav
        Corr  = np.cov(Sol.T)
        
        sMxP[iS] = Corr[0,0]
        sMyP[iS] = Corr[1,1]
        SigmaP[0,0,iS] = Corr[0,2]
        SigmaP[0,1,iS] = Corr[0,3]
        SigmaP[1,0,iS] = Corr[1,2]
        SigmaP[1,1,iS] = Corr[1,3]
         
        Mv[iS*NperP:(iS+1)*NperP, 0] = Sol[:,0]
        Mv[iS*NperP:(iS+1)*NperP, 1] = Sol[:,1]
        Nv[iS*NperP:(iS+1)*NperP, 0] = Sol[:,2]
        Nv[iS*NperP:(iS+1)*NperP, 1] = Sol[:,3]
        Iv[iS*NperP:(iS+1)*NperP] = Sol[:,4]
        #Iv[iS*NperP:(iS+1)*NperP] += muI[iS]-np.mean(Iv[iS*NperP:(iS+1)*NperP])
        
        #Quantify errors
        sIP[iS] = np.var(Iv[iS*NperP:(iS+1)*NperP]-muI[iS])
        muIP[iS] = np.mean(Iv[iS*NperP:(iS+1)*NperP])
        MuNP[0,iS] = np.mean(Nv[iS*NperP:(iS+1)*NperP, 0])
        MuNP[1,iS] = np.mean(Nv[iS*NperP:(iS+1)*NperP, 1])
        muMyP[iS] =  np.mean(Mv[iS*NperP:(iS+1)*NperP, 1])
        muMxP[iS] =  np.mean(Mv[iS*NperP:(iS+1)*NperP, 0])
           
    return(Mv, Nv, Iv, sMxP, sMyP, sIP, muMxP, muMyP, MuNP, muIP, SigmaP)

def get_network2(Sigma, sMx, sMy, sI, muMx, muMy, MuN, muI, S, NperP):
    
# =============================================================================
#     Initialize matrices
# =============================================================================
    N = S*NperP
    rank = 2
    Mv = np.zeros((N, rank))
    Nv = np.zeros((N, rank))
    Iv  =     np.zeros((N))
    sMxP = np.zeros_like(sMx)
    sMyP = np.zeros_like(sMy)
    sIP = np.zeros_like(sI)
    muMxP = np.zeros_like(muMx)
    muMyP = np.zeros_like(muMy)
    MuNP = np.zeros_like(MuN)
    muIP = np.zeros_like(muI)
    SigmaP = np.zeros_like(Sigma)
# =============================================================================
#     Go through populations
# =============================================================================
    error = 1e8
    counter =0
    Mean = np.array((0., 0.,0., 0., 0.))
    rSS = np.eye(5)
    #Take minimal finite-size out of 500 trials
    while error>1e-5 and counter<5000:
        counter+=1
        Sol = np.random.multivariate_normal(Mean, rSS, NperP)  
        MeanP = np.mean(Sol,0)
        rSSP  = np.cov(Sol.T)
        error2 = np.std(rSSP-rSS)+np.std(MeanP)
        if error2<error:
            error=error2
            Solsav = Sol
            if verbose ==True:
                print(error)

    Sol = Solsav
    for iS in range(S):   
        val = -1
        SS = Sigma[:,:,iS]
        rSS = np.zeros((5,5)) #BigSigma
        rSS[0,0] = sMx[iS] #m1^2
        rSS[1,1] = sMy[iS] #m2^2
        
        rSS[0,2] = SS[0,0]
        rSS[2,0] = rSS[0,2]
        rSS[0,3] = SS[1,0]
        rSS[3,0] = rSS[0,3]
        
        rSS[1,2] = SS[0,1]
        rSS[2,1] = rSS[1,2]
        rSS[1,3] = SS[1,1]
        rSS[3,1] = rSS[1,3]
        
        rSS[2,2] = 1.1
        rSS[3,3] = 1.1
        
        rSS[4,4] = sI[iS]
        val = np.min(np.linalg.eigvalsh(rSS))
        
        #Make BigSigma positive definite
        #mVal = np.max(np.abs(SS))
        cnt =0
        vals = []
        diag1 = []
        diag2 = []
        while val<1e-7:
            if cnt<200:
                rSS[2,2] = 1.2*rSS[2,2]
                rSS[3,3] = 1.2*rSS[3,3]
            else:
                rSS[2,2] = 1.02*rSS[2,2]
                rSS[3,3] = 1.02*rSS[3,3]
            val = np.min(np.linalg.eigvalsh(rSS))
            diag1.append(rSS[2,2])
            diag2.append(rSS[3,3])
            
            vals.append(val)
            cnt +=1
        
        Mean = np.array((muMx[iS], muMy[iS],MuN[0,iS], MuN[1,iS], muI[iS] ))
        
        error = 1e8
        counter =0
        
#        #Take minimal finite-size out of 500 trials
#        while error>0.1 and counter<500:
#            counter+=1
#            Sol = np.random.multivariate_normal(Mean, rSS, NperP)  
#            MeanP = np.mean(Sol,0)
#            rSSP  = np.cov(Sol.T)
#            rSS_corr = np.zeros_like(rSS)
#            rSS_corr[:,:] = rSS
#            rSS_corr[np.abs(rSS_corr)<1e-10] = 1e-10
#            sol      = (rSSP-rSS_corr)/rSS_corr
#            sol[np.abs(sol)>1e8] = 0.
#            sol[2,2] = 0.
#            sol[3,3] = 0.
#            error2 = np.std(sol)+np.std(Mean-MeanP)
#            if error2<error:
#                error=error2
#                Solsav = Sol

        Sol = Mean[:,None] + np.dot(sqrtm(rSS), Solsav.T)
        Sol = Sol.T
        Corr  = np.cov(Sol.T)
        
        sMxP[iS] = Corr[0,0]
        sMyP[iS] = Corr[1,1]
        SigmaP[0,0,iS] = Corr[0,2]
        SigmaP[0,1,iS] = Corr[0,3]
        SigmaP[1,0,iS] = Corr[1,2]
        SigmaP[1,1,iS] = Corr[1,3]
         
        Mv[iS*NperP:(iS+1)*NperP, 0] = Sol[:,0]
        Mv[iS*NperP:(iS+1)*NperP, 1] = Sol[:,1]
        Nv[iS*NperP:(iS+1)*NperP, 0] = Sol[:,2]
        Nv[iS*NperP:(iS+1)*NperP, 1] = Sol[:,3]
        Iv[iS*NperP:(iS+1)*NperP] = Sol[:,4]
        #Iv[iS*NperP:(iS+1)*NperP] += muI[iS]-np.mean(Iv[iS*NperP:(iS+1)*NperP])
        
        #Quantify errors
        sIP[iS] = np.var(Iv[iS*NperP:(iS+1)*NperP]-muI[iS])
        muIP[iS] = np.mean(Iv[iS*NperP:(iS+1)*NperP])
        MuNP[0,iS] = np.mean(Nv[iS*NperP:(iS+1)*NperP, 0])
        MuNP[1,iS] = np.mean(Nv[iS*NperP:(iS+1)*NperP, 1])
        muMyP[iS] =  np.mean(Mv[iS*NperP:(iS+1)*NperP, 1])
        muMxP[iS] =  np.mean(Mv[iS*NperP:(iS+1)*NperP, 0])
           
    return(Mv, Nv, Iv, sMxP, sMyP, sIP, muMxP, muMyP, MuNP, muIP, SigmaP)
    
def algorithm(dat_point, sigma, muMx, muMy, sMx, sMy, muI, sI):
    S = len(muMx) #number of populations
    K = 2 #rank
    p = np.shape(dat_point)[0] #number of points
        
    Phi0 = np.zeros((K*(K+1)*S, p*K))
    F0   = np.zeros((p*K, 1))

    for ip in range(p):
        k = dat_point[ip, :]
        k1 = dat_point[ip,0]
        k2 = dat_point[ip,1]
        SS = VP_func(k1,k2)
        F0[ip*K]   = S*(SS[0] + k1)
        F0[ip*K+1] = S*(SS[1] + k2)
        
        for iS in range(S):
            Mu = muI[iS] + k[0]*muMx[iS] + k[1]*muMy[iS]
            Delta = sI[iS] + sMx[iS]*k[0]**2 + sMy[iS]*k[1]**2
            i0 = iS*K*(K+1)
            
            Phi0[i0: i0+K, ip*K] = k*Prime(Mu, Delta)
            Phi0[i0+K: i0+2*K, ip*K+1] = k*Prime(Mu, Delta)
            Phi0[i0+2*K: i0+2*K+1, ip*K] = Phi(Mu, Delta)
            Phi0[i0+2*K+1: i0+2*K+2, ip*K+1] = Phi(Mu, Delta)
            
    sol = np.dot(F0.T, Phi0.T)
    C= np.dot(Phi0, Phi0.T)+np.eye(K*(K+1)*S)*sigma**2
    
    sigmaU = np.dot(sol, np.linalg.pinv(C))
    sigmaU = sigmaU[0,:]
        
    Sigma = np.zeros((K, K, S))
    MuN    = np.zeros((K,  S))       
    for iS in range(S):
        Sigma[0,0,iS] = sigmaU[iS*K*(K+1)]
        Sigma[0,1,iS] = sigmaU[iS*K*(K+1)+1] #sigma_m2n1
        Sigma[1,0,iS] = sigmaU[iS*K*(K+1)+2] #sigma_m1n2
        Sigma[1,1,iS] = sigmaU[iS*K*(K+1)+3]
        MuN[0, iS]     = sigmaU[iS*K*(K+1)+4]
        MuN[1, iS]     = sigmaU[iS*K*(K+1)+5]
    E = np.mean(np.abs((np.dot(sigmaU, Phi0) - F0.T)))#/F0.T))
    reconst = np.dot(sigmaU, Phi0)
    return(Sigma, MuN, E, sigmaU,reconst, F0)

    #%%
dt = 0.02
time = np.arange(0, 40, dt)
solT = np.zeros((len(time),2))

sol0 = np.array((1.,1.))
solT[0,:] = sol0

for it, ti in enumerate(time[:-1]):
    solT[it+1,:] = solT[it,:]+dt*(func(solT[it,:]))
    
#%%
x = np.linspace(-4,4, 100)
y = np.linspace(-4,4, 100)
fig = plt.figure()
ax = fig.add_subplot(111)
X, Y, E, U, V = VP_field(x,y, mu=1.)

plt.plot( solT[1200:,0], solT[1200:,1], 'k', lw=1)
plt.pcolor(x, y, np.log10(E).T, vmin = -2, vmax = 2., shading='auto')
plt.colorbar()
plt.streamplot(x, y, U.T, V.T, color='w')
plt.xlim([np.min(x), np.max(x)])
plt.ylim([np.min(y), np.max(y)])
plt.xlabel(r'$x$')
plt.ylabel(r'$y$')

rect = patches.Rectangle((-3,3),6,-6,linewidth=1,edgecolor='r',facecolor='none', zorder=5)
ax.add_patch(rect)

plt.savefig('Fig7_1n_VanderPol.pdf')
plt.show()

#%%
do_reg = False
do_fs_noreg = True
#%%
xp = np.linspace(-3, 3.01, 30)
yp = np.linspace(-3, 3.01, 30)

Xp, Yp, Ep, Up, Vp = VP_field2(xp,yp, mu=1.)
dat_point = np.vstack((np.ravel(Xp), np.ravel(Yp))).T

f0 = np.vstack((np.ravel(Up), np.ravel(Vp))).T
f0 += dat_point #Calculated without the leak
p = len(xp)*len(yp)


#%%
K = 2
pops = np.array((1, 2, 3, 4, 5, 10, 15, 20, 25, 30, 35, 40, 45, 50))#np.round(np.linspace(1, 50,10)).astype(int)#np.round(np.linspace(1, 50, 20)).astype(int)
trials = 50;#20#20#20

sigmas = [1e-8, 0.5]#[1e-8, 0.5]#[1e-8]

Ess = np.zeros((len(sigmas), trials, len(pops)))
Ess_r = np.zeros((len(sigmas), trials, len(pops)))
Ess_fs = np.zeros((len(sigmas), trials, len(pops)))
Ess_r_fs = np.zeros((len(sigmas), trials, len(pops)))


L2n = np.zeros((len(sigmas), trials, len(pops)))
Mea = np.zeros((len(sigmas), trials, len(pops)))
Var = np.zeros((len(sigmas), trials, len(pops)))
Max = np.zeros((len(sigmas), trials, len(pops)))

plott =False

for isi, sigma in enumerate(sigmas):
    for ipop, S in enumerate(pops):
        if verbose==True:
            print(' ')
            print('New pop')
            print('-.-')
        Sigma = np.zeros((K, K, S, trials))
        MuN    = np.zeros((K,  S, trials))
        for iti in range(trials):
            if np.min([np.abs(S-15), np.abs(S-35)])==0 and sigma>1e-3:
                if S==15:
                    ITI = 11
                else:
                    ITI = 15
                if iti==ITI:
                    if verbose == True:
                        print('pop: '+str(S)+' _Trial: '+str(iti))
                    
                    try:
                        fl=np.load('DataFig7/paramMF_sig_'+str(sigma)+ '_pop_'+str(S)+'_trial_'+str(iti)+'.npz')
                        Sigma = fl['name1']
                        sMx = fl['name2']
                        sMy = fl['name3']
                        sI = fl['name4']
                        muMx = fl['name5']
                        muMy = fl['name6']
                        MuN = fl['name7']
                        muI = fl['name8']
                        E = fl['name9']
                        sigmaU = fl['name10']
                        F0 = fl['name11']
                        reconst = fl['name12']
                        
                        
                    except:
                        print('no paramMF')
                        print('DataFig7/paramMF_sig_'+str(sigma)+ '_pop_'+str(S)+'_trial_'+str(iti)+'.npz')
                        sI = np.random.exponential(1, S)     
                        sMx = np.random.exponential(1, S)
                        if S>1:
                            sMx = sMx/np.std(sMx)
                        else:
                            sMx[0] = 1.
                        
                        muMx = 4*np.random.rand(S)
                        muMx = muMx-np.mean(muMx)
                        
                        muMy = 0*4*np.random.rand(S)
                        muMy = muMy-np.mean(muMy)
                        
                        muI = 4*np.random.rand(S)
                        muI = muI-np.mean(muI)
                        
                        sMy = np.random.exponential(1, S)
                        if S>1:
                            sMy = sMy/np.std(sMy)
                        else:
                            sMy[0] = 1.
                        
                        plott=False
                        Sigma, MuN, E, sigmaU, reconst, F0 = algorithm(dat_point, sigma, muMx, muMy, sMx, sMy, muI, sI)
                        import os
                        if os.path.isdir('DataFig7/')==False:
                            os.mkdir('DataFig7/')
                        np.savez('DataFig7/paramMF_sig_'+str(sigma)+ '_pop_'+str(S)+'_trial_'+str(iti),name1=Sigma, name2=sMx, name3=sMy, name4=sI, 
                                 name5=muMx, name6=muMy, name7=MuN, name8=muI, name9=E, name10=sigmaU, name11=F0, name12=reconst)

                    if sigma>0.01:
                        sol = np.zeros((len(time),2))
                        
                        sol0 = np.array((1.,1.))
                        sol[0,:] = sol0
                        
                        for it, ti in enumerate(time[:-1]):
                            sol[it+1,:] = sol[it,:]+dt*funcG(sol[it,:], Sigma, sMx, sMy, sI, muMx, muMy, MuN, muI)
                        fig = plt.figure()
                        ax = fig.add_subplot(211)
                        plt.plot(time, sol[:,0], lw=2)
                        plt.plot(time, solT[:,0], '--k')
                        
                        ax.set_ylabel(r'$\kappa_1(t)$')
                        ax.spines['top'].set_visible(False)
                        ax.spines['right'].set_visible(False)
                        ax.yaxis.set_ticks_position('left')
                        ax.xaxis.set_ticks_position('bottom')  
                        ax.set_yticks([-2, 0, 2])
                        ax = fig.add_subplot(212)
                        plt.plot(time, sol[:,1], lw=2)
                        plt.plot(time, solT[:,1], '--k')
                        
                        ax.set_xlabel('time')
                        ax.set_ylabel(r'$\kappa_2(t)$')
                        ax.spines['top'].set_visible(False)
                        ax.spines['right'].set_visible(False)
                        ax.yaxis.set_ticks_position('left')
                        ax.xaxis.set_ticks_position('bottom')
                        ax.set_yticks([-2, 0, 2])  
                        plt.savefig('Fig7_2_vanderPol_xy_MF_sig_'+str(sigma)+'_pop_'+str(S)+'_trial_'+str(iti)+'.pdf')
                        plt.show()
            
            
                        x = np.linspace(-5,5, 30)
                        y = np.linspace(-5,5, 30)       
                        try:
                            fl = np.load('DataFig7/dynMF_sig_'+str(sigma)+ '_pop_'+str(S)+'_trial_'+str(iti)+'.npz')
                            Sigma = fl['name1']
                            sMx = fl['name2']
                            sMy = fl['name3']
                            sI = fl['name4']
                            muMx = fl['name5']
                            muMy = fl['name6']
                            MuN = fl['name7']
                            muI = fl['name8']
                            Xa = fl['name9']
                            Ya = fl['name10']
                            Ea = fl['name11']
                            Ua = fl['name12']
                            Va = fl['name13']
                        except:
                            print('no dynMF')
                            print('DataFig7/dynMF_sig_'+str(sigma)+ '_pop_'+str(S)+'_trial_'+str(iti)+'.npz')
                            Xa, Ya, Ea, Ua, Va = VP_approx(x,y, Sigma, sMx, sMy, sI, muMx, muMy, MuN, muI)
                            np.savez('DataFig7/dynMF_sig_'+str(sigma)+ '_pop_'+str(S)+'_trial_'+str(iti), 
                                     name1=Sigma, name2=sMx, name3=sMy, name4=sI, name5=muMx, name6=muMy, 
                                     name7=MuN, name8=muI, name9=Xa, name10=Ya, name11=Ea, name12=Ua, name13=Va)
                        
                        
                        plt.figure()
                        plt.plot( sol[:,0], sol[:,1], lw=2)
                        plt.plot( solT[:,0], solT[:,1], '--k', lw=1)
                        plt.pcolor(x, y, np.log10(Ea.T), vmin = -2, vmax = 2., shading='auto')
                        #plt.colorbar()
                        plt.streamplot(x, y, Ua.T, Va.T, color='w')
                        plt.xlim([np.min(x), np.max(x)])
                        plt.ylim([np.min(y), np.max(y)])
                        plt.xlabel(r'$\kappa_1$')
                        plt.ylabel(r'$\kappa_2$')
                        plt.xticks([-4, -2, 0, 2, 4])  
                        if sigma<0.1:
                            plt.savefig('Fig7_1n_Gauss_approx_pops_'+str(S)+'_trial_'+ str(iti)+'.pdf')
                            plt.savefig('Fig7_1n_Gauss_approx_pops_'+str(S)+'_trial_'+ str(iti)+'.png')
                        else:
                            plt.savefig('Fig7_1Reg_Gauss_approx_pops_'+str(S)+'_trial_'+ str(iti)+'.pdf')
                            plt.savefig('Fig7_1Reg_Gauss_approx_pops_'+str(S)+'_trial_'+ str(iti)+'.png')
                        plt.show()
                    


                        NperP = 2000

                        try:
                            fl = np.load('DataFig7/dynamicsfinsiz_sig_'+str(sigma)+ '_pop_'+str(S)+'_trial_'+str(iti)+'.npz')
                            Mv = fl['name1']
                            Nv = fl['name2']
                            Xr = fl['name3']
                            Yr = fl['name4']
                            Er = fl['name5']
                            Ur = fl['name6']
                            Vr = fl['name7']
                            Iv = fl['name8']
                            #hello = fl['name9']
                        except:
                            print('calculating vectors')
                            print('DataFig7/dynamicsfinsiz_sig_'+str(sigma)+ '_pop_'+str(S)+'_trial_'+str(iti)+'.npz')
                            Mv, Nv, Iv, sMxP, sMyP, sIP, muMxP, muMyP, MuNP, muIP, SigmaP = get_network(Sigma, sMx, sMy, sI, muMx, muMy, MuN, muI, S, NperP)   
                            Xr, Yr, Er, Ur, Vr = VP_approxnum(x,y, Mv, Nv, Iv) 
                            np.savez('dynamicsfinsiz_sig_'+str(sigma)+ '_pop_'+str(S)+'_trial_'+str(iti), name1=Mv, 
                                    name2=Nv, name3=Xr, name4=Yr, name5=Er, name6=Ur, name7=Vr, name8=Iv)
                        
    
                        
                        sol = np.zeros((len(time),2))
                        sol[0,:] = sol0
            
                        for it, ti in enumerate(time[:-1]):
                            sol[it+1,:] = sol[it,:]+dt*np.array((VP_funcnum( sol[it,0], sol[it,1], Mv, Nv, Iv)))
                    
                        fig = plt.figure()
                        ax = fig.add_subplot(211)
                        plt.plot(time, sol[:,0], lw=2)
                        plt.plot(time, solT[:,0], '--k')
                        
                        ax.set_ylabel(r'$\kappa_1(t)$')
                        ax.spines['top'].set_visible(False)
                        ax.spines['right'].set_visible(False)
                        ax.yaxis.set_ticks_position('left')
                        ax.xaxis.set_ticks_position('bottom')  
                        ax.set_yticks([-2, 0, 2])
                        ax = fig.add_subplot(212)
                        plt.plot(time, sol[:,1], lw=2)
                        plt.plot(time, solT[:,1], '--k')
                        
                        ax.set_xlabel('time')
                        ax.set_ylabel(r'$\kappa_2(t)$')
                        ax.spines['top'].set_visible(False)
                        ax.spines['right'].set_visible(False)
                        ax.yaxis.set_ticks_position('left')
                        ax.xaxis.set_ticks_position('bottom')
                        ax.set_yticks([-2, 0, 2])  
                        plt.savefig('Fig7_2_vanderPol_xy_FS_sig_'+str(sigma)+'_pop_'+str(S)+'_trial_'+str(iti)+'.pdf')
                        plt.show()
                    
                        #%
                        plt.figure()
                        plt.plot( sol[:,0], sol[:,1], lw=2)
                        plt.plot( solT[:,0], solT[:,1], '--k', lw=1)
                        plt.pcolor(x, y, np.log10(Er.T), vmin = -2, vmax = 2., shading='auto')
                        #plt.colorbar()
                        plt.streamplot(x, y, Ur.T, Vr.T, color='w')
                        plt.xlim([np.min(x), np.max(x)])
                        plt.ylim([np.min(y), np.max(y)])
                        plt.xticks([-4, -2, 0, 2, 4])  
                        plt.xlabel(r'$\kappa_1$')
                        plt.ylabel(r'$\kappa_2$')
                        if sigma<0.1:
                            plt.savefig('Fig7_1n_Gauss_approx_FS_pops_'+str(S)+'_trial_'+ str(iti)+'.pdf')
                            plt.savefig('Fig7_1n_Gauss_approx_FS_pops_'+str(S)+'_trial_'+ str(iti)+'.png')
                        else:
                            plt.savefig('Fig7_1n_Gauss_approx_FSreg_pops_'+str(S)+'_trial_'+ str(iti)+'.pdf')
                            plt.savefig('Fig7_1n_Gauss_approx_FSreg_pops_'+str(S)+'_trial_'+ str(iti)+'.png')
                        plt.show()




#%%
fl = np.load('DataFig7/simulations.npz')
Ess=fl['name1']
Ess_fs = fl['name2']
Var = fl['name3']
#%%
Ess2 = np.mean(Ess, 1)/pops
sEss2 = np.std(Ess, 1)/(pops*np.sqrt(trials))#/np.sqrt(trials)

fig = plt.figure()
ax = fig.add_subplot(111)
plt.fill_between(pops, Ess2[0]-sEss2[0], Ess2[0]+sEss2[0], alpha=0.4, color='k')
plt.plot(pops, Ess2[0], lw=2, c =[0.2,0.2,0.2], label=r'$\beta=10^{-8}$')
plt.fill_between(pops, Ess2[1]-sEss2[1], Ess2[1]+sEss2[1], alpha=0.5)
plt.plot(pops, Ess2[1], lw=2, label=r'$\beta=0.5$')
plt.ylabel('error (mean field)')
plt.xlabel(r'populations $P$')
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.yaxis.set_ticks_position('left')
ax.xaxis.set_ticks_position('bottom')
plt.yscale('log')
plt.legend(frameon=False)
plt.savefig('Fig7_1n_Gauss_approx_pops_0.pdf')
plt.show()


#%%
Ess2 = np.mean(Ess_fs, 1)/pops
sEss2 = np.std(Ess_fs, 1)/(pops*np.sqrt(trials))

fig = plt.figure()
ax = fig.add_subplot(111)
plt.fill_between(pops, Ess2[0]-sEss2[0], Ess2[0]+sEss2[0], alpha=0.4, color='k')
plt.plot(pops, Ess2[0], lw=2, c =[0.2,0.2,0.2],label=r'$\beta=10^{-8}$')
plt.fill_between(pops, Ess2[1]-sEss2[1], Ess2[1]+sEss2[1], alpha=0.5)
plt.plot(pops,  Ess2[1], lw=2, label=r'$\beta=0.5$')
plt.yscale('log')
plt.ylabel('error (finite size)')
plt.xlabel(r'populations $P$')
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.yaxis.set_ticks_position('left')
ax.xaxis.set_ticks_position('bottom')
plt.legend(frameon=False)
plt.ylim([1., 100])
plt.savefig('Fig7_1n_Gauss_approx_pops_2.pdf')

#%%
Ess2 = np.mean(Var, 1)
sEss2 = np.std(Var, 1)/np.sqrt(trials)

fig = plt.figure()
ax = fig.add_subplot(111)
plt.fill_between(pops, Ess2[0]-sEss2[0], Ess2[0]+sEss2[0], alpha=0.4, color='k')
plt.plot(pops, Ess2[0], lw=2, c =[0.2,0.2,0.2],label=r'$\beta=10^{-8}$')
plt.fill_between(pops, Ess2[1]-sEss2[1], Ess2[1]+sEss2[1], alpha=0.5)
plt.plot(pops,  Ess2[1], lw=2, label=r'$\beta=0.5$')
plt.yscale('log')
plt.ylabel(r'variance $X$')
plt.xlabel(r'populations $P$')
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.yaxis.set_ticks_position('left')
ax.xaxis.set_ticks_position('bottom')
plt.legend(frameon=False)
plt.savefig('Fig7_1n_Gauss_approx_pops_3.pdf')



