#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jul  5 17:13:28 2018

@author: agiovann
"""
import numpy as np
#%%
#try:
#    profile
#except:
#    def profile(a): return a

#for i in range(q):
#    v          = max(1,t-ell)/(t+1) * lambda_[i] * Uhat[:,i] + (1+ell)/(t+1) * np.dot(x,Uhat[:,i])* x # is that OK?
#    lambda_[i] = np.sqrt(v.dot(v))#np.linalg.norm(v)
#    Uhat[:,i]  = v/lambda_[i]
#    # Orthogonalize the data against this approximate eigenvector
#    x = x - np.dot(x,Uhat[:,i]) * Uhat[:,i]
#%%
#@profile
def coord_update(double[:] x, int d, double t, double ell, double[:] lambda_, double[:,:]  Uhat, int q):
    cdef int i, kk
    cdef double v[1000]
    cdef double xU

    for i in range(q):
        xU = 0
        for kk in range(d):
            xU += x[kk]*Uhat[kk,i]

        for kk in range(d):
            v[kk] = max(1,t-ell)/(t+1) * lambda_[i] * Uhat[kk,i] + (1+ell)/(t+1) * xU*x[kk] # is that OK?

        lambda_[i] = 0
        for kk in range(d):
            lambda_[i] += v[kk]**2

        lambda_[i] = np.sqrt(lambda_[i])

#        print('AA')
        for kk in range(d):
            Uhat[kk,i]  = v[kk]/lambda_[i]
#            print(Uhat[kk,i])

        xU = 0
        # Orthogonalize the data against this approximate eigenvector
        for kk in range(d):
            xU += x[kk]*Uhat[kk,i]

#        print('AA')
        for kk in range(d):
#            print(x[kk])
            x[kk] -= xU * Uhat[kk,i]


    return Uhat, lambda_