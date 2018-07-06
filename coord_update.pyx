#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# cython: profile=True
"""
Created on Thu Jul  5 17:13:28 2018

@author: agiovann
"""
import numpy as np
from libc.math cimport sin, cos, acos, exp, sqrt, fabs, M_PI
cimport cython

#%%
#@cython.boundscheck(False)  # Deactivate bounds checking
def coord_update(double[:] x, int d, double t, double ell, double[:] lambda_, double[:,:]  Uhat, int q, double[:] v):
    '''
    Cythonized version of a coordinate update for the CCIPCA algorithm
    Parameters:
    ----------
    x: ndarray
        input vector
    d: int
        dimensionality of x
    t: double
        time step
    ell: double
        forgetting factor
    lambda_: ndarray
        eigenvalues
    Uhat: ndarray
        eigenvector
    v: ndarray
        temporary vector

    Return:
    --------
    Uhat
    lambda_

    '''

    cdef int i, kk
#    cdef double v[1000000]
    cdef double xU
    with nogil:
        for i in range(q):
            xU = 0
            for kk in range(d):
                xU += x[kk]*Uhat[kk,i]

            for kk in range(d):
                v[kk] = max(1,t-ell)/(t+1) * lambda_[i] * Uhat[kk,i] + (1+ell)/(t+1) * xU*x[kk] # is that OK?

            lambda_[i] = 0
            for kk in range(d):
                lambda_[i] += v[kk]**2

            lambda_[i] = sqrt(lambda_[i])

            for kk in range(d):
                Uhat[kk,i]  = v[kk]/lambda_[i]

            xU = 0
            # Orthogonalize the data against this approximate eigenvector
            for kk in range(d):
                xU += x[kk]*Uhat[kk,i]

            for kk in range(d):
                x[kk] -= xU * Uhat[kk,i]


    return Uhat, lambda_


def coord_update_total(double[:,:] x, int leng, int d, double t, double ell, double[:] lambda_, double[:,:]  Uhat, int q, double[:] v):
    '''
    Cythonized version of a coordinate update for the CCIPCA algorithm
    Parameters:
    ----------
    x: ndarray
        input vector
    d: int
        dimensionality of x
    t: double
        time step
    ell: double
        forgetting factor
    lambda_: ndarray
        eigenvalues
    Uhat: ndarray
        eigenvector
    v: ndarray
        temporary vector

    Return:
    --------
    Uhat
    lambda_

    '''

    cdef int i, kk
#    cdef double v[1000000]
    cdef double xU
    with nogil:
        for fr in range(leng):
            for i in range(q):
                xU = 0
                for kk in range(d):
                    xU += x[kk,fr]*Uhat[kk,i]

                for kk in range(d):
                    v[kk] = max(1,t-ell)/(t+1) * lambda_[i] * Uhat[kk,i] + (1+ell)/(t+1) * xU*x[kk,fr] # is that OK?

                lambda_[i] = 0
                for kk in range(d):
                    lambda_[i] += v[kk]**2

                lambda_[i] = sqrt(lambda_[i])

                for kk in range(d):
                    Uhat[kk,i]  = v[kk]/lambda_[i]

                xU = 0
                # Orthogonalize the data against this approximate eigenvector
                for kk in range(d):
                    xU += x[kk,fr]*Uhat[kk,i]

                for kk in range(d):
                    x[kk,fr] -= xU * Uhat[kk,i]
            t += 1


    return Uhat, lambda_