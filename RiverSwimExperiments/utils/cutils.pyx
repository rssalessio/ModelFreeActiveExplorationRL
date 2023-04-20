#! /usr/bin/env python
# -*- coding: utf-8 -*-

# cython: language_level=3
# cython: boundscheck=False
# cython: wraparound=False
# cython: cdivision=True
# cython: nonecheck=False

import numpy as numpy
cimport numpy as np

cpdef double[::1] policy_evaluation(gamma: double, P: double[:,:,:], R: double[:,:,:], pi: long[::1], atol: double = 1e-6):
    """Policy evaluation

    Args:
        gamma (float): Discount factor
        P (NDArray[np.float64]): Transition function of shape (num_states, num_actions, num_states)
        R (NDArray[np.float64]): Reward function of shape (num_states, num_actions)
        pi (Optional[NDArray[np.int64]], optional): policy
        V0 (Optional[NDArray[np.float64]], optional): Initial value function. Defaults to None.
        atol (float): Absolute tolerance

    Returns:
        NDArray[np.float64]: Value function
    """
    
    cdef long NS = P.shape[0]
    cdef long s = 0
    cdef np.ndarray[double, ndim=1, mode="c"] V = numpy.zeros(NS)
    cdef np.ndarray[double, ndim=1, mode="c"] V_next = numpy.zeros(NS)
    cdef double Delta = 0

    while True:
        Delta = 0
        for s in range(NS):
            V_next[s] = numpy.matmul(P[s, pi[s]], R[s, pi[s]] + gamma * V)
            
        Delta = numpy.max([Delta, numpy.abs(V_next - V).max()])
        V = V_next
        
        if Delta < atol:
            break
    return V # this is a memoryview
        
