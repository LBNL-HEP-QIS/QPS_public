import numpy as np
import math

def MCMC(shots, eps, g_a, g_b, na_i, nb_i, N, save= False, verbose=False):
    '''
    Executes MCsim (see below) 'shots' number of times, and returns the statistics in an array.

    Params:
        shots (int) number of times to execute the Markov chain QPS simulation
        (See MCsim for the others)

    Returns
        'emits' array, which stores probabilities for having emissions equal to the array index
    '''
    emits= np.zeros(N+1)
    for j in range(shots):
        n_emits, n_a, n_b, n_phi= MCsim(eps, g_a, g_b, na_i, nb_i, N, verbose=False)
        emits[n_emits]+= 1

    emits/= shots
    if save:
        np.save('./data/mcmc_%dstep_na=%d_nb=%d_shots=%s.npy' %(N, na_i, nb_i, '{:.0e}'.format(shots)), emits)
    return emits


def MCsim(eps, g_a, g_b, na_i, nb_i, N, verbose=False):
    '''
    A classical Markov Chain parton shower simulation with two fermion types. For a Monte Carlo simulation, use MCMC.
    
    Params:
        g_a     (float) a-type coupling
        g_b     (float) b-type coupling
        na_i    (int)   number of initial a-type particles
        nb_i    (int)   number of initial b-type particles
        N       (int)   number of steps in the simulation
        verbose (bool)  whether to print useful information while running
    Returns:
        Tuple: (n_emits, n_a, n_b, n_phi), where

        n_emits (int) = number of emissions that occur during the simulation
        n_a     (int) = number of a-type fermions at the end of the simulation
        n_b     (int) = number of b-type fermions at the end of the simulation
        n_phi   (int) = number of phis at the end of the simulation
    '''
    n_a= na_i
    n_b= nb_i
    n_phi= 0

    n_emits= 0

    for i in range(N):
        # Compute time steps
        t_up = eps ** ((i) / N)
        t_mid = eps ** ((i + 0.5) / N)
        t_low = eps ** ((i + 1) / N)
        # Compute values for emission matrices
        Delta_a = Delta_f(t_low, g_a) / Delta_f(t_up, g_a)
        Delta_b = Delta_f(t_low, g_b) / Delta_f(t_up, g_b)
        Delta_phi = Delta_bos(t_low, g_a, g_b) / Delta_bos(t_up, g_a, g_b)
        P_a, P_b, P_phi = P_f(t_mid, g_a), P_f(t_mid, g_b), P_bos(t_mid, g_a, g_b)

        P_phi_a= P_bos_g(t_mid, g_a)
        P_phi_b= P_bos_g(t_mid, g_b)

        Pemit= 1 - (Delta_a ** n_a) * (Delta_b ** n_b) * (Delta_phi ** n_phi)

        denom= (P_a * n_a) + (P_b * n_b) + (P_phi * n_phi)
        emit_a= (P_a * n_a) / denom
        emit_b= (P_b * n_b) / denom
        emit_phi= (P_phi * n_phi) / denom # = emit_phi_a + emit_phi_b
        emit_phi_a= (P_phi_a * n_phi) / denom
        emit_phi_b= (P_phi_b * n_phi) / denom 

        emit_a *= Pemit
        emit_b *= Pemit
        emit_phi*= Pemit
        emit_phi_a *= Pemit
        emit_phi_b *= Pemit

        cut_a= emit_a
        cut_b= cut_a + emit_b
        cut_phi_a= cut_b + emit_phi_a
        cut_phi_b= cut_phi_a + emit_phi_b

        r= np.random.uniform(0, 1)

        if r < cut_a:
            n_phi+= 1
        elif r < cut_b:
            n_phi+= 1
        elif r < cut_phi_a:
            n_phi-= 1
            n_a+= 2
        elif r < cut_phi_b:
            n_phi-= 1
            n_b+= 2
        else: 
            n_emits-= 1
        n_emits+= 1

        if verbose:
            print('\n\nDelta_a: ' + str(Delta_a))
            print('Delta_b: ' + str(Delta_b))
            print('Delta_phi: ' + str(Delta_phi))
            print('P_a: ' + str(P_a))
            print('P_b: ' + str(P_b))
            print('P_phi_a: ' + str(P_phi_a))
            print('P_phi_b: ' + str(P_phi_b))
            print('P_phi: ' + str(P_phi))
            print('t_mid: ' + str(t_mid))

            print('\nStep %d' %(i+1))
            print('P(emit a)= ' + str(emit_a))
            print('P(emit b)= ' + str(emit_b))
            print('P(emit phi -> aa)= ' + str(emit_phi_a))
            print('P(emit phi -> bb)= ' + str(emit_phi_b))
            print('P(emit phi)= ' + str(emit_phi))
            print('P(no emit)= ' + str(1 - Pemit))
    
    return n_emits, n_a, n_b, n_phi


def P_f(t, g):
    '''
    Splitting function for fermions.

    Params:
        t (float) scale parameter
        g (float) coupling
    '''
    return g ** 2 * Phat_f(t) / (4 * math.pi)

def Phat_f(t):
    return math.log(t)

def Phat_bos(t):
    return math.log(t)

def Delta_f(t, g):
    ''' Sudakov factor for fermions.'''
    return math.exp(P_f(t, g))

def P_bos(t, g_a, g_b):
    '''Total splitting function for bosons, given two fermion types.'''
    return g_a ** 2 *Phat_bos(t) / (4 * math.pi) + g_b ** 2 * Phat_bos(t) / (4 * math.pi)

def P_bos_g(t, g):
    '''Relative splitting function for bosons.'''
    return g ** 2 *Phat_bos(t) / (4 * math.pi)

def Delta_bos(t, g_a, g_b):
    ''' Sudakov factor for bosons/phis.'''
    return math.exp(P_bos(t, g_a, g_b))

def P(g):
    '''Generic splitting function.'''
    return g**2 / (4 * math.pi)

def Delta(lnt, g):
    '''
    Generic sudakov factor.
    
    Params:
        lnt (float) natural log of the scale parameter
        g   (float) coupling
    '''
    return math.exp(lnt * g**2 / (4 * math.pi))


def dsigma_d_t_max(lnt, lneps, g, normalized=False):
    '''
    The analytical distribution of the hardest emission, i.e. θmax.

    Params:
        lnt        (Array(float)) natural log of the scale parameter
        lneps      (float)        natural log of epsilon, as in Section IV, eq. (17).
        g          (float)        coupling
        normalized (bool)         whether to normalize as described below
    '''
    if normalized: # Normalized to -log(θmax), i.e. "conditionally" normalized on emission occuring
        return P(g) * Delta(lnt, g) / (1 - Delta(lneps, g))
    else: # Normalized to -infinity, i.e. this gives the actual probabilities --> use this for plotting
        return P(g) * Delta(lnt, g)