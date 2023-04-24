from scipy.stats import beta
import numpy as np
from numpy import random

def Triangular(a, b, c, x):
    if x < a:
        return 0
    elif x >= a and x < c:
        return (2*(x-a))/((b-a)*(c-a))
    elif x == c:
        return 2/(b-a)
    elif x > c and x <= b:
        return (2*(b-x))/((b-a)*(b-c))
    elif x > b:
        return 0
def lmda_triangular(x, x_prime, lmda):
    if lmda <= 0.5:
        return Triangular(0, 1-x_prime, 1-x_prime-(1-x_prime)/(4*lmda), x)
    elif lmda > 0.5:
        return Triangular(0, 1-x_prime, (1-x_prime)/(4*(1-lmda)))

def likelihood(I, lmdas):
    prod = 1
    for i in range(len(I)):
        for l in range(len(lmdas)):
            x = I[i,l,2]
            lmda = lmdas[l]
            ## integration by uniform discretization of \int_{t=0}^1 lmda_triangular(x, x_prime=t, lmda):

            ts, step = np.linspace(0, 1, 100,endpoint=False, retstep=True)
            integral = 0
            for t in ts:
                integral += step*(lmda_triangular(x, t, lmda))
            prod *= integral
    return prod
def posterior(I, lmdas):
    return likelihood(I, lmdas)*beta(a=1, b=100)

def metro_hast(L, I, eps, N):
    # L: number of labels
    # I: observations
    # eps: maximum transision (transition dist -> U(lmda-eps, lamda+eps))
    # N: number of samples to generate
    lmdas = [0.5] * L
    lmdas_prime = [None]*L
    samples = []
    for n in range(N):
        for l in range(len(lmdas)):
            lmda = lmdas[l]
            lmdas_prime[l] = random.uniform(np.max(0, lmda-eps), np.min(1, lmda+eps))
        acceptance = min(1, posterior(I, lmdas)/posterior(I, lmdas_prime))
        r = random.uniform(0,1)
        if r < acceptance:
            samples.append(lmdas_prime)
            lmdas = lmdas_prime
    return samples


