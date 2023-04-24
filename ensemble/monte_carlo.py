from scipy.stats import beta
import numpy as np
from numpy import random
import torch

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
    if x + x_prime > 1:
        return 0
    if lmda <= 0.5:
        return Triangular(0, 1-x_prime, 1-x_prime-(1-x_prime)/(4*lmda), x)
    elif lmda > 0.5:
        return Triangular(0, 1-x_prime, (1-x_prime)/(4*(1-lmda)),x)


def likelihood(I, lmdas):
    prod = 1.
    counter = 0
    avg_neutral = torch.mean(I[:,:,1], dim=0)
    for i in range(len(I)):
        for l in range(len(lmdas)):

            x = I[i,l,2]
            lmda = lmdas[l]
            ## integration by uniform discretization of \int_{t=0}^1 lmda_triangular(x, x_prime=t, lmda):

            ts, step = np.linspace(0, 1, 100,endpoint=False, retstep=True)
            integral = 0
            #lmda_tri = lmda_triangular(x, 0.9, lmda)
            # for t in ts:
            #     lmda_tri = (lmda_triangular(x, t, lmda))
            #     integral += step*lmda_tri
            #prod *= integral
            lmda_tri = lmda_triangular(x, avg_neutral[l], lmda)
            prod *= lmda_tri
        if counter % 10 == 0:
            print("likelihood computation:", counter, "out of", len(I))
        counter += 1
    return prod
def posterior(I, lmdas):
    prod = likelihood(I, lmdas)
    rv = beta(a=1, b=100)
    for l in range(len(lmdas)):
        prod *= rv.pdf(lmdas[l])
    return prod

def metro_hast(L, I, eps, N):
    # L: number of labels
    # I: observations
    # eps: maximum transision (transition dist -> U(lmda-eps, lamda+eps))
    # N: number of samples to generate
    device = "cuda" if torch.cuda.is_available() else "cpu"
    lmdas = np.ones(L)*0.5
    lmdas_prime = np.zeros(L)
    samples = []
    counter = 0
    while len(samples) < N :
        prev_posterior = None
        print("samples generated:", len(samples), "number epochs:", counter)
        for l in range(len(lmdas)):
            lmda = lmdas[l]
            lmdas_prime[l] = random.uniform(np.max((0., lmda-eps)), np.min((1., lmda+eps)))
        if prev_posterior == None:
            post = posterior(I, lmdas)
            post_prime = posterior(I, lmdas_prime)
            acceptance = np.min((1., post / post_prime))
        else:
            post = prev_posterior
            post_prime = posterior(I, lmdas_prime)
            acceptance = np.min((1., post/post_prime))
        r = random.uniform(0,1)
        if r < acceptance:
            samples.append(lmdas_prime)
            lmdas = lmdas_prime
            prev_posterior = post_prime
        counter += 1
    return samples


