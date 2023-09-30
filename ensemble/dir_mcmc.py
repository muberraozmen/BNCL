from scipy.stats import beta
from scipy.stats import dirichlet
import numpy as np
from numpy import random
from scipy.special import loggamma
from sympy.stats import MultivariateBeta
import torch


def triangular(a, b, c, x):
    if x < a:
        return 1e-08
    elif a <= x < c:
        return (2*(x-a))/((b-a)*(c-a))
    elif x == c:
        return 2/(b-a)
    elif c < x <= b:
        return (2*(b-x))/((b-a)*(b-c))
    elif x > b:
        return 1e-08
def ratio_dir_pdfs(x_entailment, x_neutral, x_contradiction, lambda1, lambda2):
    # if np.sum([x_entailment, x_neutral, x_contradiction]) != 1:
    #     print([x_entailment, x_neutral, x_contradiction])
    #     print(x_entailment.item())
    #     print(np.sum([x_entailment, x_neutral, x_contradiction], dtype=np.float64))
    p1 = dirichlet.pdf([x_entailment, x_neutral,x_contradiction],[lambda1, 1, 1-lambda1] )
    # print(p1)
    p2 = dirichlet.pdf([x_entailment, x_neutral,x_contradiction],[lambda2, 1, 1-lambda2] )
    return p1/p2

def ratio_pdfs(x_entailment, x_neutral, lambda1, lambda2):
    a = 0
    b = 1 - x_neutral
    if lambda1 <= 0.5:
        c1 = 1 - x_neutral - (1 - x_neutral) / (4 * lambda1)
    elif lambda1 > 0.5:
        c1 = (1-x_neutral)/(4*(1-lambda1))
    if lambda2 <= 0.5:
        c2 = 1 - x_neutral - (1 - x_neutral) / (4 * lambda2)
    elif lambda2 > 0.5:
        c2 = (1-x_neutral)/(4*(1-lambda2))
    p1 = triangular(a, b, c1, x_entailment)
    p2 = triangular(a, b, c2, x_entailment)
    return p1/p2


def beta_prior(a, b):
    return beta(a=a, b=b)


def ratio_priors(lambda1, lambda2, lambda_prior):
    return lambda_prior.pdf(lambda1)/lambda_prior.pdf(lambda2)
import time

def ratio_posterior(entailments, neutrals,contradictions, lambda1, lambda2, lambda_prior):
    r_posterior = 1
    for i in range(len(entailments)):
        x_entailment = entailments[i]
        x_neutral = neutrals[i]
        x_contradiction = contradictions[i]
        r_likelihood = ratio_dir_pdfs(x_entailment, x_neutral, x_contradiction, lambda1, lambda2)
        r_prior = ratio_priors(lambda1, lambda2, lambda_prior)
        r_posterior *= r_likelihood * r_prior
    return r_posterior

def ratio_dir_posterior(X,entailment_sum, neutral_sum, contradictions_sum, lambda1, lambda2, lambda_prior):
    alpha1 = [lambda1, 1, 1-lambda1]
    alpha2 = [lambda2, 1, 1-lambda2]
    mvb_numerator1 = 1
    mvb_numerator2 = 1
    sum_lambda1 = np.sum(alpha1)
    sum_lambda2 = np.sum(alpha2)
    for i in range(len(alpha1)):
        mvb_numerator1 += loggamma(alpha1[i])
        mvb_numerator2 += loggamma(alpha2[i])

    mvb1 = loggamma(sum_lambda1) - mvb_numerator1
    mvb2 = loggamma(sum_lambda2) - mvb_numerator2

    p1 = len(X)*mvb1 + entailment_sum*alpha1[0] + neutral_sum*alpha1[1] + contradictions_sum*alpha1[2]
    p2 = len(X) * mvb2 + entailment_sum * alpha2[0] + neutral_sum * alpha2[1] + contradictions_sum * alpha2[2]
    ratio_prior = ratio_priors(lambda1, lambda2, lambda_prior)
    exp_r = torch.exp((p1/p2))
    # exp_r = 1/
    return exp_r*ratio_prior

def update_lambdas(X, entailments_sum, neutrals_sum, contradictions_sum, lambdas_current, lambdas_suggested, lambda_prior):
    new_lambdas = np.copy(lambdas_current)
    for l in range(X.size(1)):


        acceptance = np.min((1., ratio_dir_posterior(X,entailments_sum[l], neutrals_sum[l], contradictions_sum[l], lambdas_current[l], lambdas_suggested[l], lambda_prior)))
        r = random.uniform(0, 1)
        if r < acceptance:
            new_lambdas[l] = lambdas_suggested[l]
    return new_lambdas


def transition_function(lambdas, transition_epsilon):
    lambdas_suggested = np.copy(lambdas)
    for l in range(len(lambdas)):
        lambdas_suggested[l] = random.uniform(np.max((0., lambdas[l] - transition_epsilon)),
                                              np.min((1., lambdas[l] + transition_epsilon)))
    return lambdas_suggested

import time

def metropolis_hasting(data_dir, prior_alpha=1, prior_beta=100, transition_epsilon=0.001, num_steps=10000):
    X = torch.softmax(torch.as_tensor(torch.load(data_dir)),dtype=torch.float64, dim=2)
    # avg_neutrals = torch.mean(X[:, :, 1], dim=0)
    lambda_prior = beta_prior(prior_alpha, prior_beta)
    lambdas_current = np.ones(X.size(1)) * 0.01
    samples = [lambdas_current]
    entailments_sum = torch.sum(torch.log(X[:, :, 2]), dim=0)
    neutrals_sum = torch.sum(torch.log(X[:, :, 1]), dim=0)
    contradictions_sum = torch.sum(torch.log(X[:, :, 0]), dim=0)
    for n in range(num_steps):
        trans_start = time.time()
        lambdas_suggested = transition_function(lambdas_current, transition_epsilon)
        trans_end = time.time()
        # print('trans time:', trans_end-trans_start)
        update_start = time.time()
        lambdas_current = update_lambdas(X, entailments_sum, neutrals_sum, contradictions_sum, lambdas_current, lambdas_suggested, lambda_prior)
        update_end = time.time()
        # print('update time:', update_end-update_start)
        samples.append(lambdas_current)
        if n % 100 == 0:
            print(n)
    torch.save(samples, 'data')
    return samples


data_folder = 'C:/Users/jcotn/PycharmProjects/BNCL/update/inputs'
for dataset in ["stackexchange_philosophy", "reuters"]:
    samples = metropolis_hasting(data_dir=data_folder + "/" + dataset + "/train/X.pt")
    torch.save(samples, data_folder + "/" + dataset + "/lambdas.pt")

