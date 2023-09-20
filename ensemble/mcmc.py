from scipy.stats import beta
import numpy as np
from numpy import random
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


def ratio_posterior(entailments, x_neutral, lambda1, lambda2, lambda_prior):
    r_posterior = 1
    for i in range(len(entailments)):
        x_entailment = entailments[i]
        r_likelihood = ratio_pdfs(x_entailment, x_neutral, lambda1, lambda2)
        r_prior = ratio_priors(lambda1, lambda2, lambda_prior)
        r_posterior *= r_likelihood * r_prior
    return r_posterior


def update_lambdas(X, avg_neutrals, lambdas_current, lambdas_suggested, lambda_prior):
    new_lambdas = np.copy(lambdas_current)
    for l in range(X.size(1)):
        acceptance = np.min((1., ratio_posterior(X[:, l, 2], avg_neutrals[l], lambdas_current[l], lambdas_suggested[l], lambda_prior)))
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


def metropolis_hasting(data_dir, prior_alpha=1, prior_beta=100, transition_epsilon=0.001, num_steps=100):
    X = torch.softmax(torch.as_tensor(torch.load(data_dir)), dim=2)
    avg_neutrals = torch.mean(X[:, :, 1], dim=0)
    lambda_prior = beta_prior(prior_alpha, prior_beta)
    lambdas_current = np.ones(X.size(1)) * 0.01
    samples = [lambdas_current]
    for n in range(num_steps):
        print(n)
        lambdas_suggested = transition_function(lambdas_current, transition_epsilon)
        lambdas_current = update_lambdas(X, avg_neutrals, lambdas_current, lambdas_suggested, lambda_prior)
        samples.append(lambdas_current)
    torch.save(samples, 'data')
    return samples

# l = torch.load("/Users/muberra/Documents/BNCL/inputs/reuters/lambdas.pt")

data_folder = "/Users/muberra/Documents/BNCL/inputs"
for dataset in ["stackex_philosophy", "reuters"]:
    samples = metropolis_hasting(data_dir=data_folder + "/" + dataset + "/train/X.pt")
    torch.save(samples, data_folder + "/" + dataset + "/lambdas.pt")










