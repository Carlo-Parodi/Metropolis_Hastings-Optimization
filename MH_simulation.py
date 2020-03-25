#!/usr/bin/env python
# coding: utf-8

# In[13]:

import numpy as np
import matplotlib as mp
import matplotlib.pyplot as plt
from math import erf, sqrt

#Here is the proposal distribution which originates new proposal values.
#It's different for different parameters, the average always corresponds
#to the current last value of the process
def create_new_value(res):
    alpha_prop = np.random.normal(loc = res[0], scale = 0.08)
    beta_prop = np.random.normal(loc = res[1], scale = 0.08)
    return (alpha_prop, beta_prop)

#This function determines whether to use the proposal value as next value,
#or keep using the current one
def acceptance_rule(tot1, tot2):
    ratio = tot2 - tot1
    if ratio < 0:
        u = np.random.uniform(0,1)
        if np.log(u) < ratio:
            return True
        else:
            return False
    return True

#the prior function P(ϑ) is a Gaussian with different values of mean and variance 
#which variates, depending on which parameter is to be estimated
def priori(x, param, priori_params):
    if param == 'alpha':
        mu = priori_params[0]
        sigma = priori_params[1]
    else:
        mu = priori_params[2]
        sigma = priori_params[3]
    delta = 0.0001
    upper_b = x+delta
    lower_b = x-delta
    doub_pro1 = erf( (lower_b-mu) / (sigma*sqrt(2)) )
    doub_pro2 = erf( (upper_b-mu) / (sigma*sqrt(2)) )
    p_upper1 = doub_pro1/2
    p_upper2 = doub_pro2/2
    prob = p_upper2 - p_upper1
    prob = np.log(prob)
    return prob

#This function generates the new value of the markov chain
def create_next(sample, res, priori_params):
    new_val = create_new_value(res)
    alpha_curr = res[0] 
    beta_curr = res[1]
    alpha_prop = new_val[0] 
    beta_prop = new_val[1]
    first_obs = sample[0]
    
    first_prop = -0.5*(first_obs**2 + alpha_prop**2 - 2*first_obs*alpha_prop)
    first_curr = -0.5*(first_obs**2 + alpha_curr**2 - 2*first_obs*alpha_curr)

    part_b_prop = 0
    part_c_prop = 0
    part_b_curr = 0
    part_c_curr = 0
    
    for i in range(len(sample)):
        part_b_prop = part_b_prop - 0.5*(sample[i]**2 + alpha_prop**2 + (beta_prop*sample[i-1])**2)
        part_c_prop = part_c_prop + sample[i]*alpha_prop + sample[i]*sample[i-1]*beta_prop - alpha_prop*beta_prop*sample[i-1]      

        part_b_curr = part_b_curr - 0.5*(sample[i]**2 + alpha_curr**2 + (beta_curr*sample[i-1])**2)
        part_c_curr = part_c_curr + sample[i]*alpha_curr + sample[i]*sample[i-1]*beta_curr - alpha_curr*beta_curr*sample[i-1]       

    tot_curr = first_curr + part_b_curr + part_c_curr + priori(alpha_curr, 'alpha', priori_params) + priori(beta_curr, 'beta', priori_params)   
    tot_prop = first_prop + part_b_prop + part_c_prop + priori(alpha_prop, 'alpha', priori_params) + priori(beta_prop, 'beta', priori_params)

    
    if acceptance_rule(tot_curr, tot_prop):
        return alpha_prop, beta_prop, alpha_curr, beta_curr
    else:
        return alpha_curr, beta_curr, alpha_prop, beta_prop

#This function generates a Markov Chain Monte Carlo with Metropolis Hastings' method.
#In input are required a sample, the number of new observations we want to generate,
# and the starting points from where to begin the simulation of new value for the parameters
def create_mcmc(sample, num_param_obs, starting_points, priori_params):
    res = starting_points
    barrel_alpha = np.zeros(num_param_obs)
    barrel_beta = barrel_alpha.copy()
    discard_alpha = barrel_alpha.copy()
    discard_beta = barrel_alpha.copy()
    for i in range(num_param_obs):
        res = create_next(sample, res, priori_params)
        barrel_alpha[i] = res[0]
        barrel_beta[i] = res[1]
        discard_alpha[i] = res[2]
        discard_beta[i] = res[3]
        res = (res[0], res[1])
    return barrel_alpha, barrel_beta, discard_alpha, discard_beta

#this function displays a histogram, takes in input the output of 'create_mcm'
#function here above, the parameter you want to see ('alpha' or 'beta'), and the 
#granularity of the graph, which is set to 'auto' by default
def create_hist_4_param(barrel, param, granularity = 'auto'):
    plt.style.use('dark_background')
    plt.figure(figsize=(28,14))
    if param == 'alpha':
        pos = 0
    else:
        pos = 1
    param = plt.hist(barrel[pos], bins = granularity)
    plt.show()
    maximum = np.where(param[0] == max(param[0]))[0][0]
    print('highest frequency for the param. occurred at the value: ' + str(param[1][maximum]))

def show_pattern(barrel):
    plt.figure(figsize=(40,20))
    plt.style.use('dark_background')
    plt.plot(barrel[2], 'g')
    plt.plot(barrel[3], 'y')
    plt.plot(barrel[1], 'b')
    plt.plot(barrel[0], 'r')
    plt.show()
    
#this function shows you the joint histogram, granularity parameter is an integer
#that by default is set = 100
def joint_hist(barrel, granularity = 100):
    plt.figure(figsize=(28,14))
    alfa = plt.hist2d(barrel[2], barrel[3], bins = granularity)




