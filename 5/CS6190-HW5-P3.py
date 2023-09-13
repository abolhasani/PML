
import math
import numpy as np
import torch
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from scipy.stats import norm
from scipy.optimize import minimize
import pandas as pd
from scipy.stats import multivariate_normal
from scipy.special import expit
from numpy import log, exp
import warnings
#from tabulate import tabulate

warnings.filterwarnings("ignore")

train = np.genfromtxt("train.csv", delimiter=",", dtype=float)
test = np.genfromtxt("test.csv", delimiter=",", dtype=float)

# Separate the labels from the data
train_Y = train[:, -1].reshape(-1, 1)
test_Y = test[:, -1].reshape(-1, 1)
train_X = train[:, :-1]
test_X = test[:, :-1]

# initializations

L=[10,20,50]
eps=[0.005, 0.01, 0.02, 0.05]
# to avoid getting log from 0 or 1 
tlrnc=1e-8
burn_in_it=100000
After_burn_sample =10000
pick=10
iterations = 1000
dof = 20
threshold = 1e-5

############################################################# A
# we will comstruct the required functions here. 
# According to Chapter 15-Sampling, We need H = U + K where U is the log likelihood. So:

# copied from previoius HWs
def sigmoid(W , degrees):
    return np.reshape((1 / (1+np.exp(-degrees @ W))), (-1,1)) 

# function to give log-likelihood, log-likelihood of the Bernoulli likelihood term in the joint probability of the Bayesian logistic regression
def log_like (x,w,t):
    return (t.T @ log(sigmoid(w,x)+tlrnc) + (1-t).T @ log(1-sigmoid(w,x)+tlrnc))

# from slide 63 
def U_z(r,w,t):
    w = np.reshape(w,r.shape[1])
    prior = multivariate_normal.logpdf(w,np.zeros(w.shape[0]),np.eye(w.shape[0]))
    y = sigmoid(w,r)
    log_likelihood = log_like (r,w,t).item()
    return -1*(prior + log_likelihood)

# the partial gradient function  log-likelihood of the Bernoulli likelihood term 
# in the joint probability of the Bayesian logistic regression model with respect to the weight vector
def derivative_U_Z(x,w,t):
    return np.reshape((x.T @ (sigmoid(w,x) - t)), (-1,1)) 

# from slide 63    
def K_r(r,M):
    return (0.5 * r.T @ np.linalg.inv(M) @ r)[0]

# Leapfrog function for hmc
def Leapfrog(x,z,t,r,M,L,eps):  
    for i in range(L):
        r -= 0.5 * eps * derivative_U_Z(x,z,t)
        z += eps*np.linalg.inv(M) @ r 
        r -= 0.5 * eps * derivative_U_Z(x,z,t)
    return z, r

# from slide 64
def accept_rule( x, t, r, M, z, r_prim, w_prim):
  return min([1,exp(-1*U_z(x,w_prim,t) -1* K_r(-r_prim,M) + U_z(x,z,t) + K_r(r,M))])

# two stage to run hmc up to burn in and 10k after that
def hmc(x, t, z_0, L, M, burn_in_it,After_burn_sample,eps,pick):
    # untill burn in
    w = z_0
    for i in range (burn_in_it):
        r = np.array([np.random.normal(0, M[j,j]) for j in range(len(z_0))]).reshape(-1, 1)
        w_prim, r_prim = Leapfrog(x,w,t,r,M,L,eps)
        if np.random.uniform(0,1) > accept_rule( x, t, r, M, w, r_prim, w_prim) :
            w_prim = w
        w = w_prim
    # after burn in
    total = int(After_burn_sample/pick)
    sample=np.zeros((total,w.shape[0]))
    # variable to hold number of accepted rules used for acceptance rate
    accept = 0
    # variable to get the picked 10th iteration row
    row=0
    for i in range (After_burn_sample):
        r = np.array([np.random.normal(0, M[j,j]) for j in range(len(z_0))]).reshape(-1, 1)
        w_prim, r_prim = Leapfrog(x,w,t,r,M,L,eps)
        accept += int((accept_rule(x, t, r, M, w, r_prim, w_prim) >= np.random.uniform()))
        w_prim = w if accept_rule(x, t, r, M, w, r_prim, w_prim) < np.random.uniform() else w_prim
        # pick every 10 one
        if i % pick==0 :
            sample[row,:] = np.reshape(w_prim,len(z_0))
            row +=1
    Acceptance_rate = float(accept/After_burn_sample)
    return sample, Acceptance_rate

def predictive_accuracy(x,w,t):
    err=0
    for i in range (x.shape[0]):
        # the actual prediction task
        predict_Y = np.where(sigmoid(w,x) >= 0.5, 1, 0)
        # comparing with labels to see if it is erronous
        err = np.sum(predict_Y != t)
    log_likelihood = log_like (x,w,t)
    return 1-(err/x.shape[0])  , log_likelihood/t.shape[0]

def arr_mean (arra):
    arra = np.array(arra)
    return arra.mean()

# run the algorithm for eps and L
for e in eps:
    for f in L:
        # initialization with 0
        w_0 = np.zeros((train_X.shape[1],1))
        M = np.eye(w_0.shape[0])
        selected_samples, acceptance_rate=  hmc(train_X, train_Y, w_0, f, M, burn_in_it,After_burn_sample,e,pick)
        Train_predict_acc, Train_predict_ll = zip(*[predictive_accuracy(train_X, np.reshape(w, (-1, 1)), train_Y) for w in selected_samples])
        Test_predict_acc, Test_predict_ll = zip(*[predictive_accuracy(test_X, np.reshape(w, (-1, 1)), test_Y) for w in selected_samples])
        # printing the asked values
        print("eps:", e, "L:", f, "Acceptance rate:", acceptance_rate )
        print("test\n", "predictive Accuracy:", arr_mean(Test_predict_acc), "Average Predictive Likelihood:", arr_mean(Test_predict_ll))
        print("train\n", "predictive Accuracy:", arr_mean(Train_predict_acc), "Average Predictive Likelihood:", arr_mean(Train_predict_ll), "\n")
