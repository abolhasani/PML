# U - CS6190 - Spring 23 - HW2 - P2 - u1416052 
from google.colab import drive
import io
import pandas as pd
import numpy as np
from google.colab import files
import math
import warnings
from scipy.optimize import minimize
from scipy.stats import norm
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
import scipy.stats as stats

#suppress warnings
warnings.filterwarnings('ignore')

# set maximum number of iterations and tolerance level, and regularization parameter
max_iter = 100
tolerance = 1e-5
lambda_=1
np.random.seed()

# load training and test data from my google drive in colab environment
drive.mount('/content/drive')
df_train = pd.read_csv('/content/drive/My Drive/DESB/Courses/2023-1(Spring)/CS6190/HW/2/Data/train.csv', header=None)
df_test = pd.read_csv('/content/drive/My Drive/DESB/Courses/2023-1(Spring)/CS6190/HW/2/Data/test.csv', header=None)
# leaving the last column for y and capturing the rest for x
x_train = df_train.iloc[:, :-1].values
y_train = df_train.iloc[:, -1].values
x_test= df_test.iloc[:, :-1].values
y_test = df_test.iloc[:, -1].values

# Normalize the data as advised
scaler = StandardScaler()
x_train = scaler.fit_transform(x_train)
x_test = scaler.transform(x_test)
#x_train_std = np.std(x_train, axis=0)
#x_train_mean = np.mean(x_train, axis=0)
#x_test_norm = (x_test- x_train_mean) / x_train_std
#x_test= x_test_norm

# This functuion returns the log-likelihood, gradient matrix and the hessian matrix for the logistic regression
def logistic_log_likelihood (x_train, y_train, w):
  # phi = x_train, w = w, y_n=sigmoid(wtphi), lambda =1 (regularization parameter), t_n = y_train binary variables
  # here, z is W^t*Phi
  wtphi = x_train.dot(w)
  # we then calculate E(w), i used the formula in question 5 in HW2
  log_likelihood = -np.sum(y_train*np.log(sigmoid(wtphi))+(1-y_train)*np.log(1-sigmoid(wtphi)))-0.5*np.dot(w.T,w)
  #log_likelihood = np.sum(y_train*wtphi - np.log(1 + np.exp(wtphi))) - 0.5*np.sum(w**2)
  #gradient = np.dot((sigmoid(wtphi)-y_train),x_train.T)+w
  # I calculate gradient based on formula in slide 46 of chapter 8 (generalized linear)
  gradient = np.dot(x_train.T, sigmoid(wtphi)-y_train)
  # I used the formula in slide 49 same chapter for hessian
  hessian = np.dot(x_train.T,x_train)
  #hessian = np.dot(x_train.T * sigmoid(np.dot(x_train, w)) * (1-sigmoid(np.dot(x_train, w))), x_train) + np.eye(x_train.shape[1])
  return log_likelihood, gradient, hessian

# This functuion return thes log-likelihood, gradient matrix and the hessian matrix for the probit regression
def probit_log_likelihood(w, X, y):
  # I used this website: https://www.statlect.com/fundamentals-of-statistics/probit-model-maximum-likelihood for help with the formulas
    wtphi = np.dot(X, w)
    q=2*y-1
    lambdai = q*stats.norm.pdf(q*wtphi)/stats.norm.cdf(q*wtphi)
    # log-likelihood is essentially the same just sigmoid is reoplaced by normal cdf
    #log_likelihood = np.sum(y_train*np.log(stats.norm.cdf(wtphi))+(1-y_train)*np.log(1-stats.norm.cdf(wtphi)))
    #log_likelihood = np.sum(np.log(stats.norm.cdf(q*wtphi)))
    log_likelihood = np.sum(np.log(y*wtphi))-0.5*np.dot(w.T,w)#-len(y)/2*np.log(2*np.pi)
    ####log_likelihood = np.sum(np.log(norm.cdf(y*wtphi))) - 0.5*np.dot(w, w)
    # By paying attention to log-likelihood we have: derivative of log_likelihood w.r.t. w = -X.T * y * norm.pdf(y*z) / norm.cdf(y*z) - w, for z, we use the chain rule.
    gradient = np.dot(X.T, y*norm.pdf(wtphi)/norm.cdf(y*wtphi)) - w
    #hessian = -np.dot( np.dot(X, X.T),  y**2*norm.pdf(wtphi)/norm.cdf(y*wtphi))# - np.eye(X.shape[0])
    hessian = -np.dot(X, np.dot(X.T, y*norm.pdf(wtphi)/norm.cdf(y*wtphi))) - np.eye(X.shape[0])
    # After too many failed attempts to successfully compute hessian for part c based on the formulas i found on the internet, i calculated each part seperately and then merged them together.
    cdf = norm.cdf(y*wtphi)
    pdf = y*norm.pdf(wtphi)
    hessian = np.zeros((X.shape[1], X.shape[1]))
    #for i in range(X.shape[0]):
       # hessian += (pdf[i]/cdf[i]) * np.outer(X[i], X[i])
    #hessian += np.eye(X.shape[1])
    #ypdfcdf = np.diag((pdf/cdf)**2)
    ypdfcdf = np.diag((pdf/cdf))
    #hessian = -np.dot(np.dot(X.T, ypdfcdf), X) - np.eye(X.shape[1])
    hessian = -np.dot(np.dot(X.T, np.diag((pdf/cdf))), X) - np.eye(X.shape[1])
    #print('gradient:', gradient)
    #print('hessian:',hessian)
    return log_likelihood, gradient, hessian

# this is the sigmoid function used by logistic regression to give y(phi)
def sigmoid(a):
    return 1 / (1 + np.exp(-a))

# This function provides the Newton_Raphson update with the stopping condition provided by the question
def newton_raphson (max_iter, tolerance, problem, weight, x_train, y_train, w_old):
  for i in range(max_iter):
    # Get the hessian and gradient matrixes from the respective regression models
    if problem == 'A':
      loglike, gradient, hessian = logistic_log_likelihood (x_train, y_train, w_old)
    if problem == 'C' :
      loglike, gradient, hessian = probit_log_likelihood(w_old, x_train, y_train)
    # update newton-raphson scheme weights using np.matmul operand
    w_new = w_old -np.linalg.inv(hessian) @ gradient
    # use told tolerance to check for convergence
    if np.linalg.norm(w_new - w_old) < tolerance:
      #w_old = w_new
      break
    w_old = w_new
  return w_old

def pred_results (x_test, y_test, w, problem, weight):
  if problem == 'A':
    pred_y = np.round(sigmoid(np.dot(x_test, w)))
  else :
    pred_y = np.sign(np.dot(x_test, w))
  pred_y = list(map(int, pred_y))
  for i in range(len(pred_y)):
    if pred_y[i] == -1:
      pred_y[i] = 0
  accuracy = np.mean(pred_y == y_test)
  #print("The graph of prediction Vs. Actual test results, Part {}, weights set to: {}".format(problem, weight))
  #for pred, test in zip(pred_y, y_test):
      #print(f"{pred}\t{test}")
  print("Accuracy of prediction for Part {}, weights set to: {}".format(problem, weight))
  print(accuracy)

###################################### A
# initialize weights to zeros
w = np.zeros(x_train.shape[1])
w0 = newton_raphson(max_iter, tolerance, 'A', '0', x_train, y_train, w)
pred_results(x_test, y_test, w0, 'A', '0')

# initialize weights to random Gaussian values
w = np.random.normal(size=x_train.shape[1])
w0 = newton_raphson(max_iter, tolerance, 'A', 'r', x_train, y_train, w)
pred_results(x_test, y_test, w0, 'A', 'r')

###################################### B
# optimize  with L-BFGS
w0 = np.zeros(x_train.shape[1])
maxed = minimize(probit_log_likelihood, w0, args=(x_train, y_train), method='L-BFGS-B', jac=True, options={'maxiter': max_iter, 'gtol': tolerance})
w = maxed.x
pred_results(x_test, y_test, w, 'B', '0')

w0 = np.random.normal(size=x_train.shape[1])
maxed = minimize(probit_log_likelihood, w0, args=(x_train, y_train), method='L-BFGS-B', jac=True, options={'maxiter': max_iter, 'gtol': tolerance})
w = maxed.x
pred_results(x_test, y_test, w, 'B', 'r')

###################################### C
w0 = np.zeros(x_train.shape[1])
w = newton_raphson(max_iter, tolerance, 'C', '0', x_train, y_train, w0)
pred_results(x_test, y_test, w, 'C', '0')
w1 = np.random.normal(size=x_train.shape[1])
w = newton_raphson(max_iter, tolerance, 'C', 'r', x_train, y_train, w1)
pred_results(x_test, y_test, w, 'C', 'r')