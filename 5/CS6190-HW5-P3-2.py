import numpy as np
import pandas as pd
import scipy.stats as stats

x_train=pd.read_csv('train.csv',names=np.arange(0,4),dtype=np.float64()) # modify to only include columns 0-3
y_train = np.reshape(x_train.iloc[:,-1].values,(-1,1))
x_train = x_train.iloc[:,:4].to_numpy()
x_test=pd.read_csv('test.csv',names=np.arange(0,4),dtype=np.float64()) # modify to only include columns 0-3
y_test = x_test.iloc[:,-1]
x_test = x_test.iloc[:,:4].to_numpy()

# initialization
burn_in= 100000
After_burn_in = 10000 
sigma = 1
pick = 10
tol = 1e8

# the sigmoid function used in lieu of normal
def sigmoid(x):
    return 1/(1 + np.exp(-x))

# sampler for W used in sigmoid
def prior_precision(x_train, y_train, w_init):
    W = np.eye(x_train.shape[1]) + np.sum(x_train[:, :, np.newaxis]@ x_train[:, np.newaxis, :], axis= 0)
    return W

# effect of bayesian logistic regression to replace normal distro
def marginal_z(x_train,w):
    return sigmoid(x_train@w)

# to calculate the likelihood of prediction with the original equation provided by the question
def likelihood (y_test, prediction ):
    likeli_hood = y_test * np.log(prediction + 1/tol) + (1.0 - y_test) * np.log(1.0 - prediction + 1/tol)
    return likeli_hood.mean()

# using truncated gaussian (to restrict domain) to generate samples 
def truncated_Gaussian(z_n, x_train ):
        # lower truncation point for stats.truncnorm
        lower_point = np.zeros((x_train.shape[0], 1))
        lower_point[y_train.ravel() < 0.5, :] = -tol  # use ravel() to flatten y_train
        # upper truncation point for stats.truncnorm
        upper_point= tol*np.ones ((x_train.shape[0], 1))
        upper_point[y_train.ravel() < 0.5, :] = 0      # use ravel() to flatten y_train
        # using truncnorm as advised by thee question
        # provided by scipy: a, b = (myclip_a - loc) / scale, (myclip_b - loc) / scale
        a = (lower_point - z_n) / sigma
        b = (upper_point - z_n) / sigma
        X = stats.truncnorm(a, b, loc= z_n, scale= sigma)
        # generate the random samples after truncation 
        z_n = X.rvs((x_train.shape[0],1))
        return z_n

# Gibbs sampling
def Gibbs(x_train, y_train, w_init, burn_in, After_burn_in):
    # initialize Gibbs
    w_T = np.linalg.inv(prior_precision(x_train, y_train, w_init))
    w = w_init
    z = marginal_z(x_train,w)
    # Gibbs sampling
    for i in range(burn_in):
        # sampling w posterior step
        w_new_mean = w_T @  np.sum(z*x_train, axis= 0)[:, np.newaxis]
        w = np.random.multivariate_normal(w_new_mean[:, 0], w_T, 1).T
        # updating auxiliary variable 
        z_n =marginal_z(x_train,w)
        z = truncated_Gaussian(z_n, x_train)
    w_init = w
    w_T = np.linalg.inv(prior_precision(x_train, y_train, w_init))
    w = w_init
    z = marginal_z(x_train,w)
    # array to hold final posterior sample
    samples = np.zeros((After_burn_in // pick, x_train.shape[1]))
    row = 0
    # Gibbs sampling
    for i in range(After_burn_in):
        w_new_mean = w_T @  np.sum(z*x_train, axis= 0)[:, np.newaxis]
        w = np.random.multivariate_normal(w_new_mean[:, 0], w_T, 1).T
        z_n =marginal_z(x_train,w)
        z = truncated_Gaussian(z_n, x_train)
        # taking samples from the last 10
        if i % pick == 0:
            samples[row] = w.ravel()
            row +=1
    return samples

# initiating with prior as asked by the question
w_init = np.zeros((x_train.shape[1], 1))
sampled_probit_posterior = Gibbs(x_train, y_train, w_init , burn_in, After_burn_in)
num_samples = sampled_probit_posterior.shape[0]
# pred Accuracy and pred- log-likelihood containers
prediction_accuracy = np.zeros((num_samples))
prediction_log_ll = np.zeros((num_samples))         
for i in range(num_samples):
    w_n = sampled_probit_posterior[i]
    prediction = np.squeeze(marginal_z(x_test, w_n))  # Remove single-dimensional entries from the shape
    prediction = np.where(prediction < 0.5, 0.0, 1.0)
    accuracy = np.sum(prediction == y_test)/prediction.shape[0]
    prediction_accuracy[i] = accuracy
    prediction_log_ll [i] = likelihood (y_test, prediction )
mean_accuracy = prediction_accuracy.mean()
mean_log_ll = prediction_log_ll.mean()
print("Prediction Accuracy:", mean_accuracy)
print("Prediction log-likelihood:", mean_log_ll)
