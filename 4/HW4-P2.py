import numpy as np
from scipy.stats import norm
from scipy.optimize import minimize
import pandas as pd
from google.colab import files
import warnings

warnings.filterwarnings("ignore")

uploaded = files.upload()
data_train_X=pd.read_csv('train.csv',names=np.arange(1,6))
train_Y = np.reshape(data_train_X.iloc[:,-1].values,(-1,1))
data_train = data_train_X.iloc[:,:-1]
data_test_X=pd.read_csv('test.csv',names=np.arange(1,6))
test_Y = np.reshape(data_test_X.iloc[:,-1].values,(-1,1))
data_test = data_test_X.iloc[:,:-1]

# initializations
m_0 = np.zeros((data_train.shape[1],1))
S_0 = np.eye(data_train.shape[1])
iterations = 1000
dof = 20
threshold = 1e-3

# copying the function given in the example python file, adding **kargs because of the shifted sigmoid function
def gauss_hermite_quad(f, dof, **kwargs):
    points, weights = np.polynomial.hermite.hermgauss(dof)
    f_x = f(points, **kwargs)
    F = np.sum(f_x * weights)
    return F

# sigmoid function 
def sigmoid(W , degrees):
    return 1 / (1+np.exp(-degrees @ W))

# sigmoid function fed into prediction function to acount for data shifted by means and variancen mean and var named not to be confused in calling in predictive distro
def shifted_sigmoid(W, mean, var):
    return 1 / (1 + np.exp(-np.sqrt(2 * var) * W - mean))

# prediction function that feeds mean and var to the gauss-hermite function and get the integral for each part
def predictive_distro(data_train,train_Y,m_w,S_w):
    N = len(data_train)
    likelihood = np.zeros(N)
    error_count = 0
    for i in range(N):
        phi_n = np.reshape(data_train.iloc[i,:].values,(-1,1))
        # doing the variable transformation and passing them to the gauss-hermite function. for this, the function was modified because of the change in parameters for sigmoid function
        integral = gauss_hermite_quad(f=shifted_sigmoid, dof=dof, mean=m_w.T @ phi_n, var=phi_n.T@S_w@phi_n) / np.sqrt(np.pi)
        predict_Y = int(sigmoid(m_w,phi_n.T) >= 0.5)
        y_n = train_Y[i]
        if y_n.item() == 1:
            likelihood[i] = integral
        else:
            likelihood[i] = 1 - integral
        if  predict_Y != y_n:
            error_count += 1             
    return (1-error_count/N), np.mean(likelihood)       

# calculating t labels based on the question criteria and the \sum_n t-N * ln y_n .... part in log posterior
def log_loss (data_train, W, train_Y):
    summ_sen = 0
    for i in range(len(data_train)):
        y_n = sigmoid(W,data_train.iloc[i,:].values.reshape(-1, 1).T)
        t_label = train_Y[i]
        summ_sen += t_label*np.log(y_n)+(1-t_label)*np.log(1-y_n)
    return -1*summ_sen[0]

# calculating log p(w,t) based on formula page 11 chapter 12-laplace approximation
def log_posteriors(w,data_train, train_Y, S_0, m_0):
    W_m = np.expand_dims(w, axis=1)
    logpwt = 1/2*(W_m-m_0).T @ np.linalg.inv(S_0) @ (W_m-m_0)
    logpwt = float(logpwt[0][0])
    return (logpwt+log_loss (data_train, w, train_Y))#post

# get W_MAP and S_n, similar to log_loss and problem 1 part b 
def laplace_app (data_train, train_Y, S_0, m_0):
    w_0= np.zeros((data_train.shape[1],1))
    solve = minimize(log_posteriors, w_0, args = (data_train, train_Y, S_0, m_0) , method='L-BFGS-B')
    W_MAP = solve.x # for mean
    # based on the formula in slide 11 chapter 12-Laplace approximation
    S_n = 0
    for i in range(len(data_train)):
      phi_n = np.reshape(data_train.iloc[i,:].values,(-1,1))
      y_n= sigmoid(W_MAP,np.reshape(data_train.iloc[i,:].values,(-1,1)).T)
      S_n += y_n*(1-y_n)*(phi_n @ phi_n.T)
    S_n = np.linalg.inv(np.linalg.inv(S_0) + S_n)
    return W_MAP, S_n

# lambda function from problem 1 part c used for variational logistic regression - slide 45 of chapter 13-variational
def lambda_xi(z):
    return (1/(1+np.exp(-z))-0.5)/(2*(z))

# variational logistic regression for part C and D (only E-step differs)
def vlr (data_train, train_Y, charP, S_0, m_0, iterations, threshold):
  data_X =np.reshape(data_train.values,(-1,4))
  xi = np.ones((data_train.shape[0],1))
  m_n= np.zeros((data_train.shape[1],1))
  S_n = np.eye(data_train.shape[1])
  for i in range (iterations):
    #E step (seperated for C and D)
    lambda_xi_ = np.diagflat(lambda_xi(xi))
    for j in range(data_train.shape[1]): 
        if charP=='C':
                S_n =np.linalg.inv(np.linalg.inv(S_0) + 2*(data_X.T @ lambda_xi_ @data_X))
                m_n =S_n @ (np.linalg.inv(S_0) @ m_0+ data_X.T @ (train_Y-0.5))
        else: 
                # Here, we should first compute var and mean seperately
                # S_n is computed as slide 52 chapter 13-variational
                s = 1 / (1 + np.exp(-1*np.multiply(train_Y, data_X @ m_n)))
                lambda_xi_diag = 2*np.diagflat(lambda_xi(np.sqrt(np.sum((data_X @ S_n) * data_X, axis=1))))
                S_n[j, j] = 1 / (np.sum(data_X[:, j].T * s * (1 - s)* lambda_xi_diag * data_X[:, j]) + 1/S_0[0,0])
                m_n[j] = S_n[j, j] * (np.sum(data_X[:, j] * (train_Y.ravel() - 0.5))+ 1/S_0[0,0]*m_0[j])
    #M step (the same for factorized and non-factorized)
    xi_updated = np.zeros((len(data_train), 1))
    for i in range(len(data_train)):
        phi_n= np.reshape(data_train.iloc[i,:].values,(-1,1))
        xi_updated[i] = np.sqrt( phi_n.T @ (S_n+m_n @ m_n.T) @ phi_n)
    if np.linalg.norm(xi_updated-xi) >= threshold:
        xi = xi_updated
    else:
        break  
  return m_n, S_n

############################################################################## A
print('A')
m_n, S_n = laplace_app(data_train, train_Y, S_0, m_0)
print ('m_n : ', m_n)
print('S_n : ', S_n)
accuracy , ave_pred_likelihood = predictive_distro(data_test, test_Y, m_n, S_n)
print ('Accuracy: ', accuracy)
print ('Predictive likelihood: ',  ave_pred_likelihood)
############################################################################## B
print('B')
S_n_diag = np.diag(np.diag(S_n))
print ('m_n : ', m_n)
print('S_n : ', S_n_diag)
accuracy , ave_pred_likelihood = predictive_distro(data_test, test_Y, m_n, S_n_diag)
print ('Accuracy: ', accuracy)
print ('Predictive likelihood: ',  ave_pred_likelihood)
############################################################################## C
print('C')
m_n, S_n = vlr(data_train, train_Y, 'C', S_0, m_0, iterations, threshold)  
print ('m_n : ', m_n)
print('S_n : ', S_n_diag)
accuracy , ave_pred_likelihood = predictive_distro(data_test, test_Y, m_n, S_n)
print ('Accuracy: ', accuracy)
print ('Predictive likelihood: ',  ave_pred_likelihood)
############################################################################## D
print('D')
m_n, S_n = vlr (data_train, train_Y, 'D', S_0, m_0, iterations, threshold)
print ('m_n : ', m_n)
print('S_n : ', S_n_diag)
accuracy , ave_pred_likelihood = predictive_distro(data_test, test_Y, m_n, S_n)
print ('Accuracy: ', accuracy)
print ('Predictive likelihood: ',  ave_pred_likelihood)