import numpy as np
from scipy.special import expit
from scipy.optimize import minimize
import matplotlib.pyplot as plt

# copying the function given in the example python file
def gass_hermite_quad(f, degree):
    '''
    Calculate the integral (1) numerically.
    :param f: target function, takes a array as input x = [x0, x1,...,xn], and return a array of function values f(x) = [f(x0),f(x1), ..., f(xn)]
    :param degree: integer, >=1, number of points
    :return:
    '''

    points, weights = np.polynomial.hermite.hermgauss(degree)

    # function values at given points
    f_x = f(points)

    # weighted sum of function values
    F = np.sum(f_x * weights)

    return F

# sigmoid function that is fed to the gauss-hermite quadradite function
def sigmoid(z):
    return np.exp(10 * z + 3) / (1 + np.exp(10 * z + 3))

# set the degree of freedom
degree = 100

######################################################################### A
N = gass_hermite_quad(sigmoid, degree)
print('Normalization Constant: ', N)

# calculating p(z) and plotting
z = np.linspace(-5, 5, 1000)
P_density = np.exp(-z**2)*sigmoid(z)/N
plt.plot(z, P_density)
plt.xlabel('z')
plt.ylabel('Density')
plt.title('Density Curve of P(z)')
plt.show()

######################################################################### B
from scipy.stats import norm

# calculating log p(z)
def log_dens(z):
    return -np.log(np.exp(-z**2)*sigmoid(z))

# calculating argmax log p(z) which will be the mean of the gaussian distribution N(theta| theta_0, A^-1)
solve = minimize(log_dens, 1,method='L-BFGS-B')
gaussian_mean = solve.x[0]
print ('mean = ', gaussian_mean)

# using the formula for A in slide 7 of 12-laplace-approximation chapter: A = -d/d^2 log p(z)| theta_max
# we have log p(z) above and the first derivative will be: -2z +10sigmoid'(z)/sigmoid(z), sigmoid'(z) = sigmoid(z)(1-sigmoid(z))
# second derivative will be:
def variance(z):
  return 2+100*sigmoid(z)*(1-sigmoid(z))

gaus_var = variance(gaussian_mean)
print('variance = ',1/gaus_var)

# now that we have the mean and variance of the needed gaussian distribution, we calculate the probability dencity function of the laplace approximation
laplace_app= norm.pdf(z, loc=gaussian_mean , scale=1/gaus_var)/N
plt.plot(z,laplace_app)
#plt.title('Laplace Approximation')
plt.plot(z, P_density)
plt.xlabel('z')
plt.ylabel('Density')
plt.title('Density Curve of P(z) with Laplace Approximation')
plt.show()

######################################################################### C
# 
# set the initialization params for the first iteration
epochs = 1000
S_0 = 1
S = S_0
m_0 = 0.1
m = m_0
xi=1
threshold = 1e-3

# calculating the lamda function as in slide 45 of chapter 13-variational
def lambda_xi(x):
    if x == 0:
        return 1/12
    else:
        return (1/(1+np.exp(-x))-0.5)/(2*(x))

# calculating the lower-bound sigma xi function as in slide 46 of chapter 13-variational
def lower_bound(z, xi_new):
    return np.exp(-z**2)+1/(1+np.exp(-xi_new))*np.exp((10*z+3-xi_new)/2-lambda_xi(xi_new)*((10*z+3)**2-(xi_new)**2))

# calculating the binary t label used in m_n
def calculate_t_label(z):
    p_z = np.exp(-z**2) * sigmoid(10 * z + 3)
    if np.sum(np.where(p_z >= 0.5, 1, 0)-0.5)>= 0:
      return 1
    else:
      return 0

# maximize the variational lower bound based on slide 54 of chapter 13-variational
for i in range(1, epochs):
    # E step: update q(w), q(w) = N(w|m_n, S_n)
    # m and S according to slide 52 of chapter 13-variational after simplifying:

    S = 1/(1/S_0+2*np.sum(z[:i]**2*lambda_xi(xi)))
    m = S*(np.sum((calculate_t_label(z[:i])-0.5)*z[:i])*lambda_xi(xi)+1/S_0*m_0)
    #print (m)   
    # M step: update xi
    # zi_new = \phi^2(S+m^2) according to slide 53 of chapter 13-variational
    xi_new = np.sqrt(np.sum(z[:i])**2*(S + m**2))
    #print (xi_new)
    if abs(xi-xi_new) > threshold:
        xi=xi_new
    else:
        break

LVI_density = lower_bound(z, xi_new)
print ('mean = ', m)
print('variance = ', S)

# plot the density curve of p(z), Laplace approximation, and variational approximation
plt.plot(z, P_density)
plt.plot(z, LVI_density)
plt.plot(z, laplace_app)
plt.xlabel('z')
plt.ylabel('Density')
plt.title('Density Curve of P(z) with Laplace Approximation and LVI')
plt.show()