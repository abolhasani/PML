import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm
 
###################################################################### Functions
# sigmoid function
def sigmoid(x):
    x = x*10+3
    return 1/(1 + np.exp(-x))

# copied from last homework, used for ground truth probability density
def gauss_hermite_quad(f, degree):
    points, weights = np.polynomial.hermite.hermgauss(degree)
    f_x = f(points)
    F = np.sum(f_x * weights)
    return F

# The metropolis hastings algorithm function for part A, using slide 26 chapter 15-Sampling for refrence
def metropolis_hastings(tau, burn_in, After_burn_sample, pick):
    # initialization with zero
    z = np.zeros(burn_in + After_burn_sample)
    acceptance_rate = 0
    for i in range(1, burn_in + After_burn_sample):
        z_prim = np.random.normal(z[i-1], tau) # Gaussian proposal
        accept_probability = min(1, p_z(z_prim) / p_z(z[i-1]))
        # accept rule of slide 22
        if np.random.uniform(0, 1) < accept_probability:
            z[i] = z_prim
            acceptance_rate += 1
        else:
            z[i] = z[i-1]
    return z[burn_in::pick], acceptance_rate / (burn_in + After_burn_sample)

# p(z,D), and U(Z)
def p_z(z):
    return np.exp(-z**2) * (1 / (1 + np.exp(-10*z - 3)))

# is the partial derivative of U and p(z,D) (potential energy), as denoted in slide 59 chapter 15-Sampling
# the p(z) function hiven by the question is broken to two parts (sigmoid and exp) and the derivatives
# are computed based on the chain rule. the result is used by hmc and leapfrog
def derivative_p_z(z):
    exp_part = np.exp(-z**2)
    derivative_exp = -2 * z * exp_part
    sigmoid_part = 1 / (1 + np.exp(-10*z - 3))
    derivative_sigmoid = 10 * np.exp(-10*z - 3) / (1 + np.exp(-10*z - 3))**2
    return exp_part * derivative_sigmoid + derivative_exp * sigmoid_part

# Leapfrog algorithm from the slide 59 chapter 15-Sampling and the refrenced page
def leapfrog(z, r, eps, L):
    for i in range(L):
        r -= 0.5 * eps * -1*derivative_p_z(z)
        z += eps * r # the half-step
        r -= 0.5 * eps * -1*derivative_p_z(z)
    return z, r

# Implement the HMC algorithm
def hmc(eps, L, burn_in, After_burn_sample, pick):
    # initialization with zeros
    z = np.zeros(burn_in + After_burn_sample)
    acceptance_rate = 0
    for i in range(1, burn_in + After_burn_sample):
        r = np.random.normal(0, 1)  
        z_prim, r_prim = leapfrog(z[i-1], r, eps, L) 
        # Calculate the acceptance probability. H = U + K, based on slide 51 chapter 15-Sampling
        H_current = -np.log(p_z(z[i-1])) + 0.5 * r**2    
        H_proposed = -np.log(p_z(z_prim)) + 0.5 * r_prim**2
        # getting p(z,r)
        accept_probability = np.exp(H_current - H_proposed)
        if np.random.uniform(0, 1) < accept_probability:
            z[i] = z_prim
            acceptance_rate += 1
        else:
            z[i] = z[i-1]
    return z[burn_in::pick], acceptance_rate / (burn_in + After_burn_sample)

# function to do the plottings - based on the instructions in the question
def plot_den (var, p, acceptance_rates, final_samples, ground_truth_density_values, z_values):
    # Plot the acceptance rates
    plt.figure(figsize=(10, 5))
    plt.plot(var, acceptance_rates, marker='o')
    plt.xlabel('Tau' if p=='mh' else 'Epsilon')
    plt.ylabel('Acceptance rate')
    plt.show()
    # Plot the histogram of the samples and the true density
    for i, v in enumerate(var):
        plt.figure(figsize=(10, 5))
        plt.hist(final_samples[i], bins=50, density=True, alpha=0.5, label='Samples')
        plt.plot(z_values, ground_truth_density_values, label='True density')
        plt.title(f'Tau = {v}' if p=='mh' else f'Epsilon={v}')
        plt.legend()
        plt.show()

################################################################ Initializations
taus = [0.01, 0.1, 0.2, 0.5, 1]
epsilons = [0.005, 0.01, 0.1, 0.2, 0.5]
acceptance_rates = []
final_samples = []
burn_in=100000
After_burn_sample=50000
pick=10
degree=100
L = 10
# Compute the normalization constant
N = gauss_hermite_quad(sigmoid, degree)
#print(N)
# Generate a range of z values for plotting the true density
z_values = np.linspace(-5, 5, 1000)
# Calculate the true density for these z values
ground_truth_density_values = np.exp(-z_values**2) * sigmoid(z_values)/N
############################################################################## A
for tau in taus:
    samples, acceptance_rate = metropolis_hastings(tau, burn_in, After_burn_sample, pick)
    final_samples.append(samples)
    acceptance_rates.append(acceptance_rate)
plot_den (taus, 'mh', acceptance_rates, final_samples, ground_truth_density_values, z_values)
############################################################################## B
acceptance_rates = []
final_samples = []
for eps in epsilons:
    samples, acceptance_rate = hmc(eps, L, burn_in, After_burn_sample, pick)
    final_samples.append(samples)
    acceptance_rates.append(acceptance_rate)
plot_den (epsilons, 'hmc', acceptance_rates, final_samples, ground_truth_density_values, z_values)