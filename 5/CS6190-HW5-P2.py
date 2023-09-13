import numpy as np
import matplotlib.pyplot as plt

#initialize mean and cov from the question
mean = np.array([0, 0])
cov = np.array([[3, 2.9], [2.9, 3]])

############################################################################## A
# Draw 500 samples from the distribution and split them into z1 and z2
samples = np.random.multivariate_normal(mean, cov, 500)
z1 = samples[:, 0]
z2 = samples[:, 1]

plt.figure(figsize=(8, 8))
plt.scatter(z1, z2, color = 'green')
plt.xlabel('z1')
plt.ylabel('z2')
plt.show()
############################################################################## B
# Gibbs Sampling
# initializations
samples = np.array([(-4, -4)])
z1_sample = samples[0][0]
z2_sample = samples[0][1]
iteration = 100

# sampling: for this part, I used the bivariate case properties of the conditional distributions in a multivariate normal distribution from https://online.stat.psu.edu/stat505/lesson/6/6.1
for _ in range(iteration):
    z1_n_mu = (cov[0, 1] / cov[1, 1]) * (z2_sample - mean[1]) + mean[0]
    z1_n_cov = cov[0, 0] - (cov[0, 1]**2) / cov[1, 1]
    z1_sample = np.random.normal(z1_n_mu, z1_n_cov)
    z2_n_mu = (cov[1, 0] / cov[0, 0]) * (z1_sample - mean[0]) + mean[1]
    z2_n_cov = cov[1, 1] - (cov[1, 0]**2) / cov[0, 0]
    z2_sample =np.random.normal(z2_n_mu , z2_n_cov)
    samples = np.vstack([samples, (z1_sample, z2_sample)])

# Plot the trajectory of the samples
plt.figure(figsize=(10, 10))
plt.scatter(z1, z2, color = 'green', label = 'samples')
plt.plot(samples[:, 0], samples[:, 1], color = 'red', marker='x', label = 'Gibbs Trajectory')
plt.title('Gibbs Trajectory Samples')
plt.xlabel('z1')
plt.ylabel('z2')
plt.legend()
############################################################################## C
# HMC, using the functions from problem one with minor changes. 
# initializing
samples = np.array([(-4, -4)])
z = samples[0].astype(float)
accept_probability = 0
eps = 0.1
L = 20
iterations = 100

def accept_rule(x, x_new):
    if x_new>x:
        return True
    else:
        accept=np.random.uniform(0,1)
        return (accept < x_new/(x+1e-5))

def derivative_p_z(z, mean, cov):
    return (np.linalg.inv(cov)  @ (z[np.newaxis,:] - mean[np.newaxis, :]).T)[0]

def Leapfrog(r, L, eps, z, mean, cov):
      for i in range(L):  
        r -= 0.5 * eps * -1 * derivative_p_z(z, mean, cov)
        z += r * eps
        r -= 0.5 * eps * -1 * derivative_p_z(z, mean, cov)
      return -1*r,  z

def hmc(iterations, z, mean, cov, eps, L, samples, accept_probability):
  for i in range(iterations):
    r = np.random.normal(0, 1, 2)
    r_prim,  z_prim= Leapfrog(r, L, eps, np.copy(z), mean, cov)
    p_z = 0.5 * (z[np.newaxis,:] - mean[np.newaxis, :] ) @ np.linalg.inv(cov)  @ (z[np.newaxis,:] - mean[np.newaxis, :] ).T
    H_current =  0.5 * np.sum(np.multiply(r, r)) + p_z[0][0]        
    new_p_z = 0.5 * (z_prim[np.newaxis,:] - mean[np.newaxis, :] ) @ np.linalg.inv(cov)  @ (z_prim[np.newaxis,:] - mean[np.newaxis, :] ).T
    H_proposed = 0.5 * np.sum(np.multiply(r_prim, r_prim)) + new_p_z[0][0]
    if accept_rule(np.exp(-1*H_current), np.exp(-1*H_proposed)):      
        z = z_prim
        accept_probability += 1
    samples = np.vstack([samples, z])
  return samples

samples = hmc(iterations, z, mean, cov, eps, L, samples, accept_probability)

plt.figure(figsize=(10, 10))
plt.scatter(z1, z2, color = 'green', label = 'samples')
plt.plot(samples[:, 0], samples[:, 1], color = 'red', marker='x', label = 'HMC Trajectory')
plt.title('HMC Trajectory Samples')
plt.xlabel('z1')
plt.ylabel('z2')
plt.show()