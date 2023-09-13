from scipy.optimize import minimize
from scipy.stats import norm, t
import numpy as np
import matplotlib.pyplot as plt
import math
import scipy.special


# for the asked distribution by the question, mean is 0 and standard deviaation is 2, number of observations are asked to be 30
# code to gen new random values each run
np.random.seed()
random_samples = np.random.normal(loc=0, scale=2, size=30)
zeros = [0]*30 #adding zeros to facilitate plotting scatter for our drawn random_samples

print(random_samples)
plt.scatter(random_samples,zeros, alpha=0.25, vmin=-10, vmax=10)
plt.show()

# The following Log functions get params and args from mle and return log likelihood of each distribution
def Gaussian_Log(params, random_samples):
    mu, sigma = params
    n=len(random_samples)
    llg= n*np.log(2*np.pi*(sigma**2))/2 + np.sum(((random_samples-mu)**2)/(2 * (sigma**2)))
    return llg

def Student_Log(params, random_samples):
    mu, sigma, v = params
    n = len(random_samples)
    # At first, I tried to implement the mle of student-t myself with the commands below
    summ= 0
    i=0
    for i in range (n):
      summ = summ + np.log(1+((random_samples[i]-mu)/(sigma*math.sqrt(v)))**2)
    llt = (n*(np.log(scipy.special.gamma((v + 1) / 2))-np.log(scipy.special.gamma((v/ 2)))-0.5*np.log(np.pi*v)-np.log(sigma)) - ((v+1)/2)*summ)
    # I had to  use scipy t after I couldn't get my formula to work due to shortage of time
    llt= -np.sum(t.logpdf(random_samples, df=v, loc=mu, scale=sigma))
    return llt

# Param x is setting the interval for showing the density
x = np.linspace(-10, 10, 10000)
# calculating MLE for Gaussian and Student-t distributions, because I had to use L-BFGS, I used the scipy.optimize.minimize command to pass it as the method
# According to scipy.optimize.minimize docs, I had to use L-BFGS-B as the method similar to L-BFGS but with bounds for input var
# Parameter bounds was necessary because of L-BFGS-B,
# After getting the MLE for both distributions, will calculate the pdf densities for both of them
gaussian_mle = minimize(Gaussian_Log, [0, 1], args=random_samples, method='L-BFGS-B', bounds=((-10, 10), (0.1, 10)))
# the output of minimize returns x, that has the solution array for all the attributes inside the mle for each distribution
gaussian_pdf = norm.pdf(x, loc=gaussian_mle.x[0], scale=gaussian_mle.x[1])
student_mle = minimize(Student_Log, [0, 1, 1], args=random_samples, method='L-BFGS-B', bounds=((-15, 15), (0.1, 10), (0.1, 100)))
student_pdf = t.pdf(x, df=student_mle.x[2], loc=student_mle.x[0], scale=student_mle.x[1])

# I used print to gain an insight what is happening in the minimization process and what has been fed to the pdf function
#print(gaussian_pdf)
print(gaussian_mle)
#print(student_pdf)
print(student_mle)

plt.scatter(random_samples, zeros, label='Random samples')
plt.plot(x, student_pdf, label='Student-t')
plt.plot(x, gaussian_pdf, label='Gaussian')
plt.legend()
plt.show()

# adding the noises to the previously drawn random samples array horizontally
random_samples = np.hstack((random_samples, [8, 9, 10]))
# new zeros for adjusting thwe size after noises are added
zeros = [0]*33
print(random_samples)
plt.scatter(random_samples,zeros, alpha=0.25, vmin=-10, vmax=10)
plt.show()

# Param x is updated after big noises are added to accomodate
x = np.linspace(-15, 15, 1000)

# Finding MLE and then probability density for the transformed gaussian and student-t distributions after noises were added
# repeating the previous part for both distributions after the noises were added
gaussian_mle = minimize(Gaussian_Log, [0, 1], args=random_samples, method='L-BFGS-B', bounds=((-15, 15), (0.1, 10)))
gaussian_pdf = norm.pdf(x, loc=gaussian_mle.x[0], scale=gaussian_mle.x[1])
student_mle= minimize(Student_Log, [0, 1, 1], args=random_samples, method='L-BFGS-B', bounds=((-15, 15), (0.1, 10), (0.1, 100)))
student_pdf= t.pdf(x, df=student_mle.x[2], loc=student_mle.x[0], scale=student_mle.x[1])

#print(gaussian_pdf)
print(gaussian_mle)
#print(student_pdf)
print(student_mle)

plt.scatter(random_samples, zeros, label='Random samples noised')
plt.plot(x, student_pdf, label='Student-t noised')
plt.plot(x, gaussian_pdf, label='Gaussian noised')
plt.legend()
plt.show()