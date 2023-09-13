# Problem 1
from scipy.stats import t
import numpy as np
import matplotlib.pyplot as plt


# mu = 0, lambda = 1
# I set the linspace args by trial and eror to see what combination looks better
x = np.linspace(-6, 6, 100)
plt.plot(x, t.pdf(x, 0.1, 0, 1), label='v = 0.1')
plt.plot(x, t.pdf(x, 1, 0, 1), label='v = 1')
plt.plot(x, t.pdf(x, 10, 0, 1), label='v = 10')
plt.plot(x, t.pdf(x, 100, 0, 1), label='v = 100')
plt.plot(x, t.pdf(x, 10**6, 0, 1), label='v = 1000000')
plt.plot(x, np.exp(-(x**2)/2)/np.sqrt(2*np.pi), label='Gaussian Dist(0,1)')
plt.legend()
plt.show()