# Problem 2
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import beta

# I set the linspace args by trial and eror to see what combination looks better
x = np.linspace(0, 1, 1000)
plt.plot(x, beta.pdf(x, 1,1), label='Beta(1,1)')
plt.plot(x, beta.pdf(x, 5, 5), label='Beta(5,5)')
plt.plot(x, beta.pdf(x, 10, 10), label='Beta(10,10)')
plt.legend()
plt.show()
plt.plot(x, beta.pdf(x, 1, 2), label='Beta(1,2)')
plt.plot(x, beta.pdf(x, 5, 6), label='Beta(5,6)')
plt.plot(x, beta.pdf(x, 10, 11), label='Beta(10,11)')
plt.legend()
plt.show()