# U - CS6190 - Spring 23 - HW2 - P1 - u1416052 
from six import unichr
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import multivariate_normal
import warnings

# Initialization of parameters and regression:
# Supress warnings
warnings.filterwarnings('ignore')
np.random.seed()
# Set the ground-truth parameters
w0_gt = np.array([-0.3, 0.5])
# Generate 20 random samples from the uniform distribution in [-1, 1]
x = np.random.uniform(-1, 1, size=20)
# Calculate the Gaussian noise as mentioned in the question
G_noise = np.random.normal(scale=0.2, size=20)
# Calculate the y values of the regression with noise added
y = w0_gt[0] + w0_gt[1]*x + G_noise
# Define the variables for prior and likelihood distributions given by the question
alpha = 2
beta = 25
prior_mean = np.zeros(2)
prior_covar = alpha * np.eye(2)
likelihood_covar = (1/beta) * np.eye(20)
# Sample 20 instances of w from the prior distribution
w_samples = np.random.multivariate_normal(prior_mean, prior_covar, size=20)
# Initializing two arrays to do parts B, C, D, and E
t = [1, 2, 5, 20]
u = ['B', 'C', 'D', 'E']

# Calculating the prior
def prior (prior_mean, prior_covar, alpha, beta):
  # Define the range values to evaluate the prior over, using -1 for reshape to make sure the dimensions match. This part is to make data fit for heat map
  sample_x_range = np.linspace(-1, 1, 100)
  sample_y_range = np.linspace(-1, 1, 100)
  XX, YY = np.meshgrid(sample_x_range, sample_y_range)
  w_values = np.hstack((XX.reshape(-1, 1), YY.reshape(-1, 1)))
  # Calculate the prior probability for normal distro as mentioned in the question. 
  prior_prob = (1 / np.sqrt((2 * np.pi) * np.linalg.det(prior_covar))) * np.exp(-0.5 * np.sum(np.dot((w_values - prior_mean), np.linalg.inv(prior_covar)) * (w_values - prior_mean), axis=1))
  # Reshape the prior probabilities to match the shape of the heat-map
  prior_prob = prior_prob.reshape((100, 100))
  return prior_prob

# Calculating the posterior
def posterior (X, Y, alpha, beta, prior_mean, prior_covar, problem):
  likelihood_mean = np.array([np.ones_like(X), X])
  likelihood_covar = (1/beta) * np.eye(len(X))
  #using the formula (m_n and S_n^-1) in slide 28 of chapter 8 (generalized linear)
  x = np.vstack((np.ones(X.shape), X))
  post_covar = np.linalg.inv(alpha*np.eye(2) + beta*np.matmul(x,np.transpose(x)))
  post_mean = np.matmul(post_covar, beta*np.matmul(x,Y.reshape(-1,1)))
  post_mean = post_mean.flatten()
  # Posterior computed here based on Bayes Rule
  # Using numpy matmul to perform the dot product for matrixes to make it easier to decipher.
  #post_covar = np.linalg.inv(np.linalg.inv(prior_covar) + np.matmul(np.matmul(likelihood_mean, np.linalg.inv(likelihood_covar)), likelihood_mean.T))
  #post_mean = np.matmul(post_covar, (np.matmul(np.linalg.inv(prior_covar), prior_mean) + np.matmul(np.matmul(likelihood_mean, np.linalg.inv(likelihood_covar)), Y)))
  # Posterior covariance is 2*2
  sentence = "Part {} Posterior Mean and Variance:".format(problem)
  print(sentence)
  print('Mean = ', post_mean,'Posterior Var = ', post_covar )
  return post_mean, post_covar

# Function to create a 2D posterior probability used by heatmap. 
def grid_maker (posterior_m, posterior_v):
  # Define the spacing between values in the grid, smaller = smoother
  delta = 0.01
  # Creat 1D array [-1,1]
  sample_x_range = np.arange(-1, 1 + delta, delta)
  sample_y_range = np.arange(-1, 1 + delta, delta)
  # Make a 2D array
  w0_grid, w1_grid = np.meshgrid(sample_x_range, sample_y_range)
  # Create the storage for the log probability values
  lprob_val = np.zeros_like(w0_grid)
  # Calculate log probabilities
  for i in range(w0_grid.shape[0]):
      for j in range(w0_grid.shape[1]):
          lprob_val[i, j] = multivariate_normal.logpdf([w0_grid[i, j], w1_grid[i, j]], mean=posterior_m, cov=posterior_v)
  # Convert log to prob and return it
  prob_grid_f = np.exp(lprob_val - np.max(lprob_val))
  prob_grid_f /= np.sum(prob_grid_f) * delta**2
  return prob_grid_f

# Function to print the heat map [-1,1] and [-1,1] for every part from passed probabilities, using red and blue for gradient colors
def print_heatmap(function, string, problem, ground_truth):
  sentence = "Part {} heat Map:".format(problem)
  print(sentence)
  fig = plt.figure(figsize=(10, 10))
  plt.imshow(function, origin='lower', cmap='RdBu', extent=(-1, 1, -1, 1))
  plt.title(string)
  plt.xlabel('w0')
  plt.ylabel('w1')
  if problem!='A':
    plt.plot(ground_truth[0], ground_truth[1], 'ro', label='Ground Truth')
  plt.legend()
  plt.show()

# Function to print the lines and scatter plot
def print_lines (n, x, y, posterior, samples, ground_truth, problem):
  sentence = "Part {} Scatter and lines:".format(problem)
  print(sentence)
  fig = plt.figure(figsize=(10, 10))
  plt.scatter(x[:n], y[:n])
  # a different plot for part A because it did not require posteriors
  if problem == 'A':
    for w in samples:
      plt.plot(x, w[0] + w[1] * x, 'r-', alpha=0.25)
  else:
    for i in range(samples.shape[0]):
      plt.plot([-1, 1], [samples[i, 0] - samples[i, 1], samples[i, 0] + samples[i, 1]], 'r-', alpha=0.25)
  plt.xlim(-1, 1)
  plt.ylim(-1, 1)
  plt.title('Plot of regression lines and selected values from sample W')
  plt.xlabel('X')
  plt.ylabel('Y')
  plt.legend()
  plt.show()

###################################### A
print_heatmap(prior(prior_mean, prior_covar, alpha, beta), 'Prior Distribution', 'A', w0_gt)
print_lines(20, x, y, 0, w_samples, w0_gt, 'A')

###################################### B, C, D, E
# The overall idea is to get the posterior, then get the samples and create a grid that will hold probability values
# These values are used for heat map. Then, regression lines are printed for the selected w samples
for ti, ui in zip(t, u):
  posterior_m, posterior_v = posterior (x[:ti], y[:ti], alpha, beta, prior_mean, prior_covar, ui)
  w_samples = np.random.multivariate_normal(posterior_m, posterior_v, size=20)
  grid = grid_maker(posterior_m, posterior_v)
  print_heatmap(grid,'Posterior distribution heat map of selected w samples', ui, w0_gt)
  print_lines (ti, x, y, posterior_m, w_samples, w0_gt, ui)