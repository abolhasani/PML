import numpy as np
from google.colab import files
from scipy.stats import multivariate_normal
import matplotlib.pyplot as plt

# Initialize the parameters
iterations_set = [0,1,2,5,100]
num_clusters = 2
num_iterations = 101 # to easily plot the 100th iteration

# upload the dataset to google colab because i had problems with using drive as a source folder
uploaded = files.upload()
data = np.loadtxt(next(iter(uploaded)), dtype=float)

# normalize the data as asked
data = (data - np.mean(data, axis=0)) / (np.max(data, axis=0) - np.min(data, axis=0))
num_data = data.shape[0]
num_features = data.shape[1]

def plot_updates(posteriors, num_clusters, means, data, i):
      # Assign data points to clusters based on the maximum posterior probability
      cluster_colors = np.argmax(posteriors, axis=1)
      colors = ['blue', 'red']
      # scatter the data points with their respective cluster colors
      for j in range(num_clusters):
          plt.scatter(data[cluster_colors == j, 0], data[cluster_colors == j, 1], c=colors[j], label=f'Cluster {j + 1}')
      # scatter the cluster centers
      plt.scatter(means[:, 0], means[:, 1], c='black', marker='x', s=100, label='Cluster Centers')
      plt.xlabel('Feature 1')
      plt.ylabel('Feature 2')
      plt.legend()
      plt.title(f'EM updates for GMM, iterations {i}')
      plt.show()


def EM_algo_GMM (data, iters, num_iterations, num_clusters, cluster_centers, cluster_covariances, num_data, num_features):
  # Initialize the prior probabilities and posteriors and also backup variables
  gamma_probs = np.ones(num_clusters) / num_clusters 
  posteriors = np.zeros((num_data, num_clusters)) 
  # i need to keep some backup variables and distinguish between them for the first and other iterations
  N_k = np.zeros(num_clusters)
  mu_k = np.zeros((num_clusters, data.shape[1])) 
  sigma_k = np.zeros((num_clusters, data.shape[1], data.shape[1])) 
  ccen = cluster_centers
  ccov = cluster_covariances
  # Run the EM algorithm for GMM
  for i in range(num_iterations):
    # in iteration zero i start with the initialization parameters and then swith to the updated ones
    # i did this because previously when i tried to update cluster centers and posteriors, I always had a problem of not correctly updating posterios and so cluster centers ...
    cluster_mean = cluster_centers if i == 0 else mu_k
    cluster_covariance = cluster_covariances if i == 0 else sigma_k
    old_prob = gamma_probs if i == 0 else N_k
    # E-step: using the formulas given in slide 20 of chapter 13-variational
    for k in range(num_clusters):
        posteriors[:, k] = old_prob[k] * multivariate_normal.pdf(data, mean=cluster_mean[k], cov=cluster_covariance[k])
    posteriors /= np.sum(posteriors, axis=1)[:, np.newaxis]

    # M-step: using the formulas given in slide 20 of chapter 13-variational
    N_k = np.sum(posteriors, axis=0)
    # Update cluster_centers pi_k new
    for k in range(num_clusters):
      # new pi_k
      posterior_k = posteriors[:, k]
      # creat an array for posterior to multiply elementwise with data
      posterior_k_repeated = np.repeat(posterior_k, num_features).reshape(num_data, num_features)
      # new updated mu_k
      mu_k[k] = np.sum(np.multiply(posterior_k_repeated, data), axis=0) / N_k[k]
      # (x-mu)
      difference = data - mu_k[k]
      # using matmul to get (x-mu)(x-mu)^T 
      #di2 = difference @ difference.T
      #print (di2.shape)
      # create an array for posterior to be able to dimensionally multiply it with  (x-mu)(x-mu)^T element wise
      posterior_k_repeated_2 = np.repeat(posterior_k, num_features * num_features).reshape(num_data, num_features, num_features)
      # computing the updated new sigma_k, here we have (x-mu)(x-mu)^T and i computed that with np.einsumand used the result to element-wise multiply to posterior
      sigma_k[k] = np.sum(np.multiply(posterior_k_repeated_2, np.einsum('ij,ik->ijk', difference, difference)), axis=0) /  N_k[k]

    # update prior_probs
    gamma_probs = N_k / num_data
    # Plot the results for selected iterations
    if i in iters:
      plot_updates(posteriors, num_clusters, cluster_mean, data, i)

############################################################# A
# Initialize the cluster centers and covariances
cluster_centers = np.array([[-1, 1], [1, -1]]) 
cluster_covariances = np.array([0.1 * np.eye(num_features)] * num_clusters) 
EM_algo_GMM (data, iterations_set, num_iterations, num_clusters, cluster_centers, cluster_covariances, num_data, num_features)
############################################################# B
cluster_centers = np.array([[-1, -1], [1, 1]])
cluster_covariances = np.array([0.5 * np.eye(num_features)] * num_clusters)
EM_algo_GMM (data, iterations_set, num_iterations, num_clusters, cluster_centers, cluster_covariances, num_data, num_features)