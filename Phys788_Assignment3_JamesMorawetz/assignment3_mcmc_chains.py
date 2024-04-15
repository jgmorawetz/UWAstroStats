import time
import matplotlib.pyplot as plt
import random
import numpy as np
from numpy.linalg import inv
from cosmopower_NN import cosmopower_NN
import tensorflow as tf
import gc
import emcee
from getdist import plots, MCSamples, parampriors
import getdist
from sklearn.decomposition import PCA


# Reads in the relevant data for each of the tasks
cov=np.load('data_4_assignment3/covariance.npy')
cov_inv = inv(cov)
minimum=np.load('data_4_assignment3/minimum.npy')
maximum=np.load('data_4_assignment3/maximum.npy')
reference_model = np.load('data_4_assignment3/reference_model.npy')
reference_model_noisy = np.load('data_4_assignment3/reference_model_noise.npy')
LEN = len(reference_model_noisy)
cp_nn_model = cosmopower_NN(restore=True, 
                          restore_filename='data_4_assignment3/emulator_final')
cov_num_1500 = np.load('data_4_assignment3/cov_num_1500.npy')
cov_num_3000 = np.load('data_4_assignment3/cov_num_3000.npy')
cov_num_10000 = np.load('data_4_assignment3/cov_num_10000.npy')
cov_num_1500_inv = inv(cov_num_1500)
cov_num_3000_inv = inv(cov_num_3000)
cov_num_10000_inv = inv(cov_num_10000)


 ################################### TASK 1 ###################################

# Runs an MCMC where all four parameters are being varied

total_steps = 2000 # total number of steps each walker is doing
burning_steps = 100 # how many burning steps to remove
nwalkers = 100 # number of walkers to probe parameter space
parameters = np.load('data_4_assignment2/parameters.npz')
param_names_varying = ['omega_m', 'omega_b', 'As', 'w'] # parameters to vary
ndim = len(param_names_varying)

def starting_positions(seed):
  '''Generates initial starting parameters for each of the walkers (I have 
     adjusted the starting range to encompass the likelihood more closely to 
     avoid local minima forming away from it).'''
  random.seed(seed)
  p0 = []
  for i in range(nwalkers):
    random_starts = []
    for name in param_names_varying:
      if name == 'omega_m':
        lower_edge, upper_edge = 0.32, 0.38
      elif name == 'omega_b':
        lower_edge, upper_edge = 0.03, 0.06
      elif name == 'As':
        lower_edge, upper_edge = 1.8e-9, 2.3e-9
      elif name == 'w':
        lower_edge, upper_edge = -1.3, -0.9
      # ensures all initial starting positions fall within reasonable range
      random_starts.append(random.uniform(lower_edge, upper_edge))
    p0.append(random_starts)
  return np.array(p0)

def calculate_logprior(para_dict):
  '''Computes the log of the prior based on the parameter values.'''
  lnprior = 0 # sets to zero if within range
  for name in param_names_varying:
    if ((para_dict[name] > np.max(parameters[name])) or
        (para_dict[name] < np.min(parameters[name]))):
        lnprior = -np.inf # sets to negative infinity if outside
  return lnprior

def calculate_likelihood(para):
  '''Computes the likelihood given the parameters.'''
  params = {}
  for i in range(len(param_names_varying)):
    params[param_names_varying[i]] = [para[i]]
  # obtains the predicted data vector given the parameters
  predicted_vector = cp_nn_model.predictions_np(params)[0]
  # transforms the vector appropriately
  predicted_vector = predicted_vector * maximum + minimum
  # computes the residual between observed and predicted
  delta = predicted_vector - reference_model_noisy
  likelihood = -0.5*np.matmul(delta, np.matmul(cov_inv, delta))
  lnprior = calculate_logprior(params)
  return likelihood + lnprior

p0 = starting_positions(0)
sampler = emcee.EnsembleSampler(nwalkers, ndim, calculate_likelihood)
sampler.run_mcmc(p0, total_steps, progress=True)
samples_emcee = sampler.get_chain(discard=burning_steps, flat=True)
np.save('outputs_MCMC/MCMC_task1', samples_emcee)
log_prob_samples = sampler.get_log_prob(discard=burning_steps, flat=True)
np.save('outputs_MCMC/logp_task1', log_prob_samples)


################################### TASK 2 #####################################

# We now repeat the same process as task 1, but instead use the numerical 
# covariance matrices (with and without the Hartlap correction)

total_steps = 2000 # total number of steps each walker is doing
burning_steps = 100 # how many burning steps to remove
nwalkers = 100 # number of walkers to probe parameter space
parameters = np.load('data_4_assignment2/parameters.npz')
param_names_varying = ['omega_m', 'omega_b', 'As', 'w'] # parameters to vary
ndim = len(param_names_varying)

def starting_positions(seed):
  '''Generates initial starting parameters for each of the walkers (I have 
     adjusted the starting range to encompass the likelihood more closely to 
     avoid local minima forming away from it).'''
  random.seed(seed)
  p0 = []
  for i in range(nwalkers):
    random_starts = []
    for name in param_names_varying:
      if name == 'omega_m':
        lower_edge, upper_edge = 0.32, 0.38
      elif name == 'omega_b':
        lower_edge, upper_edge = 0.03, 0.06
      elif name == 'As':
        lower_edge, upper_edge = 1.8e-9, 2.3e-9
      elif name == 'w':
        lower_edge, upper_edge = -1.3, -0.9
      # ensures all initial starting positions fall within reasonable range
      random_starts.append(random.uniform(lower_edge, upper_edge))
    p0.append(random_starts)
  return np.array(p0)

def calculate_logprior(para_dict):
  '''Computes the log of the prior based on the parameter values.'''
  lnprior = 0 # sets to zero if within range
  for name in param_names_varying:
    if ((para_dict[name] > np.max(parameters[name])) or
        (para_dict[name] < np.min(parameters[name]))):
        lnprior = -np.inf # sets to negative infinity if outside
  return lnprior

def calculate_likelihood(para, ncov, hartlap):
  '''Computes the likelihood given the parameters (but allows for selecting
     covariance matrix from finite samples and adding hartlap correction).'''
  params = {}
  for i in range(len(param_names_varying)):
    params[param_names_varying[i]] = [para[i]]
  # obtains the predicted data vector given the parameters
  predicted_vector = cp_nn_model.predictions_np(params)[0]
  # transforms the vector appropriately
  predicted_vector = predicted_vector * maximum + minimum
  # computes the residual between observed and predicted
  delta = predicted_vector - reference_model_noisy
  # selects which inverse covariance matrix to use depending on criteria
  if ncov == 1500:
    if hartlap:
      inverse_covariance = cov_num_1500_inv*(
          (ncov-LEN-2)/(ncov-1))
    else:
      inverse_covariance = cov_num_1500_inv
  elif ncov == 3000:
    if hartlap:
      inverse_covariance = cov_num_3000_inv*(
          (ncov-LEN-2)/(ncov-1))
    else:
      inverse_covariance = cov_num_3000_inv
  elif ncov == 10000:
    if hartlap:
      inverse_covariance = cov_num_10000_inv*(
          (ncov-LEN-2)/(ncov-1))
    else:
      inverse_covariance = cov_num_10000_inv
  likelihood = -0.5*np.matmul(delta, np.matmul(inverse_covariance, delta))
  lnprior = calculate_logprior(params)
  return likelihood + lnprior

# Iterates through the different combinations of number of covariance elements
# and whether to apply Hartlap correction and saves results for each
for ncov in [1500, 3000, 10000]:
  for hartlap in [True, False]:
    p0 = starting_positions(0)
    sampler = emcee.EnsembleSampler(nwalkers, ndim, calculate_likelihood,
                                    args=(ncov, hartlap))
    sampler.run_mcmc(p0, total_steps, progress=True)
    samples_emcee = sampler.get_chain(discard=burning_steps, flat=True)
    np.save('outputs_MCMC/MCMC_task2_{0}_{1}'.format(ncov, hartlap), 
            samples_emcee)
    log_prob_samples = sampler.get_log_prob(discard=burning_steps, flat=True)
    np.save('outputs_MCMC/logp_task2_{0}_{1}'.format(ncov, hartlap), 
            log_prob_samples)
    

################################### TASK 3 ####################################

# Runs MCMC chain for different number of PCA elements using noisy vector

# Generates PCA transformations for the data
N_pca = LEN
models = np.load('data_4_assignment2/models.npy')
mean = np.mean(models, axis=0)
pca = PCA(n_components=N_pca, svd_solver='full')
models_pca = pca.fit_transform(models-mean)
rotation_matrix = pca.components_.T
cov_pca = np.matmul(np.matmul(rotation_matrix.T, cov), rotation_matrix)
cov_pca_1500 = np.matmul(np.matmul(rotation_matrix.T, cov_num_1500), 
                         rotation_matrix)
reference_model_pca = pca.transform(np.array([reference_model-mean]))[0]
reference_model_noisy_pca = pca.transform(
                                  np.array([reference_model_noisy-mean]))[0]

# Creates a dictionary to store the task 3 PCA inverse covariance matrices
# (to avoid having to recompute them in the likelihood every step)
inv_cov_dict_task3 = {'analytical':{}, '1500':{}}
for n_pca in [100, 300, 500, 700, 900]:
    inv_cov_dict_task3['analytical'][n_pca] = inv(cov_pca[:n_pca, :n_pca])
    inv_cov_dict_task3['1500'][n_pca] = inv(cov_pca_1500[:n_pca, :n_pca]
                    )*(1500-n_pca-2)/(1500-1) # applies hartlap for 1500 case

total_steps = 2000 # total number of steps each walker is doing
burning_steps = 100 # how many burning steps to remove
nwalkers = 100 # number of walkers to probe parameter space
parameters = np.load('data_4_assignment2/parameters.npz')
param_names_varying = ['omega_m', 'omega_b', 'As', 'w'] # parameters to vary
ndim = len(param_names_varying)

def starting_positions(seed):
  '''Generates initial starting parameters for each of the walkers (I have 
     adjusted the starting range to encompass the likelihood more closely to 
     avoid local minima forming away from it).'''
  random.seed(seed)
  p0 = []
  for i in range(nwalkers):
    random_starts = []
    for name in param_names_varying:
      if name == 'omega_m':
        lower_edge, upper_edge = 0.32, 0.38
      elif name == 'omega_b':
        lower_edge, upper_edge = 0.03, 0.06
      elif name == 'As':
        lower_edge, upper_edge = 1.8e-9, 2.3e-9
      elif name == 'w':
        lower_edge, upper_edge = -1.3, -0.9
      # ensures all initial starting positions fall within reasonable range
      random_starts.append(random.uniform(lower_edge, upper_edge))
    p0.append(random_starts)
  return np.array(p0)

def calculate_logprior(para_dict):
  '''Computes the log of the prior based on the parameter values.'''
  lnprior = 0 # sets to zero if within range
  for name in param_names_varying:
    if ((para_dict[name] > np.max(parameters[name])) or
        (para_dict[name] < np.min(parameters[name]))):
        lnprior = -np.inf # sets to negative infinity if outside
  return lnprior

def calculate_likelihood(para, cov_type, n_pca):
    '''Computes the likelihood given the parameters (but allows for selecting
       covariance from finite samples).'''
    params = {}
    for i in range(len(param_names_varying)):
      params[param_names_varying[i]] = [para[i]]
    # obtains the predicted data vector given the parameters
    predicted_vector = cp_nn_model.predictions_np(params)[0]
    # transforms the vector appropriately
    predicted_vector = predicted_vector * maximum + minimum
    # projects into the PCA basis and selects desired number of elements
    predicted_vector = pca.transform(
                              np.array([predicted_vector-mean]))[0][:n_pca]
    # computes residual between observed and predicted
    delta = predicted_vector - reference_model_noisy_pca[:n_pca]
    # selects the appropriate inverse covariance matrix to use from dictionary
    inverse_covariance = inv_cov_dict_task3[cov_type][n_pca]
    likelihood = -0.5*np.matmul(delta, np.matmul(inverse_covariance, delta))
    lnprior = calculate_logprior(params)
    likelihood += lnprior
    return likelihood

# Iterates through the different number of PCA elements both for the analytical
# and 1500 cases
for cov_type in ['analytical', '1500']:
  for n_pca in [100, 300, 500, 700, 900]:
    p0 = starting_positions(0)
    sampler = emcee.EnsembleSampler(nwalkers, ndim, calculate_likelihood,
                                    args=(cov_type, n_pca))
    sampler.run_mcmc(p0, total_steps, progress=True)
    samples_emcee = sampler.get_chain(discard=burning_steps, flat=True)
    np.save('outputs_MCMC/MCMC_task3_{0}_{1}'.format(cov_type, n_pca), 
            samples_emcee)
    log_prob_samples = sampler.get_log_prob(discard=burning_steps, flat=True)
    np.save('outputs_MCMC/logp_task3_{0}_{1}'.format(cov_type, n_pca), 
            log_prob_samples)


################################### TASK 4a ###################################

# Runs MCMC chain for different number of PCA elements using noiseless vector

# Generates PCA transformations for the data
N_pca = LEN
models = np.load('data_4_assignment2/models.npy')
mean = np.mean(models, axis=0)
pca = PCA(n_components=N_pca, svd_solver='full')
models_pca = pca.fit_transform(models-mean)
rotation_matrix = pca.components_.T
cov_pca = np.matmul(np.matmul(rotation_matrix.T, cov), rotation_matrix)
cov_pca_1500 = np.matmul(np.matmul(rotation_matrix.T, cov_num_1500), 
                         rotation_matrix)
reference_model_pca = pca.transform(np.array([reference_model-mean]))[0]
reference_model_noisy_pca = pca.transform(
                                  np.array([reference_model_noisy-mean]))[0]

# Creates a dictionary to store the task 3 PCA inverse covariance matrices
# (to avoid having to recompute them in the likelihood every step)
inv_cov_dict_task3 = {'analytical':{}, '1500':{}}
for n_pca in [100, 300, 500, 700, 900]:
    inv_cov_dict_task3['analytical'][n_pca] = inv(cov_pca[:n_pca, :n_pca])
    inv_cov_dict_task3['1500'][n_pca] = inv(cov_pca_1500[:n_pca, :n_pca]
                    )*(1500-n_pca-2)/(1500-1) # applies hartlap for 1500 case

total_steps = 2000 # total number of steps each walker is doing
burning_steps = 100 # how many burning steps to remove
nwalkers = 100 # number of walkers to probe parameter space
parameters = np.load('data_4_assignment2/parameters.npz')
param_names_varying = ['omega_m', 'omega_b', 'As', 'w'] # parameters to vary
ndim = len(param_names_varying)

def starting_positions(seed):
  '''Generates initial starting parameters for each of the walkers (I have 
     adjusted the starting range to encompass the likelihood more closely to 
     avoid local minima forming away from it).'''
  random.seed(seed)
  p0 = []
  for i in range(nwalkers):
    random_starts = []
    for name in param_names_varying:
      if name == 'omega_m':
        lower_edge, upper_edge = 0.32, 0.38
      elif name == 'omega_b':
        lower_edge, upper_edge = 0.03, 0.06
      elif name == 'As':
        lower_edge, upper_edge = 1.8e-9, 2.3e-9
      elif name == 'w':
        lower_edge, upper_edge = -1.3, -0.9
      # ensures all initial starting positions fall within reasonable range
      random_starts.append(random.uniform(lower_edge, upper_edge))
    p0.append(random_starts)
  return np.array(p0)

def calculate_logprior(para_dict):
  '''Computes the log of the prior based on the parameter values.'''
  lnprior = 0 # sets to zero if within range
  for name in param_names_varying:
    if ((para_dict[name] > np.max(parameters[name])) or
        (para_dict[name] < np.min(parameters[name]))):
        lnprior = -np.inf # sets to negative infinity if outside
  return lnprior

def calculate_likelihood(para, cov_type, n_pca):
    '''Computes the likelihood given the parameters (but allows for selecting
       covariance from finite samples).'''
    params = {}
    for i in range(len(param_names_varying)):
      params[param_names_varying[i]] = [para[i]]
    # obtains the predicted data vector given the parameters
    predicted_vector = cp_nn_model.predictions_np(params)[0]
    # transforms the vector appropriately
    predicted_vector = predicted_vector * maximum + minimum
    # projects into the PCA basis and selects desired number of elements
    predicted_vector = pca.transform(
                              np.array([predicted_vector-mean]))[0][:n_pca]
    # computes residual between observed and predicted
    delta = predicted_vector - reference_model_pca[:n_pca]
    # selects the appropriate inverse covariance matrix to use from dictionary
    inverse_covariance = inv_cov_dict_task3[cov_type][n_pca]
    likelihood = -0.5*np.matmul(delta, np.matmul(inverse_covariance, delta))
    lnprior = calculate_logprior(params)
    likelihood += lnprior
    return likelihood

# Iterates through the different number of PCA elements both for the analytical
# and 1500 cases
for cov_type in ['analytical', '1500']:
  for n_pca in [100, 300, 500, 700, 900]:
    p0 = starting_positions(0)
    sampler = emcee.EnsembleSampler(nwalkers, ndim, calculate_likelihood,
                                    args=(cov_type, n_pca))
    sampler.run_mcmc(p0, total_steps, progress=True)
    samples_emcee = sampler.get_chain(discard=burning_steps, flat=True)
    np.save('outputs_MCMC/MCMC_task4a_{0}_{1}'.format(cov_type, n_pca), 
            samples_emcee)
    log_prob_samples = sampler.get_log_prob(discard=burning_steps, flat=True)
    np.save('outputs_MCMC/logp_task4a_{0}_{1}'.format(cov_type, n_pca), 
            log_prob_samples)


################################## TASK 4b ####################################

# Runs multiple MCMC chains for fixed PCA elements using noiseless vector

# Generates PCA transformations for the data
N_pca = LEN
models = np.load('data_4_assignment2/models.npy')
mean = np.mean(models, axis=0)
pca = PCA(n_components=N_pca, svd_solver='full')
models_pca = pca.fit_transform(models-mean)
rotation_matrix = pca.components_.T
cov_pca = np.matmul(np.matmul(rotation_matrix.T, cov), rotation_matrix)
cov_pca_1500 = np.matmul(np.matmul(rotation_matrix.T, cov_num_1500), 
                         rotation_matrix)
reference_model_pca = pca.transform(np.array([reference_model-mean]))[0]
reference_model_noisy_pca = pca.transform(
                                  np.array([reference_model_noisy-mean]))[0]

# Creates a dictionary to store the task 3 PCA inverse covariance matrices
# (to avoid having to recompute them in the likelihood every step)
inv_cov_dict_task3 = {'analytical':{}, '1500':{}}
for n_pca in [100, 300, 500, 700, 900]:
    inv_cov_dict_task3['analytical'][n_pca] = inv(cov_pca[:n_pca, :n_pca])
    inv_cov_dict_task3['1500'][n_pca] = inv(cov_pca_1500[:n_pca, :n_pca]
                    )*(1500-n_pca-2)/(1500-1) # applies hartlap for 1500 case

total_steps = 2000 # total number of steps each walker is doing
burning_steps = 100 # how many burning steps to remove
nwalkers = 100 # number of walkers to probe parameter space
parameters = np.load('data_4_assignment2/parameters.npz')
param_names_varying = ['omega_m', 'omega_b', 'As', 'w'] # parameters to vary
ndim = len(param_names_varying)

def starting_positions(seed):
  '''Generates initial starting parameters for each of the walkers (I have 
     adjusted the starting range to encompass the likelihood more closely to 
     avoid local minima forming away from it).'''
  random.seed(seed)
  p0 = []
  for i in range(nwalkers):
    random_starts = []
    for name in param_names_varying:
      if name == 'omega_m':
        lower_edge, upper_edge = 0.32, 0.38
      elif name == 'omega_b':
        lower_edge, upper_edge = 0.03, 0.06
      elif name == 'As':
        lower_edge, upper_edge = 1.8e-9, 2.3e-9
      elif name == 'w':
        lower_edge, upper_edge = -1.3, -0.9
      # ensures all initial starting positions fall within reasonable range
      random_starts.append(random.uniform(lower_edge, upper_edge))
    p0.append(random_starts)
  return np.array(p0)

def calculate_logprior(para_dict):
  '''Computes the log of the prior based on the parameter values.'''
  lnprior = 0 # sets to zero if within range
  for name in param_names_varying:
    if ((para_dict[name] > np.max(parameters[name])) or
        (para_dict[name] < np.min(parameters[name]))):
        lnprior = -np.inf # sets to negative infinity if outside
  return lnprior

def calculate_likelihood(para, cov_type, n_pca):
    '''Computes the likelihood given the parameters (but allows for selecting
       covariance from finite samples).'''
    params = {}
    for i in range(len(param_names_varying)):
      params[param_names_varying[i]] = [para[i]]
    # obtains the predicted data vector given the parameters
    predicted_vector = cp_nn_model.predictions_np(params)[0]
    # transforms the vector appropriately
    predicted_vector = predicted_vector * maximum + minimum
    # projects into the PCA basis and selects desired number of elements
    predicted_vector = pca.transform(
                              np.array([predicted_vector-mean]))[0][:n_pca]
    # computes residual between observed and predicted
    delta = predicted_vector - reference_model_pca[:n_pca]
    # selects the appropriate inverse covariance matrix to use from dictionary
    inverse_covariance = inv_cov_dict_task3[cov_type][n_pca]
    likelihood = -0.5*np.matmul(delta, np.matmul(inverse_covariance, delta))
    lnprior = calculate_logprior(params)
    likelihood += lnprior
    return likelihood

# Iterates through different MCMC chains by using a different initial seed
# for each
cov_type = '1500'
n_cpa = 300
for integer_seed in range(6):
    p0 = starting_positions(integer_seed)
    sampler = emcee.EnsembleSampler(nwalkers, ndim, calculate_likelihood,
                                    args=(cov_type, n_pca))
    sampler.run_mcmc(p0, total_steps, progress=True)
    samples_emcee = sampler.get_chain(discard=burning_steps, flat=True)
    np.save('outputs_MCMC/MCMC_task4b_{0}_{1}_{2}'.format(cov_type, n_pca, 
                                            integer_seed), samples_emcee)
    log_prob_samples = sampler.get_log_prob(discard=burning_steps, flat=True)
    np.save('outputs_MCMC/logp_task4b_{0}_{1}_{2}'.format(cov_type, n_pca, 
                                            integer_seed), log_prob_samples)


"""import random
import numpy as np
from numpy.linalg import inv
from cosmopower_NN import cosmopower_NN
import tensorflow as tf
import gc
import emcee
from sklearn.decomposition import PCA



# Loads in the relevant data
cov = np.load('data_4_assignment3/covariance.npy')
cov_inv = inv(cov)
minimum = np.load('data_4_assignment3/minimum.npy')
maximum = np.load('data_4_assignment3/maximum.npy')
reference_model = np.load('data_4_assignment3/reference_model.npy')
reference_model_noisy = np.load('data_4_assignment3/reference_model_noisy.npy')
LEN = len(reference_model_noisy)
cp_nn_model = cosmopower_NN(restore=True, restore_filename='data_4_assignment3/emulator_final')
cov_num_1500 = np.load('data_4_assignment3/cov_num_1500.npy')
cov_num_3000 = np.load('data_4_assignment3/cov_num_3000.npy')
cov_num_10000 = np.load('data_4_assignment3/cov_num_10000.npy')
cov_num_1500_inv = inv(cov_num_1500)
cov_num_3000_inv = inv(cov_num_3000)
cov_num_10000_inv = inv(cov_num_10000)

# Sets the parameters used for running the MCMC chains
total_steps = 600
burning_steps = 100
nwalkers = 100
parameters = np.load('data_4_assignment2/parameters.npz')
param_names_varying = ['omega_m', 'omega_b', 'As', 'w']
ndim = len(param_names_varying)

# Generates PCA transformations for the data
N_pca = LEN
models = np.load('data_4_assignment2/models.npy')
mean = np.mean(models, axis=0)
pca = PCA(n_components=N_pca, svd_solver='full')
models_pca = pca.fit_transform(models-mean)
rotation_matrix = pca.components_.T
cov_pca = np.matmul(np.matmul(rotation_matrix.T, cov), rotation_matrix)
cov_pca_1500 = np.matmul(np.matmul(rotation_matrix.T, cov_num_1500), rotation_matrix)
reference_model_pca = pca.transform(reference_model-mean)
reference_model_noisy_pca = pca.transform(reference_model_noisy-mean)

# creates a dictionary to store the task 3 PCA inverse covariance matrices
# (to avoid having to recompute them in the likelihood every step)
inv_cov_dict_task3 = {'analytical':{}, '1500':{}}
for n_pca in [100, 200, 300, 400, 500, 600, 700, 800, 900]:
    inv_cov_dict_task3['analytical'][n_pca] = inv(cov_pca[:n_pca, :n_pca])
    inv_cov_dict_task3['1500'][n_pca] = inv(cov_pca_1500[:n_pca, :n_pca])



# Defines the functions relevant for running the MCMC chains
def starting_positions(seed):
    '''Generates initial starting positions for each of the walkers.'''
    random.seed(seed)
    p0 = []
    for i in range(nwalkers):
        random_starts = []
        for name in param_names_varying:
            lower_edge = np.min(parameters[name])
            upper_edge = np.max(parameters[name])
            # ensures all initial starting positions fall within the range
            if lower_edge < 0:
                lower_edge *= 0.99
            else:
                lower_edge *= 1.01
            if upper_edge > 0:
                upper_edge *= 0.99
            else:
                upper_edge *= 1.01
            # randomly picks starting positions within that range
            random_starts.append(random.uniform(lower_edge, upper_edge))
        p0.append(random_starts)
    p0 = np.array(p0)
    return p0

def calculate_logprior(para_dict):
    '''Computes the log of the prior based on the parameter values.'''
    lnprior = 0 # sets to 0 if within range
    for name in param_names_varying:
        if ((para_dict[name] > np.max(parameters[name])) or 
            (para_dict[name] < np.min(parameters[name]))):
            lnprior = -np.inf # sets to negative infinity if outside range
    return lnprior
    
def calculate_likelihood_task1(para):
    '''Computes the likelihood based on the parameter values.'''
    params = {}
    for i in range(len(param_names_varying)):
        params[param_names_varying[i]] = [para[i]]
    # obtains predicted data vector given the parameters
    predicted_vector = cp_nn_model.predictions_np(params)[0]
    # transforms the vector appropriately
    predicted_vector = predicted_vector * maximum + minimum
    # computes the residual between observed and predicted
    delta = predicted_vector - reference_model_noisy
    likelihood = -0.5*np.matmul(delta, np.matmul(cov_inv, delta))
    lnprior = calculate_logprior(params)
    likelihood += lnprior
    return likelihood

def calculate_likelihood_task2(para, ncov, hartlap):
    '''Computes the likelihood given the parameters (but allows for selecting
       covariance matrix from finite samples and adding hartlap correction).'''
    params = {}
    for i in range(len(param_names_varying)):
        params[param_names_varying[i]] = [para[i]]
    # obtains predicted data vector given the parameters
    predicted_vector = cp_nn_model.predictions_np(params)[0]
    # transforms the vector appropriately
    predicted_vector = predicted_vector * maximum + minimum
    # computes the residual between observed and predicted
    delta = predicted_vector - reference_model_noisy
    # selects which inverse covariance matrix to use depending on criteria
    if ncov == 1500:
        inverse_covariance = cov_num_1500_inv
    elif ncov == 3000:
        inverse_covariance = cov_num_3000_inv
    elif ncov == 10000:
        inverse_covariance = cov_num_10000_inv
    if hartlap: # applies hartlap correction if requested
        inverse_covariance *= (ncov-LEN-2)/(ncov-1)
    likelihood = -0.5*np.matmul(delta, np.matmul(inverse_covariance, delta))
    lnprior = calculate_logprior(params)
    likelihood += lnprior
    return likelihood

def calculate_likelihood_task3(para, cov_type, n_pca):
    '''Computes the likelihood given the parameters (but allows for selecting
       covariance from finite samples).'''
    params = {}
    for i in range(len(param_names_varying)):
        params[param_names_varying[i]] = [para[i]]
    # obtains predicted data vector given the parameters
    predicted_vector = cp_nn_model.predictions(params)[0]
    # transforms the vector appropriately
    predicted_vector = predicted_vector * maximum + minimum
    # projects into the PCA basis and selects desired number of elements
    predicted_vector = pca.transform(predicted_vector-mean)[:n_pca]
    # computes residual between observed and predicted
    delta = predicted_vector - reference_model_noisy_pca[:n_pca]
    # selects the appropriate inverse covariance matrix to use from dictionary
    inverse_covariance = inv_cov_dict_task3[cov_type][n_pca]
    likelihood = -0.5*np.matmul(delta, np.matmul(inverse_covariance, delta))
    lnprior = calculate_logprior(params)
    likelihood += lnprior
    return likelihood

def calculate_likelihood_task4(para, cov_type, n_pca):
    '''Computes the likelihood given the parameters (but allows for selecting
       covariance from finite samples).'''
    params = {}
    for i in range(len(param_names_varying)):
        params[param_names_varying[i]] = [para[i]]
    # obtains predicted data vector given the parameters
    predicted_vector = cp_nn_model.predictions(params)[0]
    # transforms the vector appropriately
    predicted_vector = predicted_vector * maximum + minimum
    # projects into the PCA basis and selects desired number of elements
    predicted_vector = pca.transform(predicted_vector-mean)[:n_pca]
    # computes residual between observed and predicted
    delta = predicted_vector - reference_model_pca[:n_pca]
    # selects the appropriate inverse covariance matrix to use from dictionary
    inverse_covariance = inv_cov_dict_task3[cov_type][n_pca]
    likelihood = -0.5*np.matmul(delta, np.matmul(inverse_covariance, delta))
    lnprior = calculate_logprior(params)
    likelihood += lnprior
    return likelihood



#################################### TASK 1 ###################################

p0 = starting_positions(0)
sampler = emcee.EnsembleSampler(nwalkers, ndim, calculate_likelihood_task1)
sampler.run_mcmc(p0, total_steps, progress=True)
samples_emcee = sampler.get_chain(discard=burning_steps, flat=True)
np.save('outputs_MCMC/MCMC_task1.npy', samples_emcee)
log_prob_samples = sampler.get_log_prob(discard=burning_steps, flat=True)
np.save('outputs_MCMC/logp_task1.npy', log_prob_samples)
print('Task 1 done')



#################################### TASK 2 ###################################

# Iterates through the different number of covariance matrix elements and 
# whether to apply hartlap correction or not
for ncov in [1500, 3000, 10000]:
    for hartlap in [True, False]:
        sampler = emcee.EnsembleSampler(nwalkers, ndim, calculate_likelihood_task2,
                                        args=(ncov, hartlap))
        sampler.run_mcmc(p0, total_steps, progress=True)
        samples_emcee = sampler.get_chain(discard=burning_steps, flat=True)
        np.save('outputs_MCMC/MCMC_task2_{0}_{1}.npy'.format(ncov, hartlap),
                samples_emcee)
        log_prob_samples = sampler.get_log_prob(discard=burning_steps, flat=True)
        np.save('outputs_MCMC/logp_task2_{0}_{1}.npy'.format(ncov, hartlap),
                log_prob_samples)
        
        

################################### TASK 3 ####################################

# Iterates through different number of PCA elements and saves results for each
for cov_type in ['analytical', '1500']:
    for n_pca in [100, 200, 300, 400, 500, 600, 700, 800, 900]:
        sampler = emcee.EnsembleSampler(nwalkers, ndim, calculate_likelihood_task3,
                                        args=(cov_type, n_pca))
        sampler.run_mcmc(p0, total_steps, progress=True)
        samples_emcee = sampler.get_chain(discard=burning_steps, flat=True)
        np.save('outputs_MCMC/MCMC_task3_{0}_{1}.npy'.format(cov_type, n_pca),
                samples_emcee)
        log_prob_samples = sampler.get_log_prob(discard=burning_steps, flat=True)
        np.save('outputs_MCMC/logp_task3_{0}_{1}.npy'.format(cov_type, n_pca),
                log_prob_samples)
        print('Task 3 done for: ', cov_type, n_pca)
        
        

################################### TASK 4a ###################################

# Iterates through different number of PCA elements and saves results for each
for cov_type in ['analytical', '1500']:
    for n_pca in [100, 200, 300, 400, 500, 600, 700, 800, 900]:
        sampler = emcee.EnsembleSampler(nwalkers, ndim, calculate_likelihood_task4,
                                        args=(cov_type, n_pca))
        sampler.run_mcmc(p0, total_steps, progress=True)
        samples_emcee = sampler.get_chain(discard=burning_steps, flat=True)
        np.save('outputs_MCMC/MCMC_task4a_{0}_{1}.npy'.format(cov_type, n_pca),
                samples_emcee)
        log_prob_samples = sampler.get_log_prob(discard=burning_steps, flat=True)
        np.save('outputs_MCMC/logp_task4a_{0}_{1}.npy'.format(cov_type, n_pca),
                log_prob_samples)
        print('Task 4a done for: ', cov_type, n_pca)

    
################################### TASK 4b ###################################

cov_type = '1500' # uses only the 1500 covariance matrix for this last part
n_pca = 300 # uses a fixed number of PCA elements for this last part
# Runs multiple, separate MCMC chains (using different initial seeds)
integers = []
for integer in integers:
    p0 = starting_positions(integer)
    sampler = emcee.EnsembleSampler(nwalkers, ndim, calculate_likelihood_task4,
                                    args=(cov_type, n_pca))
    sampler.run_mcmc(p0, total_steps, progress=True)
    samples_emcee = sampler.get_chain(discard=burning_steps, flat=True)
    np.save('outputs_MCMC/MCMC_task4b_seed={}.npy'.format(integer),
            samples_emcee)
    log_prob_samples = sampler.get_log_prob(discard=burning_steps, flat=True)
    np.save('outputs_MCMC/logp_task4b_seed={}.npy'.format(integer),
            log_prob_samples)
    print('Task 4b done for: ', integer)
    """
    
    
    
    
    
    
    
    
    
    