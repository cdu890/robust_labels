import numpy as np
import pandas as pd
import sys
import glob
from scipy.stats import entropy, kstest, uniform, ks_2samp


# computes the Shannon entropy
def shannon_entropy(array):
  array = array.round(6)
  denom = len(array)
  unique, counts = np.unique(array.flatten(), return_counts=True)
  probs = counts / denom
  noise = entropy(probs, base=2)
  return noise


# Computes the normalized l1 and l2 distances within a label type
def l1_l2(label, num_classes=10):
  label = (label - np.min(label)) / (np.max(label) - np.min(label))
  l2_dist_list = []
  l1_dist_list = []
  for i in range(num_classes):
      for j in range(i + 1, num_classes):
          l2_dist_list.append(np.linalg.norm(label[i] - label[j], ord=2))
          l1_dist_list.append(np.linalg.norm(label[i] - label[j], ord=1))

  return l1_dist_list, l2_dist_list


# tests if probability distribution is normal, returns KS statistic and p-value
def check_uniform(array):
  array = array.round(6)

  lb = np.min(array)
  ub = np.max(array)
  print(lb, ub)
  sim = kstest(array, uniform(loc=lb, scale=ub).cdf)

  return sim


# tests if two distributions are equal, performs a 2-sample KS test 
# and returns the KS statistic and p-value
def check_sim(array1, array2):
  array1 = array1.round(6)
  array2 = array2.round(6)
  sim = ks_2samp(array1, array2)

  return sim