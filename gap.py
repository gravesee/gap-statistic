import pandas as pd
import numpy as np
from sklearn.cluster import KMeans, MiniBatchKMeans
from numpy.random import random_sample
from math import sqrt, log

from sklearn.datasets import load_iris

# famous iris data set
iris = load_iris()
iris_data = pd.DataFrame(iris['data'], columns=iris['feature_names'])
iris_target = iris['target']


# returns series of random values sampled between min and max values of passed col
def get_rand_data(col):
	rng = col.max() - col.min()
	return pd.Series(random_sample(len(col))*rng + col.min())

def iter_kmeans(df, n_clusters, num_iters=5):
	rng =  range(1, num_iters + 1)
	vals = pd.Series(index=rng)
	for i in rng:
		k = KMeans(n_clusters=n_clusters, n_init=3)
		k.fit(df)
		print "Ref k: %s" % k.get_params()['n_clusters']
		vals[i] = k.inertia_
	return vals

def gap_statistic(df, max_k=10):
	gaps = pd.Series(index = range(1, max_k + 1))
	for k in range(1, max_k + 1):
		km_act = KMeans(n_clusters=k, n_init=3)
		km_act.fit(df)

		# get ref dataset
		ref = df.apply(get_rand_data)
		ref_inertia = iter_kmeans(ref, n_clusters=k).mean()

		gap = log(ref_inertia) - log(km_act.inertia_)

		print "Ref: %s   Act: %s  Gap: %s" % ( ref_inertia, km_act.inertia_, gap)
		gaps[k] = gap

	return gaps
