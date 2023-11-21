# імпортуємо необхідні бібліотеки для кластеризації
import pandas as pd

import numpy as np
from numpy import where
from numpy import unique

from matplotlib import pyplot

# імпортуємо датасет
dataset = pd.read_excel(r"C:\Users\P.S.painter\Desktop\IW1_clustering\dataset.xlsx")

# переглядаємо датасет
print(dataset.shape)
print(dataset.head())
dataset.info()

# визначаємо масив даних для кластеризації
x_data = np.array(dataset.iloc[:, 1:])

# візуалізація даних
fig, axs = pyplot.subplots(2, 3, figsize=(20, 12))
fig.suptitle('Real data')

axs[0, 0].scatter(x_data[:, 9], x_data[:, 1])
axs[0, 0].set_xlabel('gdp growh')
axs[0, 0].set_ylabel('gdp per capita')

axs[0, 1].scatter(x_data[:, 4], x_data[:, 5])
axs[0, 1].set_xlabel('trade globalization')
axs[0, 1].set_ylabel('trade asymmetry')

axs[0, 2].scatter(x_data[:, 7], x_data[:, 8])
axs[0, 2].set_xlabel('consumption rate')
axs[0, 2].set_ylabel('savings rate')

axs[1, 0].scatter(x_data[:, 11], x_data[:, 1])
axs[1, 0].set_xlabel('gdp agriculture')
axs[1, 0].set_ylabel('gdp per capita')

axs[1, 1].scatter(x_data[:, 12], x_data[:, 1])
axs[1, 1].set_xlabel('gdp industry')
axs[1, 1].set_ylabel('gdp per capita')

axs[1, 2].scatter(x_data[:, 13], x_data[:, 1])
axs[1, 2].set_xlabel('gdp services')
axs[1, 2].set_ylabel('gdp per capita')

pyplot.tight_layout()
pyplot.show()

# кластеризація
# -----------------------------------------------------------------------------
# affinity propagation
damping = 0.5
max_iter = 200
convergence_iter = 15
copy = True
preference = None
affinity = 'euclidean'
verbose = False
random_state = None

from sklearn.cluster import AffinityPropagation
model = AffinityPropagation(damping = damping, 
                            max_iter = max_iter, 
                            convergence_iter = convergence_iter, 
                            copy = copy,
                            preference = preference, 
                            affinity = affinity, 
                            verbose = verbose, 
                            random_state = random_state)
model.fit(x_data)
yhat = model.predict(x_data)
clusters = unique(yhat)

fig, axs = pyplot.subplots(2, 3, figsize=(20, 12))
fig.suptitle('Affinity propagation clustering')

for cluster in clusters:
	row_ix = where(yhat == cluster)
	axs[0, 0].scatter(x_data[row_ix, 9], x_data[row_ix, 1])
axs[0, 0].set_xlabel('gdp growh')
axs[0, 0].set_ylabel('gdp per capita')

for cluster in clusters:
	row_ix = where(yhat == cluster)
	axs[0, 1].scatter(x_data[row_ix, 4], x_data[row_ix, 5])
axs[0, 1].set_xlabel('trade globalization')
axs[0, 1].set_ylabel('trade asymmetry')

for cluster in clusters:
	row_ix = where(yhat == cluster)
	axs[0, 2].scatter(x_data[row_ix, 7], x_data[row_ix, 8])
axs[0, 2].set_xlabel('consumption rate')
axs[0, 2].set_ylabel('savings rate')

for cluster in clusters:
	row_ix = where(yhat == cluster)
	axs[1, 0].scatter(x_data[row_ix, 11], x_data[row_ix, 1])
axs[1, 0].set_xlabel('gdp agriculture')
axs[1, 0].set_ylabel('gdp per capita')

for cluster in clusters:
	row_ix = where(yhat == cluster)
	axs[1, 1].scatter(x_data[row_ix, 12], x_data[row_ix, 1])
axs[1, 1].set_xlabel('gdp industry')
axs[1, 1].set_ylabel('gdp per capita')

for cluster in clusters:
	row_ix = where(yhat == cluster)
	axs[1, 2].scatter(x_data[row_ix, 13], x_data[row_ix, 1])
axs[1, 2].set_xlabel('gdp services')
axs[1, 2].set_ylabel('gdp per capita')

pyplot.tight_layout()
pyplot.show()

# -----------------------------------------------------------------------------
# agglomerative clustering
n_clusters = 4
affinity ='euclidean'
memory = None
connectivity = None
linkage = 'ward'
distance_threshold = None

from sklearn.cluster import AgglomerativeClustering
model = AgglomerativeClustering(n_clusters = n_clusters, 
                                affinity = affinity, 
                                memory = memory,
                                connectivity = connectivity, 
                                linkage = linkage,
                                distance_threshold = distance_threshold)
yhat = model.fit_predict(x_data)
clusters = unique(yhat)

fig, axs = pyplot.subplots(2, 3, figsize=(20, 12))
fig.suptitle('Agglomerative clustering')

for cluster in clusters:
	row_ix = where(yhat == cluster)
	axs[0, 0].scatter(x_data[row_ix, 9], x_data[row_ix, 1])
axs[0, 0].set_xlabel('gdp growh')
axs[0, 0].set_ylabel('gdp per capita')

for cluster in clusters:
	row_ix = where(yhat == cluster)
	axs[0, 1].scatter(x_data[row_ix, 4], x_data[row_ix, 5])
axs[0, 1].set_xlabel('trade globalization')
axs[0, 1].set_ylabel('trade asymmetry')

for cluster in clusters:
	row_ix = where(yhat == cluster)
	axs[0, 2].scatter(x_data[row_ix, 7], x_data[row_ix, 8])
axs[0, 2].set_xlabel('consumption rate')
axs[0, 2].set_ylabel('savings rate')

for cluster in clusters:
	row_ix = where(yhat == cluster)
	axs[1, 0].scatter(x_data[row_ix, 11], x_data[row_ix, 1])
axs[1, 0].set_xlabel('gdp agriculture')
axs[1, 0].set_ylabel('gdp per capita')

for cluster in clusters:
	row_ix = where(yhat == cluster)
	axs[1, 1].scatter(x_data[row_ix, 12], x_data[row_ix, 1])
axs[1, 1].set_xlabel('gdp industry')
axs[1, 1].set_ylabel('gdp per capita')

for cluster in clusters:
	row_ix = where(yhat == cluster)
	axs[1, 2].scatter(x_data[row_ix, 13], x_data[row_ix, 1])
axs[1, 2].set_xlabel('gdp services')
axs[1, 2].set_ylabel('gdp per capita')

pyplot.tight_layout()
pyplot.show()

# -----------------------------------------------------------------------------
# birch clustering
threshold = 0.01
n_clusters = 4
branching_factor = 50
compute_labels = True
copy = True

from sklearn.cluster import Birch
model = Birch(threshold = threshold, 
              n_clusters = n_clusters, 
              branching_factor = branching_factor, 
              compute_labels = compute_labels, 
              copy= copy)
model.fit(x_data)
yhat = model.predict(x_data)
clusters = unique(yhat)

fig, axs = pyplot.subplots(2, 3, figsize=(20, 12))
fig.suptitle('Birch clustering')

for cluster in clusters:
	row_ix = where(yhat == cluster)
	axs[0, 0].scatter(x_data[row_ix, 9], x_data[row_ix, 1])
axs[0, 0].set_xlabel('gdp growh')
axs[0, 0].set_ylabel('gdp per capita')

for cluster in clusters:
	row_ix = where(yhat == cluster)
	axs[0, 1].scatter(x_data[row_ix, 4], x_data[row_ix, 5])
axs[0, 1].set_xlabel('trade globalization')
axs[0, 1].set_ylabel('trade asymmetry')

for cluster in clusters:
	row_ix = where(yhat == cluster)
	axs[0, 2].scatter(x_data[row_ix, 7], x_data[row_ix, 8])
axs[0, 2].set_xlabel('consumption rate')
axs[0, 2].set_ylabel('savings rate')

for cluster in clusters:
	row_ix = where(yhat == cluster)
	axs[1, 0].scatter(x_data[row_ix, 11], x_data[row_ix, 1])
axs[1, 0].set_xlabel('gdp agriculture')
axs[1, 0].set_ylabel('gdp per capita')

for cluster in clusters:
	row_ix = where(yhat == cluster)
	axs[1, 1].scatter(x_data[row_ix, 12], x_data[row_ix, 1])
axs[1, 1].set_xlabel('gdp industry')
axs[1, 1].set_ylabel('gdp per capita')

for cluster in clusters:
	row_ix = where(yhat == cluster)
	axs[1, 2].scatter(x_data[row_ix, 13], x_data[row_ix, 1])
axs[1, 2].set_xlabel('gdp services')
axs[1, 2].set_ylabel('gdp per capita')

pyplot.tight_layout()
pyplot.show()

# -----------------------------------------------------------------------------
# dbscan clustering
eps = 0.5
min_samples = 10
metric = 'euclidean'
metric_params = None
algorithm = 'auto'
leaf_size = 30
p = None
n_jobs = None

from sklearn.cluster import DBSCAN
model = DBSCAN(eps = eps, 
               min_samples = min_samples, 
               metric=metric, 
               metric_params=metric_params, 
               algorithm=algorithm,
               leaf_size=leaf_size, 
               p=p, 
               n_jobs=n_jobs)
yhat = model.fit_predict(x_data)
clusters = unique(yhat)

fig, axs = pyplot.subplots(2, 3, figsize=(20, 12))
fig.suptitle('DBSCAN clustering')

for cluster in clusters:
	row_ix = where(yhat == cluster)
	axs[0, 0].scatter(x_data[row_ix, 9], x_data[row_ix, 1])
axs[0, 0].set_xlabel('gdp growh')
axs[0, 0].set_ylabel('gdp per capita')

for cluster in clusters:
	row_ix = where(yhat == cluster)
	axs[0, 1].scatter(x_data[row_ix, 4], x_data[row_ix, 5])
axs[0, 1].set_xlabel('trade globalization')
axs[0, 1].set_ylabel('trade asymmetry')

for cluster in clusters:
	row_ix = where(yhat == cluster)
	axs[0, 2].scatter(x_data[row_ix, 7], x_data[row_ix, 8])
axs[0, 2].set_xlabel('consumption rate')
axs[0, 2].set_ylabel('savings rate')

for cluster in clusters:
	row_ix = where(yhat == cluster)
	axs[1, 0].scatter(x_data[row_ix, 11], x_data[row_ix, 1])
axs[1, 0].set_xlabel('gdp agriculture')
axs[1, 0].set_ylabel('gdp per capita')

for cluster in clusters:
	row_ix = where(yhat == cluster)
	axs[1, 1].scatter(x_data[row_ix, 12], x_data[row_ix, 1])
axs[1, 1].set_xlabel('gdp industry')
axs[1, 1].set_ylabel('gdp per capita')

for cluster in clusters:
	row_ix = where(yhat == cluster)
	axs[1, 2].scatter(x_data[row_ix, 13], x_data[row_ix, 1])
axs[1, 2].set_xlabel('gdp services')
axs[1, 2].set_ylabel('gdp per capita')

pyplot.tight_layout()
pyplot.show()

# -----------------------------------------------------------------------------
# k-means clustering
n_clusters = 4
n_init = 10
max_iter = 300
init = 'k-means++'
tol = 0.0001
verbose = 0
random_state = None
copy_x = True
algorithm = 'elkan'

from sklearn.cluster import KMeans
model = KMeans(n_clusters = n_clusters, 
               n_init=n_init, 
               max_iter=max_iter, 
               init=init, 
               tol=tol, 
               verbose=verbose,
               random_state=random_state, 
               copy_x=copy_x, 
               algorithm=algorithm)
model.fit(x_data)
yhat = model.predict(x_data)
clusters = unique(yhat)

fig, axs = pyplot.subplots(2, 3, figsize=(20, 12))
fig.suptitle('k-means clustering')

for cluster in clusters:
	row_ix = where(yhat == cluster)
	axs[0, 0].scatter(x_data[row_ix, 9], x_data[row_ix, 1])
axs[0, 0].set_xlabel('gdp growh')
axs[0, 0].set_ylabel('gdp per capita')

for cluster in clusters:
	row_ix = where(yhat == cluster)
	axs[0, 1].scatter(x_data[row_ix, 4], x_data[row_ix, 5])
axs[0, 1].set_xlabel('trade globalization')
axs[0, 1].set_ylabel('trade asymmetry')

for cluster in clusters:
	row_ix = where(yhat == cluster)
	axs[0, 2].scatter(x_data[row_ix, 7], x_data[row_ix, 8])
axs[0, 2].set_xlabel('consumption rate')
axs[0, 2].set_ylabel('savings rate')

for cluster in clusters:
	row_ix = where(yhat == cluster)
	axs[1, 0].scatter(x_data[row_ix, 11], x_data[row_ix, 1])
axs[1, 0].set_xlabel('gdp agriculture')
axs[1, 0].set_ylabel('gdp per capita')

for cluster in clusters:
	row_ix = where(yhat == cluster)
	axs[1, 1].scatter(x_data[row_ix, 12], x_data[row_ix, 1])
axs[1, 1].set_xlabel('gdp industry')
axs[1, 1].set_ylabel('gdp per capita')

for cluster in clusters:
	row_ix = where(yhat == cluster)
	axs[1, 2].scatter(x_data[row_ix, 13], x_data[row_ix, 1])
axs[1, 2].set_xlabel('gdp services')
axs[1, 2].set_ylabel('gdp per capita')

pyplot.tight_layout()
pyplot.show()

# -----------------------------------------------------------------------------
# mini-batch k-means clustering
n_clusters = 4
n_init = 10
max_iter = 300
batch_size = 1024
init = 'k-means++'
verbose = 0
compute_labels = True
random_state = None
tol = 0.0
max_no_improvement = 10
init_size = None
reassignment_ratio = 0.01

from sklearn.cluster import MiniBatchKMeans
model = MiniBatchKMeans(n_clusters = n_clusters, 
                        n_init=n_init, 
                        max_iter=max_iter, 
                        batch_size = batch_size,
                        init=init, 
                        tol=tol, 
                        verbose=verbose, 
                        compute_labels=compute_labels, 
                        random_state=random_state,
                        max_no_improvement=max_no_improvement, 
                        init_size=init_size, 
                        reassignment_ratio=reassignment_ratio)
model.fit(x_data)
yhat = model.predict(x_data)
clusters = unique(yhat)

fig, axs = pyplot.subplots(2, 3, figsize=(20, 12))
fig.suptitle('Mini-batch k-means clustering')

for cluster in clusters:
	row_ix = where(yhat == cluster)
	axs[0, 0].scatter(x_data[row_ix, 9], x_data[row_ix, 1])
axs[0, 0].set_xlabel('gdp growh')
axs[0, 0].set_ylabel('gdp per capita')

for cluster in clusters:
	row_ix = where(yhat == cluster)
	axs[0, 1].scatter(x_data[row_ix, 4], x_data[row_ix, 5])
axs[0, 1].set_xlabel('trade globalization')
axs[0, 1].set_ylabel('trade asymmetry')

for cluster in clusters:
	row_ix = where(yhat == cluster)
	axs[0, 2].scatter(x_data[row_ix, 7], x_data[row_ix, 8])
axs[0, 2].set_xlabel('consumption rate')
axs[0, 2].set_ylabel('savings rate')

for cluster in clusters:
	row_ix = where(yhat == cluster)
	axs[1, 0].scatter(x_data[row_ix, 11], x_data[row_ix, 1])
axs[1, 0].set_xlabel('gdp agriculture')
axs[1, 0].set_ylabel('gdp per capita')

for cluster in clusters:
	row_ix = where(yhat == cluster)
	axs[1, 1].scatter(x_data[row_ix, 12], x_data[row_ix, 1])
axs[1, 1].set_xlabel('gdp industry')
axs[1, 1].set_ylabel('gdp per capita')

for cluster in clusters:
	row_ix = where(yhat == cluster)
	axs[1, 2].scatter(x_data[row_ix, 13], x_data[row_ix, 1])
axs[1, 2].set_xlabel('gdp services')
axs[1, 2].set_ylabel('gdp per capita')

pyplot.tight_layout()
pyplot.show()

# -----------------------------------------------------------------------------
# mean shift clustering
max_iter = 300
bandwidth = None
seeds = None
bin_seeding = False
min_bin_freq = 1
cluster_all = True
n_jobs = None

from sklearn.cluster import MeanShift
model = MeanShift(max_iter=max_iter, 
                  bandwidth=bandwidth, 
                  seeds=seeds, 
                  bin_seeding=bin_seeding, 
                  min_bin_freq=min_bin_freq,
                  cluster_all=cluster_all, 
                  n_jobs=n_jobs)
yhat = model.fit_predict(x_data)
clusters = unique(yhat)

fig, axs = pyplot.subplots(2, 3, figsize=(20, 12))
fig.suptitle('Mean shift clustering')

for cluster in clusters:
	row_ix = where(yhat == cluster)
	axs[0, 0].scatter(x_data[row_ix, 9], x_data[row_ix, 1])
axs[0, 0].set_xlabel('gdp growh')
axs[0, 0].set_ylabel('gdp per capita')

for cluster in clusters:
	row_ix = where(yhat == cluster)
	axs[0, 1].scatter(x_data[row_ix, 4], x_data[row_ix, 5])
axs[0, 1].set_xlabel('trade globalization')
axs[0, 1].set_ylabel('trade asymmetry')

for cluster in clusters:
	row_ix = where(yhat == cluster)
	axs[0, 2].scatter(x_data[row_ix, 7], x_data[row_ix, 8])
axs[0, 2].set_xlabel('consumption rate')
axs[0, 2].set_ylabel('savings rate')

for cluster in clusters:
	row_ix = where(yhat == cluster)
	axs[1, 0].scatter(x_data[row_ix, 11], x_data[row_ix, 1])
axs[1, 0].set_xlabel('gdp agriculture')
axs[1, 0].set_ylabel('gdp per capita')

for cluster in clusters:
	row_ix = where(yhat == cluster)
	axs[1, 1].scatter(x_data[row_ix, 12], x_data[row_ix, 1])
axs[1, 1].set_xlabel('gdp industry')
axs[1, 1].set_ylabel('gdp per capita')

for cluster in clusters:
	row_ix = where(yhat == cluster)
	axs[1, 2].scatter(x_data[row_ix, 13], x_data[row_ix, 1])
axs[1, 2].set_xlabel('gdp services')
axs[1, 2].set_ylabel('gdp per capita')

pyplot.tight_layout()
pyplot.show()

# -----------------------------------------------------------------------------
# optics clustering
eps = 0.5
min_samples = 10
max_eps = np.inf
metric = 'minkowski'
p = 2
metric_params = None
cluster_method = 'xi'
xi = 0.05
predecessor_correction = True
min_cluster_size = None
algorithm = 'auto'
leaf_size = 30
n_jobs = None

from sklearn.cluster import OPTICS
model = OPTICS(eps = eps, 
               min_samples = min_samples, 
               max_eps=max_eps, 
               metric=metric, 
               p=p, 
               metric_params=metric_params,
               cluster_method=cluster_method, 
               xi=xi, 
               predecessor_correction=predecessor_correction, 
               min_cluster_size=min_cluster_size,
               algorithm=algorithm, 
               leaf_size=leaf_size, 
               n_jobs=n_jobs)
yhat = model.fit_predict(x_data)
clusters = unique(yhat)

fig, axs = pyplot.subplots(2, 3, figsize=(20, 12))
fig.suptitle('OPTICS clustering')

for cluster in clusters:
	row_ix = where(yhat == cluster)
	axs[0, 0].scatter(x_data[row_ix, 9], x_data[row_ix, 1])
axs[0, 0].set_xlabel('gdp growh')
axs[0, 0].set_ylabel('gdp per capita')

for cluster in clusters:
	row_ix = where(yhat == cluster)
	axs[0, 1].scatter(x_data[row_ix, 4], x_data[row_ix, 5])
axs[0, 1].set_xlabel('trade globalization')
axs[0, 1].set_ylabel('trade asymmetry')

for cluster in clusters:
	row_ix = where(yhat == cluster)
	axs[0, 2].scatter(x_data[row_ix, 7], x_data[row_ix, 8])
axs[0, 2].set_xlabel('consumption rate')
axs[0, 2].set_ylabel('savings rate')

for cluster in clusters:
	row_ix = where(yhat == cluster)
	axs[1, 0].scatter(x_data[row_ix, 11], x_data[row_ix, 1])
axs[1, 0].set_xlabel('gdp agriculture')
axs[1, 0].set_ylabel('gdp per capita')

for cluster in clusters:
	row_ix = where(yhat == cluster)
	axs[1, 1].scatter(x_data[row_ix, 12], x_data[row_ix, 1])
axs[1, 1].set_xlabel('gdp industry')
axs[1, 1].set_ylabel('gdp per capita')

for cluster in clusters:
	row_ix = where(yhat == cluster)
	axs[1, 2].scatter(x_data[row_ix, 13], x_data[row_ix, 1])
axs[1, 2].set_xlabel('gdp services')
axs[1, 2].set_ylabel('gdp per capita')

pyplot.tight_layout()
pyplot.show()

# -----------------------------------------------------------------------------
# spectral clustering
n_clusters = 4
eigen_solver = 'arpack'
n_components = None
random_state = None
n_init = 10
gamma = 1.0
affinity = 'rbf'
n_neighbors = 10
eigen_tol = 0.0
assign_labels = 'kmeans'
degree = 3
coef0 = 1
kernel_params = None
n_jobs = None

from sklearn.cluster import SpectralClustering
model = SpectralClustering(n_clusters = n_clusters, 
                           eigen_solver=eigen_solver, 
                           n_components=n_components, 
                           random_state=random_state,
                           n_init=n_init, 
                           gamma=gamma, 
                           affinity=affinity, 
                           n_neighbors=n_neighbors, 
                           eigen_tol=eigen_tol,
                           assign_labels=assign_labels, 
                           degree=degree, 
                           coef0=coef0, 
                           kernel_params=kernel_params, 
                           n_jobs=n_jobs)
yhat = model.fit_predict(x_data)
clusters = unique(yhat)

fig, axs = pyplot.subplots(2, 3, figsize=(20, 12))
fig.suptitle('Spectral clustering')

for cluster in clusters:
	row_ix = where(yhat == cluster)
	axs[0, 0].scatter(x_data[row_ix, 9], x_data[row_ix, 1])
axs[0, 0].set_xlabel('gdp growh')
axs[0, 0].set_ylabel('gdp per capita')

for cluster in clusters:
	row_ix = where(yhat == cluster)
	axs[0, 1].scatter(x_data[row_ix, 4], x_data[row_ix, 5])
axs[0, 1].set_xlabel('trade globalization')
axs[0, 1].set_ylabel('trade asymmetry')

for cluster in clusters:
	row_ix = where(yhat == cluster)
	axs[0, 2].scatter(x_data[row_ix, 7], x_data[row_ix, 8])
axs[0, 2].set_xlabel('consumption rate')
axs[0, 2].set_ylabel('savings rate')

for cluster in clusters:
	row_ix = where(yhat == cluster)
	axs[1, 0].scatter(x_data[row_ix, 11], x_data[row_ix, 1])
axs[1, 0].set_xlabel('gdp agriculture')
axs[1, 0].set_ylabel('gdp per capita')

for cluster in clusters:
	row_ix = where(yhat == cluster)
	axs[1, 1].scatter(x_data[row_ix, 12], x_data[row_ix, 1])
axs[1, 1].set_xlabel('gdp industry')
axs[1, 1].set_ylabel('gdp per capita')

for cluster in clusters:
	row_ix = where(yhat == cluster)
	axs[1, 2].scatter(x_data[row_ix, 13], x_data[row_ix, 1])
axs[1, 2].set_xlabel('gdp services')
axs[1, 2].set_ylabel('gdp per capita')

pyplot.tight_layout()
pyplot.show()

# -----------------------------------------------------------------------------
# gaussian mixture clustering
n_components = 4
covariance_type = 'full'
tol = 0.001
reg_covar = 1e-06
max_iter = 100
n_init = 1
init_params = 'kmeans'
weights_init = None
means_init = None
precisions_init = None
random_state = None
warm_start = False
verbose = 0
verbose_interval = 10

from sklearn.mixture import GaussianMixture
model = GaussianMixture(n_components = n_components, 
                        covariance_type=covariance_type, 
                        tol=tol, 
                        reg_covar=reg_covar, 
                        n_init=n_init,
                        max_iter=max_iter, 
                        init_params=init_params, 
                        weights_init=weights_init, 
                        means_init=means_init,
                        precisions_init=precisions_init, 
                        random_state=random_state, 
                        warm_start=warm_start, 
                        verbose=verbose,
                        verbose_interval=verbose_interval)
model.fit(x_data)
yhat = model.predict(x_data)
clusters = unique(yhat)

fig, axs = pyplot.subplots(2, 3, figsize=(20, 12))
fig.suptitle('Gaussian mixture clustering')

for cluster in clusters:
	row_ix = where(yhat == cluster)
	axs[0, 0].scatter(x_data[row_ix, 9], x_data[row_ix, 1])
axs[0, 0].set_xlabel('gdp growh')
axs[0, 0].set_ylabel('gdp per capita')

for cluster in clusters:
	row_ix = where(yhat == cluster)
	axs[0, 1].scatter(x_data[row_ix, 4], x_data[row_ix, 5])
axs[0, 1].set_xlabel('trade globalization')
axs[0, 1].set_ylabel('trade asymmetry')

for cluster in clusters:
	row_ix = where(yhat == cluster)
	axs[0, 2].scatter(x_data[row_ix, 7], x_data[row_ix, 8])
axs[0, 2].set_xlabel('consumption rate')
axs[0, 2].set_ylabel('savings rate')

for cluster in clusters:
	row_ix = where(yhat == cluster)
	axs[1, 0].scatter(x_data[row_ix, 11], x_data[row_ix, 1])
axs[1, 0].set_xlabel('gdp agriculture')
axs[1, 0].set_ylabel('gdp per capita')

for cluster in clusters:
	row_ix = where(yhat == cluster)
	axs[1, 1].scatter(x_data[row_ix, 12], x_data[row_ix, 1])
axs[1, 1].set_xlabel('gdp industry')
axs[1, 1].set_ylabel('gdp per capita')

for cluster in clusters:
	row_ix = where(yhat == cluster)
	axs[1, 2].scatter(x_data[row_ix, 13], x_data[row_ix, 1])
axs[1, 2].set_xlabel('gdp services')
axs[1, 2].set_ylabel('gdp per capita')

pyplot.tight_layout()
pyplot.show()
