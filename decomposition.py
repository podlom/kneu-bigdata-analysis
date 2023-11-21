# імпортуємо необхідні бібліотеки для декомпозиції
import numpy as np
import pandas as pd
import os, time, pickle, gzip

from sklearn import preprocessing as pp

import matplotlib.pyplot as plt
import seaborn as sns
color = sns.color_palette()
import matplotlib as mpl

# імпортуємо датасет
dataset = pd.read_excel(r"C:\Users\podlo\OneDrive\Документы\__learn_aspirantura_kneu\_4_semestr\2023-11-20_bigdata\dataset_decomposition.xlsx")

# переглядаємо датасет
print(dataset.shape)
print(dataset.head())
print(dataset.info())

# нормалізуємо числові дані
from sklearn.preprocessing import StandardScaler
numeric_columns = dataset.select_dtypes(include=['float64', 'int64']).columns
scaler = StandardScaler()
dataset[numeric_columns] = scaler.fit_transform(dataset[numeric_columns])

# формуємо вибірки даних
data_y = dataset['capital_intencity_growh']
data_x = dataset.select_dtypes(exclude=['object']).drop(columns=['capital_intencity_growh'])

# розбиваємо вибірки даних на тестову та трейнову
y_train, y_validation, y_test = np.split(data_y.sample(frac=1), [int(.6*len(data_y)), int(.8*len(data_y))])
x_train, x_validation, x_test = np.split(data_x.sample(frac=1), [int(.6*len(data_x)), int(.8*len(data_x))])

train_index = range(0,len(x_train))
validation_index = range(len(x_train), len(x_train)+len(x_validation))
test_index = range(len(x_train)+len(x_validation), len(x_train)+len(x_validation)+len(x_test))

x_train = pd.DataFrame(data=x_train)
y_train = pd.Series(data=y_train)

x_validation = pd.DataFrame(data=x_validation)
y_validation = pd.Series(data=y_validation)

x_test = pd.DataFrame(data=x_test)
y_test = pd.Series(data=y_test)

# функція для точкового графіка
def scatterPlot(xDF, yDF, algoName):
    tempDF = pd.DataFrame(data=xDF.loc[:,0:1], index=xDF.index)
    tempDF = pd.concat((tempDF,yDF), axis=1, join="inner")
    tempDF.columns = ["First Vector", "Second Vector", "Label"]
    sns.lmplot(x="First Vector", y="Second Vector", hue="Label", data=tempDF, fit_reg=False, legend = False)
    ax = plt.gca()
    ax.set_title("Separation of Observations using "+algoName)

# зменшення розмірності

# -----------------------------------------------------------------------------
# Principal Component Analysis
from sklearn.decomposition import PCA

n_components = 40
whiten = False
random_state = 2018

pca = PCA(n_components=n_components, whiten=whiten, random_state=random_state)
x_train_PCA = pca.fit_transform(x_train)
x_train_PCA = pd.DataFrame(data=x_train_PCA, index=train_index)

print("Variance Explained by all 40 principal components: ", \
      sum(pca.explained_variance_ratio_))

importanceOfPrincipalComponents = pd.DataFrame(data=pca.explained_variance_ratio_)
importanceOfPrincipalComponents = importanceOfPrincipalComponents.T

print('Variance Captured by First 5 Principal Components: ',
      importanceOfPrincipalComponents.loc[:,0:5].sum(axis=1).values)
print('Variance Captured by First 10 Principal Components: ',
      importanceOfPrincipalComponents.loc[:,0:10].sum(axis=1).values)
print('Variance Captured by First 15 Principal Components: ',
      importanceOfPrincipalComponents.loc[:,0:15].sum(axis=1).values)
print('Variance Captured by First 20 Principal Components: ',
      importanceOfPrincipalComponents.loc[:,0:20].sum(axis=1).values)
print('Variance Captured by First 25 Principal Components: ',
      importanceOfPrincipalComponents.loc[:,0:25].sum(axis=1).values)
print('Variance Captured by First 30 Principal Components: ',
      importanceOfPrincipalComponents.loc[:,0:30].sum(axis=1).values)

scatterPlot(x_train_PCA, y_train, "PCA")

# -----------------------------------------------------------------------------
# Incremental PCA
from sklearn.decomposition import IncrementalPCA

n_components = 40
batch_size = None

incrementalPCA = IncrementalPCA(n_components=n_components, batch_size=batch_size)
x_train_incrementalPCA = incrementalPCA.fit_transform(x_train)
x_train_incrementalPCA = pd.DataFrame(data=x_train_incrementalPCA, index=train_index)
x_validation_incrementalPCA = incrementalPCA.transform(x_validation)
x_validation_incrementalPCA = pd.DataFrame(data=x_validation_incrementalPCA, index=validation_index)

scatterPlot(x_train_incrementalPCA, y_train, "Incremental PCA")

# -----------------------------------------------------------------------------
# Sparse PCA
from sklearn.decomposition import SparsePCA

n_components = 40
alpha = 0.0001
random_state = 2018
n_jobs = -1

sparsePCA = SparsePCA(n_components=n_components, alpha=alpha, random_state=random_state, n_jobs=n_jobs)
sparsePCA.fit(x_train)
x_train_sparsePCA = sparsePCA.transform(x_train)
x_train_sparsePCA = pd.DataFrame(data=x_train_sparsePCA, index=train_index)
x_validation_sparsePCA = sparsePCA.transform(x_validation)
x_validation_sparsePCA = pd.DataFrame(data=x_validation_sparsePCA, index=validation_index)

scatterPlot(x_train_sparsePCA, y_train, "Sparse PCA")

# -----------------------------------------------------------------------------
# Kernel PCA
from sklearn.decomposition import KernelPCA

n_components = 40
kernel = 'rbf'
gamma = None
random_state = 2018
n_jobs = 1

kernelPCA = KernelPCA(n_components=n_components, kernel=kernel, gamma=gamma, n_jobs=n_jobs, random_state=random_state)

kernelPCA.fit(x_train)
x_train_kernelPCA = kernelPCA.transform(x_train)
x_train_kernelPCA = pd.DataFrame(data=x_train_kernelPCA,index=train_index)
x_validation_kernelPCA = kernelPCA.transform(x_validation)
x_validation_kernelPCA = pd.DataFrame(data=x_validation_kernelPCA, index=validation_index)

scatterPlot(x_train_kernelPCA, y_train, "Kernel PCA")

# -----------------------------------------------------------------------------
# Singular Value Decomposition
from sklearn.decomposition import TruncatedSVD

n_components = 40
algorithm = 'randomized'
n_iter = 5
random_state = 2018

svd = TruncatedSVD(n_components=n_components, algorithm=algorithm, n_iter=n_iter, random_state=random_state)
x_train_svd = svd.fit_transform(x_train)
x_train_svd = pd.DataFrame(data=x_train_svd, index=train_index)
x_validation_svd = svd.transform(x_validation)
x_validation_svd = pd.DataFrame(data=x_validation_svd, index=validation_index)

scatterPlot(x_train_svd, y_train, "Singular Value Decomposition")

# -----------------------------------------------------------------------------
# Gaussian Random Projection
from sklearn.random_projection import GaussianRandomProjection

n_components = 40
eps = 0.5
random_state = 2018

GRP = GaussianRandomProjection(n_components=n_components, eps=eps, random_state=random_state)
x_train_GRP = GRP.fit_transform(x_train)
x_train_GRP = pd.DataFrame(data=x_train_GRP, index=train_index)
x_validation_GRP = GRP.transform(x_validation)
x_validation_GRP = pd.DataFrame(data=x_validation_GRP, index=validation_index)

scatterPlot(x_train_GRP, y_train, "Gaussian Random Projection")

# -----------------------------------------------------------------------------
# Sparse Random Projection
from sklearn.random_projection import SparseRandomProjection

n_components = 40
density = 'auto'
eps = 0.5
dense_output = False
random_state = 2018

SRP = SparseRandomProjection(n_components=n_components, density=density, eps=eps, dense_output=dense_output, random_state=random_state)
x_train_SRP = SRP.fit_transform(x_train)
x_train_SRP = pd.DataFrame(data=x_train_SRP, index=train_index)
x_validation_SRP = SRP.transform(x_validation)
x_validation_SRP = pd.DataFrame(data=x_validation_SRP, index=validation_index)

scatterPlot(x_train_SRP, y_train, "Sparse Random Projection")

# -----------------------------------------------------------------------------
# Isomap
from sklearn.manifold import Isomap

n_neighbors = 5
n_components = 10
n_jobs = 4

isomap = Isomap(n_neighbors=n_neighbors, n_components=n_components, n_jobs=n_jobs)

isomap.fit(x_train)
x_train_isomap = isomap.transform(x_train)
x_train_isomap = pd.DataFrame(data=x_train_isomap, index=train_index)
x_validation_isomap = isomap.transform(x_validation)
x_validation_isomap = pd.DataFrame(data=x_validation_isomap, index=validation_index)

scatterPlot(x_train_isomap, y_train, "Isomap")

# -----------------------------------------------------------------------------
# Multidimensional Scaling
from sklearn.manifold import MDS

n_components = 2
n_init = 12
max_iter = 1200
metric = True
n_jobs = 4
random_state = 2018

mds = MDS(n_components=n_components, n_init=n_init, max_iter=max_iter, metric=metric, n_jobs=n_jobs, random_state=random_state)

x_train_mds = mds.fit_transform(x_train)
x_train_mds = pd.DataFrame(data=x_train_mds, index=train_index)

scatterPlot(x_train_mds, y_train, "Multidimensional Scaling")

# -----------------------------------------------------------------------------
# Locally Linear Embedding
from sklearn.manifold import LocallyLinearEmbedding

n_neighbors = 10
n_components = 2
method = 'modified'
n_jobs = 4
random_state = 2018

lle = LocallyLinearEmbedding(n_neighbors=n_neighbors, n_components=n_components, method=method, random_state=random_state, n_jobs=n_jobs)

lle.fit(x_train)
x_train_lle = lle.transform(x_train)
x_train_lle = pd.DataFrame(data=x_train_lle, index=train_index)
x_validation_lle = lle.transform(x_validation)
x_validation_lle = pd.DataFrame(data=x_validation_lle, index=validation_index)

scatterPlot(x_train_lle, y_train, "Locally Linear Embedding")

# -----------------------------------------------------------------------------
# t-SNE
from sklearn.manifold import TSNE

n_components = 2
learning_rate = 300
perplexity = 30
early_exaggeration = 12
init = 'random'
random_state = 2018

tSNE = TSNE(n_components=n_components, learning_rate=learning_rate, perplexity=perplexity, early_exaggeration=early_exaggeration, init=init, random_state=random_state)
x_train_tSNE = tSNE.fit_transform(x_train_PCA.loc[:5000,:9])
x_train_tSNE = pd.DataFrame(data=x_train_tSNE, index=train_index[:5001])

scatterPlot(x_train_tSNE, y_train, "t-SNE")

# -----------------------------------------------------------------------------
# Mini-batch dictionary learning
from sklearn.decomposition import MiniBatchDictionaryLearning

n_components = 40
alpha = 1
batch_size = 100
n_iter = 25
random_state = 2018

miniBatchDictLearning = MiniBatchDictionaryLearning(n_components=n_components, alpha=alpha, batch_size=batch_size, n_iter=n_iter, random_state=random_state)

miniBatchDictLearning.fit(x_train)
x_train_miniBatchDictLearning = miniBatchDictLearning.fit_transform(x_train)
x_train_miniBatchDictLearning = pd.DataFrame(data=x_train_miniBatchDictLearning, index=train_index)
x_validation_miniBatchDictLearning = miniBatchDictLearning.transform(x_validation)
x_validation_miniBatchDictLearning = pd.DataFrame(data=x_validation_miniBatchDictLearning, index=validation_index)

scatterPlot(x_train_miniBatchDictLearning, y_train, "Mini-batch Dictionary Learning")

# -----------------------------------------------------------------------------
# Independent Component Analysis
from sklearn.decomposition import FastICA

n_components = 40
algorithm = 'parallel'
whiten = False
max_iter = 100
random_state = 2018

fastICA = FastICA(n_components=n_components, algorithm=algorithm, whiten=whiten, max_iter=max_iter, random_state=random_state)
x_train_fastICA = fastICA.fit_transform(x_train)
x_train_fastICA = pd.DataFrame(data=x_train_fastICA, index=train_index)
x_validation_fastICA = fastICA.transform(x_validation)
x_validation_fastICA = pd.DataFrame(data=x_validation_fastICA, index=validation_index)

scatterPlot(x_train_fastICA, y_train, "Independent Component Analysis")
