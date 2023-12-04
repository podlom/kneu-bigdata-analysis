# імпортуємо необхідні бібліотеки для класифікації
import numpy as np
import pandas as pd

# імпортуємо датасет
dataset = pd.read_excel(r"C:\Users\podlo\OneDrive\Документы\__learn_aspirantura_kneu\_4_semestr\obrobka_ta_analiz_big_data\2023-11-20_bigdata\dataset_classification.xlsx")

# переглядаємо датасет
dataset.shape
dataset.head()
dataset.info()

# перетворюємо фактор у бінарний тип
i = 0
while i<len(dataset):
    if dataset.capital_intencity_growh[i]=='decreasing':
        dataset.capital_intencity_growh[i]=0
    else:
        dataset.capital_intencity_growh[i]=1
    i=i+1

# формуємо вибірки даних
data_y = dataset['capital_intencity_growh'].astype(float).values
data_x = dataset.select_dtypes(exclude=['object']).values

# розбиваємо вибірки даних на тестову та трейнову
from sklearn.model_selection import train_test_split
train_x, test_x, train_y, test_y = train_test_split(data_x, data_y, test_size = 0.2, random_state=42)

# класифікація
# -----------------------------------------------------------------------------
# KNN
n_neighbors=10

from sklearn.neighbors import KNeighborsClassifier
model = KNeighborsClassifier(n_neighbors=n_neighbors)
model.fit(train_x, train_y)

# результати
predicted = model.predict(test_x)
predicted_p = model.predict_proba(test_x)
predicted_train = model.predict(train_x)
predicted_p_train = model.predict_proba(train_x)

# тестування класифікації
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score, roc_curve, auc

print("Test accuracy:", accuracy_score(test_y, predicted))
print("Test F1:", f1_score(test_y, predicted))
print("Train accuracy:", accuracy_score(train_y, predicted_train))
print("Train F1:", f1_score(train_y, predicted_train))

print("Test AUC:", roc_auc_score(
    y_score=predicted_p[:,1], 
    y_true=test_y)
)
print("Train AUC:", roc_auc_score(
    y_score=predicted_p_train[:,1], 
    y_true=train_y)
)

# графік порогу класифікації
tresh  = {i: f1_score(test_y, (predicted_p[:,1] > i).astype('int')) for i in np.array(list(range(101)))/100}
pd.DataFrame({
    'treshold': list(tresh.keys()),
    'f1': list(tresh.values())
}).plot.line(x='treshold', y ='f1')

# ROC крива
fpr, tpr, threshold = roc_curve(test_y, predicted_p[:,1])
roc_auc = auc(fpr, tpr)
fpr, tpr, threshold

import matplotlib.pyplot as plt

plt.title('Receiver Operating Characteristic')
plt.plot(fpr, tpr, 'b', label = 'AUC = %0.2f' % roc_auc)
plt.legend(loc = 'lower right')
plt.plot([0, 1], [0, 1],'r--')
plt.xlim([0, 1])
plt.ylim([0, 1])
plt.ylabel('True Positive Rate')
plt.xlabel('False Positive Rate')
plt.show()

# класифікація з кількістю сусідів від 2 до 20
for i in range(2, 21):
    model = KNeighborsClassifier(n_neighbors = i)
    model.fit(train_x, train_y)
    predicted_p = model.predict_proba(test_x) 
    predicted_p_train = model.predict_proba(train_x) 
    print(i, "TEST:", roc_auc_score(y_score=predicted_p[:,1], y_true=test_y), "TRAIN:",roc_auc_score(y_score=predicted_p_train[:,1], y_true=train_y))

# -----------------------------------------------------------------------------
# Logistic Regression
from sklearn.linear_model import LogisticRegression 
from sklearn.metrics import classification_report, confusion_matrix

log_reg = LogisticRegression(C=0.001, class_weight="balanced", n_jobs=-1)
lf = log_reg.fit(train_x, train_y)
y_pred = lf.predict(test_x)

print(confusion_matrix(test_y, y_pred))
print(classification_report(test_y, y_pred))
print("Accuracy: " + str(accuracy_score(test_y, y_pred)))
print("F1 score: " + str(f1_score(test_y, y_pred)))
y_pred_p = lf.predict_proba(test_x)
print("AUC: " + str(roc_auc_score(y_score=y_pred_p[:,1], y_true=test_y-1)))

# ROC крива
fpr, tpr, threshold = roc_curve(test_y, y_pred_p[:,1])
roc_auc = auc(fpr, tpr)

import matplotlib.pyplot as plt
plt.title('Receiver Operating Characteristic')
plt.plot(fpr, tpr, 'b', label = 'AUC = %0.2f' % roc_auc)
plt.legend(loc = 'lower right')
plt.plot([0, 1], [0, 1],'r--')
plt.xlim([0, 1])
plt.ylim([0, 1])
plt.ylabel('True Positive Rate')
plt.xlabel('False Positive Rate')
plt.show()

# класифікація з масивом параметрів регуляризації
for i in [1,0.5,0.1,0.01,0.001,0.0001,0.0000001]:
    log_reg = LogisticRegression(C=i, class_weight="balanced", n_jobs=-1)
    lf = log_reg.fit(train_x, train_y)
    y_pred_p = lf.predict_proba(test_x)
    print(i," - AUC: " + str(roc_auc_score(y_score=y_pred_p[:,1], y_true=test_y)))

# -----------------------------------------------------------------------------
# Naive Bayes
# GaussianNB
from sklearn.naive_bayes import GaussianNB
gnb = GaussianNB()
nb = gnb.fit(train_x, train_y)
y_pred = nb.predict_proba(test_x)
print("AUC: " + str(roc_auc_score(y_score=y_pred[:,1], y_true=test_y)))

# MultinomialNB
# from sklearn.naive_bayes import MultinomialNB
# clf = MultinomialNB()
# nb = clf.fit(train_x, train_y)
# y_pred = nb.predict_proba(test_x)
# print("AUC: " + str(roc_auc_score(y_score=y_pred[:,1], y_true=test_y)))

# BernoulliNB
from sklearn.naive_bayes import BernoulliNB
clf = BernoulliNB()
nb = clf.fit(train_x, train_y)
y_pred = nb.predict_proba(test_x)
print("AUC: " + str(roc_auc_score(y_score=y_pred[:,1], y_true=test_y)))

# ComplementNB
# from sklearn.naive_bayes import ComplementNB
# clf = ComplementNB()
# nb = clf.fit(train_x, train_y)
# y_pred = nb.predict_proba(test_x)
# print("AUC: " + str(roc_auc_score(y_score=y_pred[:,1], y_true=test_y)))

# CategoricalNB
# from sklearn.naive_bayes import CategoricalNB
# clf = CategoricalNB()
# nb = clf.fit(train_x, train_y)
# y_pred = nb.predict_proba(test_x)
# print("AUC: " + str(roc_auc_score(y_score=y_pred[:,1], y_true=test_y)))

# -----------------------------------------------------------------------------
# Decision Tree
from sklearn.tree import DecisionTreeClassifier

tree_clf = DecisionTreeClassifier(
    max_depth=3, 
    min_samples_leaf=21, 
    max_features=0.9, 
    criterion="gini",                                    
    random_state=2)

dt = tree_clf.fit(train_x, train_y)
y_pred = dt.predict_proba(test_x)
print("AUC: " + str(roc_auc_score(y_score=y_pred[:,1], y_true=test_y)))

# будуємо граф дерева рішень
from sklearn.tree import plot_tree
import matplotlib.pyplot as plt
plt.rcParams["figure.figsize"] = (20,10)
plt.figure()
plot_tree(dt, filled=True, class_names=['no','yes'], 
          feature_names=dataset.select_dtypes(exclude=['object']).columns)
plt.show()

# класифікація з масивами гіперпараметрів дерева рішень
for i in range(5,11):
    for j in range(10,21):
        tree_clf = DecisionTreeClassifier(max_depth=i, 
                                          min_samples_leaf=j,
                                          max_features=0.9, 
                                          criterion="gini", 
                                          random_state=2)  
        dt = tree_clf.fit(train_x, train_y)
        y_pred = dt.predict_proba(test_x)
        print("Depth: "+str(i)+" Leaf: "+str(j)+"- AUC: " + str(roc_auc_score(y_score=y_pred[:,1], y_true=test_y-1)))

# -----------------------------------------------------------------------------
# MLP
from sklearn.neural_network import MLPClassifier

# lbfgs solver
nn_clf = MLPClassifier(solver='lbfgs', alpha=1,
                    hidden_layer_sizes=(5, 20), random_state=1, max_iter=10000)
nn_clf.fit(train_x, train_y)

print("AUC: " + str(roc_auc_score(y_score=nn_clf.predict_proba(test_x)[:, 1], y_true=test_y)))

# adam solver
nn_clf = MLPClassifier(solver='adam', learning_rate_init=0.00001, alpha=1,
                    hidden_layer_sizes=(13, 3), random_state=1, max_iter=100000, verbose = True)
nn_clf.fit(train_x, train_y)

print("AUC: " + str(roc_auc_score(y_score=nn_clf.predict_proba(test_x)[:, 1], y_true=test_y)))

# -----------------------------------------------------------------------------
# LDA & QDA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis, QuadraticDiscriminantAnalysis
from sklearn.metrics import roc_auc_score

# QuadraticDiscriminantAnalysis
clf = QuadraticDiscriminantAnalysis(reg_param=0.0001)
clf.fit(train_x, train_y)

y_pred = clf.predict_proba(test_x)

print("AUC: " + str(roc_auc_score(y_score=y_pred[:,1], y_true=test_y)))

# LinearDiscriminantAnalysis
clf = LinearDiscriminantAnalysis(solver='lsqr', shrinkage='auto')
clf.fit(train_x, train_y)

y_pred = clf.predict_proba(test_x)

print("AUC: " + str(roc_auc_score(y_score=y_pred[:,1], y_true=test_y)))

# -----------------------------------------------------------------------------
# SVM
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, PolynomialFeatures
from sklearn.svm import LinearSVC

svm_clf = Pipeline((
        ("poly_features", PolynomialFeatures(degree=2)),
        ("scaler", StandardScaler()),
        ("svm_clf", LinearSVC(C=0.0000001))
))
svm_clf.fit(train_x, train_y)
p = svm_clf.predict(test_x)

print(confusion_matrix(test_y, p))
print(classification_report(test_y, p))
print("Accuracy:", str(accuracy_score(test_y, p)))
