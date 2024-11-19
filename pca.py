import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.decomposition import PCA
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import validation_curve
sns.set()

'''
#Example of data compression using Principal Component Analysis(PCA)

def myPCA(energValue, X):
    pcaModel = PCA()
    pcaModel.fit(X)
    nc = np.where(pcaModel.explained_variance_ratio_.cumsum()>energValue)[0][0]
    pcaModel = PCA(n_components=nc)
    pcaModel.fit(X)
    X2 = pcaModel.transform(X)
    return X2
    
X=np.random.rand(200, 50)
X2 = myPCA(0.90, X)

print(X.shape)
print(X2.shape)
'''

#polynomial fitting PR: Polynomial regression
def PR(degree=2, **kwargs):
    p = make_pipeline(PolynomialFeatures(degree), LinearRegression(**kwargs))
    return p

'''
#Linear regression Prediciton
X = np.arange(30)
y = 3*(X**2)-10*X+34
y_pred = PR(2).fit(X[:,np.newaxis], y).predict(X[:,np.newaxis])

plt.scatter(X,y)
plt.plot(X,y_pred, color = 'red')
plt.show()
'''

#Cross validation of the model
X = np.arange(30)
y = 3*(X**2)-10*X+34
# Define degrees to test
degree = np.arange(0, 15)

# Perform validation curve
train_score, val_score = validation_curve(
    PR(),
    X[:, np.newaxis],
    y,
    param_name='polynomialfeatures__degree',
    param_range=degree,
    cv=3
)

# Plot the results
plt.plot(degree, np.median(train_score, 1), color='green', label='Training Score (TRS)')
plt.plot(degree, np.median(val_score, 1), color='red', label='Validation Score (VS)')
plt.legend(loc='best')
#plt.ylim(0, 1)
plt.xlabel("Degree")
plt.ylabel("Score")
plt.show()
