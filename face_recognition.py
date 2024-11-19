###############################################
#Face recognition using Support vector machines
###############################################

from sklearn.datasets import fetch_lfw_people
from sklearn.svm import SVC
from sklearn.decomposition import PCA
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

faces = fetch_lfw_people(min_faces_per_person=50)

print(faces.data.shape)
print(faces.images[0].shape)
print(faces.target_names)


'''
fig, ax = plt.subplots(2,4)

for idx, axidx in enumerate(ax.flat):
    axidx.imshow(faces.images[idx], cmap='bone')
    axidx.set(xticks=[], yticks=[], xlabel=faces.target_names[faces.target[idx]])

plt.show()
'''

pcaModel = PCA(n_components=150, whiten=True)
svmModel = SVC(kernel='rbf', class_weight='balanced')

mdl = make_pipeline(pcaModel, svmModel)

Xtrain, Xtest, ytrain, ytest = train_test_split(faces.data, faces.target, test_size=0.2)
param_grid = {'svc__C':[1, 5, 15, 30], 'svc__gamma':[0.00001, 0.00005, 0.0001, 0.0005]}
grid = GridSearchCV(mdl, param_grid)

grid.fit(Xtrain, ytrain)

'''
# Create PCA model
pcaModel = PCA(n_components=150, whiten=True)

# Create SVM model
svmModel = SVC(kernel='rbf', class_weight='balanced')

# Combine PCA and SVM in a pipeline
mdl = make_pipeline(pcaModel, svmModel)

# Split the data into training and testing sets
Xtrain, Xtest, ytrain, ytest = train_test_split(faces.data, faces.target, test_size=0.2, random_state=42)

# Define the parameter grid for GridSearchCV
param_grid = {
    'svc__C': [1, 5, 15, 30],
    'svc__gamma': [0.00001, 0.00005, 0.0001, 0.0005]
}

# Create GridSearchCV object
grid = GridSearchCV(mdl, param_grid, cv=5)

# Fit the model to the training data
grid.fit(Xtrain, ytrain)
'''

# Print the best parameters and best score
print("Best parameters found: ", grid.best_params_)
print("Best cross-validation score: {:.2f}".format(grid.best_score_))

mdl = grid.best_estimator_

y_pred = mdl.predict(Xtest)

print(ytest, y_pred)

fig, ax = plt.subplots(2,4)
for idx, axidx in enumerate(ax.flat):
    axidx.imshow(Xtest[idx].reshape(62,47), cmap='bone')
    axidx.set(xticks=[], yticks=[])
    axidx.set_ylabel(faces.target_names[y_pred[idx]].split()[-1], color='green' if y_pred[idx] == ytest[idx] else 'red')
    fig.suptitle("Wrong are in red", size=14)

print(classification_report(ytest, y_pred, target_names=faces.target_names))
mat = confusion_matrix(ytest, y_pred)
sns.heatmap(mat.T, square=True, annot=True, fmt='d', cbar=False, xticklabels=faces.target_names, yticklabels=faces.target_names)
plt.xlabel("True Label")
plt.ylabel("predicted Label")
plt.show()
