from sklearn.model_selection import RandomizedSearchCV
from sklearn.model_selection import RepeatedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVR
from sklearn.model_selection import train_test_split
from scipy.stats import loguniform
import pandas as pd
import numpy as np

CSV_PATH = "abalone_train.csv"

COLS = ["Length", "Diameter", "Height", "Whole weight",
	"Shucked weight", "Viscera weight", "Shell weight", "Age"]

print("[INFO] loading data...")
dataset = pd.read_csv(CSV_PATH, names=COLS)
dataX = dataset[dataset.columns[:-1]]
dataY = dataset[dataset.columns[-1]]
(trainX, testX, trainY, testY) = train_test_split(dataX,
	dataY, random_state=3, test_size=0.15)

scaler = StandardScaler()
trainX = scaler.fit_transform(trainX)
testX = scaler.transform(testX)

model_rand = SVR()
kernel = ["linear", "rbf", "sigmoid", "poly"]
tolerance = np.arange(1e-6, 1e-3, 9e-6)
C = np.arange(1, 3, 0.1)
random = dict(kernel=kernel, tol=tolerance, C=C)

model_grid = SVR()
kernel = ["linear", "rbf", "sigmoid", "poly"]
tolerance = [1e-3, 1e-4, 1e-5, 1e-6]
C = [1, 1.5, 2, 2.5, 3]
grid = dict(kernel=kernel, tol=tolerance, C=C)

print("[INFO] random searching over the hyperparameters...")
cvFold = RepeatedKFold(n_splits=20, n_repeats=3, random_state=1)
randomSearch = RandomizedSearchCV(estimator=model, n_jobs=-1,
	cv=cvFold, param_distributions=grid,
	scoring="neg_mean_squared_error")
searchResults = randomSearch.fit(trainX, trainY)

print("[INFO] grid searching over the hyperparameters...")
cvFold = RepeatedKFold(n_splits=20, n_repeats=3, random_state=1)
gridSearch = GridSearchCV(estimator=model, param_grid = grid, n_jobs=-1,
	cv=cvFold, scoring="neg_mean_squared_error")
searchResults = randomSearch.fit(trainX, trainY)

print("[INFO] evaluating random ...")
bestModel_rand = searchResults.best_estimator_
print("R2: {:.2f}".format(bestModel_rand.score(testX, testY)))

print("[INFO] evaluating grid ...")
bestModel_grid = searchResults.best_estimator_
print("R2: {:.2f}".format(bestModel_grid.score(testX, testY)))

model = SVR()
model.fit(trainX, trainY)

print("[INFO] evaluating standard ...")
print("R2: {:.2f}".format(model.score(testX, testY)))
