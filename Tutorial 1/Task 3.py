########### Task 3 #################
#########  Replace the labels of 1,-1 with 1 & 0.



import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle


class SigmoidPerceptron(object):
    def __init__(self, eta=0.01, n_iter=50):
        self.eta = eta
        self.n_iter = n_iter

    def weighted_sum(self, X):
        return np.dot(X, self.w_[1:]) + self.w_[0]

    def sigmoid(self, z):
        return 1.0 / (1.0 + np.exp(-np.clip(z, -250, 250)))

    def fit(self, X, y):
        self.w_ = np.zeros(1 + X.shape[1])
        self.errors_ = []

        for _ in range(self.n_iter):
            errors = 0
            for xi, target in zip(X, y):
                net_input = self.weighted_sum(xi)
                output = self.sigmoid(net_input)


                update = self.eta * (target - output)
                self.w_[1:] += update * xi
                self.w_[0] += update


                y_pred = 1 if output >= 0.5 else 0
                errors += int(target != y_pred)

            self.errors_.append(errors)
        return self

    def predict(self, X):
        net_input = self.weighted_sum(X)
        prob = self.sigmoid(net_input)
        return np.where(prob >= 0.5, 1, 0)



url = "https://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data"
df = pd.read_csv(url, header=None)
df = shuffle(df)

X = df.iloc[:, 0:4].values
y = df.iloc[:, 4].values


y = np.where(y == "Iris-setosa", 1, 0)


train_data, test_data, train_labels, test_labels = train_test_split(
    X, y, test_size=0.25
)
model = SigmoidPerceptron(eta=0.01, n_iter=50)
model.fit(train_data, train_labels)

print("\n--- Iris Flower Classification (Sigmoid + Binary Labels 1/0) ---")
sepal_length = float(input("Enter Sepal Length (cm): "))
sepal_width = float(input("Enter Sepal Width (cm): "))
petal_length = float(input("Enter Petal Length (cm): "))
petal_width = float(input("Enter Petal Width (cm): "))

manual_input = np.array([[sepal_length, sepal_width, petal_length, petal_width]])
prediction = model.predict(manual_input)

print("\nPrediction Result:")
if prediction[0] == 1:
    print("The flower is Iris-setosa.")
else:
    print("The flower is NOT Iris-setosa.")
