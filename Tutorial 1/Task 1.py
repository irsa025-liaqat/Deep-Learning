######## TASK 1 #############
 #### Take manual input, use the model to predict and display the prediction result #######

import numpy as np
import pandas as pd
from sklearn.linear_model import Perceptron
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle


url = "https://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data"
df = pd.read_csv(url, header=None)
df = shuffle(df)


X = df.iloc[:, 0:4].values
y = df.iloc[:, 4].values


y = np.where(y == "Iris-setosa", 1, -1)


train_data, test_data, train_labels, test_labels = train_test_split(
    X, y, test_size=0.25
)
model = Perceptron(eta0=0.1, max_iter=10)
model.fit(train_data, train_labels)



print("\n--- Iris Flower Classification ---")
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
