# import modules
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
import pandas as pd
from sklearn import preprocessing

# load data
data = pd.read_csv("car.data")

# encode each column of our data into integers
label_encoder = preprocessing.LabelEncoder()

# take each of our columns and return an array containing our new values
buying = label_encoder.fit_transform(list(data["buying"]))
maint = label_encoder.fit_transform(list(data["maint"]))
door = label_encoder.fit_transform(list(data["door"]))
persons = label_encoder.fit_transform(list(data["persons"]))
lug_boot = label_encoder.fit_transform(list(data["lug_boot"]))
safety = label_encoder.fit_transform(list(data["safety"]))
cls = label_encoder.fit_transform(list(data["class"]))

predict = "class"  # optional

# features
X = list(zip(buying, maint, door, persons, lug_boot, safety))

# labels
y = list(cls)

# split data into training and testing
x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.1)

# create KNN classifier and choose number of neighbors
model = KNeighborsClassifier(n_neighbors=9)

# train model
model.fit(x_train, y_train)

# compute the accuracy of the model
accuracy = model.score(x_test, y_test)
print(accuracy)

# use model to predict test data
predicted = model.predict(x_test)

for x in range(len(predicted)):
    print("Predicted:", predicted[x], "Data:", x_test[x], "Actual:", y_test[x])

    # see the neighbors of each point in our testing data
    n = model.kneighbors([x_test[x]], 9, True)
    # print("N: ", n)
