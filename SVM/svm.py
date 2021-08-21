#import modules
import sklearn
from sklearn import datasets
from sklearn import svm
from sklearn import metrics
from sklearn.neighbors import KNeighborsClassifier

#load data
cancer = datasets.load_breast_cancer()

#features
x = cancer.data

#labels
y = cancer.target

#split data into training and testing
x_train, x_test, y_train, y_test = sklearn.model_selection.train_test_split(x, y, test_size=0.2)

#fit model (kernel can be linear, poly, rbf, sigmoid, precomputed)
clf = svm.SVC(kernel="linear")
clf.fit(x_train, y_train)

#calculate predicted values
y_pred = clf.predict(x_test)

#measure accuracy
accuracy = metrics.accuracy_score(y_test, y_pred)

print(accuracy)