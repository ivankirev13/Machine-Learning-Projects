from tensorflow import keras # noqa
import matplotlib.pyplot as plt
import numpy as np

# load data fashion mnist
data = keras.datasets.fashion_mnist

# split data into training and testing parts
(train_images, train_labels), (test_images, test_labels) = data.load_data()

# create labels
class_names = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat',
               'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']

# normalize data
train_images = train_images/255.0
test_images = test_images/255.0

# flatten the data (get a list instead of a 28x28 matrix)
model = keras.Sequential([
    keras.layers.Flatten(input_shape=(28, 28)),  # first layer
    keras.layers.Dense(128, activation="relu"),  # second layer
    keras.layers.Dense(10, activation="softmax")  # output layer
    ])

model.compile(optimizer="adam",
              loss="sparse_categorical_crossentropy",
              metrics=["accuracy"])

# train model
model.fit(train_images, train_labels, epochs=5)

# test model
test_loss, test_acc = model.evaluate(test_images, test_labels)
print('\nTest accuracy:', test_acc)  # print accuracy

predictions = model.predict(test_images)

# print first 5 pictures
plt.figure(figsize=(5, 5))
for i in range(5):
    plt.grid(False)
    plt.imshow(test_images[i], cmap=plt.cm.binary)
    # actual label
    plt.xlabel("Actual: " + class_names[test_labels[i]])
    # predicted label
    plt.title("Prediction: " + class_names[np.argmax(predictions[i])])
    plt.show()
