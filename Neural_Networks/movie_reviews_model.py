from tensorflow import keras  # noqa


# load imdb data
data = keras.datasets.imdb

# split data into training and testing
(train_data, train_labels), (test_data, test_labels) = data.load_data(num_words=88000)  # noqa

# A dictionary mapping words to an integer index
_word_index = data.get_word_index()

word_index = {k: (v+3) for k, v in _word_index.items()}
word_index["<PAD>"] = 0
word_index["<START>"] = 1
word_index["<UNK>"] = 2  # unknown
word_index["<UNUSED>"] = 3

reverse_word_index = dict([(value, key) for (key, value) in word_index.items()])  # noqa

train_data = keras.preprocessing.sequence.pad_sequences(train_data,
                                                        value=word_index["<PAD>"],  # noqa
                                                        padding="post",
                                                        maxlen=250)
test_data = keras.preprocessing.sequence.pad_sequences(test_data,
                                                       value=word_index["<PAD>"],  # noqa
                                                       padding="post",
                                                       maxlen=250)


def decode_review(text):
    """Return the decoded (human readable) reviews."""
    return " ".join([reverse_word_index.get(i, "?") for i in text])


model = keras.Sequential()
model.add(keras.layers.Embedding(88000, 16))
model.add(keras.layers.GlobalAveragePooling1D())
model.add(keras.layers.Dense(16, activation="relu"))
model.add(keras.layers.Dense(1, activation="sigmoid"))  # final layer (0 or 1)

# prints a summary of the model
model.summary()

model.compile(optimizer="adam",
              loss="binary_crossentropy",
              metrics=["accuracy"])

# split training data into validation data
x_val = train_data[:10000]
x_train = train_data[10000:]
y_val = train_labels[:10000]
y_train = train_labels[10000:]

# fit model
fit_model = model.fit(x_train,
                      y_train,
                      epochs=40,
                      batch_size=512,  # how many reviews at once
                      validation_data=(x_val, y_val),
                      verbose=1)

results = model.evaluate(test_data, test_labels)
print(results)

model.save("model.h5")

# test_review = test_data[1]
# predict = model.predict([test_review])
# print("Review: ")
# print(decode_review(test_review))
# print("Prediction: " + str(predict[1]))
# print("Actual: " + str(test_labels[1]))
