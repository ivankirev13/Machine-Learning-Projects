from tensorflow import keras  # noqa

data = keras.datasets.imdb

model = keras.models.load_model("model.h5")

_word_index = data.get_word_index()

word_index = {k: (v+3) for k, v in _word_index.items()}
word_index["<PAD>"] = 0
word_index["<START>"] = 1
word_index["<UNK>"] = 2  # unknown
word_index["<UNUSED>"] = 3

reverse_word_index = dict([(val, key) for (key, val) in word_index.items()])


def review_encode(s):
    """
    Encode words into integers.

    Return 2 if work is not familiar.
    """
    encoded = [1]

    for word in s:
        if word.lower() in word_index:
            encoded.append(word_index[word.lower()])
        else:
            encoded.append(2)

    return encoded


# read in lion king review
with open("lion_king.txt", encoding="utf-8") as f:
    for line in f.readlines():
        nline = line.replace(",", "")
        nline = nline.replace(".", "")
        nline = nline.replace("(", "")
        nline = nline.replace(")", "")
        nline = nline.replace(":", "")
        nline = nline.replace("\"", "").strip().split(" ")
        encode = review_encode(nline)
        # make the data 250 words long
        encode = keras.preprocessing.sequence.pad_sequences(
                                                     [encode],
                                                     value=word_index["<PAD>"],
                                                     padding="post",
                                                     maxlen=250
                                                     )
        predict = model.predict(encode)
        print("Review: " + line)
        print("Encoding: " + encode)
        print("Prediction: " + predict[0])  # 0 is negative and 1 positive
