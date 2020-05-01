import pickle as pkl
import json
import numpy as np
from keras.preprocessing.text import Tokenizer
# from sklearn.linear_model import LogisticRegression
from keras.models import Sequential
from keras import layers
from utils import generate_results


if __name__ == '__main__':
    train_pos_path = "D:\\workspace\\vscode\\python\\Text-Classification-Pytorch\\data\\theguardian\\train_pos_clean_1154.pkl"
    train_neg_path = "D:\\workspace\\vscode\\python\\Text-Classification-Pytorch\\data\\theguardian\\guardian_clean_2000.pkl"
    dev_path = "D:\\workspace\\vscode\\python\\Text-Classification\\data\\dev_100.pkl"
    test_path = "D:\\workspace\\vscode\\python\\Text-Classification\\data\\test_1410.pkl"
    with open(train_pos_path, "rb") as f:
        train_pos = pkl.load(f)
    with open(train_neg_path, 'rb') as f:
        train_neg = pkl.load(f)
    with open(dev_path, "rb") as f:
        dev = pkl.load(f)
    with open(test_path, "rb") as f:
        test = pkl.load(f)

    X_train = train_pos + train_neg
    y_train = [1 for _ in range(len(train_pos))] + [0 for _ in range(len(train_neg))]

    X_dev = dev["text"]
    y_dev = dev["label"]
    print(len(X_dev), len(y_dev))
    X_test = test

    tokenizer = Tokenizer(oov_token="<UNK>")
    tokenizer.fit_on_texts(X_train)

    x_train = tokenizer.texts_to_matrix(X_train, mode="count") #BOW representation
    x_dev = tokenizer.texts_to_matrix(X_dev, mode="count") #BOW representation
    x_test = tokenizer.texts_to_matrix(X_test, mode="count") #BOW representation

    vocab_size = x_train.shape[1]
    print("Vocab size =", vocab_size)
    print(x_train[0])

    # #model definition
    model = Sequential(name="feedforward-bow-input")
    model.add(layers.Dense(10, input_dim=vocab_size, activation='relu'))
    model.add(layers.Dense(1, activation='sigmoid'))

    # #since it's a binary classification problem, we use a binary cross entropy loss here
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
    model.summary()

    # #training
    model.fit(x_train, y_train, epochs=20, verbose=True, validation_data=(x_dev, y_dev), batch_size=10)

    loss, accuracy = model.evaluate(x_dev, y_dev, verbose=False)
    print("\nTesting Accuracy:  {:.4f}".format(accuracy))

    predictions = model.predict(x_test)
    generate_results(predictions)
