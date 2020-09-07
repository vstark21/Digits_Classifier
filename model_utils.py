import os
import numpy as np

def build_model():
    
    files = os.listdir()
    if "mnist_model.h5" not in files:
    
        from sklearn.model_selection import train_test_split
        import tensorflow as tf
        from tensorflow import keras
        from keras.datasets import mnist
        from keras.models import load_model
        import pandas as pd

        (feat_tkeras, lab_tkeras), (feat_val, lab_val) = mnist.load_data()

        feat_tkeras.shape, lab_tkeras.shape, feat_val.shape, lab_val.shape

        train_data = pd.read_csv("train.csv")

        lab_train = train_data["label"]
        feat_train = np.array(train_data.drop("label", axis=1)).reshape((-1, 28, 28, 1))

        features_train = np.concatenate((np.expand_dims(feat_tkeras, axis=-1), feat_train), axis=0)
        labels_train = np.concatenate((lab_tkeras, lab_train), axis=0)

        mnist_model = tf.keras.Sequential([
                        tf.keras.layers.Conv2D(32, (3, 3), input_shape=(28, 28, 1), activation="relu"),
                        tf.keras.layers.MaxPooling2D((2, 2)),
                        tf.keras.layers.Conv2D(64, (3, 3), activation="relu"),
                        tf.keras.layers.MaxPooling2D((2, 2)),
                        tf.keras.layers.Conv2D(128, (3, 3), activation="relu"),
                        tf.keras.layers.MaxPooling2D((2, 2)),
                        tf.keras.layers.Flatten(),
                        tf.keras.layers.Dense(512, activation="relu"),
                        tf.keras.layers.Dropout(0.3),
                        tf.keras.layers.Dense(256, activation="relu"),
                        tf.keras.layers.Dropout(0.2),
                        tf.keras.layers.Dense(128, activation="relu"),
                        tf.keras.layers.Dropout(0.1),
                        tf.keras.layers.Dense(10, activation="softmax")
        ])

        mnist_model.compile(optimizer=tf.keras.optimizers.SGD(), loss="sparse_categorical_crossentropy", metrics=["accuracy"])
        mnist_model.summary()

        mnist_model.fit(features_train, labels_train, epochs=69, validation_data=(np.expand_dims(feat_val, axis=-1), lab_val))

        mnist_model.save("mnist_model.h5")

    else:
        from tensorflow.keras.models import load_model
        mnist_model = load_model("mnist_model.h5")

    return mnist_model

def predict(mnist_model, images):

    return np.argmax(mnist_model.predict(images), axis=1)

