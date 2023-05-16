import numpy as np
import tensorflow as tf
from tensorflow.keras import layers

from AbstractClassifier import AbstractClassifier
from data_utils import evaluate
from utils import plot_histogram_data, plot_confusion_matrix


class ANNSepiaClassifier(AbstractClassifier):

    def train_classifier(self, train_inputs, train_outputs):
        classifier = tf.keras.Sequential()
        classifier.add(layers.Flatten(input_shape=(128, 128, 3)))
        classifier.add(layers.Dense(128, activation='relu'))
        classifier.add(layers.Dense(64, activation='relu'))
        classifier.add(layers.Dense(32, activation='relu'))
        classifier.add(layers.Dense(1, activation='sigmoid'))

        classifier.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
        classifier.fit(train_inputs, train_outputs, epochs=100)

        return classifier

    def run_classifier(self, train_inputs, train_outputs, test_inputs, test_outputs):
        output_names = ["original", "sepia"]

        train_inputs = np.asarray(train_inputs)
        train_outputs = np.asarray(train_outputs)
        test_inputs = np.asarray(test_inputs)
        test_outputs = np.asarray(test_outputs)

        plot_histogram_data(train_outputs, output_names, 'original and sepia images')
        classifier = self.train_classifier(train_inputs, train_outputs)

        computed_outputs = classifier.predict(test_inputs)
        computed_outputs = np.round(computed_outputs)

        acc, precision, recall, conf_matrix = evaluate(test_outputs, computed_outputs, output_names)
        print("Accuracy: ", acc)
        print("Precision: ", precision)
        print("Recall: ", recall)
        plot_confusion_matrix(conf_matrix, output_names, "Sepia ANN classification")
